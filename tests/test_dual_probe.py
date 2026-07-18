from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch as t

from methylation_latent.domain import (
    NonEmptyProbeSet,
    ProbeLocus,
    parse_fraction,
    parse_latent_dimension,
    parse_non_negative_weight,
    parse_positive_int,
)
from methylation_latent.dual_probe import (
    AlphaSchedule,
    AlphaScheduleMode,
    DualInitialization,
    DualProbeLatent,
    DualTrainingConfig,
    DualValidationData,
    LeastSquaresAudit,
    dual_probe_objective,
    normal_equation_projection,
    refit_dual_probe,
    tune_dual_probe,
)
from methylation_latent.dual_probe_experiment import (
    hybrid_seen_by_held_out_predictions,
)
from methylation_latent.dual_probe_protocol import load_dual_probe_protocol
from methylation_latent.evaluation import PairIndices, PairPopulation
from methylation_latent.model import normalize_rows_strict, normalize_vector_strict
from methylation_latent.storage import EmbeddingMatrix
from methylation_latent.targets import (
    CorrelationMatrix,
    CorrelationVector,
    TargetGeometry,
    build_target_geometry,
)


def _fixed_alpha(value: float) -> AlphaSchedule:
    alpha = parse_fraction(value)
    return AlphaSchedule(f"fixed-{value:g}", AlphaScheduleMode.FIXED, alpha, alpha)


def test_alpha_schedules_have_exact_endpoints_and_reject_invalid_state() -> None:
    upward = AlphaSchedule(
        "linear_0_to_1",
        AlphaScheduleMode.LINEAR,
        parse_fraction(0.0),
        parse_fraction(1.0),
    )
    assert float(upward.value(1, 5)) == 0.0
    assert float(upward.value(3, 5)) == 0.5
    assert float(upward.value(5, 5)) == 1.0
    assert float(_fixed_alpha(0.25).value(4, 5)) == 0.25
    with pytest.raises(ValueError, match="equal endpoints"):
        AlphaSchedule(
            "bad",
            AlphaScheduleMode.FIXED,
            parse_fraction(0.0),
            parse_fraction(1.0),
        )
    with pytest.raises(ValueError, match="distinct endpoints"):
        AlphaSchedule(
            "bad",
            AlphaScheduleMode.LINEAR,
            parse_fraction(0.5),
            parse_fraction(0.5),
        )
    with pytest.raises(ValueError, match="step"):
        upward.value(0, 5)


def test_normal_equation_projection_satisfies_ols_stationarity() -> None:
    generator = t.Generator().manual_seed(14)
    embeddings = t.randn((40, 4), generator=generator)
    learned = normalize_rows_strict(t.randn((40, 3), generator=generator))
    weight, audit = normal_equation_projection(
        embeddings,
        learned,
        condition_ceiling=1.0e6,
        relative_residual_ceiling=1.0e-10,
        discarded_cross_moment_ceiling=1.0,
    )
    residual = embeddings @ weight.mT - learned
    stationarity = embeddings.mT @ residual
    assert weight.shape == (3, 4)
    assert (
        t.linalg.vector_norm(stationarity) / t.linalg.vector_norm(embeddings.mT @ learned) < 2.0e-6
    )
    assert audit.fit_row_count == 40
    assert audit.relative_retained_normal_residual < 1.0e-10
    assert audit.effective_rank == 4
    assert -1.0 <= audit.initial_mean_cosine <= 1.0
    with pytest.raises(ValueError, match="positive definite"):
        normal_equation_projection(
            t.ones((8, 4)),
            normalize_rows_strict(t.randn((8, 2), generator=generator)),
            condition_ceiling=1.0e6,
            relative_residual_ceiling=1.0e-6,
            discarded_cross_moment_ceiling=1.0,
        )


def test_normal_equation_projection_truncates_only_ill_conditioned_directions() -> None:
    generator = t.Generator().manual_seed(141)
    embeddings = t.randn((80, 3), generator=generator)
    embeddings[:, 2] *= 1.0e-5
    learned = normalize_rows_strict(t.randn((80, 2), generator=generator))
    _, audit = normal_equation_projection(
        embeddings,
        learned,
        condition_ceiling=1.0e6,
        relative_residual_ceiling=1.0e-10,
        discarded_cross_moment_ceiling=1.0,
    )
    assert audit.raw_condition_number > 1.0e6
    assert audit.retained_condition_number <= 1.0e6
    assert audit.effective_rank == 2
    assert audit.discarded_spectral_mass_fraction < 1.0e-8


def _zero_geometry_catch_gradients(alpha: float) -> tuple[t.Tensor, t.Tensor]:
    learned_raw = t.tensor([[1.0, 0.2], [0.1, 1.0]], requires_grad=True)
    sequence_raw = t.tensor([[0.3, 1.0], [1.0, -0.2]], requires_grad=True)
    age_direction = t.tensor([0.4, 0.8], requires_grad=True)
    learned = normalize_rows_strict(learned_raw)
    sequence = normalize_rows_strict(sequence_raw)
    age_target = learned.detach() @ normalize_vector_strict(age_direction.detach())
    terms = dual_probe_objective(
        learned,
        sequence,
        age_direction,
        CorrelationMatrix(learned.detach() @ learned.detach().mT),
        CorrelationVector(age_target),
        lambda_age=parse_non_negative_weight(1.0),
        catch_weight=parse_non_negative_weight(1.0),
        alpha=parse_fraction(alpha),
    )
    terms.total.backward()
    if learned_raw.grad is None or sequence_raw.grad is None:
        raise AssertionError("routed catch objective omitted an expected gradient tensor")
    return learned_raw.grad, sequence_raw.grad


def test_alpha_routes_catch_gradient_to_exact_branch() -> None:
    learned_at_zero, sequence_at_zero = _zero_geometry_catch_gradients(0.0)
    assert float(t.linalg.vector_norm(learned_at_zero).item()) < 1.0e-7
    assert float(t.linalg.vector_norm(sequence_at_zero).item()) > 1.0e-3
    learned_at_one, sequence_at_one = _zero_geometry_catch_gradients(1.0)
    assert float(t.linalg.vector_norm(learned_at_one).item()) > 1.0e-3
    assert float(t.linalg.vector_norm(sequence_at_one).item()) < 1.0e-7


def _training_fixture(
    make_probe: Callable[..., ProbeLocus],
) -> tuple[
    NonEmptyProbeSet,
    EmbeddingMatrix,
    TargetGeometry,
    DualValidationData,
    DualInitialization,
]:
    fit_count = 260
    sample_count = 5
    generator = t.Generator().manual_seed(22)
    probes = NonEmptyProbeSet(
        tuple(
            make_probe(index + 1, chromosome=1, position=10_000 + 100 * index)
            for index in range(fit_count)
        )
    )
    validation_probes = NonEmptyProbeSet(
        tuple(
            make_probe(
                fit_count + index + 1,
                chromosome=2,
                position=10_000 + 100 * index,
            )
            for index in range(4)
        )
    )
    embeddings = EmbeddingMatrix(t.randn((fit_count, 256), generator=generator, dtype=t.float16))
    validation_embeddings = EmbeddingMatrix(t.randn((4, 256), generator=generator, dtype=t.float16))
    targets = build_target_geometry(
        t.rand((fit_count, sample_count), generator=generator, dtype=t.float64),
        t.linspace(20.0, 80.0, sample_count, dtype=t.float64),
    )
    validation_targets = build_target_geometry(
        t.rand((4, sample_count), generator=generator, dtype=t.float64),
        t.linspace(20.0, 80.0, sample_count, dtype=t.float64),
    )
    learned = normalize_rows_strict(t.randn((fit_count, 2), generator=generator, dtype=t.float32))
    projection = t.randn((2, 256), generator=generator, dtype=t.float32) * 0.01
    age_direction = t.randn((2,), generator=generator, dtype=t.float32)
    initialization = DualInitialization(
        learned,
        projection,
        age_direction,
        LeastSquaresAudit(
            fit_row_count=fit_count,
            embedding_dimension=256,
            latent_dimension=2,
            effective_rank=256,
            raw_condition_number=2.0,
            retained_condition_number=2.0,
            discarded_spectral_mass_fraction=0.0,
            discarded_cross_moment_fraction=0.0,
            relative_retained_normal_residual=1.0e-12,
            initial_mean_cosine=0.1,
        ),
    )
    return (
        probes,
        embeddings,
        targets,
        DualValidationData(validation_probes, validation_embeddings, validation_targets),
        initialization,
    )


def _training_config(alpha: float) -> DualTrainingConfig:
    return DualTrainingConfig(
        latent_dimension=parse_latent_dimension(2),
        lambda_age=parse_non_negative_weight(0.1),
        catch_weight=parse_non_negative_weight(0.1),
        alpha_schedule=_fixed_alpha(alpha),
        batch_size=parse_positive_int(4),
        neighbourhood_width=parse_positive_int(100_000),
        steps=parse_positive_int(2),
        validation_interval=parse_positive_int(1),
        validation_pair_chunk_size=parse_positive_int(2),
        learning_rate=1.0e-3,
        seed=31,
        device="cpu",
    )


def test_alpha_one_refit_leaves_sequence_projection_exactly_fixed(
    make_probe: Callable[..., ProbeLocus],
) -> None:
    probes, embeddings, targets, _, initialization = _training_fixture(make_probe)
    refit = refit_dual_probe(
        probes,
        embeddings,
        targets,
        initialization,
        config=_training_config(1.0),
        selected_steps=2,
    )
    assert t.equal(refit.model.projection.weight.detach(), initialization.projection_weight)
    moving = refit_dual_probe(
        probes,
        embeddings,
        targets,
        initialization,
        config=_training_config(0.0),
        selected_steps=2,
    )
    assert not t.equal(moving.model.projection.weight.detach(), initialization.projection_weight)


def test_tuning_has_no_learned_validation_table_and_rejects_overlap(
    make_probe: Callable[..., ProbeLocus],
) -> None:
    probes, embeddings, targets, validation, initialization = _training_fixture(make_probe)
    result = tune_dual_probe(
        probes,
        embeddings,
        targets,
        validation,
        initialization,
        config=_training_config(0.0),
    )
    assert tuple(record.step for record in result.validation_history) == (0, 1, 2)
    assert result.selected_step in {0, 1, 2}
    assert DualProbeLatent(initialization).fit_probe_count == len(probes)
    overlapping = DualValidationData(
        NonEmptyProbeSet(probes.probes[:4]),
        validation.embeddings,
        validation.targets,
    )
    with pytest.raises(ValueError, match="must be disjoint"):
        tune_dual_probe(
            probes,
            embeddings,
            targets,
            overlapping,
            initialization,
            config=_training_config(0.0),
        )


def test_hybrid_prediction_requires_learned_seen_and_unlearned_held_out() -> None:
    learned = normalize_rows_strict(t.tensor([[1.0, 1.0], [1.0, -1.0]]))
    sequence = normalize_rows_strict(t.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 2.0], [-2.0, 1.0]]))
    pairs = PairIndices(
        PairPopulation.SEEN_BY_HELD_OUT,
        left=t.tensor([0, 1], dtype=t.int64),
        right=t.tensor([2, 3], dtype=t.int64),
        distance_class=t.tensor([7, 7], dtype=t.int64),
        total_possible_pairs=4,
        seed=9,
    )
    prediction = hybrid_seen_by_held_out_predictions(
        learned,
        t.tensor([0, 1], dtype=t.int64),
        sequence,
        pairs,
        chunk_size=1,
    )
    expected = t.tensor([3.0, -3.0]) / t.sqrt(t.tensor(10.0))
    assert t.allclose(prediction, expected)
    with pytest.raises(ValueError, match="held-out side"):
        hybrid_seen_by_held_out_predictions(
            t.cat((learned, sequence[2:3])),
            t.tensor([0, 1, 2], dtype=t.int64),
            sequence,
            pairs,
            chunk_size=1,
        )


def test_frozen_dual_protocol_parses_complete_candidate_grid() -> None:
    config = load_dual_probe_protocol(Path("configs/dual-probe-latent-v1.toml"))
    assert tuple(schedule.name for schedule in config.alpha_schedules()) == (
        "fixed_0",
        "fixed_0.25",
        "fixed_0.5",
        "fixed_0.75",
        "fixed_1",
        "linear_0_to_1",
        "linear_1_to_0",
    )
    assert config.parent("held-out-chromosome", 4096).latent_dimension == 256
