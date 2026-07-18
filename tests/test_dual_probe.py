from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
import torch as t
from jaxtyping import TypeCheckError

import methylation_latent.dual_probe as dual_probe
import methylation_latent.dual_probe_protocol as dual_protocol
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
    DualObjectiveTerms,
    DualProbeLatent,
    DualRefitResult,
    DualTrainingConfig,
    DualTrainingStep,
    DualTuningResult,
    DualValidationData,
    DualValidationStep,
    LeastSquaresAudit,
    build_dual_initialization,
    dual_probe_objective,
    normal_equation_projection,
    refit_dual_probe,
    tune_dual_probe,
)
from methylation_latent.dual_probe_experiment import (
    assert_dual_split_roles,
    held_out_age_predictions,
    hybrid_seen_by_held_out_predictions,
    sequence_latent_from_projection,
    subset_dual_partition,
)
from methylation_latent.dual_probe_protocol import (
    load_dual_probe_protocol,
)
from methylation_latent.evaluation import PairIndices, PairPopulation
from methylation_latent.experiment_data import SplitArtifact
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


def _small_audit() -> LeastSquaresAudit:
    return LeastSquaresAudit(
        fit_row_count=2,
        embedding_dimension=2,
        latent_dimension=2,
        effective_rank=2,
        raw_condition_number=2.0,
        retained_condition_number=2.0,
        discarded_spectral_mass_fraction=0.0,
        discarded_cross_moment_fraction=0.0,
        relative_retained_normal_residual=0.0,
        initial_mean_cosine=0.5,
    )


def _small_initialization() -> DualInitialization:
    return DualInitialization(
        learned_vectors=t.eye(2, dtype=t.float32),
        projection_weight=t.eye(2, dtype=t.float32),
        age_direction=t.tensor((1.0, 0.0), dtype=t.float32),
        least_squares=_small_audit(),
    )


def _training_step(step: int = 1) -> DualTrainingStep:
    return DualTrainingStep(
        step=step,
        alpha=0.5,
        total_loss=1.0,
        geometry_total=0.8,
        pair_mse=0.5,
        age_mse=0.3,
        catch=0.2,
        mean_alignment=0.4,
    )


def _validation_step(step: int = 0) -> DualValidationStep:
    return DualValidationStep(step=step, selection_score=0.8, pair_mse=0.5, age_mse=0.3)


def test_dual_scalar_and_least_squares_audits_reject_drift() -> None:
    dual_probe._require_finite_float(t.ones(2), "values")
    with pytest.raises(TypeError, match="floating"):
        dual_probe._require_finite_float(t.ones(2, dtype=t.int64), "values")
    with pytest.raises(ValueError, match="finite"):
        dual_probe._require_finite_float(t.tensor((1.0, float("nan"))), "values")
    valid = _small_audit()
    with pytest.raises(ValueError, match="dimensions"):
        replace(valid, fit_row_count=1)
    with pytest.raises(ValueError, match="finite"):
        replace(valid, raw_condition_number=float("inf"))
    with pytest.raises(ValueError, match="effective rank"):
        replace(valid, effective_rank=0)
    with pytest.raises(ValueError, match="condition numbers"):
        replace(valid, raw_condition_number=1.0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        replace(valid, discarded_cross_moment_fraction=1.1)
    with pytest.raises(ValueError, match="non-negative"):
        replace(valid, relative_retained_normal_residual=-0.1)
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        replace(valid, initial_mean_cosine=1.1)


def test_normal_equation_projection_rejects_axes_dtype_threshold_and_cross_moment() -> None:
    thresholds = {
        "condition_ceiling": 1.0e6,
        "relative_residual_ceiling": 1.0e-8,
        "discarded_cross_moment_ceiling": 1.0,
    }
    with pytest.raises(ValueError, match="non-empty matrices"):
        normal_equation_projection(t.empty((0, 2)), t.empty((0, 1)), **thresholds)
    with pytest.raises(ValueError, match="at least as many rows"):
        normal_equation_projection(t.ones((2, 3)), t.ones((2, 1)), **thresholds)
    with pytest.raises(TypeError, match="share a dtype"):
        normal_equation_projection(t.ones((3, 2)), t.ones((3, 1), dtype=t.float64), **thresholds)
    with pytest.raises(ValueError, match="thresholds"):
        normal_equation_projection(
            t.eye(2),
            t.eye(2),
            condition_ceiling=0.5,
            relative_residual_ceiling=1.0e-8,
            discarded_cross_moment_ceiling=1.0,
        )
    with pytest.raises(ValueError, match="cross moment"):
        normal_equation_projection(t.eye(2), t.zeros((2, 1)), **thresholds)
    generator = t.Generator().manual_seed(91)
    embeddings = t.randn((80, 3), generator=generator)
    embeddings[:, 2] *= 1.0e-5
    learned = normalize_rows_strict(t.randn((80, 2), generator=generator))
    with pytest.raises(ValueError, match="discarded cross-moment"):
        normal_equation_projection(
            embeddings,
            learned,
            condition_ceiling=1.0e6,
            relative_residual_ceiling=1.0e-8,
            discarded_cross_moment_ceiling=1.0e-20,
        )


def test_dual_initialization_tensor_contracts_reject_drift() -> None:
    valid = _small_initialization()
    with pytest.raises(TypeError, match="float32"):
        replace(valid, age_direction=valid.age_direction.double())
    with pytest.raises(ValueError, match="finite"):
        replace(valid, age_direction=t.tensor((1.0, float("nan"))))
    with pytest.raises(ValueError, match="rank two"):
        replace(valid, learned_vectors=t.ones(2))
    with pytest.raises(ValueError, match="rank one"):
        replace(valid, age_direction=t.ones((1, 2)))
    with pytest.raises(ValueError, match="audit dimensions"):
        replace(valid, projection_weight=t.ones((2, 3)))
    with pytest.raises(ValueError, match="unit norm"):
        replace(valid, learned_vectors=t.ones((2, 2)))
    with pytest.raises(ValueError, match="must not be zero"):
        replace(valid, age_direction=t.zeros(2))


def test_build_dual_initialization_covers_cpu_path_and_rejects_device_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embeddings = EmbeddingMatrix(
        t.randn((280, 256), generator=t.Generator().manual_seed(92), dtype=t.float16)
    )
    initialization = build_dual_initialization(
        embeddings,
        latent_dimension=parse_latent_dimension(2),
        seed=93,
        device="cpu",
        condition_ceiling=1.0e8,
        relative_residual_ceiling=1.0e-5,
        discarded_cross_moment_ceiling=0.005,
    )
    assert initialization.learned_vectors.shape == (280, 2)
    assert initialization.projection_weight.shape == (2, 256)
    with pytest.raises(ValueError, match="cannot exceed"):
        build_dual_initialization(
            embeddings,
            latent_dimension=parse_latent_dimension(257),
            seed=93,
            device="cpu",
            condition_ceiling=1.0e8,
            relative_residual_ceiling=1.0e-5,
            discarded_cross_moment_ceiling=0.005,
        )
    monkeypatch.setattr(t.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA initialization"):
        build_dual_initialization(
            embeddings,
            latent_dimension=parse_latent_dimension(2),
            seed=93,
            device="cuda",
            condition_ceiling=1.0e8,
            relative_residual_ceiling=1.0e-5,
            discarded_cross_moment_ceiling=0.005,
        )


def test_dual_model_methods_and_objective_terms_reject_invalid_axes() -> None:
    model = DualProbeLatent(_small_initialization())
    assert model.all_learned().shape == (2, 2)
    assert model.age(model.all_learned()).shape == (2,)
    with pytest.raises(ValueError, match="non-empty vector"):
        model.learned(t.empty(0, dtype=t.int64))
    with pytest.raises(IndexError, match="outside"):
        model.learned(t.tensor((2,), dtype=t.int64))
    with pytest.raises(ValueError, match="non-empty matrix"):
        model.sequence(t.empty((0, 2)))
    with pytest.raises(ValueError, match="width differs"):
        model.sequence(t.ones((1, 3)))
    with pytest.raises(TypeError, match="share a dtype"):
        model.sequence(t.ones((1, 2), dtype=t.float64))
    with pytest.raises(ValueError, match="finite"):
        model.sequence(t.tensor(((1.0, float("nan")),)))
    scalar = t.tensor(0.0)
    terms = DualObjectiveTerms(*(scalar for _ in range(8)))
    assert terms.total.ndim == 0
    with pytest.raises(ValueError, match="floating scalar"):
        replace(terms, total=t.ones(1))
    with pytest.raises(TypeCheckError, match="sequence_latent"):
        dual_probe_objective(
            t.eye(2),
            t.ones((3, 2)),
            t.ones(2),
            CorrelationMatrix(t.eye(2)),
            CorrelationVector(t.zeros(2)),
            lambda_age=parse_non_negative_weight(0.1),
            catch_weight=parse_non_negative_weight(0.1),
            alpha=parse_fraction(0.5),
        )


def test_dual_training_result_dataclasses_reject_invalid_state() -> None:
    config = _training_config(0.5)
    with pytest.raises(ValueError, match="learning rate"):
        replace(config, learning_rate=0.0)
    with pytest.raises(ValueError, match="interval"):
        replace(config, validation_interval=parse_positive_int(3))
    with pytest.raises(ValueError, match="positive catch"):
        replace(config, catch_weight=parse_non_negative_weight(0.0))
    training = _training_step()
    validation_zero = _validation_step()
    validation_one = _validation_step(1)
    with pytest.raises(ValueError, match="step or alpha"):
        replace(training, step=0)
    with pytest.raises(ValueError, match="finite and non-negative"):
        replace(training, pair_mse=-0.1)
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        replace(training, mean_alignment=2.0)
    with pytest.raises(ValueError, match="cannot be negative"):
        replace(validation_zero, step=-1)
    with pytest.raises(ValueError, match="finite and non-negative"):
        replace(validation_zero, age_mse=float("nan"))
    tuning = DualTuningResult((training,), (validation_zero, validation_one), selected_step=1)
    assert tuning.selected_step == 1
    with pytest.raises(ValueError, match="non-empty"):
        replace(tuning, training_history=())
    with pytest.raises(ValueError, match="unique and increasing"):
        replace(tuning, validation_history=(validation_one, validation_zero))
    with pytest.raises(ValueError, match="absent"):
        replace(tuning, selected_step=2)
    model = DualProbeLatent(_small_initialization())
    zero_refit = DualRefitResult(model, (), 0)
    with pytest.raises(ValueError, match="cannot be negative"):
        replace(zero_refit, refit_steps=-1)
    with pytest.raises(ValueError, match="empty history"):
        replace(zero_refit, training_history=(training,))
    with pytest.raises(ValueError, match="end at"):
        DualRefitResult(model, (_training_step(1),), 2)
    with pytest.raises(ValueError, match="unique and increasing"):
        DualRefitResult(model, (_training_step(2), _training_step(1)), 1)
    with pytest.raises(ValueError, match="learning rate"):
        dual_probe._make_optimizers(model, 0.0)


def test_dual_axis_validators_reject_mismatches(
    make_probe: Callable[..., ProbeLocus],
) -> None:
    probes, embeddings, targets, _, initialization = _training_fixture(make_probe)
    with pytest.raises(ValueError, match="axes must align"):
        dual_probe._validate_axes(
            NonEmptyProbeSet(probes.probes[:-1]),
            embeddings,
            targets,
            initialization,
        )
    with pytest.raises(ValueError, match="row axis"):
        short_initialization = DualInitialization(
            learned_vectors=initialization.learned_vectors[:-1],
            projection_weight=initialization.projection_weight,
            age_direction=initialization.age_direction,
            least_squares=replace(initialization.least_squares, fit_row_count=259),
        )
        dual_probe._validate_axes(
            probes,
            embeddings,
            targets,
            short_initialization,
        )
    with pytest.raises(ValueError, match="embedding width"):
        audit = replace(initialization.least_squares, embedding_dimension=255, effective_rank=255)
        projection = initialization.projection_weight[:, :255]
        narrow_initialization = DualInitialization(
            learned_vectors=initialization.learned_vectors,
            projection_weight=projection,
            age_direction=initialization.age_direction,
            least_squares=audit,
        )
        dual_probe._validate_axes(
            probes,
            embeddings,
            targets,
            narrow_initialization,
        )
    with pytest.raises(ValueError, match="latent dimension"):
        dual_probe._validate_config_initialization(
            replace(_training_config(0.5), latent_dimension=parse_latent_dimension(3)),
            initialization,
        )


def test_tune_and_refit_reject_partition_budget_and_device_drift(
    make_probe: Callable[..., ProbeLocus],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probes, embeddings, targets, validation, initialization = _training_fixture(make_probe)
    mismatched_targets = build_target_geometry(
        t.rand((4, 6), generator=t.Generator().manual_seed(97), dtype=t.float64),
        t.linspace(20.0, 80.0, 6, dtype=t.float64),
    )
    with pytest.raises(ValueError, match="sample axes"):
        tune_dual_probe(
            probes,
            embeddings,
            targets,
            DualValidationData(validation.probes, validation.embeddings, mismatched_targets),
            initialization,
            config=_training_config(0.5),
        )
    too_large = replace(_training_config(0.5), batch_size=parse_positive_int(len(probes) + 1))
    with pytest.raises(ValueError, match="batch size"):
        tune_dual_probe(
            probes,
            embeddings,
            targets,
            validation,
            initialization,
            config=too_large,
        )
    with pytest.raises(ValueError, match="refit steps"):
        refit_dual_probe(
            probes,
            embeddings,
            targets,
            initialization,
            config=_training_config(0.5),
            selected_steps=3,
        )
    with pytest.raises(ValueError, match="batch size"):
        refit_dual_probe(
            probes,
            embeddings,
            targets,
            initialization,
            config=too_large,
            selected_steps=1,
        )
    monkeypatch.setattr(t.cuda, "is_available", lambda: False)
    cuda_config = replace(_training_config(0.5), device="cuda")
    with pytest.raises(ValueError, match="CUDA dual tuning"):
        tune_dual_probe(
            probes,
            embeddings,
            targets,
            validation,
            initialization,
            config=cuda_config,
        )
    with pytest.raises(ValueError, match="CUDA dual refit"):
        refit_dual_probe(
            probes,
            embeddings,
            targets,
            initialization,
            config=cuda_config,
            selected_steps=1,
        )


def _small_split() -> SplitArtifact:
    return SplitArtifact(
        name="split",
        universe_size=5,
        optimization_indices=t.tensor((0,), dtype=t.int64),
        validation_indices=t.tensor((1,), dtype=t.int64),
        test_indices=t.tensor((2,), dtype=t.int64),
        primary_buffer_indices=t.tensor((3,), dtype=t.int64),
        validation_buffer_indices=t.tensor((4,), dtype=t.int64),
    )


def test_dual_partition_subset_and_split_role_contracts(
    make_probe: Callable[..., ProbeLocus],
) -> None:
    probes, embeddings, targets, _, _ = _training_fixture(make_probe)
    subset = subset_dual_partition(
        probes,
        embeddings,
        targets,
        t.tensor((0, 2), dtype=t.int64),
    )
    assert subset.global_indices.tolist() == [0, 2]
    with pytest.raises(TypeError, match="non-empty int64"):
        replace(subset, global_indices=t.tensor((0.0, 2.0)))
    with pytest.raises(ValueError, match="unique"):
        replace(subset, global_indices=t.tensor((0, 0), dtype=t.int64))
    with pytest.raises(ValueError, match="axes differ"):
        replace(subset, global_indices=t.tensor((0,), dtype=t.int64))
    with pytest.raises(ValueError, match="different probe axes"):
        subset_dual_partition(
            NonEmptyProbeSet(probes.probes[:-1]), embeddings, targets, t.tensor((0,), dtype=t.int64)
        )
    with pytest.raises(TypeError, match="non-empty int64"):
        subset_dual_partition(probes, embeddings, targets, t.tensor((0.0,)))
    with pytest.raises(IndexError, match="outside"):
        subset_dual_partition(probes, embeddings, targets, t.tensor((len(probes),), dtype=t.int64))
    with pytest.raises(ValueError, match="unique"):
        subset_dual_partition(probes, embeddings, targets, t.tensor((0, 0), dtype=t.int64))
    split = _small_split()
    assert_dual_split_roles(split)
    object.__setattr__(split, "optimization_indices", split.test_indices.clone())
    with pytest.raises(ValueError, match="contains test"):
        assert_dual_split_roles(split)
    split = _small_split()
    object.__setattr__(split, "optimization_indices", split.validation_indices.clone())
    with pytest.raises(ValueError, match="overlap"):
        assert_dual_split_roles(split)


def test_sequence_projection_and_age_prediction_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embeddings = EmbeddingMatrix(
        t.randn((4, 256), generator=t.Generator().manual_seed(98), dtype=t.float16)
    )
    weight = t.randn((2, 256), generator=t.Generator().manual_seed(99), dtype=t.float32)
    latent = sequence_latent_from_projection(embeddings, weight, device="cpu", row_chunk_size=2)
    assert latent.shape == (4, 2)
    assert t.allclose(t.linalg.vector_norm(latent, dim=1), t.ones(4), atol=2.0e-5, rtol=0.0)
    with pytest.raises(ValueError, match="not aligned"):
        sequence_latent_from_projection(embeddings, weight[:, :255], device="cpu", row_chunk_size=2)
    with pytest.raises(ValueError, match="finite"):
        invalid = weight.clone()
        invalid[0, 0] = float("nan")
        sequence_latent_from_projection(embeddings, invalid, device="cpu", row_chunk_size=2)
    with pytest.raises(ValueError, match="positive"):
        sequence_latent_from_projection(embeddings, weight, device="cpu", row_chunk_size=0)
    monkeypatch.setattr(t.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA sequence"):
        sequence_latent_from_projection(embeddings, weight, device="cuda", row_chunk_size=2)
    age_direction = t.tensor((1.0, 1.0), dtype=t.float32)
    prediction = held_out_age_predictions(latent, age_direction, t.tensor((1, 3), dtype=t.int64))
    assert prediction.shape == (2,)
    with pytest.raises(TypeError, match="float32 matrix"):
        held_out_age_predictions(latent.double(), age_direction, t.tensor((1,), dtype=t.int64))
    with pytest.raises(TypeError, match="aligned float32"):
        held_out_age_predictions(latent, t.ones(3), t.tensor((1,), dtype=t.int64))
    with pytest.raises(TypeError, match="int64"):
        held_out_age_predictions(latent, age_direction, t.tensor((1.0,)))
    with pytest.raises(IndexError, match="outside"):
        held_out_age_predictions(latent, age_direction, t.tensor((4,), dtype=t.int64))


def _hybrid_fixture() -> tuple[t.Tensor, t.Tensor, t.Tensor, PairIndices]:
    learned = normalize_rows_strict(t.tensor(((1.0, 1.0), (1.0, -1.0))))
    sequence = normalize_rows_strict(t.tensor(((1.0, 0.0), (0.0, 1.0), (1.0, 2.0), (-2.0, 1.0))))
    indices = t.tensor((0, 1), dtype=t.int64)
    pairs = PairIndices(
        PairPopulation.SEEN_BY_HELD_OUT,
        left=t.tensor((0, 1), dtype=t.int64),
        right=t.tensor((2, 3), dtype=t.int64),
        distance_class=t.tensor((7, 7), dtype=t.int64),
        total_possible_pairs=4,
        seed=9,
    )
    return learned, indices, sequence, pairs


def test_hybrid_predictions_reject_population_axis_type_norm_and_lookup_drift() -> None:
    learned, indices, sequence, pairs = _hybrid_fixture()
    held_pairs = PairIndices(
        PairPopulation.HELD_OUT_BY_HELD_OUT,
        left=t.tensor((2,), dtype=t.int64),
        right=t.tensor((3,), dtype=t.int64),
        distance_class=t.tensor((7,), dtype=t.int64),
        total_possible_pairs=1,
        seed=9,
    )
    with pytest.raises(ValueError, match="seen-by-held-out"):
        hybrid_seen_by_held_out_predictions(learned, indices, sequence, held_pairs, chunk_size=1)
    with pytest.raises(ValueError, match="chunk size"):
        hybrid_seen_by_held_out_predictions(learned, indices, sequence, pairs, chunk_size=0)
    with pytest.raises(ValueError, match="latent axes"):
        hybrid_seen_by_held_out_predictions(learned[:, :1], indices, sequence, pairs, chunk_size=1)
    with pytest.raises(ValueError, match="global indices"):
        hybrid_seen_by_held_out_predictions(learned, indices.float(), sequence, pairs, chunk_size=1)
    with pytest.raises(TypeError, match="float32"):
        hybrid_seen_by_held_out_predictions(
            learned.double(), indices, sequence.double(), pairs, chunk_size=1
        )
    with pytest.raises(ValueError, match="finite"):
        invalid = learned.clone()
        invalid[0, 0] = float("nan")
        hybrid_seen_by_held_out_predictions(invalid, indices, sequence, pairs, chunk_size=1)
    with pytest.raises(ValueError, match="unit norm"):
        hybrid_seen_by_held_out_predictions(learned * 2.0, indices, sequence, pairs, chunk_size=1)
    with pytest.raises(IndexError, match="outside"):
        hybrid_seen_by_held_out_predictions(
            learned, t.tensor((0, 4), dtype=t.int64), sequence, pairs, chunk_size=1
        )
    with pytest.raises(ValueError, match="seen-side"):
        hybrid_seen_by_held_out_predictions(learned[:1], indices[:1], sequence, pairs, chunk_size=1)


def test_dual_protocol_helpers_dataclasses_and_lookup_reject_drift() -> None:
    config = load_dual_probe_protocol(Path("configs/dual-probe-latent-v1.toml"))
    with pytest.raises(ValueError, match="fields differ"):
        dual_protocol._exact_keys({"a": 1}, {"b"}, "record")
    with pytest.raises(TypeError, match="must be a table"):
        dual_protocol._table({"table": 1}, "table")
    with pytest.raises(TypeError, match="non-empty string"):
        dual_protocol._string("", "value")
    with pytest.raises(TypeError, match="Boolean"):
        dual_protocol._boolean(1, "value")
    with pytest.raises(TypeError, match="integer"):
        dual_protocol._integer(True, "value")
    with pytest.raises(TypeError, match="numeric"):
        dual_protocol._number("1", "value")
    with pytest.raises(ValueError, match="finite"):
        dual_protocol._number(float("inf"), "value")
    with pytest.raises(TypeError, match="non-empty array"):
        dual_protocol._list([], "value")
    with pytest.raises(ValueError, match="SHA-256"):
        dual_protocol._sha256("bad", "value")
    with pytest.raises(ValueError, match="thresholds"):
        replace(config.initialization, condition_ceiling=1.0)
    with pytest.raises(ValueError, match="schedule"):
        replace(config.optimization, steps=parse_positive_int(1))
    parent = config.parents[0]
    with pytest.raises(ValueError, match="outside"):
        replace(parent, split="bad")
    with pytest.raises(ValueError, match="age weight"):
        replace(parent, lambda_age=parse_non_negative_weight(0.2))
    with pytest.raises(ValueError, match="fingerprints"):
        replace(parent, selection_sha256="bad")
    with pytest.raises(ValueError, match="protocol ID"):
        replace(config, protocol_id="bad")
    with pytest.raises(ValueError, match="parent protocol ID"):
        replace(config, parent_protocol_id="bad")
    with pytest.raises(ValueError, match="fingerprints"):
        replace(config, data_bundle_sha256="bad")
    with pytest.raises(ValueError, match="fixed-alpha"):
        replace(config, fixed_alphas=(parse_fraction(0.0),))
    with pytest.raises(ValueError, match="alpha schedules"):
        replace(config, schedule_names=("bad",))
    with pytest.raises(ValueError, match="frozen grid"):
        replace(config, parents=config.parents[:-1])
    with pytest.raises(ValueError, match="absent"):
        config.parent("diverse-blocks", 16_384)
    with pytest.raises(TypeError, match="must be a table"):
        dual_protocol._parse_parent("bad")


@pytest.mark.parametrize(
    ("old", "new", "message"),
    (
        ('status = "frozen"', 'status = "wrong"', "status or experiment axes"),
        (
            "sequence_direct_target_weight = 0.0",
            "sequence_direct_target_weight = 0.1",
            "objective differs",
        ),
        (
            "validation_or_test_rows_allowed = false",
            "validation_or_test_rows_allowed = true",
            "initialization contract",
        ),
        ('dense_optimizer = "adam"', 'dense_optimizer = "sgd"', "optimizer kinds"),
        (
            'refit_partition = "complete_primary_train"',
            'refit_partition = "wrong"',
            "selection policy",
        ),
    ),
)
def test_dual_protocol_loader_rejects_reviewed_contract_drift(
    tmp_path: Path,
    old: str,
    new: str,
    message: str,
) -> None:
    source = Path("configs/dual-probe-latent-v1.toml").read_text(encoding="utf-8")
    assert source.count(old) == 1
    path = tmp_path / "dual.toml"
    path.write_text(source.replace(old, new), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_dual_probe_protocol(path)
