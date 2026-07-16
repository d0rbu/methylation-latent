from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest
import torch as t
from jaxtyping import TypeCheckError

from methylation_latent.batching import GenomicBatchSampler, ProbeBatch
from methylation_latent.domain import (
    NonEmptyProbeSet,
    ProbeLocus,
    parse_latent_dimension,
    parse_non_negative_weight,
    parse_positive_int,
    parse_probe_id,
)
from methylation_latent.model import (
    LatentMetric,
    ObjectiveTerms,
    PairObjectiveInput,
    make_optimizer,
    metric_objective,
    normalize_rows_strict,
    normalize_vector_strict,
)
from methylation_latent.storage import EmbeddingMatrix
from methylation_latent.targets import (
    CorrelationMatrix,
    CorrelationVector,
    TargetGeometry,
    build_target_geometry,
)
from methylation_latent.training import (
    RefitMetric,
    TrainedMetric,
    TrainingConfig,
    TrainingMode,
    TrainingStep,
    ValidationData,
    ValidationStep,
    _cuda_fork_devices,
    refit_latent_metric,
    train_latent_metric,
)


def _probe_universe(make_probe: Callable[..., ProbeLocus], count: int = 8) -> NonEmptyProbeSet:
    return NonEmptyProbeSet(
        tuple(make_probe(index + 1, position=1_000 + index * 100) for index in range(count))
    )


def test_genomic_batch_sampler_is_deterministic_unique_and_local(
    make_probe: Callable[..., ProbeLocus],
) -> None:
    probes = _probe_universe(make_probe)
    first = GenomicBatchSampler(
        probes,
        parse_positive_int(6),
        parse_positive_int(250),
        seed=7,
    )
    second = GenomicBatchSampler(
        probes,
        parse_positive_int(6),
        parse_positive_int(250),
        seed=7,
    )
    batch = first.sample()
    assert t.equal(batch.indices, second.sample().indices)
    assert len(set(batch.probe_ids)) == 6
    assert batch.indices.dtype == t.int64
    with pytest.raises(ValueError, match="cannot exceed"):
        GenomicBatchSampler(
            probes,
            parse_positive_int(9),
            parse_positive_int(100),
            seed=1,
        )


def test_probe_batch_rejects_duplicate_and_misaligned_indices() -> None:
    with pytest.raises(ValueError, match="unique"):
        ProbeBatch(
            t.tensor([0, 0], dtype=t.int64),
            (parse_probe_id("cg00000001"), parse_probe_id("cg00000002")),
        )
    with pytest.raises(ValueError, match="lengths"):
        ProbeBatch(t.tensor([0], dtype=t.int64), ())
    with pytest.raises(ValueError, match="non-empty int64"):
        ProbeBatch(t.tensor([0.0]), (parse_probe_id("cg00000001"),))
    with pytest.raises(ValueError, match="probe IDs must be unique"):
        ProbeBatch(
            t.tensor([0, 1]),
            (parse_probe_id("cg00000001"), parse_probe_id("cg00000001")),
        )


def test_strict_normalization_rejects_zero_nonfinite_and_wrong_shapes() -> None:
    rows = normalize_rows_strict(t.tensor([[3.0, 4.0], [1.0, 0.0]], dtype=t.float32))
    assert t.allclose(t.linalg.vector_norm(rows, dim=1), t.ones(2))
    vector = normalize_vector_strict(t.tensor([3.0, 4.0], dtype=t.float32))
    assert t.linalg.vector_norm(vector) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="zero latent rows"):
        normalize_rows_strict(t.zeros((1, 2)))
    with pytest.raises(ValueError, match="finite"):
        normalize_rows_strict(t.tensor([[float("nan"), 0.0]]))
    with pytest.raises(TypeCheckError, match="values"):
        normalize_rows_strict(t.ones(2))
    with pytest.raises(ValueError, match="zero latent direction"):
        normalize_vector_strict(t.zeros(2))
    with pytest.raises(TypeCheckError, match="values"):
        normalize_vector_strict(t.ones((1, 2)))


def test_latent_metric_shapes_cosines_and_scale_invariance() -> None:
    t.manual_seed(2)
    model = LatentMetric(embedding_dimension=8, latent_dimension=parse_latent_dimension(4))
    embeddings = t.randn((5, 8), dtype=t.float32)
    latent, pairs, age = model.predict(embeddings)
    assert latent.shape == (5, 4)
    assert pairs.shape == (5, 5)
    assert age.shape == (5,)
    assert t.allclose(pairs, pairs.mT)
    assert t.allclose(pairs.diagonal(), t.ones(5), atol=1e-6)
    assert bool(t.all(age.abs() <= 1.0 + 1e-6).item())
    before = pairs.detach().clone(), age.detach().clone()
    with t.no_grad():
        model.projection.weight.mul_(3.0)
        model.age_direction.mul_(7.0)
    _, scaled_pairs, scaled_age = model.predict(embeddings)
    assert t.allclose(before[0], scaled_pairs, atol=1e-6)
    assert t.allclose(before[1], scaled_age, atol=1e-6)
    assert model.projection.bias is None


def test_cosine_objective_gradient_is_orthogonal_to_projection_scale() -> None:
    t.manual_seed(3)
    model = LatentMetric(embedding_dimension=6, latent_dimension=parse_latent_dimension(4))
    embeddings = t.randn((5, 6), dtype=t.float32)
    latent, pairs, age = model.predict(embeddings)
    pair_target = CorrelationMatrix(t.eye(5, dtype=t.float32))
    age_target = CorrelationVector(t.linspace(-0.5, 0.5, 5, dtype=t.float32))
    terms = metric_objective(
        age,
        age_target,
        lambda_age=parse_non_negative_weight(1.0),
        pair=PairObjectiveInput(pairs, pair_target),
    )
    terms.total.backward()
    gradient = model.projection.weight.grad
    assert gradient is not None
    gradient_dot_weight = t.sum(gradient * model.projection.weight)
    assert float(gradient_dot_weight.abs().item()) < 2e-6


def test_age_only_objective_never_requires_a_gram_prediction() -> None:
    prediction = t.tensor([0.1, -0.2], dtype=t.float32, requires_grad=True)
    target = CorrelationVector(t.tensor([0.0, 0.0], dtype=t.float32))
    terms = metric_objective(
        prediction,
        target,
        lambda_age=parse_non_negative_weight(2.0),
        pair=None,
    )
    assert terms.pair_mse is None
    assert float(terms.total.item()) == pytest.approx(2 * 0.025)
    terms.total.backward()
    assert prediction.grad is not None


def test_objective_and_model_reject_misaligned_or_illegal_inputs() -> None:
    with pytest.raises(ValueError, match="embedding dimension"):
        LatentMetric(embedding_dimension=0, latent_dimension=parse_latent_dimension(1))
    with pytest.raises(ValueError, match="cannot exceed"):
        LatentMetric(embedding_dimension=2, latent_dimension=parse_latent_dimension(3))
    model = LatentMetric(embedding_dimension=3, latent_dimension=parse_latent_dimension(2))
    with pytest.raises(ValueError, match="width"):
        model.latent(t.ones((2, 4)))
    with pytest.raises(TypeError, match="dtype"):
        model.latent(t.ones((2, 3), dtype=t.float64))
    with pytest.raises(ValueError, match="finite"):
        model.latent(t.tensor([[1.0, float("nan"), 0.0]]))
    with t.no_grad():
        model.projection.weight.zero_()
    with pytest.raises(ValueError, match="zero latent rows"):
        model.latent(t.ones((1, 3)))
    model = LatentMetric(embedding_dimension=3, latent_dimension=parse_latent_dimension(2))
    with t.no_grad():
        model.age_direction.zero_()
    with pytest.raises(ValueError, match="zero latent direction"):
        model.predict_age_from_latent(t.tensor([[1.0, 0.0]]))
    with pytest.raises(ValueError, match="shapes differ"):
        metric_objective(
            t.ones(2),
            CorrelationVector(t.zeros(3)),
            lambda_age=parse_non_negative_weight(1),
            pair=None,
        )
    with pytest.raises(ValueError, match="shapes differ"):
        PairObjectiveInput(t.eye(2), CorrelationMatrix(t.eye(3)))
    with pytest.raises(TypeError, match="dtypes differ"):
        PairObjectiveInput(t.eye(2, dtype=t.float64), CorrelationMatrix(t.eye(2)))
    with pytest.raises(ValueError, match="finite"):
        PairObjectiveInput(t.tensor([[1.0, float("nan")], [0.0, 1.0]]), CorrelationMatrix(t.eye(2)))
    with pytest.raises(TypeError, match="dtypes differ"):
        metric_objective(
            t.ones(2, dtype=t.float64),
            CorrelationVector(t.zeros(2)),
            lambda_age=parse_non_negative_weight(1),
            pair=None,
        )
    with pytest.raises(ValueError, match="positive"):
        make_optimizer(model, learning_rate=0.0)


def test_optimizer_has_zero_weight_decay() -> None:
    model = LatentMetric(embedding_dimension=4, latent_dimension=parse_latent_dimension(2))
    optimizer = make_optimizer(model, learning_rate=1e-3)
    assert all(float(group["weight_decay"]) == 0.0 for group in optimizer.param_groups)
    with pytest.raises(ValueError, match="finite and positive"):
        make_optimizer(model, learning_rate=float("nan"))


def _tiny_training_inputs(
    make_probe: Callable[..., ProbeLocus],
) -> tuple[NonEmptyProbeSet, EmbeddingMatrix, TargetGeometry, ValidationData]:
    probes = _probe_universe(make_probe, count=8)
    generator = t.Generator().manual_seed(8)
    embedding_values = t.randn((8, 256), generator=generator, dtype=t.float16)
    beta = t.rand((8, 5), generator=generator, dtype=t.float64)
    age = t.linspace(20, 80, 5, dtype=t.float64)
    training = NonEmptyProbeSet(probes.probes[:6])
    validation = NonEmptyProbeSet(probes.probes[6:])
    return (
        training,
        EmbeddingMatrix(embedding_values[:6]),
        build_target_geometry(beta[:6], age),
        ValidationData(
            probes=validation,
            embeddings=EmbeddingMatrix(embedding_values[6:]),
            targets=build_target_geometry(beta[6:], age),
        ),
    )


@pytest.mark.parametrize("mode", [TrainingMode.AGE_ONLY, TrainingMode.FULL])
def test_fixed_step_training_consumes_cached_embeddings(
    mode: TrainingMode,
    make_probe: Callable[..., ProbeLocus],
) -> None:
    probes, embeddings, targets, validation = _tiny_training_inputs(make_probe)
    config = TrainingConfig(
        mode=mode,
        latent_dimension=parse_latent_dimension(4),
        lambda_age=parse_non_negative_weight(1.0),
        batch_size=parse_positive_int(4),
        neighbourhood_width=parse_positive_int(300),
        steps=parse_positive_int(2),
        validation_interval=parse_positive_int(1),
        validation_pair_chunk_size=parse_positive_int(2),
        learning_rate=1e-3,
        seed=91,
        device="cpu",
    )
    trained = train_latent_metric(
        probes,
        embeddings,
        targets,
        validation,
        config=config,
    )
    assert len(trained.history) == 2
    assert tuple(record.step for record in trained.validation_history) == (0, 1, 2)
    assert trained.selected_step in {0, 1, 2}
    assert all(
        (step.pair_mse is None) == (mode == TrainingMode.AGE_ONLY) for step in trained.history
    )
    assert all(
        (step.pair_mse is None) == (mode == TrainingMode.AGE_ONLY)
        for step in trained.validation_history
    )
    assert all(parameter.dtype == t.float32 for parameter in trained.model.parameters())


def test_final_refit_uses_only_the_selected_nonnegative_step_count(
    make_probe: Callable[..., ProbeLocus],
) -> None:
    probes, embeddings, targets, _ = _tiny_training_inputs(make_probe)
    config = TrainingConfig(
        mode=TrainingMode.FULL,
        latent_dimension=parse_latent_dimension(4),
        lambda_age=parse_non_negative_weight(1.0),
        batch_size=parse_positive_int(4),
        neighbourhood_width=parse_positive_int(300),
        steps=parse_positive_int(2),
        validation_interval=parse_positive_int(1),
        validation_pair_chunk_size=parse_positive_int(2),
        learning_rate=1e-3,
        seed=91,
        device="cpu",
    )
    refit = refit_latent_metric(
        probes,
        embeddings,
        targets,
        config=config,
        selected_steps=1,
    )
    assert refit.refit_steps == 1
    assert tuple(step.step for step in refit.history) == (1,)
    untrained = refit_latent_metric(
        probes,
        embeddings,
        targets,
        config=config,
        selected_steps=0,
    )
    assert untrained.history == ()
    with pytest.raises(ValueError, match="between zero"):
        refit_latent_metric(
            probes,
            embeddings,
            targets,
            config=config,
            selected_steps=3,
        )
    with pytest.raises(ValueError, match="refit history"):
        RefitMetric(refit.model, refit.history, 0)


def test_training_configs_and_records_reject_invalid_states(
    make_probe: Callable[..., ProbeLocus],
) -> None:
    with pytest.raises(ValueError, match="positive age-loss"):
        TrainingConfig(
            TrainingMode.AGE_ONLY,
            parse_latent_dimension(2),
            parse_non_negative_weight(0),
            parse_positive_int(2),
            parse_positive_int(10),
            parse_positive_int(1),
            parse_positive_int(1),
            parse_positive_int(2),
            1e-3,
            1,
            "cpu",
        )
    with pytest.raises(ValueError, match="learning rate"):
        TrainingConfig(
            TrainingMode.FULL,
            parse_latent_dimension(2),
            parse_non_negative_weight(1),
            parse_positive_int(2),
            parse_positive_int(10),
            parse_positive_int(1),
            parse_positive_int(1),
            parse_positive_int(2),
            float("nan"),
            1,
            "cpu",
        )
    with pytest.raises(ValueError, match="validation interval"):
        TrainingConfig(
            TrainingMode.FULL,
            parse_latent_dimension(2),
            parse_non_negative_weight(1),
            parse_positive_int(2),
            parse_positive_int(10),
            parse_positive_int(1),
            parse_positive_int(2),
            parse_positive_int(2),
            1e-3,
            1,
            "cpu",
        )
    with pytest.raises(ValueError, match="losses"):
        TrainingStep(1, -1.0, 0.0, None)
    with pytest.raises(ValueError, match="pair loss"):
        TrainingStep(1, 1.0, 0.0, -1.0)
    with pytest.raises(ValueError, match="positive"):
        TrainingStep(0, 1.0, 0.0, None)
    with pytest.raises(ValueError, match="validation losses"):
        ValidationStep(0, -1.0, 0.0, 0.0, None)
    with pytest.raises(ValueError, match="cannot be negative"):
        ValidationStep(-1, 0.0, 0.0, 0.0, None)
    with pytest.raises(ValueError, match="pair loss"):
        ValidationStep(0, 0.0, 0.0, 0.0, -1.0)
    model = LatentMetric(embedding_dimension=2, latent_dimension=parse_latent_dimension(2))
    with pytest.raises(ValueError, match="non-empty"):
        TrainedMetric(model, (), (), 0)
    training_step = TrainingStep(1, 0.0, 0.0, None)
    validation_step_zero = ValidationStep(0, 0.0, 0.0, 0.0, None)
    validation_step_one = ValidationStep(1, 0.0, 0.0, 0.0, None)
    with pytest.raises(ValueError, match="unique and increasing"):
        TrainedMetric(
            model,
            (training_step,),
            (validation_step_one, validation_step_zero),
            0,
        )
    with pytest.raises(ValueError, match="absent"):
        TrainedMetric(model, (training_step,), (validation_step_zero,), 1)
    with pytest.raises(ValueError, match="floating scalar"):
        ObjectiveTerms(t.ones(1), None, t.tensor(0.0))
    with pytest.raises(ValueError, match="pair_mse"):
        ObjectiveTerms(t.tensor(1.0), t.ones(1), t.tensor(0.0))
    probes, embeddings, targets, validation = _tiny_training_inputs(make_probe)
    short_probes = NonEmptyProbeSet(probes.probes[:-1])
    config = TrainingConfig(
        TrainingMode.FULL,
        parse_latent_dimension(4),
        parse_non_negative_weight(1),
        parse_positive_int(4),
        parse_positive_int(100),
        parse_positive_int(1),
        parse_positive_int(1),
        parse_positive_int(2),
        1e-3,
        1,
        "cpu",
    )
    with pytest.raises(ValueError, match="align"):
        train_latent_metric(
            short_probes,
            embeddings,
            targets,
            validation,
            config=config,
        )
    too_large_batch = replace(config, batch_size=parse_positive_int(7))
    with pytest.raises(ValueError, match="batch size"):
        train_latent_metric(
            probes,
            embeddings,
            targets,
            validation,
            config=too_large_batch,
        )
    too_wide = replace(config, latent_dimension=parse_latent_dimension(5))
    with pytest.raises(ValueError, match="useful ceiling"):
        train_latent_metric(
            probes,
            embeddings,
            targets,
            validation,
            config=too_wide,
        )
    with pytest.raises(ValueError, match="validation probe"):
        ValidationData(
            probes=probes,
            embeddings=validation.embeddings,
            targets=validation.targets,
        )
    overlapping_validation = replace(
        validation,
        probes=NonEmptyProbeSet(probes.probes[:2]),
    )
    with pytest.raises(ValueError, match="must be disjoint"):
        train_latent_metric(
            probes,
            embeddings,
            targets,
            overlapping_validation,
            config=config,
        )
    wrong_width_embeddings = object.__new__(EmbeddingMatrix)
    object.__setattr__(
        wrong_width_embeddings,
        "tensor",
        t.ones((2, 255), dtype=t.float16),
    )
    wrong_width_validation = replace(validation, embeddings=wrong_width_embeddings)
    with pytest.raises(ValueError, match="embedding widths"):
        train_latent_metric(
            probes,
            embeddings,
            targets,
            wrong_width_validation,
            config=config,
        )
    wrong_sample_validation = replace(
        validation,
        targets=build_target_geometry(
            t.rand((2, 6), dtype=t.float64),
            t.linspace(20, 70, 6, dtype=t.float64),
        ),
    )
    with pytest.raises(ValueError, match="sample axes"):
        train_latent_metric(
            probes,
            embeddings,
            targets,
            wrong_sample_validation,
            config=config,
        )


def test_cuda_device_selection_and_unavailable_training_fail_loudly(
    monkeypatch: pytest.MonkeyPatch,
    make_probe: Callable[..., ProbeLocus],
) -> None:
    assert _cuda_fork_devices(t.device("cpu")) == []
    assert _cuda_fork_devices(t.device("cuda:2")) == [2]
    probes, embeddings, targets, validation = _tiny_training_inputs(make_probe)
    config = TrainingConfig(
        TrainingMode.FULL,
        parse_latent_dimension(4),
        parse_non_negative_weight(1),
        parse_positive_int(4),
        parse_positive_int(100),
        parse_positive_int(1),
        parse_positive_int(1),
        parse_positive_int(2),
        1e-3,
        1,
        "cuda",
    )
    monkeypatch.setattr(t.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA.*unavailable"):
        train_latent_metric(
            probes,
            embeddings,
            targets,
            validation,
            config=config,
        )
