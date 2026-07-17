from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest
import torch as t
from jaxtyping import TypeCheckError

from methylation_latent.direct_age import (
    DirectAgeData,
    DirectAgeStep,
    DirectAgeTrainingConfig,
    DirectTanhAgeRegressor,
    RefitDirectAge,
    TrainedDirectAge,
    _cuda_fork_devices,
    _make_optimizer,
    direct_age_mse,
    refit_direct_tanh_age,
    train_direct_tanh_age,
)
from methylation_latent.domain import (
    NonEmptyProbeSet,
    ProbeLocus,
    parse_positive_int,
)
from methylation_latent.storage import EmbeddingMatrix
from methylation_latent.targets import CorrelationVector


def _data(
    make_probe: Callable[..., ProbeLocus],
) -> tuple[DirectAgeData, DirectAgeData]:
    probes = tuple(make_probe(index + 1, position=1_000 + 100 * index) for index in range(8))
    values = t.zeros((8, 256), dtype=t.float16)
    values[:, 0] = t.linspace(-1.0, 1.0, 8, dtype=t.float16)
    values[:, 1] = 1.0
    rho = t.tanh(0.5 * values[:, 0].to(t.float32) - 0.1)
    return (
        DirectAgeData(
            NonEmptyProbeSet(probes[:6]),
            EmbeddingMatrix(values[:6]),
            CorrelationVector(rho[:6]),
        ),
        DirectAgeData(
            NonEmptyProbeSet(probes[6:]),
            EmbeddingMatrix(values[6:]),
            CorrelationVector(rho[6:]),
        ),
    )


def _config() -> DirectAgeTrainingConfig:
    return DirectAgeTrainingConfig(
        batch_size=parse_positive_int(4),
        neighbourhood_width=parse_positive_int(300),
        steps=parse_positive_int(3),
        validation_interval=parse_positive_int(1),
        learning_rate=1.0e-2,
        seed=71,
        device="cpu",
    )


def test_direct_tanh_head_is_one_bounded_scalar_with_zero_initialization() -> None:
    model = DirectTanhAgeRegressor(3)
    values = t.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]], dtype=t.float32)
    assert t.equal(model(values), t.zeros(2))
    with t.no_grad():
        model.linear.weight[0] = t.tensor([1.0, 0.0, 0.0])
        model.linear.bias[0] = 0.5
    prediction = model(values)
    assert prediction.shape == (2,)
    assert t.allclose(prediction, t.tanh(t.tensor([1.5, -0.5])))
    assert bool(t.all(prediction.abs() < 1.0).item())


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (t.ones(3), "embeddings"),
        (t.empty((0, 3)), "non-empty"),
        (t.ones((2, 4)), "expected embedding width"),
        (t.tensor([[1.0, float("nan"), 0.0]]), "finite"),
        (t.ones((2, 3), dtype=t.float64), "dtypes differ"),
    ],
)
def test_direct_tanh_head_rejects_invalid_embeddings(values: t.Tensor, message: str) -> None:
    model = DirectTanhAgeRegressor(3)
    with pytest.raises((TypeCheckError, TypeError, ValueError), match=message):
        model(values)
    with pytest.raises(ValueError, match="positive"):
        DirectTanhAgeRegressor(0)


def test_direct_age_mse_is_the_unweighted_mean() -> None:
    prediction = t.tensor([0.1, -0.2], dtype=t.float32)
    target = CorrelationVector(t.tensor([0.0, 0.2], dtype=t.float32))
    observed = direct_age_mse(prediction, target)
    assert observed.item() == pytest.approx((0.01 + 0.16) / 2)
    with pytest.raises(ValueError, match="shapes differ"):
        direct_age_mse(prediction, CorrelationVector(t.tensor([0.0])))
    with pytest.raises(ValueError, match="finite"):
        direct_age_mse(t.tensor([float("nan")]), CorrelationVector(t.tensor([0.0])))
    with pytest.raises(TypeError, match="dtypes differ"):
        direct_age_mse(t.tensor([0.0]), CorrelationVector(t.tensor([0.0], dtype=t.float64)))
    with pytest.raises(TypeCheckError):
        direct_age_mse(t.tensor([0]), CorrelationVector(t.tensor([0.0])))


def test_direct_age_data_and_config_reject_invalid_axes(
    make_probe: Callable[..., ProbeLocus],
) -> None:
    training, validation = _data(make_probe)
    with pytest.raises(ValueError, match="axes must align"):
        replace(training, rho=CorrelationVector(training.rho.tensor[:-1]))
    with pytest.raises(ValueError, match="finite and positive"):
        replace(_config(), learning_rate=float("nan"))
    with pytest.raises(ValueError, match="cannot exceed"):
        replace(_config(), validation_interval=parse_positive_int(4))


def test_training_and_refit_are_deterministic_and_have_no_lambda_or_d(
    make_probe: Callable[..., ProbeLocus],
) -> None:
    training, validation = _data(make_probe)
    first = train_direct_tanh_age(training, validation, config=_config())
    second = train_direct_tanh_age(training, validation, config=_config())
    assert tuple(record.step for record in first.history) == (1, 2, 3)
    assert tuple(record.step for record in first.validation_history) == (0, 1, 2, 3)
    assert first.selected_step in {0, 1, 2, 3}
    assert first.validation_history == second.validation_history
    assert all(
        t.equal(first.model.state_dict()[name], second.model.state_dict()[name])
        for name in first.model.state_dict()
    )
    refit = refit_direct_tanh_age(training, config=_config(), selected_steps=2)
    assert refit.refit_steps == 2
    assert tuple(record.step for record in refit.history) == (1, 2)
    untrained = refit_direct_tanh_age(training, config=_config(), selected_steps=0)
    assert untrained.history == ()


def test_training_rejects_overlap_batch_size_steps_and_unavailable_cuda(
    make_probe: Callable[..., ProbeLocus],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training, validation = _data(make_probe)
    with pytest.raises(ValueError, match="must be disjoint"):
        train_direct_tanh_age(training, training, config=_config())
    oversized = replace(_config(), batch_size=parse_positive_int(7))
    with pytest.raises(ValueError, match="cannot exceed"):
        train_direct_tanh_age(training, validation, config=oversized)
    with pytest.raises(ValueError, match="cannot exceed"):
        refit_direct_tanh_age(training, config=oversized, selected_steps=1)
    with pytest.raises(ValueError, match="within the tuning budget"):
        refit_direct_tanh_age(training, config=_config(), selected_steps=4)
    monkeypatch.setattr("methylation_latent.direct_age.t.cuda.is_available", lambda: False)
    cuda = replace(_config(), device="cuda")
    with pytest.raises(ValueError, match="CUDA direct-age"):
        train_direct_tanh_age(training, validation, config=cuda)
    assert _cuda_fork_devices(t.device("cpu")) == []


def test_direct_age_records_and_optimizer_fail_loudly() -> None:
    model = DirectTanhAgeRegressor(2)
    optimizer = _make_optimizer(model, learning_rate=1.0e-3)
    assert all(float(group["weight_decay"]) == 0.0 for group in optimizer.param_groups)
    with pytest.raises(ValueError, match="step cannot be negative"):
        DirectAgeStep(-1, 0.1)
    with pytest.raises(ValueError, match="finite and non-negative"):
        DirectAgeStep(0, -0.1)
    valid = DirectAgeStep(0, 0.1)
    with pytest.raises(ValueError, match="non-empty histories"):
        TrainedDirectAge(model, (), (valid,), 0)
    with pytest.raises(ValueError, match="unique and increasing"):
        TrainedDirectAge(model, (DirectAgeStep(1, 0.1),), (valid, valid), 0)
    with pytest.raises(ValueError, match="absent"):
        TrainedDirectAge(model, (DirectAgeStep(1, 0.1),), (valid,), 1)
    with pytest.raises(ValueError, match="exactly cover"):
        RefitDirectAge(model, (), 1)
    with pytest.raises(ValueError, match="does not end"):
        RefitDirectAge(model, (DirectAgeStep(1, 0.1), DirectAgeStep(1, 0.1)), 2)
