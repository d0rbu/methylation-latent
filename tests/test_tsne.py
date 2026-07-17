from __future__ import annotations

from dataclasses import replace

import pytest
import torch as t

from methylation_latent.tsne import (
    TsneConfig,
    TsneProjection,
    _joint_probabilities,
    deterministic_balanced_indices,
    exact_tsne,
)


def _config() -> TsneConfig:
    return TsneConfig(
        perplexity=3.0,
        probability_search_steps=20,
        optimization_steps=8,
        early_exaggeration_steps=2,
        early_exaggeration=4.0,
        learning_rate=2.0,
        seed=17,
    )


def test_exact_tsne_is_deterministic_centered_and_finite() -> None:
    values = t.tensor(
        [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [2.0, 2.0], [2.1, 2.0], [2.0, 2.1]],
        dtype=t.float32,
    )
    first = exact_tsne(values, config=_config())
    second = exact_tsne(values, config=_config())
    assert t.equal(first.coordinates, second.coordinates)
    assert first.final_kl_divergence == second.final_kl_divergence
    assert t.allclose(first.coordinates.mean(dim=0), t.zeros(2, dtype=t.float64), atol=1e-12)
    assert bool(t.any(first.coordinates != 0.0).item())


def test_joint_probabilities_are_symmetric_zero_diagonal_and_normalized() -> None:
    values = t.arange(12, dtype=t.float64).reshape(6, 2)
    joint = _joint_probabilities(values, _config())
    assert t.equal(joint, joint.mT)
    assert t.equal(t.diag(joint), t.zeros(6, dtype=t.float64))
    assert joint.sum().item() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"perplexity": 0.0}, "must be positive"),
        ({"early_exaggeration": float("nan")}, "must be positive"),
        ({"probability_search_steps": 0}, "inconsistent"),
        ({"optimization_steps": 0}, "inconsistent"),
        ({"early_exaggeration_steps": 9}, "inconsistent"),
    ],
)
def test_tsne_config_rejects_invalid_state(change: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_config(), **change)


@pytest.mark.parametrize(
    ("values", "error", "message"),
    [
        (t.ones(3), ValueError, "shape"),
        (t.ones((1, 2)), ValueError, "shape"),
        (t.ones((4, 0)), ValueError, "shape"),
        (t.ones((4, 2), dtype=t.int64), TypeError, "floating"),
        (t.tensor([[0.0, 1.0], [float("nan"), 0.0], [1.0, 1.0], [2.0, 2.0]]), ValueError, "finite"),
    ],
)
def test_exact_tsne_rejects_invalid_inputs(
    values: t.Tensor,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        exact_tsne(values, config=_config())
    with pytest.raises(ValueError, match="perplexity"):
        exact_tsne(t.ones((3, 2)), config=_config())


def test_projection_record_rejects_invalid_shape_and_kl() -> None:
    with pytest.raises(TypeError, match="shape"):
        TsneProjection(t.ones((2, 3), dtype=t.float64), 1.0)
    with pytest.raises(ValueError, match="finite"):
        TsneProjection(t.tensor([[0.0, 0.0], [float("nan"), 0.0]], dtype=t.float64), 1.0)
    with pytest.raises(ValueError, match="KL"):
        TsneProjection(t.zeros((2, 2), dtype=t.float64), -1.0)


def test_balanced_indices_are_deterministic_sorted_and_balanced() -> None:
    labels = t.tensor([0, 1, 2, 3] * 5, dtype=t.int64)
    first = deterministic_balanced_indices(
        labels,
        group_count=4,
        maximum_points=12,
        seed=81,
    )
    second = deterministic_balanced_indices(
        labels,
        group_count=4,
        maximum_points=12,
        seed=81,
    )
    assert t.equal(first, second)
    assert t.equal(first, t.unique(first, sorted=True))
    assert t.equal(
        t.bincount(labels.index_select(0, first), minlength=4),
        t.full((4,), 3, dtype=t.int64),
    )


@pytest.mark.parametrize(
    ("labels", "group_count", "maximum_points", "error", "message"),
    [
        (t.ones(4), 1, 4, TypeError, "int64"),
        (t.ones((2, 2), dtype=t.int64), 1, 4, TypeError, "vector"),
        (t.empty(0, dtype=t.int64), 1, 1, TypeError, "non-empty"),
        (t.zeros(4, dtype=t.int64), 0, 4, ValueError, "positive"),
        (t.zeros(4, dtype=t.int64), 1, 5, ValueError, "exceed"),
        (t.zeros(4, dtype=t.int64), 3, 4, ValueError, "evenly"),
        (t.tensor([0, 0, 2, 2]), 2, 4, ValueError, "outside"),
        (t.tensor([0, 0, 0, 1]), 2, 4, ValueError, "has 1"),
    ],
)
def test_balanced_indices_reject_illegal_states(
    labels: t.Tensor,
    group_count: int,
    maximum_points: int,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        deterministic_balanced_indices(
            labels,
            group_count=group_count,
            maximum_points=maximum_points,
            seed=1,
        )
