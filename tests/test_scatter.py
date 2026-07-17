from __future__ import annotations

from dataclasses import replace

import pytest
import torch as t

from methylation_latent.scatter import (
    CorrelationScatterSample,
    deterministic_display_indices,
)


def _sample() -> CorrelationScatterSample:
    indices = t.tensor([0, 3], dtype=t.int64)
    return CorrelationScatterSample(
        source_count=4,
        seed=7,
        sample_indices=indices,
        target=t.tensor([-0.2, 0.3], dtype=t.float64),
        predictions={"model": t.tensor([-0.1, 0.4], dtype=t.float64)},
    )


def test_deterministic_display_indices_are_target_blind_sorted_and_bounded() -> None:
    first = deterministic_display_indices(100, 10, seed=19)
    second = deterministic_display_indices(100, 10, seed=19)
    assert t.equal(first, second)
    assert first.numel() == 10
    assert t.equal(first, t.unique(first, sorted=True))
    assert t.equal(deterministic_display_indices(3, 5, seed=2), t.arange(3))
    with pytest.raises(ValueError, match="positive"):
        deterministic_display_indices(0, 5, seed=2)
    with pytest.raises(ValueError, match="positive"):
        deterministic_display_indices(5, 0, seed=2)


def test_scatter_sample_serializes_parallel_vectors() -> None:
    sample = _sample()
    assert sample.as_json() == {
        "source_count": 4,
        "display_count": 2,
        "seed": 7,
        "sample_indices": [0, 3],
        "target": [-0.2, 0.3],
        "predictions": {"model": [-0.1, 0.4]},
    }


@pytest.mark.parametrize(
    ("change", "error", "message"),
    [
        ({"source_count": 0}, ValueError, "positive source"),
        ({"predictions": {}}, ValueError, "prediction models"),
        ({"sample_indices": t.tensor([0.0])}, TypeError, "int64"),
        ({"sample_indices": t.tensor([1, 1])}, ValueError, "unique"),
        ({"sample_indices": t.tensor([0, 4])}, IndexError, "outside"),
        ({"target": t.tensor([0.0, 0.1])}, TypeError, "float64"),
        ({"target": t.tensor([0.0, float("nan")], dtype=t.float64)}, ValueError, "finite"),
        ({"target": t.tensor([0.0, 1.1], dtype=t.float64)}, ValueError, r"\[-1,1\]"),
        ({"predictions": {"": t.zeros(2, dtype=t.float64)}}, ValueError, "names"),
    ],
)
def test_scatter_sample_rejects_invalid_state(
    change: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        replace(_sample(), **change)
