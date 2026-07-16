from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from beartype import beartype
from hypothesis import given
from hypothesis import strategies as st
from jaxtyping import Float64, TypeCheckError, jaxtyped
from phantom import Phantom

Vector = Float64[np.ndarray, "n"]


def _is_probability(value: float) -> bool:
    return 0.0 <= value <= 1.0


class Probability(float, Phantom[float], predicate=_is_probability, bound=float):
    """Example phantom type for a float in the closed interval [0, 1]."""

    @classmethod
    def __register_strategy__(cls) -> Any:
        return st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


def parse_probability(value: float | int | str) -> Probability:
    raw = float(value) if isinstance(value, int | str) else value
    return Probability.parse(raw)


@jaxtyped(typechecker=beartype)
def normalize_weights(weights: Vector) -> Vector:
    if weights.size == 0:
        raise ValueError("weights must not be empty")
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must be finite")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")

    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("at least one weight must be positive")

    return weights / total


@jaxtyped(typechecker=beartype)
def weighted_mean(values: Vector, weights: Vector) -> Probability:
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("values must be probabilities")

    result = float(np.dot(values, normalize_weights(weights)))
    return parse_probability(min(1.0, max(0.0, result)))


@pytest.mark.property
@given(st.from_type(Probability))
def test_phantom_type_strategy_generates_valid_probabilities(value: Probability) -> None:
    assert isinstance(value, Probability)
    assert 0.0 <= value <= 1.0


@pytest.mark.parametrize("raw", [0, 0.5, "0.75"])
def test_parse_probability_accepts_valid_raw_values(raw: float | int | str) -> None:
    assert isinstance(parse_probability(raw), Probability)


@pytest.mark.parametrize("raw", [-0.1, 1.1, "not-a-float"])
def test_parse_probability_rejects_invalid_raw_values(raw: float | str) -> None:
    with pytest.raises((TypeError, ValueError)):
        parse_probability(raw)


@pytest.mark.property
@given(
    st.lists(
        st.floats(min_value=1.0e-6, max_value=1.0e6, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=32,
    )
)
def test_normalize_weights_returns_probability_vector(raw_weights: list[float]) -> None:
    weights = np.array(raw_weights, dtype=np.float64)

    normalized = normalize_weights(weights)

    assert normalized.dtype == np.float64
    assert normalized.shape == weights.shape
    assert np.all(normalized >= 0.0)
    assert float(np.sum(normalized)) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        (np.array([], dtype=np.float64), "empty"),
        (np.array([1.0, np.nan], dtype=np.float64), "finite"),
        (np.array([1.0, -0.5], dtype=np.float64), "non-negative"),
        (np.array([0.0, 0.0], dtype=np.float64), "positive"),
    ],
)
def test_normalize_weights_rejects_invalid_vectors(
    weights: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_weights(weights)


def test_weighted_mean_returns_probability() -> None:
    result = weighted_mean(
        np.array([0.25, 0.75], dtype=np.float64),
        np.array([1.0, 3.0], dtype=np.float64),
    )

    assert isinstance(result, Probability)
    assert result == pytest.approx(0.625)


def test_weighted_mean_rejects_shape_mismatch() -> None:
    with pytest.raises(TypeCheckError, match="weights"):
        weighted_mean(
            np.array([0.5, 0.25], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
        )


def test_weighted_mean_rejects_non_probability_values() -> None:
    with pytest.raises(ValueError, match="probabilities"):
        weighted_mean(
            np.array([0.5, 1.5], dtype=np.float64),
            np.array([1.0, 1.0], dtype=np.float64),
        )
