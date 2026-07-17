"""Deterministic display-only samples of bounded correlation predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch as t

from methylation_latent.artifacts import JsonValue


def deterministic_display_indices(
    source_count: int,
    maximum_points: int,
    *,
    seed: int,
) -> t.Tensor:
    """Choose a sorted, reproducible display subset without fitting or target access."""

    if source_count <= 0 or maximum_points <= 0:
        raise ValueError("scatter source and display counts must be positive")
    if source_count <= maximum_points:
        return t.arange(source_count, dtype=t.int64)
    generator = t.Generator(device="cpu").manual_seed(seed)
    return t.sort(t.randperm(source_count, generator=generator)[:maximum_points]).values


@dataclass(frozen=True, slots=True)
class CorrelationScatterSample:
    source_count: int
    seed: int
    sample_indices: t.Tensor
    target: t.Tensor
    predictions: dict[str, t.Tensor]

    def __post_init__(self) -> None:
        if self.source_count <= 0 or not self.predictions:
            raise ValueError("scatter sample requires a positive source and prediction models")
        if (
            self.sample_indices.dtype != t.int64
            or self.sample_indices.ndim != 1
            or self.sample_indices.numel() == 0
        ):
            raise TypeError("scatter indices must be a non-empty int64 vector")
        if not t.equal(self.sample_indices, t.unique(self.sample_indices, sorted=True)):
            raise ValueError("scatter indices must be unique and increasing")
        if bool(
            t.any((self.sample_indices < 0) | (self.sample_indices >= self.source_count)).item()
        ):
            raise IndexError("scatter index is outside its source population")
        vectors = (self.target, *self.predictions.values())
        if any(
            vector.dtype != t.float64
            or vector.ndim != 1
            or vector.shape != self.sample_indices.shape
            for vector in vectors
        ):
            raise TypeError("scatter targets and predictions must be aligned float64 vectors")
        if any(not bool(t.isfinite(vector).all().item()) for vector in vectors):
            raise ValueError("scatter targets and predictions must be finite")
        if any(bool(t.any(vector.abs() > 1.0 + 1.0e-12).item()) for vector in vectors):
            raise ValueError("scatter correlations must lie in [-1,1]")
        if any(not name for name in self.predictions):
            raise ValueError("scatter prediction model names must be non-empty")

    def as_json(self) -> dict[str, JsonValue]:
        return {
            "source_count": self.source_count,
            "display_count": self.sample_indices.numel(),
            "seed": self.seed,
            "sample_indices": self.sample_indices.tolist(),
            "target": self.target.tolist(),
            "predictions": cast(
                dict[str, JsonValue],
                {name: values.tolist() for name, values in self.predictions.items()},
            ),
        }
