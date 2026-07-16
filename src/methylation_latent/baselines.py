"""Training-only fits for durable age-head baselines."""

from __future__ import annotations

from dataclasses import dataclass

import torch as t
from beartype import beartype
from jaxtyping import Float64, jaxtyped

from methylation_latent.targets import CorrelationVector

FeatureMatrix = Float64[t.Tensor, "probes features"]


def _require_float64_finite(tensor: t.Tensor, name: str) -> None:
    if tensor.dtype != t.float64:
        raise TypeError(f"{name} must be float64, got {tensor.dtype}")
    if not bool(t.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must contain only finite values")


@dataclass(frozen=True, slots=True)
class SequenceAgeBaseline:
    """Intercept, CpG-density, and GC-content ordinary-least-squares fit."""

    coefficients: t.Tensor

    def __post_init__(self) -> None:
        if self.coefficients.shape != (3,) or self.coefficients.dtype != t.float64:
            raise ValueError("baseline coefficients must be a float64 vector of length three")
        if not bool(t.isfinite(self.coefficients).all().item()):
            raise ValueError("baseline coefficients must be finite")

    @jaxtyped(typechecker=beartype)
    def predict(self, features: FeatureMatrix) -> t.Tensor:
        design = _design_matrix(features)
        predictions = design @ self.coefficients
        if not bool(t.isfinite(predictions).all().item()):
            raise RuntimeError("baseline produced non-finite predictions")
        return predictions


@jaxtyped(typechecker=beartype)
def _design_matrix(features: FeatureMatrix) -> t.Tensor:
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("baseline features must be a non-empty rank-2 tensor")
    if features.shape[1] != 2:
        raise ValueError("baseline requires exactly CpG-density and GC-content columns")
    _require_float64_finite(features, "baseline features")
    if not bool(t.all((features >= 0.0) & (features <= 1.0)).item()):
        raise ValueError("sequence features must lie in [0, 1]")
    return t.cat((t.ones((features.shape[0], 1), dtype=t.float64), features), dim=1)


@jaxtyped(typechecker=beartype)
def fit_sequence_age_baseline(
    train_features: FeatureMatrix,
    train_rho: CorrelationVector,
) -> SequenceAgeBaseline:
    """Fit full-rank float64 OLS on training probes only."""

    if train_features.shape[0] != train_rho.tensor.numel():
        raise ValueError("baseline feature and target probe axes differ")
    if train_rho.tensor.dtype != t.float64:
        raise TypeError("baseline target must be float64")
    design = _design_matrix(train_features)
    rank = int(t.linalg.matrix_rank(design).item())
    if rank != design.shape[1]:
        raise ValueError(
            f"baseline design is rank deficient: rank={rank}, columns={design.shape[1]}"
        )
    solution = t.linalg.lstsq(design, train_rho.tensor.unsqueeze(1)).solution.squeeze(1)
    residual = design @ solution - train_rho.tensor
    normal_error = float((design.mT @ residual).abs().max().item())
    tolerance = 1.0e-10 * max(1.0, float(design.abs().max().item())) * design.shape[0]
    if normal_error > tolerance:
        raise RuntimeError(
            f"OLS normal-equation audit failed: error={normal_error}, tolerance={tolerance}"
        )
    return SequenceAgeBaseline(coefficients=solution)
