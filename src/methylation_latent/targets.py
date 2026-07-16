"""Construction and validation of correlation targets."""

from __future__ import annotations

from dataclasses import dataclass

import torch as t
from beartype import beartype
from jaxtyping import Float64, Int64, jaxtyped

from methylation_latent.domain import LatentDimension, PositiveInt

BetaMatrix = Float64[t.Tensor, "probes samples"]
AgeVector = Float64[t.Tensor, "samples"]
IndexVector = Int64[t.Tensor, "selected"]


def _semantic_atol(tensor: t.Tensor) -> float:
    if tensor.dtype == t.float64:
        return 1.0e-10
    if tensor.dtype == t.float32:
        return 2.0e-5
    raise TypeError(f"semantic target tensors must be float32 or float64, got {tensor.dtype}")


def _require_finite(tensor: t.Tensor, name: str) -> None:
    if not bool(t.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must contain only finite values")


@dataclass(frozen=True, slots=True)
class UnitNormRows:
    """A finite non-empty matrix with zero-mean, unit-L2-norm rows."""

    tensor: t.Tensor

    def __post_init__(self) -> None:
        if self.tensor.ndim != 2 or 0 in self.tensor.shape:
            raise ValueError("UnitNormRows requires a non-empty rank-2 tensor")
        _require_finite(self.tensor, "UnitNormRows")
        atol = _semantic_atol(self.tensor)
        row_means = self.tensor.mean(dim=1)
        row_norms = t.linalg.vector_norm(self.tensor, dim=1)
        if not bool(t.all(row_means.abs() <= atol).item()):
            error = float(row_means.abs().max().item())
            raise ValueError(f"rows are not zero-mean; maximum absolute mean={error}")
        if not bool(t.allclose(row_norms, t.ones_like(row_norms), atol=atol, rtol=0.0)):
            error = float((row_norms - 1.0).abs().max().item())
            raise ValueError(f"rows are not unit norm; maximum absolute error={error}")

    @property
    def n_rows(self) -> int:
        return self.tensor.shape[0]

    @property
    def n_samples(self) -> int:
        return self.tensor.shape[1]


@dataclass(frozen=True, slots=True)
class UnitNormVector:
    """A finite non-empty zero-mean vector with unit L2 norm."""

    tensor: t.Tensor

    def __post_init__(self) -> None:
        if self.tensor.ndim != 1 or self.tensor.numel() == 0:
            raise ValueError("UnitNormVector requires a non-empty rank-1 tensor")
        _require_finite(self.tensor, "UnitNormVector")
        atol = _semantic_atol(self.tensor)
        mean = float(self.tensor.mean().abs().item())
        norm_error = float((t.linalg.vector_norm(self.tensor) - 1.0).abs().item())
        if mean > atol:
            raise ValueError(f"vector is not zero-mean; absolute mean={mean}")
        if norm_error > atol:
            raise ValueError(f"vector is not unit norm; absolute error={norm_error}")


@dataclass(frozen=True, slots=True)
class CorrelationVector:
    """A finite non-empty correlation vector bounded by one up to roundoff."""

    tensor: t.Tensor

    def __post_init__(self) -> None:
        if self.tensor.ndim != 1 or self.tensor.numel() == 0:
            raise ValueError("CorrelationVector requires a non-empty rank-1 tensor")
        _require_finite(self.tensor, "CorrelationVector")
        atol = _semantic_atol(self.tensor)
        if not bool(t.all(self.tensor.abs() <= 1.0 + atol).item()):
            maximum = float(self.tensor.abs().max().item())
            raise ValueError(f"correlation magnitude exceeds one: {maximum}")


@dataclass(frozen=True, slots=True)
class CorrelationMatrix:
    """A finite symmetric correlation matrix with unit diagonal."""

    tensor: t.Tensor

    def __post_init__(self) -> None:
        if self.tensor.ndim != 2 or self.tensor.shape[0] == 0:
            raise ValueError("CorrelationMatrix requires a non-empty rank-2 tensor")
        if self.tensor.shape[0] != self.tensor.shape[1]:
            raise ValueError("CorrelationMatrix must be square")
        _require_finite(self.tensor, "CorrelationMatrix")
        atol = _semantic_atol(self.tensor)
        if not bool(t.all(self.tensor.abs() <= 1.0 + atol).item()):
            maximum = float(self.tensor.abs().max().item())
            raise ValueError(f"correlation magnitude exceeds one: {maximum}")
        if not bool(t.allclose(self.tensor, self.tensor.mT, atol=atol, rtol=0.0)):
            error = float((self.tensor - self.tensor.mT).abs().max().item())
            raise ValueError(f"correlation matrix is not symmetric; maximum error={error}")
        diagonal = self.tensor.diagonal()
        if not bool(t.allclose(diagonal, t.ones_like(diagonal), atol=atol, rtol=0.0)):
            error = float((diagonal - 1.0).abs().max().item())
            raise ValueError(f"correlation matrix diagonal is not one; maximum error={error}")


@dataclass(frozen=True, slots=True)
class TargetGeometry:
    """Cached standardized probe rows and their age-correlation vector."""

    methylation: UnitNormRows
    age: UnitNormVector
    rho: CorrelationVector

    def __post_init__(self) -> None:
        if self.methylation.n_samples != self.age.tensor.numel():
            raise ValueError("methylation and age sample axes do not match")
        if self.methylation.n_rows != self.rho.tensor.numel():
            raise ValueError("methylation and rho probe axes do not match")

    @property
    def rank_upper_bound(self) -> int:
        return self.methylation.n_samples - 1


@jaxtyped(typechecker=beartype)
def standardize_rows(values: BetaMatrix) -> UnitNormRows:
    """Center each float64 row and divide by its L2 norm."""

    if values.dtype != t.float64:
        raise TypeError(f"beta matrix must be float64, got {values.dtype}")
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError("beta matrix must have non-empty probe and sample axes")
    _require_finite(values, "beta matrix")
    if not bool(t.all((values >= 0.0) & (values <= 1.0)).item()):
        raise ValueError("beta matrix values must lie in [0, 1]")

    centered = values - values.mean(dim=1, keepdim=True)
    norms = t.linalg.vector_norm(centered, dim=1, keepdim=True)
    if bool(t.any(norms == 0.0).item()):
        indices = t.nonzero(norms.squeeze(1) == 0.0).flatten().tolist()
        raise ValueError(f"constant beta rows cannot be standardized: {indices[:10]}")
    return UnitNormRows(centered / norms)


@jaxtyped(typechecker=beartype)
def standardize_vector(values: AgeVector) -> UnitNormVector:
    """Center a float64 vector and divide by its L2 norm."""

    if values.dtype != t.float64:
        raise TypeError(f"age vector must be float64, got {values.dtype}")
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("age vector must be non-empty")
    _require_finite(values, "age vector")
    centered = values - values.mean()
    norm = t.linalg.vector_norm(centered)
    if float(norm.item()) == 0.0:
        raise ValueError("constant age vector cannot be standardized")
    return UnitNormVector(centered / norm)


@jaxtyped(typechecker=beartype)
def build_target_geometry(beta: BetaMatrix, age: AgeVector) -> TargetGeometry:
    methylation = standardize_rows(beta)
    standardized_age = standardize_vector(age)
    if methylation.n_samples != standardized_age.tensor.numel():
        raise ValueError("beta sample axis and age length do not match")
    rho = CorrelationVector(methylation.tensor @ standardized_age.tensor)
    return TargetGeometry(methylation=methylation, age=standardized_age, rho=rho)


@jaxtyped(typechecker=beartype)
def correlation_block(rows: UnitNormRows, indices: IndexVector) -> CorrelationMatrix:
    if indices.dtype != t.int64:
        raise TypeError(f"indices must be int64, got {indices.dtype}")
    if indices.ndim != 1 or indices.numel() == 0:
        raise ValueError("correlation block indices must be non-empty")
    if bool(t.any((indices < 0) | (indices >= rows.n_rows)).item()):
        raise IndexError("correlation block index is out of bounds")
    selected = rows.tensor.index_select(0, indices)
    return CorrelationMatrix(selected @ selected.mT)


@jaxtyped(typechecker=beartype)
def assert_correlation_identity(
    raw_beta: BetaMatrix,
    rows: UnitNormRows,
    indices: IndexVector,
    *,
    atol: float = 1.0e-10,
) -> float:
    """Compare standardized dot products with Torch's Pearson implementation."""

    if raw_beta.shape != rows.tensor.shape:
        raise ValueError("raw and standardized beta matrices must have identical shapes")
    block = raw_beta.index_select(0, indices)
    expected = t.corrcoef(block)
    actual = correlation_block(rows, indices).tensor
    maximum_error = float((expected - actual).abs().max().item())
    if maximum_error > atol:
        raise ValueError(
            f"correlation identity failed: maximum error {maximum_error} exceeds {atol}"
        )
    return maximum_error


@beartype
def target_rank_upper_bound(n_samples: PositiveInt | int) -> int:
    value = int(n_samples)
    if value < 2:
        raise ValueError("at least two samples are required for correlation targets")
    return value - 1


@beartype
def validate_latent_dimension(
    dimension: LatentDimension,
    *,
    embedding_dimension: PositiveInt | int,
    n_samples: PositiveInt | int,
) -> int:
    embedding_ceiling = int(embedding_dimension)
    target_ceiling = target_rank_upper_bound(n_samples)
    useful_ceiling = min(embedding_ceiling, target_ceiling)
    if int(dimension) > useful_ceiling:
        raise ValueError(
            f"latent dimension {int(dimension)} exceeds useful ceiling {useful_ceiling} "
            f"(embedding ceiling={embedding_ceiling}, target ceiling={target_ceiling})"
        )
    return useful_ceiling
