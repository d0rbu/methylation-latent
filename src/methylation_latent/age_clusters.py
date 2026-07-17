"""Deterministic post-hoc diagnostics for age-association scatter lobes."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch as t
from beartype import beartype
from jaxtyping import Float64, Int64, jaxtyped

Float64Matrix = Float64[t.Tensor, "rows columns"]
Float64Vector = Float64[t.Tensor, "rows"]
Float64ColumnVector = Float64[t.Tensor, "columns"]
Int64Vector = Int64[t.Tensor, "rows"]
Int64ContingencyMatrix = Int64[t.Tensor, "clusters categories"]


def _require_float64_matrix(values: t.Tensor, name: str) -> None:
    if values.dtype != t.float64 or values.ndim != 2 or 0 in values.shape:
        raise TypeError(f"{name} must be a non-empty float64 matrix")
    if not bool(t.isfinite(values).all().item()):
        raise ValueError(f"{name} must be finite")


def _require_binary_labels(labels: t.Tensor, *, expected_count: int | None = None) -> None:
    if labels.dtype != t.int64 or labels.ndim != 1 or labels.numel() == 0:
        raise TypeError("cluster labels must be a non-empty int64 vector")
    if expected_count is not None and labels.numel() != expected_count:
        raise ValueError("cluster labels do not align to the observations")
    if not t.equal(t.unique(labels), t.tensor((0, 1), dtype=t.int64)):
        raise ValueError("cluster labels must contain exactly the values zero and one")


@beartype
@dataclass(frozen=True, slots=True)
class KMeansTwoConfig:
    restarts: int
    maximum_iterations: int
    seed: int

    def __post_init__(self) -> None:
        if self.restarts <= 0 or self.maximum_iterations <= 0:
            raise ValueError("k-means restart and iteration counts must be positive")


@dataclass(frozen=True, slots=True)
class KMeansTwoResult:
    labels: Int64Vector
    centers: Float64Matrix
    standardized_centers: Float64Matrix
    inertia: float
    iterations: int

    def __post_init__(self) -> None:
        _require_binary_labels(self.labels)
        for values, name in (
            (self.centers, "k-means centers"),
            (self.standardized_centers, "standardized k-means centers"),
        ):
            _require_float64_matrix(values, name)
            if values.shape != (2, 2):
                raise ValueError(f"{name} must have shape [2,2]")
        if not math.isfinite(self.inertia) or self.inertia < 0.0 or self.iterations <= 0:
            raise ValueError("k-means inertia and iteration count are invalid")
        if not float(self.centers[0, 0].item()) < float(self.centers[1, 0].item()):
            raise ValueError("k-means clusters must be ordered by empirical age association")


@dataclass(frozen=True, slots=True)
class CategoricalAssociation:
    contingency: Int64ContingencyMatrix
    cramer_v: float

    def __post_init__(self) -> None:
        if (
            self.contingency.dtype != t.int64
            or self.contingency.ndim != 2
            or self.contingency.shape[0] != 2
            or self.contingency.shape[1] == 0
            or bool(t.any(self.contingency < 0).item())
            or int(self.contingency.sum().item()) == 0
        ):
            raise ValueError("categorical contingency must be a non-empty 2-by-k count matrix")
        if not math.isfinite(self.cramer_v) or not 0.0 <= self.cramer_v <= 1.0:
            raise ValueError("Cramer's V must lie in [0,1]")


@dataclass(frozen=True, slots=True)
class ContinuousContrast:
    means: tuple[float, float]
    standard_deviations: tuple[float, float]
    standardized_mean_difference: float

    def __post_init__(self) -> None:
        values = (*self.means, *self.standard_deviations, self.standardized_mean_difference)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("continuous cluster contrast must be finite")
        if any(value < 0.0 for value in self.standard_deviations):
            raise ValueError("continuous cluster standard deviations must be non-negative")


@dataclass(frozen=True, slots=True)
class SubgroupCorrelations:
    correlations: Float64Vector
    sample_count: int

    def __post_init__(self) -> None:
        if (
            self.correlations.dtype != t.float64
            or self.correlations.ndim != 1
            or self.correlations.numel() == 0
        ):
            raise TypeError("subgroup correlations must be a non-empty float64 vector")
        if self.sample_count < 3:
            raise ValueError("a subgroup correlation requires at least three samples")
        if not bool(t.isfinite(self.correlations).all().item()) or bool(
            t.any(t.abs(self.correlations) > 1.0 + 8.0 * t.finfo(t.float64).eps).item()
        ):
            raise ValueError("subgroup correlations must be finite and bounded")


@jaxtyped(typechecker=beartype)
def _standardize_columns(
    values: Float64Matrix,
) -> tuple[Float64Matrix, Float64ColumnVector, Float64ColumnVector]:
    mean = values.mean(dim=0)
    scale = values.std(dim=0, correction=0)
    if bool(t.any(scale == 0.0).item()):
        raise ValueError("k-means input contains a constant column")
    return (values - mean) / scale, mean, scale


@jaxtyped(typechecker=beartype)
def fit_kmeans_two(values: t.Tensor, *, config: KMeansTwoConfig) -> KMeansTwoResult:
    """Fit seeded k=2 means in standardized 2D scatter space with multiple restarts."""

    _require_float64_matrix(values, "k-means input")
    if values.shape[0] < 4 or values.shape[1] != 2:
        raise ValueError("k=2 scatter clustering requires shape [at least four,2]")
    standardized, mean, scale = _standardize_columns(values)
    generator = t.Generator(device="cpu").manual_seed(config.seed)
    best: tuple[float, t.Tensor, t.Tensor, int] | None = None
    for _ in range(config.restarts):
        first = int(t.randint(values.shape[0], size=(), generator=generator).item())
        squared = t.sum(t.square(standardized - standardized[first]), dim=1)
        if float(squared.sum().item()) == 0.0:
            raise ValueError("k-means input contains only one distinct point")
        second = int(t.multinomial(squared, 1, generator=generator).item())
        centers = standardized[t.tensor((first, second), dtype=t.int64)].clone()
        previous: t.Tensor | None = None
        for _iteration in range(1, config.maximum_iterations + 1):
            distances = t.sum(t.square(standardized[:, None, :] - centers[None, :, :]), dim=2)
            labels = t.argmin(distances, dim=1)
            if t.unique(labels).numel() != 2:
                raise RuntimeError("k-means produced an empty cluster")
            if previous is not None and t.equal(labels, previous):
                break
            previous = labels
            centers = t.stack(
                tuple(standardized[labels == cluster].mean(dim=0) for cluster in (0, 1))
            )
        else:
            raise RuntimeError("k-means failed to converge within the declared iteration limit")
        distances = t.sum(t.square(standardized - centers.index_select(0, labels)), dim=1)
        inertia = float(distances.sum().item())
        if best is None or inertia < best[0]:
            best = inertia, labels.clone(), centers.clone(), _iteration
    if best is None:
        raise RuntimeError("k-means did not execute any restart")
    inertia, labels, standardized_centers, iterations = best
    centers = standardized_centers * scale + mean
    order = t.argsort(centers[:, 0], stable=True)
    if float(centers[order[0], 0].item()) == float(centers[order[1], 0].item()):
        raise RuntimeError("k-means centers cannot be ordered by empirical association")
    inverse = t.empty_like(order)
    inverse[order] = t.arange(2, dtype=t.int64)
    ordered_labels = inverse.index_select(0, labels)
    return KMeansTwoResult(
        labels=ordered_labels,
        centers=centers.index_select(0, order),
        standardized_centers=standardized_centers.index_select(0, order),
        inertia=inertia,
        iterations=iterations,
    )


@jaxtyped(typechecker=beartype)
def categorical_association(labels: t.Tensor, categories: t.Tensor) -> CategoricalAssociation:
    """Compute the exact two-by-k table and uncorrected Cramer's V."""

    _require_binary_labels(labels)
    if categories.dtype != t.int64 or categories.ndim != 1:
        raise TypeError("categories must be an int64 vector")
    if categories.shape != labels.shape or bool(t.any(categories < 0).item()):
        raise ValueError("categories must align to labels and be non-negative")
    unique, inverse = t.unique(categories, sorted=True, return_inverse=True)
    category_count = unique.numel()
    contingency = t.bincount(
        labels * category_count + inverse, minlength=2 * category_count
    ).reshape(2, category_count)
    if category_count == 1:
        return CategoricalAssociation(contingency, 0.0)
    observed = contingency.to(t.float64)
    expected = observed.sum(dim=1, keepdim=True) @ observed.sum(dim=0, keepdim=True)
    expected /= observed.sum()
    if bool(t.any(expected == 0.0).item()):
        raise RuntimeError("categorical association has an empty expected cell margin")
    chi_squared = t.sum(t.square(observed - expected) / expected)
    cramer_v = math.sqrt(float((chi_squared / observed.sum()).item()))
    return CategoricalAssociation(contingency, cramer_v)


@jaxtyped(typechecker=beartype)
def continuous_contrast(labels: t.Tensor, values: t.Tensor) -> ContinuousContrast:
    """Summarize a continuous feature by cluster with signed pooled Cohen's d."""

    _require_binary_labels(labels)
    if values.dtype != t.float64 or values.ndim != 1:
        raise TypeError("continuous feature values must be a float64 vector")
    if values.shape != labels.shape or not bool(t.isfinite(values).all().item()):
        raise ValueError("continuous feature values must be aligned and finite")
    groups = tuple(values[labels == cluster] for cluster in (0, 1))
    if any(group.numel() < 2 for group in groups):
        raise ValueError("continuous contrast requires at least two observations per cluster")
    means = (float(groups[0].mean().item()), float(groups[1].mean().item()))
    deviations = (
        float(groups[0].std(correction=1).item()),
        float(groups[1].std(correction=1).item()),
    )
    numerator = t.stack(
        tuple((group.numel() - 1) * group.var(correction=1) for group in groups)
    ).sum()
    denominator = sum(group.numel() for group in groups) - 2
    pooled = t.sqrt(numerator / denominator)
    if float(pooled.item()) == 0.0:
        raise ValueError("continuous feature is constant within both clusters")
    effect = (means[1] - means[0]) / float(pooled.item())
    return ContinuousContrast(means, deviations, effect)


@jaxtyped(typechecker=beartype)
def permutation_invariant_label_agreement(first: t.Tensor, second: t.Tensor) -> float:
    _require_binary_labels(first)
    _require_binary_labels(second, expected_count=first.numel())
    direct = float((first == second).to(t.float64).mean().item())
    return max(direct, 1.0 - direct)


@jaxtyped(typechecker=beartype)
def adjusted_rand_index(first: t.Tensor, second: t.Tensor) -> float:
    """Adjusted Rand index for two non-degenerate binary partitions."""

    _require_binary_labels(first)
    _require_binary_labels(second, expected_count=first.numel())
    table = t.bincount(first * 2 + second, minlength=4).reshape(2, 2).to(t.float64)

    def choose_two(value: t.Tensor) -> t.Tensor:
        return value * (value - 1.0) / 2.0

    pairs = choose_two(table).sum()
    row_pairs = choose_two(table.sum(dim=1)).sum()
    column_pairs = choose_two(table.sum(dim=0)).sum()
    total_pairs = choose_two(t.tensor(float(first.numel()), dtype=t.float64))
    expected = row_pairs * column_pairs / total_pairs
    maximum = (row_pairs + column_pairs) / 2.0
    denominator = maximum - expected
    if float(denominator.item()) == 0.0:
        raise ValueError("adjusted Rand index is undefined for these partitions")
    result = float(((pairs - expected) / denominator).item())
    if not -1.0 <= result <= 1.0:
        raise RuntimeError("adjusted Rand index escaped its mathematical bounds")
    return result


@jaxtyped(typechecker=beartype)
def subgroup_age_correlations(
    standardized_methylation: t.Tensor,
    standardized_age: t.Tensor,
    subgroup_mask: t.Tensor,
) -> SubgroupCorrelations:
    """Recenter and unit-normalize probes and age within one sample subgroup."""

    _require_float64_matrix(standardized_methylation, "standardized methylation")
    if (
        standardized_age.dtype != t.float64
        or standardized_age.ndim != 1
        or standardized_age.numel() != standardized_methylation.shape[1]
        or not bool(t.isfinite(standardized_age).all().item())
    ):
        raise TypeError("standardized age must be a finite aligned float64 vector")
    if subgroup_mask.dtype != t.bool or subgroup_mask.shape != standardized_age.shape:
        raise TypeError("subgroup mask must be an aligned boolean vector")
    sample_count = int(subgroup_mask.sum().item())
    if sample_count < 3:
        raise ValueError("a subgroup correlation requires at least three selected samples")
    methylation = standardized_methylation[:, subgroup_mask]
    methylation = methylation - methylation.mean(dim=1, keepdim=True)
    methylation_norms = t.linalg.vector_norm(methylation, dim=1)
    if bool(t.any(methylation_norms == 0.0).item()):
        rows = t.nonzero(methylation_norms == 0.0).flatten().tolist()
        raise ValueError(f"subgroup contains constant methylation rows: {rows[:10]}")
    age = standardized_age[subgroup_mask]
    age = age - age.mean()
    age_norm = t.linalg.vector_norm(age)
    if float(age_norm.item()) == 0.0:
        raise ValueError("subgroup age is constant")
    correlations = (methylation / methylation_norms[:, None]) @ (age / age_norm)
    tolerance = 8.0 * t.finfo(t.float64).eps
    if bool(t.any(t.abs(correlations) > 1.0 + tolerance).item()):
        raise RuntimeError("subgroup dot products escaped correlation bounds")
    return SubgroupCorrelations(t.clamp(correlations, -1.0, 1.0), sample_count)
