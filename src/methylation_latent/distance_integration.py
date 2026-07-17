"""PSD distance kernels and exact simplex-constrained kernel mixtures."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import torch as t
from beartype import beartype
from jaxtyping import Float64, Int64, jaxtyped

from methylation_latent.domain import NonEmptyProbeSet

KernelValues = Float64[t.Tensor, "pairs kernels"]
PairValues = Float64[t.Tensor, "pairs"]
PairIndices = Int64[t.Tensor, "pairs"]


@dataclass(frozen=True, slots=True)
class KernelDesign:
    """Aligned finite pairwise kernel evaluations with explicit column names."""

    values: t.Tensor
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.values.dtype != t.float64 or self.values.ndim != 2:
            raise TypeError("kernel design must be a rank-2 float64 tensor")
        if self.values.shape[0] == 0 or self.values.shape[1] == 0:
            raise ValueError("kernel design axes must be non-empty")
        if self.values.shape[1] != len(self.names):
            raise ValueError("kernel design width and names differ")
        if len(set(self.names)) != len(self.names) or any(not name for name in self.names):
            raise ValueError("kernel names must be non-empty and unique")
        if not bool(t.isfinite(self.values).all().item()):
            raise ValueError("kernel design must be finite")
        if bool(t.any(self.values.abs() > 1.0 + 1.0e-12).item()):
            raise ValueError("kernel values must lie in [-1,1]")


@dataclass(frozen=True, slots=True)
class SimplexKernelFit:
    """Globally optimal least-squares weights on one kernel simplex."""

    names: tuple[str, ...]
    weights: tuple[float, ...]
    mse: float

    def __post_init__(self) -> None:
        if not self.names or len(self.names) != len(self.weights):
            raise ValueError("kernel-fit names and weights must be aligned and non-empty")
        values = t.tensor(self.weights, dtype=t.float64)
        if not bool(t.isfinite(values).all().item()) or self.mse < 0.0:
            raise ValueError("kernel-fit weights and MSE must be finite and non-negative")
        if bool(t.any(values < 0.0).item()):
            raise ValueError("kernel-fit weights must be non-negative")
        if not bool(t.isclose(values.sum(), t.tensor(1.0, dtype=t.float64), atol=1.0e-10).item()):
            raise ValueError("kernel-fit weights must sum to one")


def _validate_pair_indices(
    probes: NonEmptyProbeSet,
    left: t.Tensor,
    right: t.Tensor,
) -> None:
    if left.dtype != t.int64 or right.dtype != t.int64 or left.ndim != 1 or right.ndim != 1:
        raise TypeError("pair indices must be rank-1 int64 tensors")
    if left.shape != right.shape or left.numel() == 0:
        raise ValueError("pair indices must be aligned and non-empty")
    if bool(t.any((left < 0) | (right < 0) | (left >= len(probes)) | (right >= len(probes))).item()):
        raise IndexError("pair index is outside the probe universe")


@jaxtyped(typechecker=beartype)
def distance_kernel_design(
    probes: NonEmptyProbeSet,
    left: PairIndices,
    right: PairIndices,
    *,
    length_scales: tuple[int, ...],
) -> KernelDesign:
    """Evaluate a global kernel and chromosome-block Laplacian kernels."""

    _validate_pair_indices(probes, left, right)
    if not length_scales or any(scale <= 0 for scale in length_scales):
        raise ValueError("distance-kernel length scales must be positive and non-empty")
    if tuple(sorted(set(length_scales))) != length_scales:
        raise ValueError("distance-kernel length scales must be unique and increasing")

    chromosomes = t.tensor(
        tuple(int(probe.chromosome) for probe in probes.probes),
        dtype=t.int64,
    )
    positions = t.tensor(
        tuple(int(probe.position) for probe in probes.probes),
        dtype=t.int64,
    )
    same_chromosome = chromosomes.index_select(0, left) == chromosomes.index_select(0, right)
    distance = (
        positions.index_select(0, left) - positions.index_select(0, right)
    ).abs().to(t.float64)
    same = same_chromosome.to(t.float64)
    values = [t.ones(left.numel(), dtype=t.float64)]
    values.extend(same * t.exp(-distance / scale) for scale in length_scales)
    return KernelDesign(
        values=t.stack(values, dim=1),
        names=("global", *(f"cis_laplacian_{scale}" for scale in length_scales)),
    )


@jaxtyped(typechecker=beartype)
def append_sequence_kernel(
    distance: KernelDesign,
    sequence_similarity: PairValues,
) -> KernelDesign:
    """Append one bounded sequence-cosine kernel to an aligned distance design."""

    if sequence_similarity.dtype != t.float64 or sequence_similarity.ndim != 1:
        raise TypeError("sequence similarity must be a rank-1 float64 tensor")
    if sequence_similarity.shape[0] != distance.values.shape[0]:
        raise ValueError("sequence similarities and distance design are misaligned")
    if not bool(t.isfinite(sequence_similarity).all().item()):
        raise ValueError("sequence similarities must be finite")
    if bool(t.any(sequence_similarity.abs() > 1.0 + 1.0e-6).item()):
        raise ValueError("sequence similarities fall outside cosine bounds")
    return KernelDesign(
        values=t.cat((distance.values, sequence_similarity[:, None]), dim=1),
        names=(*distance.names, "sequence_cosine"),
    )


def _candidate_weights(
    gram: t.Tensor,
    cross: t.Tensor,
    active: tuple[int, ...],
    *,
    tolerance: float,
) -> t.Tensor | None:
    indices = t.tensor(active, dtype=t.int64)
    active_gram = gram.index_select(0, indices).index_select(1, indices)
    active_cross = cross.index_select(0, indices)
    size = len(active)
    system = t.zeros((size + 1, size + 1), dtype=t.float64)
    system[:size, :size] = active_gram
    system[:size, size] = 1.0
    system[size, :size] = 1.0
    right_hand_side = t.cat((active_cross, t.ones(1, dtype=t.float64)))
    solution = t.linalg.lstsq(system, right_hand_side).solution
    residual = system @ solution - right_hand_side
    if float(residual.abs().max().item()) > tolerance:
        return None
    active_weights = solution[:size]
    if float(active_weights.min().item()) < -tolerance:
        return None
    active_weights = t.where(active_weights.abs() <= tolerance, 0.0, active_weights)
    weights = t.zeros(gram.shape[0], dtype=t.float64)
    weights.index_copy_(0, indices, active_weights)
    weights /= weights.sum()
    return weights


@jaxtyped(typechecker=beartype)
def fit_simplex_kernel(
    design: KernelDesign,
    target: PairValues,
) -> SimplexKernelFit:
    """Solve convex least squares exactly by enumerating all active kernel sets."""

    if target.dtype != t.float64 or target.ndim != 1:
        raise TypeError("kernel-fit target must be a rank-1 float64 tensor")
    if target.shape[0] != design.values.shape[0] or target.numel() < 2:
        raise ValueError("kernel-fit design and target must align on at least two pairs")
    if not bool(t.isfinite(target).all().item()):
        raise ValueError("kernel-fit target must be finite")
    if bool(t.any(target.abs() > 1.0 + 1.0e-12).item()):
        raise ValueError("kernel-fit correlations must lie in [-1,1]")

    count = target.numel()
    gram = design.values.mT @ design.values / count
    cross = design.values.mT @ target / count
    target_square = t.mean(t.square(target))
    tolerance = 1.0e-9
    candidates: list[tuple[float, t.Tensor]] = []
    for size in range(1, design.values.shape[1] + 1):
        for active in combinations(range(design.values.shape[1]), size):
            weights = _candidate_weights(gram, cross, active, tolerance=tolerance)
            if weights is None:
                continue
            objective = weights @ gram @ weights - 2.0 * weights @ cross + target_square
            candidates.append((float(objective.item()), weights))
    if not candidates:
        raise RuntimeError("simplex kernel fit found no feasible active set")
    _, weights = min(candidates, key=lambda item: (item[0], tuple(item[1].tolist())))
    prediction = design.values @ weights
    mse = float(t.mean(t.square(target - prediction)).item())

    active = weights > tolerance
    gradient = gram @ weights - cross
    multiplier = -gradient[active].mean()
    reduced_gradient = gradient + multiplier
    if (
        float(reduced_gradient[active].abs().max().item()) > 1.0e-7
        or (bool((~active).any().item()) and float(reduced_gradient[~active].min().item()) < -1.0e-7)
    ):
        raise RuntimeError("simplex kernel fit failed its global KKT audit")
    return SimplexKernelFit(
        names=design.names,
        weights=tuple(float(value) for value in weights.tolist()),
        mse=mse,
    )


@jaxtyped(typechecker=beartype)
def predict_kernel_mixture(
    design: KernelDesign,
    fit: SimplexKernelFit,
) -> PairValues:
    """Apply an exact named simplex fit to aligned kernel evaluations."""

    if design.names != fit.names:
        raise ValueError("kernel design and fitted weight names differ")
    weights = t.tensor(fit.weights, dtype=t.float64)
    prediction = design.values @ weights
    if not bool(t.isfinite(prediction).all().item()) or bool(
        t.any(prediction.abs() > 1.0 + 1.0e-12).item()
    ):
        raise RuntimeError("convex kernel prediction escaped correlation bounds")
    return prediction
