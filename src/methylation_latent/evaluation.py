"""Immutable pair populations, distance baselines, metrics, and latent projections."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum, StrEnum

import torch as t
from beartype import beartype
from jaxtyping import Float, Int64, jaxtyped

from methylation_latent.domain import NonEmptyProbeSet
from methylation_latent.targets import UnitNormRows

PairIndexVector = Int64[t.Tensor, "pairs"]
ValueVector = Float[t.Tensor, "pairs"]
LatentMatrix = Float[t.Tensor, "probes latent"]


class PairPopulation(StrEnum):
    """Evaluation populations that must never be pooled together."""

    SEEN_BY_HELD_OUT = "seen_by_held_out"
    HELD_OUT_BY_HELD_OUT = "held_out_by_held_out"
    TRAINING_BY_TRAINING = "training_by_training"


class DistanceClass(IntEnum):
    """Frozen cis distance bins plus a distinct trans class."""

    CIS_0_1KB = 0
    CIS_1_4KB = 1
    CIS_4_16KB = 2
    CIS_16_64KB = 3
    CIS_64_256KB = 4
    CIS_256KB_1MB = 5
    CIS_1MB_PLUS = 6
    TRANS = 7


DISTANCE_CLASS_LABELS: tuple[str, ...] = (
    "cis_0_1kb",
    "cis_1_4kb",
    "cis_4_16kb",
    "cis_16_64kb",
    "cis_64_256kb",
    "cis_256kb_1mb",
    "cis_1mb_plus",
    "trans",
)
_CIS_BOUNDARIES = (1_000, 4_000, 16_000, 64_000, 256_000, 1_000_000)


@dataclass(frozen=True, slots=True)
class PairIndices:
    """A deterministic pair-index artifact over one ordered probe universe."""

    population: PairPopulation
    left: t.Tensor
    right: t.Tensor
    distance_class: t.Tensor
    total_possible_pairs: int
    seed: int

    def __post_init__(self) -> None:
        vectors = (self.left, self.right, self.distance_class)
        if any(vector.dtype != t.int64 or vector.ndim != 1 for vector in vectors):
            raise ValueError("pair indices and distance classes must be int64 vectors")
        if self.left.numel() == 0:
            raise ValueError("pair-index artifact must not be empty")
        if not (self.left.shape == self.right.shape == self.distance_class.shape):
            raise ValueError("pair-index vectors must have identical shapes")
        if bool(t.any((self.left < 0) | (self.right < 0)).item()):
            raise ValueError("pair indices must be non-negative")
        if bool(t.any(self.left == self.right).item()):
            raise ValueError("evaluation pairs must not contain diagonal entries")
        if bool(
            t.any(
                (self.distance_class < int(DistanceClass.CIS_0_1KB))
                | (self.distance_class > int(DistanceClass.TRANS))
            ).item()
        ):
            raise ValueError("pair artifact contains an unknown distance class")
        if self.total_possible_pairs < self.left.numel():
            raise ValueError("sampled pair count exceeds total possible pairs")
        encoding_base = max(int(self.left.max().item()), int(self.right.max().item())) + 1
        encoded = self.left * encoding_base + self.right
        if t.unique(encoded).numel() != encoded.numel():
            raise ValueError("pair-index artifact contains duplicate ordered pairs")
        if self.population != PairPopulation.SEEN_BY_HELD_OUT and bool(
            t.any(self.left >= self.right).item()
        ):
            raise ValueError("within-partition pairs must be stored once with left < right")

    @property
    def count(self) -> int:
        return self.left.numel()


def _validate_partition_indices(indices: t.Tensor, universe_size: int, name: str) -> None:
    if indices.dtype != t.int64 or indices.ndim != 1 or indices.numel() == 0:
        raise ValueError(f"{name} indices must be a non-empty int64 vector")
    if bool(t.any((indices < 0) | (indices >= universe_size)).item()):
        raise IndexError(f"{name} partition index is outside the probe universe")
    if t.unique(indices).numel() != indices.numel():
        raise ValueError(f"{name} partition indices must be unique")


def _validate_pair_index_vector(indices: t.Tensor, universe_size: int, name: str) -> None:
    if indices.dtype != t.int64 or indices.ndim != 1 or indices.numel() == 0:
        raise ValueError(f"{name} pair indices must be a non-empty int64 vector")
    if bool(t.any((indices < 0) | (indices >= universe_size)).item()):
        raise IndexError(f"{name} pair index is outside the probe universe")


def _floyd_sample(population_size: int, sample_size: int, generator: t.Generator) -> t.Tensor:
    """Uniformly sample sorted unique integer ranks using O(sample_size) memory."""

    if not 0 < sample_size <= population_size:
        raise ValueError("sample size must lie in [1, population size]")
    selected: set[int] = set()
    for upper in range(population_size - sample_size, population_size):
        candidate = int(t.randint(upper + 1, size=(), generator=generator).item())
        selected.add(upper if candidate in selected else candidate)
    if len(selected) != sample_size:
        raise RuntimeError("Floyd sampler did not produce the requested number of ranks")
    return t.tensor(sorted(selected), dtype=t.int64)


def _combination_from_rank(rank: int, size: int) -> tuple[int, int]:
    """Map a lexicographic rank to one unordered pair without floating arithmetic."""

    if rank < 0 or rank >= size * (size - 1) // 2:
        raise IndexError("combination rank is out of bounds")
    low = 0
    high = size - 1
    while low + 1 < high:
        middle = (low + high) // 2
        cumulative = middle * (2 * size - middle - 1) // 2
        if cumulative <= rank:
            low = middle
        else:
            high = middle
    left = low
    cumulative = left * (2 * size - left - 1) // 2
    right = left + 1 + rank - cumulative
    if not 0 <= left < right < size:
        raise RuntimeError("invalid pair produced from combination rank")
    return left, right


@beartype
def classify_distances(
    probes: NonEmptyProbeSet,
    left: t.Tensor,
    right: t.Tensor,
) -> t.Tensor:
    """Assign frozen cis distance bins and never give trans pairs a fake distance."""

    _validate_pair_index_vector(left, len(probes), "left")
    _validate_pair_index_vector(right, len(probes), "right")
    if left.shape != right.shape:
        raise ValueError("left and right pair vectors must have identical shapes")
    chromosomes = t.tensor(tuple(int(probe.chromosome) for probe in probes.probes), dtype=t.int64)
    positions = t.tensor(tuple(int(probe.position) for probe in probes.probes), dtype=t.int64)
    left_chromosomes = chromosomes.index_select(0, left)
    right_chromosomes = chromosomes.index_select(0, right)
    is_trans = left_chromosomes != right_chromosomes
    distances = (positions.index_select(0, left) - positions.index_select(0, right)).abs()
    boundaries = t.tensor(_CIS_BOUNDARIES, dtype=t.int64)
    classes = t.bucketize(distances, boundaries, right=True)
    classes[is_trans] = int(DistanceClass.TRANS)
    return classes


def _pair_limit(total: int, maximum_pairs: int | None) -> int:
    if total <= 0:
        raise ValueError("pair population must contain at least one possible pair")
    if maximum_pairs is None:
        if total > 10_000_000:
            raise ValueError(
                "pair population exceeds 10,000,000; configure an explicit deterministic cap"
            )
        return total
    if maximum_pairs <= 0:
        raise ValueError("maximum pair count must be positive")
    return min(total, maximum_pairs)


@beartype
def build_cross_partition_pairs(
    probes: NonEmptyProbeSet,
    seen_indices: t.Tensor,
    held_out_indices: t.Tensor,
    *,
    maximum_pairs: int | None,
    seed: int,
) -> PairIndices:
    """Build or uniformly subsample the seen-by-held-out Cartesian product."""

    _validate_partition_indices(seen_indices, len(probes), "seen")
    _validate_partition_indices(held_out_indices, len(probes), "held-out")
    if bool(t.isin(seen_indices, held_out_indices).any().item()):
        raise ValueError("seen and held-out partitions must be disjoint")
    total = seen_indices.numel() * held_out_indices.numel()
    count = _pair_limit(total, maximum_pairs)
    generator = t.Generator(device="cpu")
    generator.manual_seed(seed)
    flat = (
        t.arange(total, dtype=t.int64) if count == total else _floyd_sample(total, count, generator)
    )
    right_size = held_out_indices.numel()
    left = seen_indices.index_select(0, flat // right_size)
    right = held_out_indices.index_select(0, flat % right_size)
    return PairIndices(
        population=PairPopulation.SEEN_BY_HELD_OUT,
        left=left,
        right=right,
        distance_class=classify_distances(probes, left, right),
        total_possible_pairs=total,
        seed=seed,
    )


@beartype
def build_within_partition_pairs(
    probes: NonEmptyProbeSet,
    indices: t.Tensor,
    *,
    population: PairPopulation,
    maximum_pairs: int | None,
    seed: int,
) -> PairIndices:
    """Build or uniformly subsample unique unordered pairs within one partition."""

    if population == PairPopulation.SEEN_BY_HELD_OUT:
        raise ValueError("cross-partition population is invalid for within-partition pairs")
    _validate_partition_indices(indices, len(probes), "within-partition")
    if indices.numel() < 2:
        raise ValueError("within-partition evaluation requires at least two probes")
    size = indices.numel()
    total = size * (size - 1) // 2
    count = _pair_limit(total, maximum_pairs)
    generator = t.Generator(device="cpu")
    generator.manual_seed(seed)
    if count == total:
        local_left, local_right = t.triu_indices(size, size, offset=1)
    else:
        ranks = _floyd_sample(total, count, generator)
        combinations = tuple(_combination_from_rank(int(rank), size) for rank in ranks.tolist())
        local_left = t.tensor(tuple(pair[0] for pair in combinations), dtype=t.int64)
        local_right = t.tensor(tuple(pair[1] for pair in combinations), dtype=t.int64)
    left = indices.index_select(0, local_left)
    right = indices.index_select(0, local_right)
    canonical_left = t.minimum(left, right)
    canonical_right = t.maximum(left, right)
    order = t.argsort(canonical_left * len(probes) + canonical_right, stable=True)
    canonical_left = canonical_left.index_select(0, order)
    canonical_right = canonical_right.index_select(0, order)
    return PairIndices(
        population=population,
        left=canonical_left,
        right=canonical_right,
        distance_class=classify_distances(probes, canonical_left, canonical_right),
        total_possible_pairs=total,
        seed=seed,
    )


def _sorted_partition(
    probes: NonEmptyProbeSet,
    indices: t.Tensor,
    name: str,
) -> dict[int, tuple[tuple[int, ...], tuple[int, ...]]]:
    _validate_partition_indices(indices, len(probes), name)
    grouped: defaultdict[int, list[int]] = defaultdict(list)
    for index in indices.tolist():
        grouped[int(probes.probes[index].chromosome)].append(index)
    result: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    for chromosome, chromosome_indices in grouped.items():
        ordered = tuple(
            sorted(
                chromosome_indices,
                key=lambda index: (
                    int(probes.probes[index].position),
                    probes.probes[index].probe_id,
                ),
            )
        )
        result[chromosome] = (
            ordered,
            tuple(int(probes.probes[index].position) for index in ordered),
        )
    return result


def _sample_segments(
    segment_left: list[int],
    segment_right_indices: list[tuple[int, ...]],
    segment_starts: list[int],
    segment_stops: list[int],
    *,
    maximum_pairs: int,
    generator: t.Generator,
) -> tuple[t.Tensor, t.Tensor, int]:
    if not (
        len(segment_left) == len(segment_right_indices) == len(segment_starts) == len(segment_stops)
    ):
        raise RuntimeError("pair-sampling segment fields are misaligned")
    counts = tuple(stop - start for start, stop in zip(segment_starts, segment_stops, strict=True))
    if any(count <= 0 for count in counts):
        raise RuntimeError("pair-sampling segments must be non-empty")
    cumulative: list[int] = []
    total = 0
    for count in counts:
        total += count
        cumulative.append(total)
    if total == 0:
        return t.empty(0, dtype=t.int64), t.empty(0, dtype=t.int64), 0
    count = min(total, maximum_pairs)
    ranks = (
        t.arange(total, dtype=t.int64) if count == total else _floyd_sample(total, count, generator)
    )
    sampled_left: list[int] = []
    sampled_right: list[int] = []
    for rank in ranks.tolist():
        segment = bisect_right(cumulative, rank)
        previous = 0 if segment == 0 else cumulative[segment - 1]
        offset = rank - previous
        sampled_left.append(segment_left[segment])
        sampled_right.append(segment_right_indices[segment][segment_starts[segment] + offset])
    return (
        t.tensor(sampled_left, dtype=t.int64),
        t.tensor(sampled_right, dtype=t.int64),
        total,
    )


def _within_cis_segments(
    partition: dict[int, tuple[tuple[int, ...], tuple[int, ...]]],
    *,
    lower: int,
    upper: int | None,
) -> tuple[list[int], list[tuple[int, ...]], list[int], list[int]]:
    segment_left: list[int] = []
    segment_right_indices: list[tuple[int, ...]] = []
    segment_starts: list[int] = []
    segment_stops: list[int] = []
    for indices, positions in partition.values():
        for local_left, (global_left, position) in enumerate(zip(indices, positions, strict=True)):
            start = bisect_left(positions, position + lower, lo=local_left + 1)
            stop = (
                len(positions)
                if upper is None
                else bisect_left(positions, position + upper, lo=local_left + 1)
            )
            if start < stop:
                segment_left.append(global_left)
                segment_right_indices.append(indices)
                segment_starts.append(start)
                segment_stops.append(stop)
    return segment_left, segment_right_indices, segment_starts, segment_stops


def _cross_cis_segments(
    left_partition: dict[int, tuple[tuple[int, ...], tuple[int, ...]]],
    right_partition: dict[int, tuple[tuple[int, ...], tuple[int, ...]]],
    *,
    lower: int,
    upper: int | None,
) -> tuple[list[int], list[tuple[int, ...]], list[int], list[int]]:
    segment_left: list[int] = []
    segment_right_indices: list[tuple[int, ...]] = []
    segment_starts: list[int] = []
    segment_stops: list[int] = []
    for chromosome in sorted(left_partition.keys() & right_partition.keys()):
        left_indices, left_positions = left_partition[chromosome]
        right_indices, right_positions = right_partition[chromosome]
        for global_left, position in zip(left_indices, left_positions, strict=True):
            left_start = 0 if upper is None else bisect_left(right_positions, position - upper + 1)
            left_stop = (
                bisect_left(right_positions, position)
                if lower == 0
                else bisect_left(right_positions, position - lower + 1)
            )
            right_start = bisect_left(right_positions, position + lower)
            right_stop = (
                len(right_positions)
                if upper is None
                else bisect_left(right_positions, position + upper)
            )
            for start, stop in ((left_start, left_stop), (right_start, right_stop)):
                if start < stop:
                    segment_left.append(global_left)
                    segment_right_indices.append(right_indices)
                    segment_starts.append(start)
                    segment_stops.append(stop)
    return segment_left, segment_right_indices, segment_starts, segment_stops


def _sample_trans_blocks(
    left_partition: dict[int, tuple[tuple[int, ...], tuple[int, ...]]],
    right_partition: dict[int, tuple[tuple[int, ...], tuple[int, ...]]],
    *,
    within_partition: bool,
    maximum_pairs: int,
    generator: t.Generator,
) -> tuple[t.Tensor, t.Tensor, int]:
    blocks: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for left_chromosome, (left_indices, _) in left_partition.items():
        for right_chromosome, (right_indices, _) in right_partition.items():
            allowed = (
                left_chromosome < right_chromosome
                if within_partition
                else left_chromosome != right_chromosome
            )
            if allowed:
                blocks.append((left_indices, right_indices))
    block_counts = tuple(len(left) * len(right) for left, right in blocks)
    cumulative: list[int] = []
    total = 0
    for count in block_counts:
        total += count
        cumulative.append(total)
    if total == 0:
        return t.empty(0, dtype=t.int64), t.empty(0, dtype=t.int64), 0
    count = min(total, maximum_pairs)
    ranks = (
        t.arange(total, dtype=t.int64) if count == total else _floyd_sample(total, count, generator)
    )
    sampled_left: list[int] = []
    sampled_right: list[int] = []
    for rank in ranks.tolist():
        block = bisect_right(cumulative, rank)
        previous = 0 if block == 0 else cumulative[block - 1]
        within_rank = rank - previous
        left_indices, right_indices = blocks[block]
        right_size = len(right_indices)
        sampled_left.append(left_indices[within_rank // right_size])
        sampled_right.append(right_indices[within_rank % right_size])
    return (
        t.tensor(sampled_left, dtype=t.int64),
        t.tensor(sampled_right, dtype=t.int64),
        total,
    )


def _distance_bounds(distance_class: DistanceClass) -> tuple[int, int | None]:
    if distance_class == DistanceClass.TRANS:
        raise ValueError("trans pairs do not have finite distance bounds")
    boundaries = (0, *_CIS_BOUNDARIES)
    lower = boundaries[int(distance_class)]
    upper = (
        None
        if distance_class == DistanceClass.CIS_1MB_PLUS
        else boundaries[int(distance_class) + 1]
    )
    return lower, upper


def _finalize_stratified_pairs(
    probes: NonEmptyProbeSet,
    *,
    population: PairPopulation,
    left_parts: list[t.Tensor],
    right_parts: list[t.Tensor],
    class_parts: list[t.Tensor],
    total_possible_pairs: int,
    seed: int,
) -> PairIndices:
    if not left_parts:
        raise ValueError("distance-stratified sampling found no possible pairs")
    left = t.cat(left_parts)
    right = t.cat(right_parts)
    classes = t.cat(class_parts)
    if population != PairPopulation.SEEN_BY_HELD_OUT:
        canonical_left = t.minimum(left, right)
        canonical_right = t.maximum(left, right)
        left, right = canonical_left, canonical_right
    order = t.argsort(classes * len(probes) * len(probes) + left * len(probes) + right)
    left = left.index_select(0, order)
    right = right.index_select(0, order)
    classes = classes.index_select(0, order)
    observed_classes = classify_distances(probes, left, right)
    if not t.equal(classes, observed_classes):
        raise RuntimeError("distance-stratified pair construction disagrees with its classifier")
    return PairIndices(
        population=population,
        left=left,
        right=right,
        distance_class=classes,
        total_possible_pairs=total_possible_pairs,
        seed=seed,
    )


@beartype
def build_stratified_within_partition_pairs(
    probes: NonEmptyProbeSet,
    indices: t.Tensor,
    *,
    population: PairPopulation,
    maximum_pairs_per_distance_class: int,
    seed: int,
) -> PairIndices:
    """Uniformly sample within each cis bin and trans without enumerating all pairs."""

    if population == PairPopulation.SEEN_BY_HELD_OUT:
        raise ValueError("cross-partition population is invalid for within-partition pairs")
    if maximum_pairs_per_distance_class <= 0:
        raise ValueError("maximum pairs per distance class must be positive")
    partition = _sorted_partition(probes, indices, "within-partition")
    generator = t.Generator(device="cpu").manual_seed(seed)
    left_parts: list[t.Tensor] = []
    right_parts: list[t.Tensor] = []
    class_parts: list[t.Tensor] = []
    for distance_class in tuple(DistanceClass)[:-1]:
        lower, upper = _distance_bounds(distance_class)
        left, right, _ = _sample_segments(
            *_within_cis_segments(partition, lower=lower, upper=upper),
            maximum_pairs=maximum_pairs_per_distance_class,
            generator=generator,
        )
        if left.numel() > 0:
            left_parts.append(left)
            right_parts.append(right)
            class_parts.append(t.full_like(left, fill_value=int(distance_class), dtype=t.int64))
    trans_left, trans_right, _ = _sample_trans_blocks(
        partition,
        partition,
        within_partition=True,
        maximum_pairs=maximum_pairs_per_distance_class,
        generator=generator,
    )
    if trans_left.numel() > 0:
        left_parts.append(trans_left)
        right_parts.append(trans_right)
        class_parts.append(
            t.full_like(trans_left, fill_value=int(DistanceClass.TRANS), dtype=t.int64)
        )
    total = indices.numel() * (indices.numel() - 1) // 2
    return _finalize_stratified_pairs(
        probes,
        population=population,
        left_parts=left_parts,
        right_parts=right_parts,
        class_parts=class_parts,
        total_possible_pairs=total,
        seed=seed,
    )


@beartype
def build_stratified_cross_partition_pairs(
    probes: NonEmptyProbeSet,
    seen_indices: t.Tensor,
    held_out_indices: t.Tensor,
    *,
    maximum_pairs_per_distance_class: int,
    seed: int,
) -> PairIndices:
    """Uniformly sample seen-by-held-out pairs separately within every distance class."""

    if maximum_pairs_per_distance_class <= 0:
        raise ValueError("maximum pairs per distance class must be positive")
    _validate_partition_indices(seen_indices, len(probes), "seen")
    _validate_partition_indices(held_out_indices, len(probes), "held-out")
    if bool(t.isin(seen_indices, held_out_indices).any().item()):
        raise ValueError("seen and held-out partitions must be disjoint")
    seen = _sorted_partition(probes, seen_indices, "seen")
    held_out = _sorted_partition(probes, held_out_indices, "held-out")
    generator = t.Generator(device="cpu").manual_seed(seed)
    left_parts: list[t.Tensor] = []
    right_parts: list[t.Tensor] = []
    class_parts: list[t.Tensor] = []
    for distance_class in tuple(DistanceClass)[:-1]:
        lower, upper = _distance_bounds(distance_class)
        left, right, _ = _sample_segments(
            *_cross_cis_segments(seen, held_out, lower=lower, upper=upper),
            maximum_pairs=maximum_pairs_per_distance_class,
            generator=generator,
        )
        if left.numel() > 0:
            left_parts.append(left)
            right_parts.append(right)
            class_parts.append(t.full_like(left, fill_value=int(distance_class), dtype=t.int64))
    trans_left, trans_right, _ = _sample_trans_blocks(
        seen,
        held_out,
        within_partition=False,
        maximum_pairs=maximum_pairs_per_distance_class,
        generator=generator,
    )
    if trans_left.numel() > 0:
        left_parts.append(trans_left)
        right_parts.append(trans_right)
        class_parts.append(
            t.full_like(trans_left, fill_value=int(DistanceClass.TRANS), dtype=t.int64)
        )
    return _finalize_stratified_pairs(
        probes,
        population=PairPopulation.SEEN_BY_HELD_OUT,
        left_parts=left_parts,
        right_parts=right_parts,
        class_parts=class_parts,
        total_possible_pairs=seen_indices.numel() * held_out_indices.numel(),
        seed=seed,
    )


@jaxtyped(typechecker=beartype)
def gather_pair_targets(rows: UnitNormRows, pairs: PairIndices) -> t.Tensor:
    return gather_pair_targets_chunked(rows, pairs, chunk_size=pairs.count)


@beartype
def gather_pair_targets_chunked(
    rows: UnitNormRows,
    pairs: PairIndices,
    *,
    chunk_size: int,
) -> t.Tensor:
    """Gather exact dot products without materializing two full pair-by-sample matrices."""

    if chunk_size <= 0:
        raise ValueError("pair-target chunk size must be positive")
    if bool(
        t.any(
            (pairs.left < 0)
            | (pairs.right < 0)
            | (pairs.left >= rows.n_rows)
            | (pairs.right >= rows.n_rows)
        ).item()
    ):
        raise IndexError("pair target index is outside the standardized probe matrix")
    chunks = tuple(
        t.sum(
            rows.tensor.index_select(0, pairs.left[start:stop])
            * rows.tensor.index_select(0, pairs.right[start:stop]),
            dim=1,
        )
        for start in range(0, pairs.count, chunk_size)
        for stop in (min(start + chunk_size, pairs.count),)
    )
    return t.cat(chunks)


@jaxtyped(typechecker=beartype)
def gather_pair_predictions(
    normalized_latent: LatentMatrix,
    pairs: PairIndices,
) -> t.Tensor:
    return gather_pair_predictions_chunked(
        normalized_latent,
        pairs,
        chunk_size=pairs.count,
    )


@beartype
def gather_pair_predictions_chunked(
    normalized_latent: t.Tensor,
    pairs: PairIndices,
    *,
    chunk_size: int,
) -> t.Tensor:
    """Gather latent dot products with bounded pair-by-latent temporary storage."""

    if chunk_size <= 0:
        raise ValueError("pair-prediction chunk size must be positive")
    if normalized_latent.ndim != 2 or 0 in normalized_latent.shape:
        raise ValueError("latent matrix must be non-empty and rank two")
    if not normalized_latent.is_floating_point() or not bool(
        t.isfinite(normalized_latent).all().item()
    ):
        raise ValueError("latent matrix must be finite and floating")
    norms = t.linalg.vector_norm(normalized_latent, dim=1)
    tolerance = 1.0e-10 if normalized_latent.dtype == t.float64 else 2.0e-5
    if not bool(t.allclose(norms, t.ones_like(norms), atol=tolerance, rtol=0.0)):
        raise ValueError("latent rows must be unit norm before pair prediction")
    if bool(
        t.any(
            (pairs.left < 0)
            | (pairs.right < 0)
            | (pairs.left >= normalized_latent.shape[0])
            | (pairs.right >= normalized_latent.shape[0])
        ).item()
    ):
        raise IndexError("pair prediction index is outside the latent matrix")
    chunks = tuple(
        t.sum(
            normalized_latent.index_select(0, pairs.left[start:stop])
            * normalized_latent.index_select(0, pairs.right[start:stop]),
            dim=1,
        )
        for start in range(0, pairs.count, chunk_size)
        for stop in (min(start + chunk_size, pairs.count),)
    )
    return t.cat(chunks)


@dataclass(frozen=True, slots=True)
class DistanceBaseline:
    """Frozen training-only mean target for every distance class."""

    means: t.Tensor
    counts: t.Tensor

    def __post_init__(self) -> None:
        if self.means.shape != (len(DistanceClass),) or not self.means.is_floating_point():
            raise ValueError("distance baseline means have an invalid shape or dtype")
        if self.counts.shape != (len(DistanceClass),) or self.counts.dtype != t.int64:
            raise ValueError("distance baseline counts must be an int64 vector of length eight")
        if not bool(t.isfinite(self.means).all().item()):
            raise ValueError("distance baseline means must be finite")
        if bool(t.any(self.counts <= 0).item()):
            raise ValueError("every distance class must have at least one training pair")

    def predict(self, distance_class: t.Tensor) -> t.Tensor:
        if distance_class.dtype != t.int64 or distance_class.ndim != 1:
            raise ValueError("distance classes must be an int64 vector")
        if bool(t.any((distance_class < 0) | (distance_class >= len(DistanceClass))).item()):
            raise ValueError("unknown distance class")
        return self.means.index_select(0, distance_class)


@jaxtyped(typechecker=beartype)
def fit_distance_baseline(
    training_pairs: PairIndices,
    training_targets: ValueVector,
) -> DistanceBaseline:
    if training_pairs.population != PairPopulation.TRAINING_BY_TRAINING:
        raise ValueError("distance baseline must use training-by-training pairs")
    if training_targets.ndim != 1 or training_targets.shape != training_pairs.left.shape:
        raise ValueError("distance baseline target shape does not match pair indices")
    if not training_targets.is_floating_point() or not bool(
        t.isfinite(training_targets).all().item()
    ):
        raise ValueError("distance baseline targets must be finite and floating")
    counts = t.bincount(training_pairs.distance_class, minlength=len(DistanceClass))
    if bool(t.any(counts == 0).item()):
        missing = tuple(
            DISTANCE_CLASS_LABELS[index] for index in t.nonzero(counts == 0).flatten().tolist()
        )
        raise ValueError(f"training pairs do not cover distance classes: {missing}")
    sums = t.zeros(len(DistanceClass), dtype=training_targets.dtype)
    sums.scatter_add_(0, training_pairs.distance_class, training_targets)
    return DistanceBaseline(means=sums / counts.to(training_targets.dtype), counts=counts)


@dataclass(frozen=True, slots=True)
class RegressionMetrics:
    """Complete scalar report for one non-empty evaluation stratum."""

    count: int
    target_mean: float
    target_standard_deviation: float
    prediction_mean: float
    prediction_standard_deviation: float
    mse: float
    pearson: float
    r_squared: float


@jaxtyped(typechecker=beartype)
def regression_metrics(target: ValueVector, prediction: ValueVector) -> RegressionMetrics:
    if target.ndim != 1 or prediction.ndim != 1 or target.shape != prediction.shape:
        raise ValueError("metric target and prediction must be equal-length vectors")
    if target.numel() < 2:
        raise ValueError("at least two values are required for correlation metrics")
    if target.dtype != prediction.dtype:
        raise TypeError("metric target and prediction dtypes differ")
    if not target.is_floating_point() or not bool(
        t.isfinite(target).all().item() and t.isfinite(prediction).all().item()
    ):
        raise ValueError("metric vectors must be finite and floating")
    target_centered = target - target.mean()
    prediction_centered = prediction - prediction.mean()
    target_ss = t.sum(t.square(target_centered))
    prediction_ss = t.sum(t.square(prediction_centered))
    if float(target_ss.item()) == 0.0:
        raise ValueError("R-squared and Pearson are undefined for a constant target")
    if float(prediction_ss.item()) == 0.0:
        raise ValueError("Pearson is undefined for a constant prediction")
    residual_ss = t.sum(t.square(target - prediction))
    pearson = t.sum(target_centered * prediction_centered) / t.sqrt(target_ss * prediction_ss)
    return RegressionMetrics(
        count=target.numel(),
        target_mean=float(target.mean().item()),
        target_standard_deviation=float(target.std(unbiased=False).item()),
        prediction_mean=float(prediction.mean().item()),
        prediction_standard_deviation=float(prediction.std(unbiased=False).item()),
        mse=float(t.mean(t.square(target - prediction)).item()),
        pearson=float(pearson.item()),
        r_squared=float((1.0 - residual_ss / target_ss).item()),
    )


@jaxtyped(typechecker=beartype)
def metrics_by_distance(
    pairs: PairIndices,
    target: ValueVector,
    prediction: ValueVector,
) -> dict[DistanceClass, RegressionMetrics]:
    if target.shape != pairs.left.shape or prediction.shape != pairs.left.shape:
        raise ValueError("metric values must align with pair-index artifact")
    reports: dict[DistanceClass, RegressionMetrics] = {}
    for distance_class in DistanceClass:
        mask = pairs.distance_class == int(distance_class)
        if bool(mask.any().item()):
            reports[distance_class] = regression_metrics(target[mask], prediction[mask])
    if not reports:
        raise RuntimeError("distance stratification produced no reports")
    return reports


@dataclass(frozen=True, slots=True)
class Projection2D:
    """Training-fitted deterministic PCA axes and centring vector."""

    mean: t.Tensor
    axes: t.Tensor

    def __post_init__(self) -> None:
        if self.mean.ndim != 1 or self.axes.shape != (2, self.mean.numel()):
            raise ValueError("projection mean/axes shapes are inconsistent")
        if self.mean.dtype != self.axes.dtype or not self.mean.is_floating_point():
            raise TypeError("projection mean and axes must share a floating dtype")
        if not bool(t.isfinite(self.mean).all().item() and t.isfinite(self.axes).all().item()):
            raise ValueError("projection parameters must be finite")
        gram = self.axes @ self.axes.mT
        tolerance = 1.0e-10 if self.axes.dtype == t.float64 else 2.0e-5
        if not bool(t.allclose(gram, t.eye(2, dtype=self.axes.dtype), atol=tolerance, rtol=0.0)):
            raise ValueError("projection axes must be orthonormal")

    def transform(self, latent: t.Tensor) -> t.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.mean.numel():
            raise ValueError("latent matrix is incompatible with projection")
        if latent.dtype != self.mean.dtype or not bool(t.isfinite(latent).all().item()):
            raise ValueError("latent matrix must be finite and match projection dtype")
        return (latent - self.mean) @ self.axes.mT


@jaxtyped(typechecker=beartype)
def fit_projection_2d(train_latent: LatentMatrix) -> Projection2D:
    """Fit axes on training loci only and orient signs deterministically."""

    if train_latent.ndim != 2 or train_latent.shape[0] < 3 or train_latent.shape[1] < 2:
        raise ValueError("2D projection requires at least three rows and two latent dimensions")
    if not train_latent.is_floating_point() or not bool(t.isfinite(train_latent).all().item()):
        raise ValueError("training latent matrix must be finite and floating")
    mean = train_latent.mean(dim=0)
    centered = train_latent - mean
    _, singular_values, right_vectors = t.linalg.svd(centered, full_matrices=False)
    if singular_values.numel() < 2 or float(singular_values[1].item()) == 0.0:
        raise ValueError("training latent matrix has rank below two")
    axes = right_vectors[:2].clone()
    for axis in axes:
        pivot = int(t.argmax(axis.abs()).item())
        if float(axis[pivot].item()) < 0.0:
            axis.mul_(-1.0)
    return Projection2D(mean=mean, axes=axes)
