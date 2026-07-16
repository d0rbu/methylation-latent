"""Target-blind genomic splits and exact sequence-overlap audits."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum

import torch as t
from beartype import beartype

from methylation_latent.domain import (
    Autosome,
    GenomicContext,
    NonEmptyProbeSet,
    PositiveInt,
    ProbeId,
    ProbeLocus,
    WindowSize,
)
from methylation_latent.genome import GenomicInterval


class SplitKind(StrEnum):
    """Pre-registered split families."""

    DIVERSE_BLOCKS = "diverse_blocks"
    HELD_OUT_CHROMOSOME = "held_out_chromosome"


@beartype
@dataclass(frozen=True, slots=True)
class HeldOutBlock:
    """A target-blind physical test block selected around one anchor locus."""

    anchor_probe_id: ProbeId
    anchor_context: GenomicContext
    interval: GenomicInterval


@dataclass(frozen=True, slots=True)
class OverlapAudit:
    """Result of an exhaustive-by-nearest-neighbour overlap assertion."""

    window_size: WindowSize
    shared_chromosomes: int
    minimum_cytosine_distance: int | None


@dataclass(frozen=True, slots=True)
class GenomicSplit:
    """Disjoint train/test probe sets plus probes removed only by the buffer."""

    kind: SplitKind
    train: NonEmptyProbeSet
    test: NonEmptyProbeSet
    buffer_excluded: tuple[ProbeLocus, ...]
    blocks: tuple[HeldOutBlock, ...]
    window_sizes: tuple[WindowSize, ...]
    seed: int
    overlap_audits: tuple[OverlapAudit, ...]

    def __post_init__(self) -> None:
        if not self.window_sizes:
            raise ValueError("split must record at least one window size")
        if len(set(map(int, self.window_sizes))) != len(self.window_sizes):
            raise ValueError("split window sizes must be unique")
        partitions = (
            {probe.probe_id for probe in self.train.probes},
            {probe.probe_id for probe in self.test.probes},
            {probe.probe_id for probe in self.buffer_excluded},
        )
        if (
            partitions[0] & partitions[1]
            or partitions[0] & partitions[2]
            or partitions[1] & partitions[2]
        ):
            raise ValueError("train, test, and buffer-excluded probe IDs must be disjoint")
        if tuple(audit.window_size for audit in self.overlap_audits) != self.window_sizes:
            raise ValueError("overlap audits must exactly match the configured window order")


@dataclass(frozen=True, slots=True)
class NestedGenomicSplit:
    """Primary test split plus a separately buffered validation split inside its train side."""

    primary: GenomicSplit
    validation_within_primary_train: GenomicSplit

    def __post_init__(self) -> None:
        validation = self.validation_within_primary_train
        if validation.window_sizes != self.primary.window_sizes:
            raise ValueError("primary and validation splits must use identical window sweeps")
        original_train_ids = {probe.probe_id for probe in self.primary.train.probes}
        repartitioned_ids = {
            probe.probe_id
            for probe in (
                *validation.train.probes,
                *validation.test.probes,
                *validation.buffer_excluded,
            )
        }
        if original_train_ids != repartitioned_ids:
            raise ValueError("validation split does not exactly repartition the primary train side")
        assert_no_window_overlap(
            validation.test,
            self.primary.test,
            self.primary.window_sizes,
        )

    @property
    def optimization(self) -> NonEmptyProbeSet:
        return self.validation_within_primary_train.train

    @property
    def validation(self) -> NonEmptyProbeSet:
        return self.validation_within_primary_train.test

    @property
    def test(self) -> NonEmptyProbeSet:
        return self.primary.test


@beartype
def add_diverse_validation_split(
    primary: GenomicSplit,
    *,
    chromosome_lengths: Mapping[int, int],
    block_width: PositiveInt | int,
    anchors_per_context: PositiveInt | int,
    seed: int,
) -> NestedGenomicSplit:
    """Select validation blocks without ever reopening the sealed primary test universe."""

    validation = build_diverse_block_split(
        primary.train,
        chromosome_lengths=chromosome_lengths,
        block_width=block_width,
        anchors_per_context=anchors_per_context,
        window_sizes=primary.window_sizes,
        seed=seed,
    )
    return NestedGenomicSplit(
        primary=primary,
        validation_within_primary_train=validation,
    )


def _by_chromosome(probes: Iterable[ProbeLocus]) -> dict[int, tuple[ProbeLocus, ...]]:
    grouped: defaultdict[int, list[ProbeLocus]] = defaultdict(list)
    for probe in probes:
        grouped[int(probe.chromosome)].append(probe)
    return {
        chromosome: tuple(sorted(loci, key=lambda probe: (int(probe.position), probe.probe_id)))
        for chromosome, loci in grouped.items()
    }


def _positions(probes: tuple[ProbeLocus, ...]) -> t.Tensor:
    return t.tensor(tuple(int(probe.position) for probe in probes), dtype=t.int64)


def _nearest_distances(query: t.Tensor, reference: t.Tensor) -> t.Tensor:
    """Return exact nearest absolute distances for sorted non-empty int64 vectors."""

    if query.ndim != 1 or reference.ndim != 1 or query.numel() == 0 or reference.numel() == 0:
        raise ValueError("nearest-distance inputs must be non-empty vectors")
    if query.dtype != t.int64 or reference.dtype != t.int64:
        raise TypeError("nearest-distance inputs must be int64")
    if not bool(t.all(query[1:] >= query[:-1]).item()):
        raise ValueError("query positions must be sorted")
    if not bool(t.all(reference[1:] >= reference[:-1]).item()):
        raise ValueError("reference positions must be sorted")

    insertion = t.searchsorted(reference, query)
    right_indices = insertion.clamp(max=reference.numel() - 1)
    left_indices = (insertion - 1).clamp(min=0)
    right = (query - reference.index_select(0, right_indices)).abs()
    left = (query - reference.index_select(0, left_indices)).abs()
    return t.minimum(left, right)


@beartype
def assert_no_window_overlap(
    train: NonEmptyProbeSet,
    test: NonEmptyProbeSet,
    window_sizes: tuple[WindowSize, ...],
) -> tuple[OverlapAudit, ...]:
    """Fail if any equally sized train/test input windows overlap."""

    if not window_sizes:
        raise ValueError("overlap audit requires at least one window size")
    train_ids = {probe.probe_id for probe in train.probes}
    test_ids = {probe.probe_id for probe in test.probes}
    if train_ids & test_ids:
        raise ValueError("overlap audit received duplicate train/test probe IDs")

    train_by_chromosome = _by_chromosome(train.probes)
    test_by_chromosome = _by_chromosome(test.probes)
    shared = sorted(train_by_chromosome.keys() & test_by_chromosome.keys())
    chromosome_minima = tuple(
        int(
            _nearest_distances(
                _positions(train_by_chromosome[chromosome]),
                _positions(test_by_chromosome[chromosome]),
            )
            .min()
            .item()
        )
        for chromosome in shared
    )
    minimum = min(chromosome_minima) if chromosome_minima else None

    audits: list[OverlapAudit] = []
    for window_size in window_sizes:
        if minimum is not None and minimum < int(window_size):
            raise ValueError(
                f"train/test sequence windows overlap for width {int(window_size)}: "
                f"minimum same-chromosome cytosine distance is {minimum}"
            )
        audits.append(
            OverlapAudit(
                window_size=window_size,
                shared_chromosomes=len(shared),
                minimum_cytosine_distance=minimum,
            )
        )
    return tuple(audits)


@beartype
def apply_maximum_window_buffer(
    candidates: NonEmptyProbeSet,
    test: NonEmptyProbeSet,
    maximum_window_size: WindowSize,
) -> tuple[NonEmptyProbeSet, tuple[ProbeLocus, ...]]:
    """Remove candidates at cytosine distance <= the maximum test-window width."""

    candidate_ids = {probe.probe_id for probe in candidates.probes}
    test_ids = {probe.probe_id for probe in test.probes}
    if candidate_ids & test_ids:
        raise ValueError("buffer candidates and test probes must be disjoint")

    test_by_chromosome = _by_chromosome(test.probes)
    retained: list[ProbeLocus] = []
    excluded: list[ProbeLocus] = []
    for chromosome, chromosome_candidates in _by_chromosome(candidates.probes).items():
        if chromosome not in test_by_chromosome:
            retained.extend(chromosome_candidates)
            continue
        distances = _nearest_distances(
            _positions(chromosome_candidates),
            _positions(test_by_chromosome[chromosome]),
        )
        for probe, distance in zip(chromosome_candidates, distances.tolist(), strict=True):
            destination = excluded if distance <= int(maximum_window_size) else retained
            destination.append(probe)

    if not retained:
        raise ValueError("maximum-window buffer removed every training probe")
    return NonEmptyProbeSet.from_iterable(retained), tuple(excluded)


def _block_for_probe(
    probe: ProbeLocus,
    *,
    block_width: int,
    chromosome_length: int,
) -> HeldOutBlock | None:
    cytosine_zero = int(probe.position) - 1
    start = cytosine_zero - block_width // 2
    end = start + block_width
    if start < 0 or end > chromosome_length:
        return None
    return HeldOutBlock(
        anchor_probe_id=probe.probe_id,
        anchor_context=probe.context,
        interval=GenomicInterval(
            chromosome=f"chr{int(probe.chromosome)}",
            start=start,
            end=end,
        ),
    )


def _torch_permutation(values: tuple[int, ...], generator: t.Generator) -> tuple[int, ...]:
    if not values:
        return ()
    order = t.randperm(len(values), generator=generator).tolist()
    return tuple(values[index] for index in order)


def _choose_blocks(
    probes: NonEmptyProbeSet,
    *,
    chromosome_lengths: Mapping[int, int],
    block_width: int,
    anchors_per_context: int,
    generator: t.Generator,
) -> tuple[HeldOutBlock, ...]:
    selected: list[HeldOutBlock] = []
    for context in GenomicContext:
        eligible_by_chromosome: defaultdict[int, list[HeldOutBlock]] = defaultdict(list)
        for probe in probes.probes:
            chromosome = int(probe.chromosome)
            if probe.context != context:
                continue
            if chromosome not in chromosome_lengths:
                raise KeyError(f"missing chromosome length for chromosome {chromosome}")
            block = _block_for_probe(
                probe,
                block_width=block_width,
                chromosome_length=chromosome_lengths[chromosome],
            )
            if block is not None:
                eligible_by_chromosome[chromosome].append(block)
        if len(eligible_by_chromosome) < anchors_per_context:
            raise ValueError(
                f"context {context.value} has eligible anchors on only "
                f"{len(eligible_by_chromosome)} chromosomes; need {anchors_per_context}"
            )

        chosen_for_context = 0
        chromosome_order = _torch_permutation(tuple(sorted(eligible_by_chromosome)), generator)
        for chromosome in chromosome_order:
            candidates = tuple(eligible_by_chromosome[chromosome])
            candidate_order = _torch_permutation(tuple(range(len(candidates))), generator)
            chosen = next(
                (
                    candidates[index]
                    for index in candidate_order
                    if all(
                        not candidates[index].interval.overlaps(block.interval)
                        for block in selected
                    )
                ),
                None,
            )
            if chosen is None:
                continue
            selected.append(chosen)
            chosen_for_context += 1
            if chosen_for_context == anchors_per_context:
                break
        if chosen_for_context != anchors_per_context:
            raise ValueError(
                f"could not select {anchors_per_context} non-overlapping blocks for "
                f"context {context.value}"
            )
    return tuple(selected)


def _probe_in_blocks(probe: ProbeLocus, blocks: tuple[HeldOutBlock, ...]) -> bool:
    chromosome = f"chr{int(probe.chromosome)}"
    cytosine_zero = int(probe.position) - 1
    return any(
        block.interval.chromosome == chromosome
        and block.interval.start <= cytosine_zero < block.interval.end
        for block in blocks
    )


def _require_context_coverage(probes: NonEmptyProbeSet, partition: str) -> None:
    observed = {probe.context for probe in probes.probes}
    missing = set(GenomicContext) - observed
    if missing:
        rendered = ", ".join(sorted(context.value for context in missing))
        raise ValueError(f"{partition} split lacks required contexts: {rendered}")


@beartype
def build_diverse_block_split(
    probes: NonEmptyProbeSet,
    *,
    chromosome_lengths: Mapping[int, int],
    block_width: PositiveInt | int,
    anchors_per_context: PositiveInt | int,
    window_sizes: tuple[WindowSize, ...],
    seed: int,
) -> GenomicSplit:
    """Construct chromosome-spread, context-stratified held-out physical blocks."""

    width = int(block_width)
    per_context = int(anchors_per_context)
    if width <= 0 or per_context <= 0:
        raise ValueError("block width and anchors per context must be positive")
    if not window_sizes:
        raise ValueError("diverse split requires at least one window size")
    if min(chromosome_lengths.values(), default=0) <= 0:
        raise ValueError("chromosome lengths must be positive and non-empty")

    generator = t.Generator(device="cpu")
    generator.manual_seed(seed)
    blocks = _choose_blocks(
        probes,
        chromosome_lengths=chromosome_lengths,
        block_width=width,
        anchors_per_context=per_context,
        generator=generator,
    )
    test_probes = tuple(probe for probe in probes.probes if _probe_in_blocks(probe, blocks))
    test_ids = {probe.probe_id for probe in test_probes}
    candidate_probes = tuple(probe for probe in probes.probes if probe.probe_id not in test_ids)
    test = NonEmptyProbeSet(test_probes)
    candidates = NonEmptyProbeSet(candidate_probes)
    train, excluded = apply_maximum_window_buffer(candidates, test, max(window_sizes, key=int))
    _require_context_coverage(train, "train")
    _require_context_coverage(test, "test")
    audits = assert_no_window_overlap(train, test, window_sizes)
    return GenomicSplit(
        kind=SplitKind.DIVERSE_BLOCKS,
        train=train,
        test=test,
        buffer_excluded=excluded,
        blocks=blocks,
        window_sizes=window_sizes,
        seed=seed,
        overlap_audits=audits,
    )


@beartype
def build_held_out_chromosome_split(
    probes: NonEmptyProbeSet,
    *,
    held_out_chromosome: Autosome,
    window_sizes: tuple[WindowSize, ...],
    seed: int,
) -> GenomicSplit:
    """Hold out every eligible probe on one autosome."""

    test_probes = tuple(probe for probe in probes.probes if probe.chromosome == held_out_chromosome)
    train_probes = tuple(
        probe for probe in probes.probes if probe.chromosome != held_out_chromosome
    )
    train = NonEmptyProbeSet(train_probes)
    test = NonEmptyProbeSet(test_probes)
    audits = assert_no_window_overlap(train, test, window_sizes)
    return GenomicSplit(
        kind=SplitKind.HELD_OUT_CHROMOSOME,
        train=train,
        test=test,
        buffer_excluded=(),
        blocks=(),
        window_sizes=window_sizes,
        seed=seed,
        overlap_audits=audits,
    )
