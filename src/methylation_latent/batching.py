"""Target-blind batches assembled from local genomic neighbourhoods."""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass, field

import torch as t
from beartype import beartype
from jaxtyping import Int64, jaxtyped

from methylation_latent.domain import NonEmptyProbeSet, PositiveInt, ProbeId

BatchIndices = Int64[t.Tensor, "batch"]


@dataclass(frozen=True, slots=True)
class ProbeBatch:
    """Unique probe indices and IDs in a target-blind sampling order."""

    indices: t.Tensor
    probe_ids: tuple[ProbeId, ...]

    def __post_init__(self) -> None:
        if self.indices.dtype != t.int64 or self.indices.ndim != 1 or self.indices.numel() == 0:
            raise ValueError("batch indices must be a non-empty int64 vector")
        if self.indices.numel() != len(self.probe_ids):
            raise ValueError("batch index and probe-ID lengths differ")
        if t.unique(self.indices).numel() != self.indices.numel():
            raise ValueError("batch indices must be unique")
        if len(set(self.probe_ids)) != len(self.probe_ids):
            raise ValueError("batch probe IDs must be unique")


@beartype
@dataclass(slots=True)
class GenomicBatchSampler:
    """Repeatable local-neighbourhood sampler that never receives target values."""

    probes: NonEmptyProbeSet
    batch_size: PositiveInt
    neighbourhood_width: PositiveInt
    seed: int
    _generator: t.Generator = field(init=False, repr=False)
    _indices_by_chromosome: dict[int, tuple[int, ...]] = field(init=False, repr=False)
    _positions_by_chromosome: dict[int, tuple[int, ...]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if int(self.batch_size) > len(self.probes):
            raise ValueError("batch size cannot exceed the probe universe")
        self._generator = t.Generator(device="cpu")
        self._generator.manual_seed(self.seed)
        grouped: defaultdict[int, list[int]] = defaultdict(list)
        for index, probe in enumerate(self.probes.probes):
            grouped[int(probe.chromosome)].append(index)
        self._indices_by_chromosome = {
            chromosome: tuple(
                sorted(
                    indices,
                    key=lambda index: (
                        int(self.probes.probes[index].position),
                        self.probes.probes[index].probe_id,
                    ),
                )
            )
            for chromosome, indices in grouped.items()
        }
        self._positions_by_chromosome = {
            chromosome: tuple(int(self.probes.probes[index].position) for index in indices)
            for chromosome, indices in self._indices_by_chromosome.items()
        }

    @jaxtyped(typechecker=beartype)
    def sample(self) -> ProbeBatch:
        """Fill one unique batch by repeatedly choosing an unused random anchor."""

        selected: list[int] = []
        selected_set: set[int] = set()
        half_left = int(self.neighbourhood_width) // 2
        half_right = int(self.neighbourhood_width) - half_left

        while len(selected) < int(self.batch_size):
            if len(selected_set) == len(self.probes):
                raise RuntimeError("sampler exhausted probes before filling the batch")
            anchor_index = int(
                t.randint(len(self.probes), size=(), generator=self._generator).item()
            )
            while anchor_index in selected_set:
                anchor_index = int(
                    t.randint(len(self.probes), size=(), generator=self._generator).item()
                )
            anchor = self.probes.probes[anchor_index]
            anchor_position = int(anchor.position)
            lower = anchor_position - half_left
            upper = anchor_position + half_right
            chromosome = int(anchor.chromosome)
            positions = self._positions_by_chromosome[chromosome]
            chromosome_indices = self._indices_by_chromosome[chromosome]
            start = bisect_left(positions, lower)
            stop = bisect_left(positions, upper)
            neighbourhood = tuple(
                index for index in chromosome_indices[start:stop] if index not in selected_set
            )
            ordered = tuple(
                sorted(
                    neighbourhood,
                    key=lambda index: (
                        index != anchor_index,
                        abs(int(self.probes.probes[index].position) - anchor_position),
                        int(self.probes.probes[index].position),
                        self.probes.probes[index].probe_id,
                    ),
                )
            )
            if not ordered or anchor_index not in ordered:
                raise RuntimeError("anchor is absent from its own genomic neighbourhood")
            remaining = int(self.batch_size) - len(selected)
            additions = ordered[:remaining]
            selected.extend(additions)
            selected_set.update(additions)

        indices = t.tensor(selected, dtype=t.int64)
        probe_ids = tuple(self.probes.probes[index].probe_id for index in selected)
        return ProbeBatch(indices=indices, probe_ids=probe_ids)
