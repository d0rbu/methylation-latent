"""Target-blind split and deterministic sequence-feature artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch as t
from beartype import beartype

from methylation_latent.artifacts import (
    JsonValue,
    sha256_ordered_strings,
    write_canonical_json_exclusive,
)
from methylation_latent.domain import (
    Autosome,
    NonEmptyProbeSet,
    PositiveInt,
    ProbeLocus,
    WindowSize,
)
from methylation_latent.genome import IndexedFasta, extract_cpg_window
from methylation_latent.splits import (
    GenomicSplit,
    NestedGenomicSplit,
    add_diverse_validation_split,
    build_diverse_block_split,
    build_held_out_chromosome_split,
)
from methylation_latent.storage import load_exact_safetensors, save_safetensors_exclusive

_SPLIT_SCHEMA = "methylation-latent.nested-genomic-split.v1"
_FEATURE_SCHEMA = "methylation-latent.sequence-features.v1"
_SPLIT_KEYS = {
    "optimization_indices",
    "validation_indices",
    "test_indices",
    "primary_buffer_indices",
    "validation_buffer_indices",
}


def _probe_order_sha256(probes: NonEmptyProbeSet) -> str:
    return sha256_ordered_strings(str(probe.probe_id) for probe in probes.probes)


@dataclass(frozen=True, slots=True)
class SplitArtifact:
    """One exhaustive partition of a fixed global probe universe."""

    name: str
    universe_size: int
    optimization_indices: t.Tensor
    validation_indices: t.Tensor
    test_indices: t.Tensor
    primary_buffer_indices: t.Tensor
    validation_buffer_indices: t.Tensor

    def __post_init__(self) -> None:
        if not self.name or self.universe_size <= 0:
            raise ValueError("split name and positive universe size are required")
        named = (
            ("optimization", self.optimization_indices),
            ("validation", self.validation_indices),
            ("test", self.test_indices),
            ("primary buffer", self.primary_buffer_indices),
            ("validation buffer", self.validation_buffer_indices),
        )
        if any(values.dtype != t.int64 or values.ndim != 1 for _, values in named):
            raise ValueError("split partitions must be int64 vectors")
        if any(
            values.numel() == 0
            for name, values in named
            if name in {"optimization", "validation", "test"}
        ):
            raise ValueError("optimization, validation, and test partitions must be non-empty")
        for name, values in named:
            if bool(t.any((values < 0) | (values >= self.universe_size)).item()):
                raise IndexError(f"{name} split index is outside the global probe universe")
            if t.unique(values).numel() != values.numel():
                raise ValueError(f"{name} split indices contain duplicates")
        concatenated = t.cat(tuple(values for _, values in named))
        expected = t.arange(self.universe_size, dtype=t.int64)
        if concatenated.numel() != self.universe_size or not t.equal(
            t.sort(concatenated).values,
            expected,
        ):
            raise ValueError("split partitions do not exactly and disjointly cover the universe")

    def probe_set(self, probes: NonEmptyProbeSet, partition: str) -> NonEmptyProbeSet:
        if len(probes) != self.universe_size:
            raise ValueError("probe universe size differs from split artifact")
        by_name = {
            "optimization": self.optimization_indices,
            "validation": self.validation_indices,
            "test": self.test_indices,
        }
        if partition not in by_name:
            raise ValueError(f"unknown non-empty split partition: {partition!r}")
        return NonEmptyProbeSet(
            tuple(probes.probes[index] for index in by_name[partition].tolist())
        )

    @property
    def primary_train_indices(self) -> t.Tensor:
        """All non-test, non-primary-buffer probes used for the final refit."""

        values = t.cat(
            (
                self.optimization_indices,
                self.validation_indices,
                self.validation_buffer_indices,
            )
        )
        return t.sort(values).values


def _indices_for_loci(
    probes: NonEmptyProbeSet,
    loci: tuple[ProbeLocus, ...],
) -> t.Tensor:
    by_probe_id = {str(probe.probe_id): index for index, probe in enumerate(probes.probes)}
    probe_ids = tuple(str(locus.probe_id) for locus in loci)
    missing = set(probe_ids) - set(by_probe_id)
    if missing:
        raise ValueError(f"split loci are absent from the global universe: {sorted(missing)[:10]}")
    return t.tensor(tuple(by_probe_id[probe_id] for probe_id in probe_ids), dtype=t.int64)


@beartype
def split_artifact_from_nested(
    name: str,
    probes: NonEmptyProbeSet,
    nested: NestedGenomicSplit,
) -> SplitArtifact:
    validation_split = nested.validation_within_primary_train
    return SplitArtifact(
        name=name,
        universe_size=len(probes),
        optimization_indices=_indices_for_loci(
            probes,
            tuple(nested.optimization.probes),
        ),
        validation_indices=_indices_for_loci(
            probes,
            tuple(nested.validation.probes),
        ),
        test_indices=_indices_for_loci(probes, tuple(nested.test.probes)),
        primary_buffer_indices=_indices_for_loci(
            probes,
            tuple(nested.primary.buffer_excluded),
        ),
        validation_buffer_indices=_indices_for_loci(
            probes,
            tuple(validation_split.buffer_excluded),
        ),
    )


@dataclass(frozen=True, slots=True)
class BuiltSplits:
    diverse: SplitArtifact
    held_out_chromosome: SplitArtifact
    diverse_nested: NestedGenomicSplit
    held_out_chromosome_nested: NestedGenomicSplit


@beartype
def build_primary_splits(
    probes: NonEmptyProbeSet,
    *,
    chromosome_lengths: dict[int, int],
    block_width: PositiveInt,
    anchors_per_context: PositiveInt,
    held_out_chromosome: Autosome,
    window_sizes: tuple[WindowSize, ...],
    primary_seed: int,
    validation_seed: int,
) -> BuiltSplits:
    """Build both split families with separately buffered validation partitions."""

    diverse_primary = build_diverse_block_split(
        probes,
        chromosome_lengths=chromosome_lengths,
        block_width=block_width,
        anchors_per_context=anchors_per_context,
        window_sizes=window_sizes,
        seed=primary_seed,
    )
    diverse_nested = add_diverse_validation_split(
        diverse_primary,
        chromosome_lengths=chromosome_lengths,
        block_width=block_width,
        anchors_per_context=anchors_per_context,
        seed=validation_seed,
    )
    chromosome_primary = build_held_out_chromosome_split(
        probes,
        held_out_chromosome=held_out_chromosome,
        window_sizes=window_sizes,
        seed=primary_seed,
    )
    chromosome_nested = add_diverse_validation_split(
        chromosome_primary,
        chromosome_lengths=chromosome_lengths,
        block_width=block_width,
        anchors_per_context=anchors_per_context,
        seed=validation_seed,
    )
    return BuiltSplits(
        diverse=split_artifact_from_nested(
            "diverse_blocks",
            probes,
            diverse_nested,
        ),
        held_out_chromosome=split_artifact_from_nested(
            f"held_out_chr{int(held_out_chromosome)}",
            probes,
            chromosome_nested,
        ),
        diverse_nested=diverse_nested,
        held_out_chromosome_nested=chromosome_nested,
    )


def _split_summary(split: GenomicSplit) -> dict[str, JsonValue]:
    return {
        "kind": split.kind.value,
        "seed": split.seed,
        "train_probes": len(split.train),
        "test_probes": len(split.test),
        "buffer_excluded_probes": len(split.buffer_excluded),
        "window_sizes": [int(window) for window in split.window_sizes],
        "blocks": [
            {
                "anchor_probe_id": str(block.anchor_probe_id),
                "anchor_context": block.anchor_context.value,
                "chromosome": block.interval.chromosome,
                "start": block.interval.start,
                "end": block.interval.end,
            }
            for block in split.blocks
        ],
        "overlap_audits": [
            {
                "window_size": int(audit.window_size),
                "shared_chromosomes": audit.shared_chromosomes,
                "minimum_cytosine_distance": audit.minimum_cytosine_distance,
            }
            for audit in split.overlap_audits
        ],
    }


@beartype
def save_split_artifact_exclusive(
    tensor_path: Path,
    metadata_path: Path,
    split: SplitArtifact,
    nested: NestedGenomicSplit,
    probes: NonEmptyProbeSet,
) -> None:
    if split.universe_size != len(probes):
        raise ValueError("split artifact and probe table sizes differ")
    save_safetensors_exclusive(
        tensor_path,
        {
            "optimization_indices": split.optimization_indices,
            "validation_indices": split.validation_indices,
            "test_indices": split.test_indices,
            "primary_buffer_indices": split.primary_buffer_indices,
            "validation_buffer_indices": split.validation_buffer_indices,
        },
    )
    write_canonical_json_exclusive(
        metadata_path,
        {
            "schema": _SPLIT_SCHEMA,
            "name": split.name,
            "tensor_file": tensor_path.name,
            "probe_order_sha256": _probe_order_sha256(probes),
            "universe_size": split.universe_size,
            "primary": _split_summary(nested.primary),
            "validation_within_primary_train": _split_summary(
                nested.validation_within_primary_train
            ),
        },
    )


@beartype
def load_split_artifact(
    tensor_path: Path,
    metadata_path: Path,
    *,
    probes: NonEmptyProbeSet,
) -> SplitArtifact:
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "schema",
        "name",
        "tensor_file",
        "probe_order_sha256",
        "universe_size",
        "primary",
        "validation_within_primary_train",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("split metadata envelope differs")
    if (
        raw["schema"] != _SPLIT_SCHEMA
        or raw["tensor_file"] != tensor_path.name
        or raw["probe_order_sha256"] != _probe_order_sha256(probes)
        or raw["universe_size"] != len(probes)
    ):
        raise ValueError("split metadata identity differs from the requested probe universe")
    tensors = load_exact_safetensors(tensor_path, _SPLIT_KEYS)
    return SplitArtifact(
        name=str(raw["name"]),
        universe_size=len(probes),
        optimization_indices=tensors["optimization_indices"],
        validation_indices=tensors["validation_indices"],
        test_indices=tensors["test_indices"],
        primary_buffer_indices=tensors["primary_buffer_indices"],
        validation_buffer_indices=tensors["validation_buffer_indices"],
    )


@dataclass(frozen=True, slots=True)
class SequenceFeatureArtifact:
    """CpG density and GC content aligned to one probe order for every window."""

    probe_count: int
    by_window: dict[int, t.Tensor]

    def __post_init__(self) -> None:
        if self.probe_count <= 0 or not self.by_window:
            raise ValueError("sequence-feature artifact must be non-empty")
        if any(window <= 0 or window % 2 != 0 for window in self.by_window):
            raise ValueError("sequence-feature windows must be positive and even")
        for features in self.by_window.values():
            if features.shape != (self.probe_count, 2) or features.dtype != t.float64:
                raise ValueError("sequence features must have shape [probes, 2] and float64 dtype")
            if not bool(
                t.isfinite(features).all().item()
                and t.all((features >= 0.0) & (features <= 1.0)).item()
            ):
                raise ValueError("sequence features must be finite values in [0, 1]")


@beartype
def compute_sequence_feature_artifact(
    reference: IndexedFasta,
    probes: NonEmptyProbeSet,
    window_sizes: tuple[WindowSize, ...],
) -> SequenceFeatureArtifact:
    if not window_sizes or len(set(map(int, window_sizes))) != len(window_sizes):
        raise ValueError("sequence-feature windows must be non-empty and unique")
    maximum = max(window_sizes, key=int)
    rows = {int(window): t.empty((len(probes), 2), dtype=t.float64) for window in window_sizes}
    maximum_width = int(maximum)
    for probe_index, probe in enumerate(probes.probes):
        maximum_sequence = extract_cpg_window(reference, probe, maximum)
        for window in window_sizes:
            width = int(window)
            offset = (maximum_width - width) // 2
            sequence = maximum_sequence[offset : offset + width]
            if sequence[width // 2 : width // 2 + 2] != "CG":
                raise RuntimeError("nested sequence-feature window lost its centred CpG")
            gc_count = sequence.count("G") + sequence.count("C")
            # "CG" has no proper self-overlap, so str.count equals the number of
            # adjacent C→G transitions used by the Torch reference implementation.
            cpg_count = sequence.count("CG")
            rows[width][probe_index] = t.tensor(
                (cpg_count / (width - 1), gc_count / width),
                dtype=t.float64,
            )
    return SequenceFeatureArtifact(
        probe_count=len(probes),
        by_window=rows,
    )


@beartype
def save_sequence_features_exclusive(
    tensor_path: Path,
    metadata_path: Path,
    features: SequenceFeatureArtifact,
    probes: NonEmptyProbeSet,
) -> None:
    if features.probe_count != len(probes):
        raise ValueError("sequence-feature artifact and probe table sizes differ")
    save_safetensors_exclusive(
        tensor_path,
        {f"window_{window}": values for window, values in features.by_window.items()},
    )
    write_canonical_json_exclusive(
        metadata_path,
        {
            "schema": _FEATURE_SCHEMA,
            "tensor_file": tensor_path.name,
            "probe_order_sha256": _probe_order_sha256(probes),
            "probe_count": len(probes),
            "window_sizes": sorted(features.by_window),
            "columns": ["cpg_density", "gc_content"],
        },
    )


@beartype
def load_sequence_features(
    tensor_path: Path,
    metadata_path: Path,
    *,
    probes: NonEmptyProbeSet,
) -> SequenceFeatureArtifact:
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "schema",
        "tensor_file",
        "probe_order_sha256",
        "probe_count",
        "window_sizes",
        "columns",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("sequence-feature metadata envelope differs")
    windows_raw = raw["window_sizes"]
    if not isinstance(windows_raw, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in windows_raw
    ):
        raise ValueError("sequence-feature window list is malformed")
    windows = tuple(int(value) for value in windows_raw)
    if (
        raw["schema"] != _FEATURE_SCHEMA
        or raw["tensor_file"] != tensor_path.name
        or raw["probe_order_sha256"] != _probe_order_sha256(probes)
        or raw["probe_count"] != len(probes)
        or raw["columns"] != ["cpg_density", "gc_content"]
    ):
        raise ValueError("sequence-feature identity differs from the requested probe universe")
    tensors = load_exact_safetensors(
        tensor_path,
        {f"window_{window}" for window in windows},
    )
    return SequenceFeatureArtifact(
        probe_count=len(probes),
        by_window={window: tensors[f"window_{window}"] for window in windows},
    )
