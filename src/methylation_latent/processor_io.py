"""Strict processor sample lists and little-endian matrix ingestion."""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import torch as t
from beartype import beartype

from methylation_latent.artifacts import sha256_ordered_strings
from methylation_latent.metadata import Gse87571RawSampleSet

_PROCESSOR_OUTPUT_SCHEMA_FILES = frozenset(
    {
        "probe_ids.txt",
        "sample_order.txt",
        "environment.txt",
    }
)


def _assert_idat_magic(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 4:
        raise ValueError(f"processor IDAT input is absent or empty: {path}")
    with path.open(mode="rb") as handle:
        if handle.read(4) != b"IDAT":
            raise ValueError(f"processor input lacks IDAT magic bytes: {path}")


@beartype
def write_sesame_sample_list_exclusive(
    path: Path,
    samples: Gse87571RawSampleSet,
    *,
    idat_directory: Path,
) -> None:
    """Write exact Sentrix/prefix rows after checking both materialized channels."""

    lines = ["sentrix_identity\tprefix\n"]
    for sample in samples.samples:
        prefix = idat_directory / str(sample.sentrix_identity)
        _assert_idat_magic(Path(f"{prefix}_Grn.idat"))
        _assert_idat_magic(Path(f"{prefix}_Red.idat"))
        lines.append(f"{sample.sentrix_identity}\t{prefix}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode="x", encoding="utf-8", newline="") as handle:
        handle.writelines(lines)
        handle.flush()


@beartype
def write_methylprep_sample_sheet_exclusive(
    path: Path,
    samples: Gse87571RawSampleSet,
    *,
    idat_directory: Path,
) -> None:
    """Write a minimal ordered methylprep sample sheet after auditing both channels."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode="x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("Sample_Name", "Sentrix_ID", "Sentrix_Position"))
        for sample in samples.samples:
            sentrix_id, sentrix_position = str(sample.sentrix_identity).split("_", maxsplit=1)
            prefix = idat_directory / str(sample.sentrix_identity)
            _assert_idat_magic(Path(f"{prefix}_Grn.idat"))
            _assert_idat_magic(Path(f"{prefix}_Red.idat"))
            writer.writerow((str(sample.sentrix_identity), sentrix_id, sentrix_position))
        handle.flush()


@dataclass(frozen=True, slots=True)
class ProcessorOutput:
    """Aligned normalized beta, detection-p, and invariant quality-mask outputs."""

    probe_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    beta: t.Tensor
    detection_p: t.Tensor
    quality_excluded: t.Tensor

    def __post_init__(self) -> None:
        if not self.probe_ids or not self.sample_ids:
            raise ValueError("processor output axes must not be empty")
        if len(set(self.probe_ids)) != len(self.probe_ids):
            raise ValueError("processor output contains duplicate probe IDs")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("processor output contains duplicate sample IDs")
        expected_shape = (len(self.probe_ids), len(self.sample_ids))
        if (
            self.beta.shape != expected_shape
            or self.detection_p.shape != expected_shape
            or self.quality_excluded.shape != expected_shape
        ):
            raise ValueError("processor matrix shapes differ from recorded axes")
        if self.beta.dtype != t.float64 or self.detection_p.dtype != t.float64:
            raise TypeError("processor beta and detection-p matrices must be float64")
        if self.quality_excluded.dtype != t.bool:
            raise ValueError("processor quality mask must be a boolean matrix")
        if not bool(
            t.isfinite(self.beta).all().item() and t.isfinite(self.detection_p).all().item()
        ):
            raise ValueError("processor beta and detection-p matrices must be finite")
        if not bool(
            t.all((self.beta >= 0.0) & (self.beta <= 1.0)).item()
            and t.all((self.detection_p >= 0.0) & (self.detection_p <= 1.0)).item()
        ):
            raise ValueError("processor beta and detection-p values must lie in [0, 1]")

    @property
    def probe_order_sha256(self) -> str:
        return sha256_ordered_strings(self.probe_ids)

    @property
    def sample_order_sha256(self) -> str:
        return sha256_ordered_strings(self.sample_ids)


def _read_nonempty_unique_lines(path: Path, name: str) -> tuple[str, ...]:
    values = tuple(path.read_text(encoding="utf-8").splitlines())
    if not values or any(not value for value in values):
        raise ValueError(f"{name} must contain non-empty lines")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate values")
    return values


def _load_exact_vector(path: Path, *, count: int, dtype: t.dtype) -> t.Tensor:
    if sys.byteorder != "little":
        raise RuntimeError("processor binary ingestion requires a little-endian host")
    element_size = t.empty((), dtype=dtype).element_size()
    expected_bytes = count * element_size
    if not path.is_file() or path.stat().st_size != expected_bytes:
        observed = None if not path.is_file() else path.stat().st_size
        raise ValueError(
            f"processor binary size differs for {path}: "
            f"expected={expected_bytes}, observed={observed}"
        )
    values = t.from_file(str(path), shared=False, size=count, dtype=dtype)
    if values.shape != (count,):
        raise RuntimeError("processor binary mapping returned an unexpected shape")
    return values


def _load_binary_processor_output(
    directory: Path,
    *,
    expected_sample_ids: tuple[str, ...],
    processor_name: str,
) -> ProcessorOutput:
    probe_ids = _read_nonempty_unique_lines(directory / "probe_ids.txt", "probe IDs")
    sample_ids = _read_nonempty_unique_lines(directory / "sample_order.txt", "sample order")
    if sample_ids != expected_sample_ids:
        raise ValueError(
            f"{processor_name} sample order differs: "
            f"expected_hash={sha256_ordered_strings(expected_sample_ids)}, "
            f"observed_hash={sha256_ordered_strings(sample_ids)}"
        )
    expected_files = {
        *_PROCESSOR_OUTPUT_SCHEMA_FILES,
        *(
            name
            for sample_id in sample_ids
            for name in (
                f"{sample_id}.beta.f64",
                f"{sample_id}.detection_p.f64",
                f"{sample_id}.quality_excluded.u8",
            )
        ),
    }
    observed_files = {path.name for path in directory.iterdir() if path.is_file()}
    if observed_files != expected_files:
        raise ValueError(
            f"{processor_name} output file inventory differs: "
            f"missing={sorted(expected_files - observed_files)[:10]}, "
            f"unknown={sorted(observed_files - expected_files)[:10]}"
        )

    probe_count = len(probe_ids)
    beta_columns = tuple(
        _load_exact_vector(
            directory / f"{sample_id}.beta.f64",
            count=probe_count,
            dtype=t.float64,
        )
        for sample_id in sample_ids
    )
    detection_columns = tuple(
        _load_exact_vector(
            directory / f"{sample_id}.detection_p.f64",
            count=probe_count,
            dtype=t.float64,
        )
        for sample_id in sample_ids
    )
    quality_columns = tuple(
        _load_exact_vector(
            directory / f"{sample_id}.quality_excluded.u8",
            count=probe_count,
            dtype=t.uint8,
        )
        for sample_id in sample_ids
    )
    quality_raw = t.stack(quality_columns, dim=1)
    if not bool(t.all((quality_raw == 0) | (quality_raw == 1)).item()):
        raise ValueError(f"{processor_name} quality mask contains values outside {{0, 1}}")
    return ProcessorOutput(
        probe_ids=probe_ids,
        sample_ids=sample_ids,
        beta=t.stack(beta_columns, dim=1),
        detection_p=t.stack(detection_columns, dim=1),
        quality_excluded=quality_raw.to(t.bool),
    )


@beartype
def load_sesame_output(
    directory: Path,
    *,
    expected_sample_ids: tuple[str, ...],
) -> ProcessorOutput:
    """Load only the exact files emitted by the pinned seSAMe preprocessing script."""

    return _load_binary_processor_output(
        directory,
        expected_sample_ids=expected_sample_ids,
        processor_name="seSAMe",
    )


@beartype
def load_methylprep_output(
    directory: Path,
    *,
    expected_sample_ids: tuple[str, ...],
) -> ProcessorOutput:
    """Load only the exact files emitted by the pinned methylprep audit script."""

    return _load_binary_processor_output(
        directory,
        expected_sample_ids=expected_sample_ids,
        processor_name="methylprep",
    )


@beartype
def align_processor_output_to_probe_order(
    output: ProcessorOutput,
    probe_ids: tuple[str, ...],
) -> ProcessorOutput:
    """Reindex a processor output only after requiring an identical probe-ID set."""

    if set(output.probe_ids) != set(probe_ids):
        raise ValueError(
            "processor probe sets differ: "
            f"missing={sorted(set(probe_ids) - set(output.probe_ids))[:10]}, "
            f"unknown={sorted(set(output.probe_ids) - set(probe_ids))[:10]}"
        )
    return select_processor_output_probes(output, probe_ids)


@beartype
def select_processor_output_probes(
    output: ProcessorOutput,
    probe_ids: tuple[str, ...],
) -> ProcessorOutput:
    """Select an ordered non-empty processor subset with no silent missing probes."""

    if not probe_ids or len(set(probe_ids)) != len(probe_ids):
        raise ValueError("requested processor probe order must be non-empty and unique")
    missing = set(probe_ids) - set(output.probe_ids)
    if missing:
        raise ValueError(
            f"requested probes are absent from processor output: {sorted(missing)[:10]}"
        )
    by_probe = {probe_id: index for index, probe_id in enumerate(output.probe_ids)}
    indices = t.tensor(
        tuple(by_probe[probe_id] for probe_id in probe_ids),
        dtype=t.int64,
    )
    return ProcessorOutput(
        probe_ids=probe_ids,
        sample_ids=output.sample_ids,
        beta=output.beta.index_select(0, indices),
        detection_p=output.detection_p.index_select(0, indices),
        quality_excluded=output.quality_excluded.index_select(0, indices),
    )
