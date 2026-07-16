"""hg19 coordinate conversion, indexed FASTA access, and sequence features."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from io import BufferedReader
from pathlib import Path
from types import TracebackType

import torch as t
from beartype import beartype
from jaxtyping import UInt8, jaxtyped

from methylation_latent.artifacts import sha256_ordered_strings
from methylation_latent.domain import (
    Fraction,
    NonEmptyProbeSet,
    ProbeLocus,
    WindowSize,
    parse_fraction,
)

EncodedDna = UInt8[t.Tensor, "bases"]
_VALID_BASE_BYTES = frozenset(b"ACGT")
UCSC_HG19_FASTA_ARCHIVE_MD5 = "806c02398f5ac5da8ffd6da2d1d5d1a9"
UCSC_HG19_FASTA_PAYLOAD_SHA256 = "92b96d16b307d824f3b6b9dc63feb237a75226f82ca2166d51ecec039bee4449"
PRIMARY_REFERENCE_INPUT_PROBES = 405_610
PRIMARY_REFERENCE_ELIGIBLE_PROBES = 403_573
PRIMARY_REFERENCE_BOUNDARY_EXCLUSIONS = 15
PRIMARY_REFERENCE_NON_ACGT_EXCLUSIONS = 2_022
PRIMARY_REFERENCE_ELIGIBLE_ORDER_SHA256 = (
    "6a647c74a664bc693f67013c51046d163499839c843486f1c26ce6c5aa4ab4cf"
)


@beartype
@dataclass(frozen=True, slots=True)
class GenomicInterval:
    """A zero-based, half-open interval on one named reference contig."""

    chromosome: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not self.chromosome:
            raise ValueError("interval chromosome must not be empty")
        if self.start < 0:
            raise ValueError("interval start must be non-negative")
        if self.end <= self.start:
            raise ValueError("interval end must be greater than start")

    @property
    def width(self) -> int:
        return self.end - self.start

    def overlaps(self, other: GenomicInterval) -> bool:
        return (
            self.chromosome == other.chromosome
            and self.start < other.end
            and other.start < self.end
        )


@beartype
@dataclass(frozen=True, slots=True)
class FastaIndexEntry:
    """One standard samtools `.fai` record."""

    name: str
    length: int
    offset: int
    line_bases: int
    line_width: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("FASTA contig name must not be empty")
        if min(self.length, self.line_bases, self.line_width) <= 0 or self.offset < 0:
            raise ValueError("FASTA index contains non-positive dimensions or a negative offset")
        if self.line_width < self.line_bases:
            raise ValueError("FASTA index line width cannot be smaller than line bases")


def parse_fasta_index(path: Path) -> dict[str, FastaIndexEntry]:
    """Parse an `.fai` file and reject duplicate or malformed contigs."""

    records: dict[str, FastaIndexEntry] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            fields = raw_line.rstrip("\r\n").split("\t")
            if len(fields) != 5:
                raise ValueError(f"invalid FASTA index record at line {line_number}")
            name, length, offset, line_bases, line_width = fields
            record = FastaIndexEntry(
                name=name,
                length=int(length),
                offset=int(offset),
                line_bases=int(line_bases),
                line_width=int(line_width),
            )
            if record.name in records:
                raise ValueError(f"duplicate FASTA contig in index: {record.name}")
            records[record.name] = record
    if not records:
        raise ValueError("FASTA index must contain at least one contig")
    return records


class IndexedFasta:
    """Minimal read-only random access for an uncompressed indexed FASTA."""

    def __init__(self, fasta_path: Path, index_path: Path | None = None) -> None:
        self.fasta_path = fasta_path
        self.index_path = index_path or Path(f"{fasta_path}.fai")
        self.entries = parse_fasta_index(self.index_path)
        self._handle: BufferedReader | None = None

    def __enter__(self) -> IndexedFasta:
        if self._handle is not None:
            raise RuntimeError("IndexedFasta context is already open")
        self._handle = self.fasta_path.open("rb")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        if self._handle is None:
            raise RuntimeError("IndexedFasta context is not open")
        self._handle.close()
        self._handle = None

    def contig_length(self, chromosome: str) -> int:
        if chromosome not in self.entries:
            raise KeyError(f"FASTA index has no contig {chromosome!r}")
        return self.entries[chromosome].length

    def read_interval(self, interval: GenomicInterval) -> str:
        if self._handle is None:
            raise RuntimeError("IndexedFasta must be used as a context manager")
        if interval.chromosome not in self.entries:
            raise KeyError(f"FASTA index has no contig {interval.chromosome!r}")
        entry = self.entries[interval.chromosome]
        if interval.end > entry.length:
            raise ValueError(
                f"interval {interval.chromosome}:{interval.start}-{interval.end} exceeds "
                f"contig length {entry.length}"
            )

        chunks: list[bytes] = []
        position = interval.start
        remaining = interval.width
        while remaining:
            within_line = position % entry.line_bases
            take = min(remaining, entry.line_bases - within_line)
            byte_offset = (
                entry.offset + (position // entry.line_bases) * entry.line_width + within_line
            )
            self._handle.seek(byte_offset)
            chunk = self._handle.read(take)
            if len(chunk) != take:
                raise ValueError("FASTA ended before indexed interval was read")
            if b"\n" in chunk or b"\r" in chunk:
                raise ValueError("FASTA index points into a line terminator")
            chunks.append(chunk)
            position += take
            remaining -= take

        sequence = b"".join(chunks).decode("ascii").upper()
        if len(sequence) != interval.width:
            raise ValueError("FASTA interval length does not match requested width")
        return sequence


@dataclass(frozen=True, slots=True)
class ReferenceWindowAudit:
    """Exhaustive plus-strand coordinate and maximum-window eligibility counts."""

    input_probes: int
    coordinate_cpgs_verified: int
    eligible_probes: int
    boundary_exclusions: int
    non_acgt_exclusions: int
    eligible_probe_order_sha256: str

    def __post_init__(self) -> None:
        if self.input_probes <= 0 or self.coordinate_cpgs_verified != self.input_probes:
            raise ValueError("reference audit must verify every non-empty input coordinate")
        if (
            self.eligible_probes <= 0
            or min(self.boundary_exclusions, self.non_acgt_exclusions) < 0
            or self.eligible_probes + self.boundary_exclusions + self.non_acgt_exclusions
            != self.input_probes
        ):
            raise ValueError("reference eligibility counts do not partition the input probes")
        if len(self.eligible_probe_order_sha256) != 64:
            raise ValueError("eligible probe order must carry a SHA-256 fingerprint")


@beartype
def audit_reference_windows(
    reference: IndexedFasta,
    probes: NonEmptyProbeSet,
    maximum_window_size: WindowSize,
) -> ReferenceWindowAudit:
    """Verify every MAPINFO CpG, then classify full plus-strand maximum windows."""

    width = int(maximum_window_size)
    half = width // 2
    boundary_exclusions = 0
    non_acgt_exclusions = 0
    eligible_probe_ids: list[str] = []
    for probe in probes.probes:
        chromosome = f"chr{int(probe.chromosome)}"
        chromosome_length = reference.contig_length(chromosome)
        cytosine_zero = int(probe.position) - 1
        dinucleotide = reference.read_interval(
            GenomicInterval(chromosome, cytosine_zero, cytosine_zero + 2)
        )
        if dinucleotide != "CG":
            raise ValueError(
                f"probe {probe.probe_id} MAPINFO is not a plus-strand CpG: "
                f"observed {dinucleotide!r}"
            )
        start = cytosine_zero - half
        end = start + width
        if start < 0 or end > chromosome_length:
            boundary_exclusions += 1
            continue
        sequence = reference.read_interval(GenomicInterval(chromosome, start, end))
        if sequence[half : half + 2] != "CG":
            raise RuntimeError("maximum-window extraction disagrees with coordinate CpG audit")
        if set(sequence.encode("ascii")) - _VALID_BASE_BYTES:
            non_acgt_exclusions += 1
            continue
        eligible_probe_ids.append(str(probe.probe_id))
    return ReferenceWindowAudit(
        input_probes=len(probes),
        coordinate_cpgs_verified=len(probes),
        eligible_probes=len(eligible_probe_ids),
        boundary_exclusions=boundary_exclusions,
        non_acgt_exclusions=non_acgt_exclusions,
        eligible_probe_order_sha256=sha256_ordered_strings(eligible_probe_ids),
    )


@beartype
def cpg_window_interval(
    probe: ProbeLocus,
    window_size: WindowSize,
    *,
    chromosome_prefix: str = "chr",
    chromosome_length: int,
) -> GenomicInterval:
    """Convert a one-based CpG cytosine to an even, centred half-open window."""

    if chromosome_length <= 0:
        raise ValueError("chromosome length must be positive")
    cytosine_zero = int(probe.position) - 1
    start = cytosine_zero - int(window_size) // 2
    end = start + int(window_size)
    if start < 0 or end > chromosome_length:
        raise ValueError(
            f"probe {probe.probe_id} cannot provide a full {int(window_size)}-base window"
        )
    return GenomicInterval(
        chromosome=f"{chromosome_prefix}{int(probe.chromosome)}",
        start=start,
        end=end,
    )


@beartype
def extract_cpg_window(
    reference: IndexedFasta,
    probe: ProbeLocus,
    window_size: WindowSize,
) -> str:
    chromosome = f"chr{int(probe.chromosome)}"
    interval = cpg_window_interval(
        probe,
        window_size,
        chromosome_length=reference.contig_length(chromosome),
    )
    sequence = reference.read_interval(interval)
    centre = int(window_size) // 2
    if sequence[centre : centre + 2] != "CG":
        observed = sequence[centre : centre + 2]
        raise ValueError(
            f"probe {probe.probe_id} does not centre on plus-strand CG: observed {observed!r}"
        )
    invalid = sorted(set(sequence.encode("ascii")) - _VALID_BASE_BYTES)
    if invalid:
        rendered = "".join(chr(value) for value in invalid)
        raise ValueError(f"probe {probe.probe_id} window contains non-ACGT bases: {rendered}")
    return sequence


@jaxtyped(typechecker=beartype)
def encode_dna(sequence: str) -> EncodedDna:
    upper = sequence.upper()
    invalid = sorted(set(upper.encode("ascii")) - _VALID_BASE_BYTES)
    if invalid:
        rendered = "".join(chr(value) for value in invalid)
        raise ValueError(f"DNA sequence contains non-ACGT bases: {rendered}")
    if not upper:
        raise ValueError("DNA sequence must not be empty")
    return t.tensor(tuple(upper.encode("ascii")), dtype=t.uint8)


@dataclass(frozen=True, slots=True)
class SequenceFeatures:
    """Hand-crafted deterministic features for the age baseline."""

    cpg_density: Fraction
    gc_content: Fraction

    def as_float64_tensor(self) -> t.Tensor:
        return t.tensor(
            [float(self.cpg_density), float(self.gc_content)],
            dtype=t.float64,
        )


@beartype
def sequence_features(sequence: str) -> SequenceFeatures:
    encoded = encode_dna(sequence)
    if encoded.numel() < 2:
        raise ValueError("sequence must contain at least two bases for CpG density")
    gc = ((encoded == ord("G")) | (encoded == ord("C"))).to(t.float64).mean()
    cpg = ((encoded[:-1] == ord("C")) & (encoded[1:] == ord("G"))).to(t.float64).mean()
    return SequenceFeatures(
        cpg_density=parse_fraction(float(cpg.item())),
        gc_content=parse_fraction(float(gc.item())),
    )


@beartype
def assert_probe_windows(
    reference: IndexedFasta,
    probes: Iterable[ProbeLocus],
    window_sizes: tuple[WindowSize, ...],
) -> int:
    """Exhaustively assert all probe/window sequence contracts."""

    if not window_sizes:
        raise ValueError("at least one sequence-window width is required")
    count = 0
    for probe in probes:
        for window_size in window_sizes:
            extract_cpg_window(reference, probe, window_size)
            count += 1
    if count == 0:
        raise ValueError("sequence audit requires at least one probe")
    return count


def chromosome_lengths(reference: IndexedFasta) -> Mapping[int, int]:
    """Return hg-style autosomal lengths and reject absent autosomes."""

    return {chromosome: reference.contig_length(f"chr{chromosome}") for chromosome in range(1, 23)}
