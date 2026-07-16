"""GPL13534 parsing and provenance-preserving probe exclusion unions."""

from __future__ import annotations

import csv
import gzip
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TextIO

from beartype import beartype

from methylation_latent.artifacts import sha256_file
from methylation_latent.domain import (
    InfiniumDesign,
    ManifestStrand,
    NonEmptyProbeSet,
    ProbeId,
    ProbeLocus,
    parse_autosome,
    parse_genomic_context,
    parse_one_based_position,
    parse_probe_id,
)

GPL13534_V11_SHA256 = "df3d5009d9b5b878507ba330bbcd7d2012455f6d0aa92c2df897f53a28452c33"
CHEN2013_CROSS_REACTIVE_URL = (
    "https://raw.githubusercontent.com/sirselim/illumina450k_filtering/"
    "52ff77bb9dc631305e82cbeaac2027f361a7361f/48639-non-specific-probes-Illumina450k.csv"
)
CHEN2013_CROSS_REACTIVE_SHA256 = "4e962d36821f6f6fcd8b81cc0558090c028e54fbdb2c039a5712f9b471d9d89e"
CHEN2013_CROSS_REACTIVE_CPGS = 29_233
ZHOU2017_MASK_GENERAL_URL = (
    "https://zhouserver.research.chop.edu/InfiniumAnnotation/20180808/hm450/"
    "hm450.hg19.manifest.tsv.gz"
)
ZHOU2017_MASK_GENERAL_SHA256 = "94aaa52274738bbf8d767e56512b8fc36893c2cabffa202870531950bf522f0e"
ZHOU2017_MANIFEST_ROWS = 485_577
ZHOU2017_MASK_GENERAL_ROWS = 65_574
ZHOU2017_MASK_GENERAL_CPGS = 64_987
_CONTROL_NAMES = frozenset(
    {
        "BISULFITE CONVERSION I",
        "BISULFITE CONVERSION II",
        "EXTENSION",
        "HYBRIDIZATION",
        "NEGATIVE",
        "NON-POLYMORPHIC",
        "NORM_A",
        "NORM_C",
        "NORM_G",
        "NORM_T",
        "SPECIFICITY I",
        "SPECIFICITY II",
        "STAINING",
        "TARGET REMOVAL",
    }
)
_REQUIRED_COLUMNS = frozenset(
    {
        "IlmnID",
        "Name",
        "Infinium_Design_Type",
        "Forward_Sequence",
        "Genome_Build",
        "CHR",
        "MAPINFO",
        "Strand",
        "UCSC_CpG_Islands_Name",
        "Relation_to_UCSC_CpG_Island",
    }
)


class ManifestExclusionReason(StrEnum):
    SNP_ASSAY = "snp_assay"
    NON_CPG_ASSAY = "non_cpg_assay"
    CONTROL = "control"
    SEX_CHROMOSOME = "sex_chromosome"
    UNMAPPED = "unmapped"


@dataclass(frozen=True, slots=True)
class ProbeExclusion:
    raw_identifier: str
    reason: str
    source: str

    def __post_init__(self) -> None:
        if not self.raw_identifier or not self.reason or not self.source:
            raise ValueError("probe exclusion identifier, reason, and source are required")


@dataclass(frozen=True, slots=True)
class ManifestParseResult:
    probes: NonEmptyProbeSet
    exclusions: tuple[ProbeExclusion, ...]
    total_assay_rows: int

    def __post_init__(self) -> None:
        if self.total_assay_rows != len(self.probes) + len(self.exclusions):
            raise ValueError("manifest rows are not exactly partitioned into retained/excluded")


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8-sig", newline="")
    return path.open(mode="rt", encoding="utf-8-sig", newline="")


def _manifest_rows(path: Path) -> tuple[tuple[str, ...], Iterable[dict[str, str]]]:
    handle = _open_text(path)
    reader = csv.reader(handle)
    for fields in reader:
        if fields and fields[0] == "[Assay]":
            break
    else:
        handle.close()
        raise ValueError("manifest has no [Assay] section")
    try:
        header = tuple(next(reader))
    except StopIteration:
        handle.close()
        raise ValueError("manifest [Assay] section has no header") from None
    if len(set(header)) != len(header):
        handle.close()
        raise ValueError("manifest assay header contains duplicate column names")
    missing = _REQUIRED_COLUMNS - set(header)
    if missing:
        handle.close()
        raise ValueError(f"manifest lacks required columns: {sorted(missing)}")

    def dictionaries() -> Iterable[dict[str, str]]:
        with handle:
            for line_number, fields in enumerate(reader, start=1):
                if not any(fields):
                    continue
                if fields[0].startswith("[") and fields[0].endswith("]"):
                    break
                if len(fields) != len(header):
                    raise ValueError(
                        f"manifest assay row {line_number} has {len(fields)} fields; "
                        f"expected {len(header)}"
                    )
                yield dict(zip(header, fields, strict=True))

    return header, dictionaries()


def _manifest_level_exclusion(row: dict[str, str]) -> ManifestExclusionReason | None:
    name = row["Name"]
    if name.startswith("rs"):
        return ManifestExclusionReason.SNP_ASSAY
    if name.startswith("ch"):
        return ManifestExclusionReason.NON_CPG_ASSAY
    if name in _CONTROL_NAMES:
        return ManifestExclusionReason.CONTROL
    if not name.startswith("cg"):
        raise ValueError(f"unrecognized non-CpG manifest assay name: {name!r}")
    chromosome = row["CHR"]
    if chromosome in {"X", "Y"}:
        return ManifestExclusionReason.SEX_CHROMOSOME
    if chromosome == "" or row["MAPINFO"] == "" or row["Genome_Build"] == "":
        return ManifestExclusionReason.UNMAPPED
    return None


@beartype
def parse_gpl13534_manifest(path: Path) -> ManifestParseResult:
    """Parse the reviewed v1.1 schema and retain only mapped autosomal CpG loci."""

    _, rows = _manifest_rows(path)
    retained: list[ProbeLocus] = []
    exclusions: list[ProbeExclusion] = []
    seen_ilmn_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        ilmn_id = row["IlmnID"]
        if not ilmn_id:
            raise ValueError(f"manifest row {row_number} has an empty IlmnID")
        if ilmn_id in seen_ilmn_ids:
            raise ValueError(f"manifest contains duplicate IlmnID: {ilmn_id}")
        seen_ilmn_ids.add(ilmn_id)
        exclusion_reason = _manifest_level_exclusion(row)
        if exclusion_reason is not None:
            exclusions.append(
                ProbeExclusion(
                    raw_identifier=ilmn_id,
                    reason=exclusion_reason.value,
                    source="GPL13534_v1.1",
                )
            )
            continue
        if row["Name"] != ilmn_id:
            raise ValueError(f"mapped CpG Name/IlmnID mismatch at row {row_number}")
        if row["Genome_Build"] != "37":
            raise ValueError(
                f"mapped CpG {ilmn_id} has build {row['Genome_Build']!r}, expected '37'"
            )
        forward_sequence = row["Forward_Sequence"].upper()
        if forward_sequence.count("[CG]") != 1:
            raise ValueError(f"mapped CpG {ilmn_id} must contain exactly one [CG] marker")
        retained.append(
            ProbeLocus(
                probe_id=parse_probe_id(ilmn_id),
                chromosome=parse_autosome(row["CHR"]),
                position=parse_one_based_position(row["MAPINFO"]),
                context=parse_genomic_context(
                    row["Relation_to_UCSC_CpG_Island"], row["UCSC_CpG_Islands_Name"]
                ),
                design=InfiniumDesign(row["Infinium_Design_Type"]),
                manifest_strand=ManifestStrand(row["Strand"]),
            )
        )
    return ManifestParseResult(
        probes=NonEmptyProbeSet.from_iterable(retained),
        exclusions=tuple(exclusions),
        total_assay_rows=len(seen_ilmn_ids),
    )


@dataclass(frozen=True, slots=True)
class PublishedExclusionSource:
    """A pinned set of platform probe IDs from one reviewed exclusion list."""

    name: str
    version: str
    sha256: str
    probe_ids: frozenset[ProbeId]

    def __post_init__(self) -> None:
        if not self.name or not self.version or len(self.sha256) != 64 or not self.probe_ids:
            raise ValueError("published exclusion source metadata must be complete")


@dataclass(frozen=True, slots=True)
class PublishedExclusionResult:
    retained: NonEmptyProbeSet
    ledger: tuple[ProbeExclusion, ...]
    unmatched_by_source: dict[str, tuple[ProbeId, ...]]


@beartype
def load_chen2013_cross_reactive(path: Path) -> PublishedExclusionSource:
    """Load the byte-pinned 29,233-probe Chen non-specific-probe supplement."""

    observed_hash = sha256_file(path)
    if observed_hash != CHEN2013_CROSS_REACTIVE_SHA256:
        raise ValueError(
            "Chen 2013 exclusion hash differs: "
            f"expected={CHEN2013_CROSS_REACTIVE_SHA256}, observed={observed_hash}"
        )
    expected_header = ("TargetID", "47", "48", "49", "50")
    probe_ids: set[ProbeId] = set()
    with _open_text(path) as handle:
        reader = csv.reader(handle)
        header = tuple(next(reader, ()))
        if header != expected_header:
            raise ValueError(f"Chen 2013 header differs: observed={header}")
        for line_number, fields in enumerate(reader, start=2):
            if len(fields) != len(expected_header):
                raise ValueError(f"Chen 2013 row {line_number} has the wrong field count")
            raw_probe_id, *raw_match_counts = fields
            probe_id = parse_probe_id(raw_probe_id)
            if probe_id in probe_ids:
                raise ValueError(f"Chen 2013 contains duplicate probe ID: {probe_id}")
            match_counts = tuple(map(int, raw_match_counts))
            if any(count < 0 for count in match_counts) or sum(match_counts) == 0:
                raise ValueError(f"Chen 2013 row {line_number} has invalid match counts")
            probe_ids.add(probe_id)
    if len(probe_ids) != CHEN2013_CROSS_REACTIVE_CPGS:
        raise ValueError(
            "Chen 2013 probe count differs: "
            f"expected={CHEN2013_CROSS_REACTIVE_CPGS}, observed={len(probe_ids)}"
        )
    return PublishedExclusionSource(
        name="Chen2013_cross_reactive",
        version="48639@mirror-52ff77bb9dc631305e82cbeaac2027f361a7361f",
        sha256=CHEN2013_CROSS_REACTIVE_SHA256,
        probe_ids=frozenset(probe_ids),
    )


@beartype
def load_zhou2017_mask_general(path: Path) -> PublishedExclusionSource:
    """Load MASK_general from the byte-pinned hg19 HM450 annotation table."""

    observed_hash = sha256_file(path)
    if observed_hash != ZHOU2017_MASK_GENERAL_SHA256:
        raise ValueError(
            "Zhou MASK_general hash differs: "
            f"expected={ZHOU2017_MASK_GENERAL_SHA256}, observed={observed_hash}"
        )
    seen_raw_ids: set[str] = set()
    masked_cpgs: set[ProbeId] = set()
    masked_rows = 0
    with _open_text(path) as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = tuple(next(reader, ()))
        if len(set(header)) != len(header) or not {"probeID", "MASK_general"} <= set(header):
            raise ValueError("Zhou annotation header lacks unique probeID/MASK_general columns")
        probe_index = header.index("probeID")
        mask_index = header.index("MASK_general")
        for line_number, fields in enumerate(reader, start=2):
            if len(fields) != len(header):
                raise ValueError(f"Zhou annotation row {line_number} has the wrong field count")
            raw_probe_id = fields[probe_index]
            if not raw_probe_id or raw_probe_id in seen_raw_ids:
                raise ValueError(
                    f"Zhou annotation has empty/duplicate probe ID at row {line_number}"
                )
            seen_raw_ids.add(raw_probe_id)
            masked = fields[mask_index]
            if masked not in {"TRUE", "FALSE"}:
                raise ValueError(f"Zhou annotation row {line_number} has invalid MASK_general")
            if masked == "TRUE":
                masked_rows += 1
                if raw_probe_id.startswith("cg"):
                    masked_cpgs.add(parse_probe_id(raw_probe_id))
    observed_counts = (len(seen_raw_ids), masked_rows, len(masked_cpgs))
    expected_counts = (
        ZHOU2017_MANIFEST_ROWS,
        ZHOU2017_MASK_GENERAL_ROWS,
        ZHOU2017_MASK_GENERAL_CPGS,
    )
    if observed_counts != expected_counts:
        raise ValueError(
            "Zhou MASK_general counts differ: "
            f"expected={expected_counts}, observed={observed_counts}"
        )
    return PublishedExclusionSource(
        name="Zhou2017_MASK.general",
        version="InfiniumAnnotation-20180808-hg19",
        sha256=ZHOU2017_MASK_GENERAL_SHA256,
        probe_ids=frozenset(masked_cpgs),
    )


@beartype
def apply_published_exclusions(
    probes: NonEmptyProbeSet,
    sources: tuple[PublishedExclusionSource, ...],
) -> PublishedExclusionResult:
    """Apply the union while preserving every overlapping source/reason row."""

    if not sources:
        raise ValueError("at least one published exclusion source is required")
    if len({source.name for source in sources}) != len(sources):
        raise ValueError("published exclusion source names must be unique")
    manifest_ids = {probe.probe_id for probe in probes.probes}
    excluded_union: set[ProbeId] = set()
    ledger: list[ProbeExclusion] = []
    unmatched: dict[str, tuple[ProbeId, ...]] = {}
    for source in sources:
        matched = source.probe_ids & manifest_ids
        excluded_union.update(matched)
        ledger.extend(
            ProbeExclusion(
                raw_identifier=probe_id,
                reason="published_exclusion",
                source=f"{source.name}@{source.version}:{source.sha256}",
            )
            for probe_id in sorted(matched)
        )
        unmatched[source.name] = tuple(sorted(source.probe_ids - manifest_ids))
    retained = tuple(probe for probe in probes.probes if probe.probe_id not in excluded_union)
    return PublishedExclusionResult(
        retained=NonEmptyProbeSet(retained),
        ledger=tuple(ledger),
        unmatched_by_source=unmatched,
    )
