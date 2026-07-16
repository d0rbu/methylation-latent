"""Target-independent public-genotype audits for duplicate sample identities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import torch as t
from beartype import beartype

from methylation_latent.artifacts import sha256_file, sha256_ordered_strings
from methylation_latent.domain import GsmAccession, parse_gsm_accession
from methylation_latent.metadata import (
    GSE87571_ADDITIONAL_CHARACTERISTICS_SHA256,
    Gse87571RawSampleSet,
)

_SPREADSHEET_NAMESPACE = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_CELL_REFERENCE = re.compile(r"^([A-Z]+)([0-9]+)$", flags=re.ASCII)
_GENOTYPE_HEADER = re.compile(
    r"^characteristics: (?:rs[0-9]+(?::.*)?|Chr[0-9]+:.*|X[0-9]+:.*|exm[0-9]+)$",
    flags=re.ASCII,
)
_EXPECTED_GSE87571_GENOTYPE_VARIANTS = 104


def _column_index(cell_reference: str) -> int:
    match = _CELL_REFERENCE.fullmatch(cell_reference)
    if match is None:
        raise ValueError(f"invalid XLSX cell reference: {cell_reference!r}")
    letters = match.group(1)
    value = 0
    for letter in letters:
        value = value * 26 + ord(letter) - ord("A") + 1
    return value - 1


def _shared_strings(archive: ZipFile) -> tuple[str, ...]:
    path = "xl/sharedStrings.xml"
    if path not in archive.namelist():
        return ()
    root = ElementTree.fromstring(archive.read(path))
    namespace = {"m": _SPREADSHEET_NAMESPACE}
    return tuple(
        "".join(node.text or "" for node in item.iterfind(".//m:t", namespace))
        for item in root.findall("m:si", namespace)
    )


def _cell_value(
    cell: ElementTree.Element,
    *,
    shared_strings: tuple[str, ...],
) -> str:
    namespace = {"m": _SPREADSHEET_NAMESPACE}
    cell_type = cell.attrib.get("t")
    value = cell.find("m:v", namespace)
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.iterfind(".//m:t", namespace))
    raw = "" if value is None else value.text or ""
    if cell_type == "s":
        index = int(raw)
        if not 0 <= index < len(shared_strings):
            raise ValueError("XLSX shared-string index is out of bounds")
        return shared_strings[index]
    return raw


@dataclass(frozen=True, slots=True)
class PublicGenotypePanel:
    """Ordered public imputed-dosage panel used only for identity diagnostics."""

    gsm_accessions: tuple[GsmAccession, ...]
    variant_names: tuple[str, ...]
    dosages: t.Tensor
    observed: t.Tensor

    def __post_init__(self) -> None:
        if not self.gsm_accessions or not self.variant_names:
            raise ValueError("public genotype panel axes must not be empty")
        if len(set(self.gsm_accessions)) != len(self.gsm_accessions):
            raise ValueError("public genotype panel contains duplicate GSM accessions")
        if len(set(self.variant_names)) != len(self.variant_names):
            raise ValueError("public genotype panel contains duplicate variants")
        if self.dosages.shape != (
            len(self.gsm_accessions),
            len(self.variant_names),
        ):
            raise ValueError("public genotype dosage shape differs from recorded axes")
        if self.observed.shape != self.dosages.shape or self.observed.dtype != t.bool:
            raise ValueError("public genotype observation mask must align and be boolean")
        if self.dosages.dtype != t.float64 or not bool(t.isfinite(self.dosages).all().item()):
            raise ValueError("public genotype dosages must be finite float64 values")
        if not bool(t.all((self.dosages >= 0.0) & (self.dosages <= 2.0)).item()):
            raise ValueError("public genotype dosages must lie in [0, 2]")
        if not bool(t.all(self.dosages[~self.observed] == 0.0).item()):
            raise ValueError("missing public genotype dosages must use zero plus an explicit mask")

    @property
    def order_sha256(self) -> str:
        return sha256_ordered_strings(
            (
                *(str(accession) for accession in self.gsm_accessions),
                *(self.variant_names),
            )
        )


@beartype
def parse_gse87571_public_genotypes(
    path: Path,
    samples: Gse87571RawSampleSet,
) -> PublicGenotypePanel:
    """Parse the byte-pinned XLSX without pandas, NumPy, or inferred row joins."""

    observed_hash = sha256_file(path)
    if observed_hash != GSE87571_ADDITIONAL_CHARACTERISTICS_SHA256:
        raise ValueError(
            "GSE87571 additional-characteristics fingerprint differs: "
            f"expected={GSE87571_ADDITIONAL_CHARACTERISTICS_SHA256}, "
            f"observed={observed_hash}"
        )
    with ZipFile(path) as archive:
        shared = _shared_strings(archive)
        sheet_path = "xl/worksheets/sheet1.xml"
        if sheet_path not in archive.namelist():
            raise ValueError("GSE87571 XLSX lacks worksheet sheet1.xml")
        root = ElementTree.fromstring(archive.read(sheet_path))
    namespace = {"m": _SPREADSHEET_NAMESPACE}
    rows = root.findall(".//m:sheetData/m:row", namespace)
    by_number = {int(row.attrib["r"]): row for row in rows}
    if len(by_number) != len(rows) or 3 not in by_number:
        raise ValueError("GSE87571 XLSX row numbers are duplicate or lack header row 3")

    def row_values(row: ElementTree.Element) -> dict[int, str]:
        values: dict[int, str] = {}
        for cell in row.findall("m:c", namespace):
            column = _column_index(cell.attrib["r"])
            if column in values:
                raise ValueError("GSE87571 XLSX row contains duplicate cell columns")
            values[column] = _cell_value(cell, shared_strings=shared)
        return values

    header = row_values(by_number[3])
    if header.get(0) != "geo accession":
        raise ValueError("GSE87571 XLSX first header is not geo accession")
    genotype_columns = tuple(range(3, 3 + _EXPECTED_GSE87571_GENOTYPE_VARIANTS))
    if len(genotype_columns) != _EXPECTED_GSE87571_GENOTYPE_VARIANTS:
        raise RuntimeError("frozen GSE87571 genotype-column range has the wrong width")
    invalid_headers = tuple(
        header.get(column, "")
        for column in genotype_columns
        if _GENOTYPE_HEADER.fullmatch(header.get(column, "")) is None
    )
    if invalid_headers:
        raise ValueError(f"GSE87571 public genotype headers differ: {invalid_headers[:5]}")
    variant_names = tuple(
        header[column].removeprefix("characteristics: ") for column in genotype_columns
    )
    data_rows = tuple(by_number[number] for number in sorted(by_number) if number > 3)
    if len(data_rows) != len(samples):
        raise ValueError(
            f"GSE87571 XLSX/sample count differs: xlsx={len(data_rows)}, samples={len(samples)}"
        )

    gsm_accessions: list[GsmAccession] = []
    dosage_rows: list[t.Tensor] = []
    observed_rows: list[t.Tensor] = []
    for expected_sample, row in zip(samples.samples, data_rows, strict=True):
        values = row_values(row)
        observed_gsm = parse_gsm_accession(values.get(0, ""))
        if observed_gsm != expected_sample.gsm_accession:
            raise ValueError(
                "GSE87571 XLSX/series GSM order differs: "
                f"expected={expected_sample.gsm_accession}, observed={observed_gsm}"
            )
        raw_dosages = tuple(values.get(column, "") for column in genotype_columns)
        if any(value == "" for value in raw_dosages):
            raise ValueError(f"GSE87571 public genotypes contain missing values for {observed_gsm}")
        gsm_accessions.append(observed_gsm)
        observed = tuple(value != "NA" for value in raw_dosages)
        dosage_rows.append(
            t.tensor(
                tuple(0.0 if value == "NA" else float(value) for value in raw_dosages),
                dtype=t.float64,
            )
        )
        observed_rows.append(t.tensor(observed, dtype=t.bool))
    return PublicGenotypePanel(
        gsm_accessions=tuple(gsm_accessions),
        variant_names=variant_names,
        dosages=t.stack(dosage_rows),
        observed=t.stack(observed_rows),
    )


@dataclass(frozen=True, slots=True)
class DuplicateIdentityAudit:
    """Nearest pair under dosage MAD and genotype-call agreement."""

    nearest_mad_left: int
    nearest_mad_right: int
    minimum_mean_absolute_dosage_difference: float
    nearest_mad_shared_variants: int
    maximum_genotype_call_agreement: float
    maximum_agreement_left: int
    maximum_agreement_right: int
    maximum_agreement_shared_variants: int
    excluded_insufficient_samples: tuple[int, ...]

    def __post_init__(self) -> None:
        indices = (
            self.nearest_mad_left,
            self.nearest_mad_right,
            self.maximum_agreement_left,
            self.maximum_agreement_right,
        )
        if any(index < 0 for index in indices):
            raise ValueError("duplicate-identity audit indices must be non-negative")
        if not 0.0 <= self.maximum_genotype_call_agreement <= 1.0:
            raise ValueError("genotype-call agreement must lie in [0, 1]")
        if self.minimum_mean_absolute_dosage_difference < 0.0:
            raise ValueError("mean absolute dosage difference must be non-negative")
        if (
            min(
                self.nearest_mad_shared_variants,
                self.maximum_agreement_shared_variants,
            )
            <= 0
        ):
            raise ValueError("duplicate-identity audit requires shared observed variants")
        if any(index < 0 for index in self.excluded_insufficient_samples):
            raise ValueError("excluded duplicate-audit sample indices must be non-negative")


@beartype
def audit_duplicate_public_genotypes(panel: PublicGenotypePanel) -> DuplicateIdentityAudit:
    """Reject exact duplicate genotype profiles and report the nearest observed pairs."""

    _, variant_count = panel.dosages.shape
    observed_per_sample = panel.observed.sum(dim=1)
    included_indices = t.nonzero(observed_per_sample >= 100).flatten()
    excluded_indices = t.nonzero(observed_per_sample < 100).flatten()
    if included_indices.numel() < 2 or variant_count < 2:
        raise ValueError("duplicate-identity audit requires at least two samples and variants")
    dosages = panel.dosages.index_select(0, included_indices)
    observed_matrix = panel.observed.index_select(0, included_indices)
    sample_count = included_indices.numel()
    dosage_difference = t.zeros((sample_count, sample_count), dtype=t.float64)
    call_agreement = t.zeros((sample_count, sample_count), dtype=t.int64)
    shared_variants = t.zeros((sample_count, sample_count), dtype=t.int64)
    calls = t.round(dosages).to(t.int64)
    for variant in range(variant_count):
        dosage = dosages[:, variant]
        observed = observed_matrix[:, variant]
        shared = observed[:, None] & observed[None, :]
        dosage_difference += (dosage[:, None] - dosage[None, :]).abs() * shared
        call = calls[:, variant]
        call_agreement += (call[:, None] == call[None, :]) & shared
        shared_variants += shared
    minimum_shared = int(shared_variants[~t.eye(sample_count, dtype=t.bool)].min().item())
    if minimum_shared < 2:
        raise ValueError(
            "public genotype pairs share fewer than two observed variants: "
            f"minimum={minimum_shared}"
        )
    dosage_difference /= shared_variants.clamp_min(1)
    agreement_fraction = call_agreement.to(t.float64) / shared_variants.clamp_min(1)
    diagonal = t.arange(sample_count)
    dosage_difference[diagonal, diagonal] = float("inf")
    agreement_fraction[diagonal, diagonal] = -1.0

    nearest_flat = int(t.argmin(dosage_difference).item())
    agreement_flat = int(t.argmax(agreement_fraction).item())
    nearest_local_left, nearest_local_right = divmod(nearest_flat, sample_count)
    agreement_local_left, agreement_local_right = divmod(agreement_flat, sample_count)
    minimum_mad = float(dosage_difference[nearest_local_left, nearest_local_right].item())
    maximum_agreement = float(
        agreement_fraction[agreement_local_left, agreement_local_right].item()
    )
    nearest_shared = int(shared_variants[nearest_local_left, nearest_local_right].item())
    agreement_shared = int(shared_variants[agreement_local_left, agreement_local_right].item())
    nearest_agreement = float(agreement_fraction[nearest_local_left, nearest_local_right].item())
    nearest_left = int(included_indices[nearest_local_left].item())
    nearest_right = int(included_indices[nearest_local_right].item())
    agreement_left = int(included_indices[agreement_local_left].item())
    agreement_right = int(included_indices[agreement_local_right].item())
    if minimum_mad <= 1.0e-12 and nearest_agreement == 1.0 and nearest_shared >= 100:
        raise ValueError(
            "public genotype panel contains an exact duplicate sample identity: "
            f"{panel.gsm_accessions[nearest_left]} and {panel.gsm_accessions[nearest_right]}"
        )
    return DuplicateIdentityAudit(
        nearest_mad_left=nearest_left,
        nearest_mad_right=nearest_right,
        minimum_mean_absolute_dosage_difference=minimum_mad,
        nearest_mad_shared_variants=nearest_shared,
        maximum_genotype_call_agreement=maximum_agreement,
        maximum_agreement_left=agreement_left,
        maximum_agreement_right=agreement_right,
        maximum_agreement_shared_variants=agreement_shared,
        excluded_insufficient_samples=tuple(excluded_indices.tolist()),
    )
