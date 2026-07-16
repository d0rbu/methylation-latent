from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
import torch as t

from methylation_latent import cohort_diagnostics as cohort_module
from methylation_latent.artifacts import sha256_file
from methylation_latent.cohort_diagnostics import (
    PublicGenotypePanel,
    _column_index,
    audit_duplicate_public_genotypes,
    parse_gse87571_public_genotypes,
)
from methylation_latent.domain import (
    parse_age_years,
    parse_gsm_accession,
    parse_positive_int,
    parse_sentrix_identity,
)
from methylation_latent.metadata import Gender, Gse87571RawSample, Gse87571RawSampleSet


def _excel_column(index: int) -> str:
    value = index + 1
    letters = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return letters


def _samples() -> Gse87571RawSampleSet:
    rows = []
    for index in range(3):
        gsm = f"GSM{index + 1}"
        sentrix = f"581528400{index + 1}_R0{index + 1}C01"
        base = f"ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM0nnn/{gsm}/suppl/{gsm}_{sentrix}"
        rows.append(
            Gse87571RawSample(
                gsm_accession=parse_gsm_accession(gsm),
                subject_id=parse_positive_int(index + 1),
                sentrix_identity=parse_sentrix_identity(sentrix),
                age=parse_age_years(30 + index),
                gender=Gender.FEMALE,
                tissue="whole blood",
                disease_state="normal",
                green_url=f"{base}_Grn.idat.gz",
                red_url=f"{base}_Red.idat.gz",
            )
        )
    return Gse87571RawSampleSet(tuple(rows))


def _inline_cell(reference: str, value: str) -> str:
    return f'<c r="{reference}" t="inlineStr"><is><t>{value}</t></is></c>'


def _numeric_cell(reference: str, value: float) -> str:
    return f'<c r="{reference}"><v>{value}</v></c>'


def _write_xlsx(
    path: Path,
    samples: Gse87571RawSampleSet,
    *,
    malformed_header: bool = False,
) -> None:
    header = [_inline_cell("A3", "geo accession")]
    header.extend(
        _inline_cell(
            f"{_excel_column(column)}3",
            (
                "characteristics: bad"
                if malformed_header and column == 3
                else f"characteristics: rs{column}"
            ),
        )
        for column in range(3, 107)
    )
    rows = [f'<row r="3">{"".join(header)}</row>']
    for row_number, sample in enumerate(samples.samples, start=4):
        cells = [_inline_cell(f"A{row_number}", str(sample.gsm_accession))]
        for column in range(3, 107):
            value = float((row_number + column) % 3)
            cells.append(_numeric_cell(f"{_excel_column(column)}{row_number}", value))
        rows.append(f'<row r="{row_number}">{"".join(cells)}</row>')
    worksheet = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(rows)}</sheetData></worksheet>"
    )
    with ZipFile(path, mode="w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("xl/worksheets/sheet1.xml", worksheet)


def test_public_genotype_xlsx_parser_and_duplicate_audit_are_target_independent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = _samples()
    path = tmp_path / "characteristics.xlsx"
    _write_xlsx(path, samples)
    monkeypatch.setattr(
        cohort_module,
        "GSE87571_ADDITIONAL_CHARACTERISTICS_SHA256",
        sha256_file(path),
    )
    panel = parse_gse87571_public_genotypes(path, samples)
    assert panel.dosages.shape == (3, 104)
    assert panel.observed.all()
    assert len(panel.order_sha256) == 64
    audit = audit_duplicate_public_genotypes(panel)
    assert audit.nearest_mad_shared_variants == 104
    assert audit.maximum_agreement_shared_variants == 104
    assert audit.excluded_insufficient_samples == ()


def test_public_genotype_audit_rejects_exact_duplicates_and_invalid_panels() -> None:
    samples = _samples()
    dosages = t.stack(
        (
            t.zeros(104, dtype=t.float64),
            t.zeros(104, dtype=t.float64),
            t.ones(104, dtype=t.float64),
        )
    )
    panel = PublicGenotypePanel(
        gsm_accessions=tuple(sample.gsm_accession for sample in samples.samples),
        variant_names=tuple(f"rs{index}" for index in range(104)),
        dosages=dosages,
        observed=t.ones_like(dosages, dtype=t.bool),
    )
    with pytest.raises(ValueError, match="exact duplicate"):
        audit_duplicate_public_genotypes(panel)
    with pytest.raises(ValueError, match="explicit mask"):
        replace(
            panel,
            dosages=t.ones_like(dosages),
            observed=t.zeros_like(dosages, dtype=t.bool),
        )
    with pytest.raises(ValueError, match="at least two"):
        audit_duplicate_public_genotypes(
            replace(
                panel,
                dosages=t.zeros_like(dosages),
                observed=t.zeros_like(dosages, dtype=t.bool),
            )
        )


def test_public_genotype_parser_rejects_hash_order_and_header_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = _samples()
    path = tmp_path / "characteristics.xlsx"
    _write_xlsx(path, samples)
    with pytest.raises(ValueError, match="fingerprint"):
        parse_gse87571_public_genotypes(path, samples)
    monkeypatch.setattr(
        cohort_module,
        "GSE87571_ADDITIONAL_CHARACTERISTICS_SHA256",
        sha256_file(path),
    )
    reversed_samples = Gse87571RawSampleSet(tuple(reversed(samples.samples)))
    with pytest.raises(ValueError, match="GSM order"):
        parse_gse87571_public_genotypes(path, reversed_samples)
    malformed = tmp_path / "malformed.xlsx"
    _write_xlsx(malformed, samples, malformed_header=True)
    monkeypatch.setattr(
        cohort_module,
        "GSE87571_ADDITIONAL_CHARACTERISTICS_SHA256",
        sha256_file(malformed),
    )
    with pytest.raises(ValueError, match="headers"):
        parse_gse87571_public_genotypes(malformed, samples)


@pytest.mark.parametrize(
    ("reference", "expected"),
    [("A1", 0), ("Z1", 25), ("AA1", 26), ("DC99", 106)],
)
def test_xlsx_column_index_is_exact(reference: str, expected: int) -> None:
    assert _column_index(reference) == expected
    with pytest.raises(ValueError, match="invalid"):
        _column_index("1A")
