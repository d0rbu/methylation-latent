from __future__ import annotations

import csv
import gzip
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
import torch as t
from jaxtyping import TypeCheckError

from methylation_latent import manifest as manifest_module
from methylation_latent.domain import (
    GenomicContext,
    NonEmptyProbeSet,
    ProbeLocus,
    parse_age_years,
    parse_fraction,
    parse_gsm_accession,
    parse_positive_int,
    parse_probe_id,
    parse_sentrix_identity,
)
from methylation_latent.manifest import (
    ManifestExclusionReason,
    ManifestParseResult,
    ProbeExclusion,
    PublishedExclusionSource,
    apply_published_exclusions,
    load_chen2013_cross_reactive,
    load_zhou2017_mask_general,
    parse_gpl13534_manifest,
)
from methylation_latent.metadata import (
    Gender,
    OrderedSampleSet,
    SampleMetadata,
    SeriesSample,
    _parse_geo_header_row,
    assert_gzip_integrity,
    audit_idat_inventory,
    join_sample_key,
    parse_series_matrix_metadata,
    select_parity_sample_set,
)
from methylation_latent.preprocessing import (
    DetectionQcResult,
    _validate_probability_matrix,
    apply_detection_qc,
    assert_processor_parity,
)


def _write_gzip(path: Path, text: str) -> None:
    with gzip.open(path, mode="wt", encoding="utf-8", newline="") as handle:
        handle.write(text)


def _series_text() -> str:
    rows = (
        '!Sample_title\t"age 67y 1001"\t"age 42y 1002"\n',
        '!Sample_geo_accession\t"GSM1"\t"GSM2"\n',
        '!Sample_source_name_ch1\t"X1001"\t"X1002"\n',
        '!Sample_characteristics_ch1\t"age (y): 67"\t"age (y): 42"\n',
        '!Sample_characteristics_ch1\t"source: UCSD"\t"source: USC"\n',
        '!Sample_characteristics_ch1\t"plate: 1"\t"plate: 2"\n',
        '!Sample_characteristics_ch1\t"gender: F"\t"gender: M"\n',
        '!Sample_characteristics_ch1\t"ethnicity: A"\t"ethnicity: B"\n',
        '!Sample_characteristics_ch1\t"tissue: whole blood"\t"tissue: whole blood"\n',
        "!series_matrix_table_begin\n",
        '"ID_REF"\t"GSM1"\t"GSM2"\n',
        "!series_matrix_table_end\n",
    )
    return "".join(rows)


def _series_fixture(path: Path) -> None:
    _write_gzip(path, _series_text())


def test_series_metadata_and_sample_key_join_are_explicit(tmp_path: Path) -> None:
    series_path = tmp_path / "series.txt.gz"
    _series_fixture(series_path)
    assert assert_gzip_integrity(series_path) > 0
    series = parse_series_matrix_metadata(series_path, expected_sample_count=2)
    assert [float(sample.age) for sample in series] == [67.0, 42.0]
    assert [sample.gender for sample in series] == [Gender.FEMALE, Gender.MALE]
    key_path = tmp_path / "key.txt.gz"
    _write_gzip(
        key_path,
        "1\t1001\t5815284001_R01C01\n2\t1002\t5815284001_R02C01\n",
    )
    joined = join_sample_key(series, key_path)
    assert len(joined) == 2
    assert joined.samples[0].gsm_accession == "GSM1"
    assert joined.samples[1].sentrix_identity == "5815284001_R02C01"
    assert len(joined.order_sha256) == 64


def test_series_parser_rejects_metadata_drift(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_text(
        '!Sample_title\t"age 67y 1001"\n'
        '!Sample_geo_accession\t"GSM1"\n'
        '!Sample_source_name_ch1\t"X1001"\n'
        '!Sample_characteristics_ch1\t"age (y): 66"\n'
        '!Sample_characteristics_ch1\t"source: UCSD"\n'
        '!Sample_characteristics_ch1\t"plate: 1"\n'
        '!Sample_characteristics_ch1\t"gender: F"\n'
        '!Sample_characteristics_ch1\t"ethnicity: A"\n'
        '!Sample_characteristics_ch1\t"tissue: whole blood"\n'
        "!series_matrix_table_begin\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="age mismatch"):
        parse_series_matrix_metadata(path, expected_sample_count=1)
    with pytest.raises(ValueError, match="gzip"):
        assert_gzip_integrity(path)


def test_series_value_objects_and_gzip_audit_reject_empty_or_duplicate_identity(
    tmp_path: Path,
) -> None:
    sample = SeriesSample(
        parse_gsm_accession("GSM1"),
        parse_positive_int(1),
        parse_age_years(30),
        "source",
        parse_positive_int(1),
        Gender.FEMALE,
        "ethnicity",
        "whole blood",
    )
    with pytest.raises(ValueError, match="source and ethnicity"):
        replace(sample, source="")
    with pytest.raises(ValueError, match="tissue"):
        replace(sample, tissue="saliva")
    with pytest.raises(ValueError, match="must not be empty"):
        OrderedSampleSet(())
    metadata = _sample_set().samples[0]
    with pytest.raises(ValueError, match="duplicate GSM"):
        OrderedSampleSet((metadata, metadata))
    empty = tmp_path / "empty.gz"
    _write_gzip(empty, "")
    with pytest.raises(ValueError, match="empty payload"):
        assert_gzip_integrity(empty)
    full = tmp_path / "full.gz"
    _write_gzip(full, "x")
    with pytest.raises(ValueError, match="chunk size"):
        assert_gzip_integrity(full, chunk_size=0)
    with pytest.raises(ValueError, match="no sample values"):
        _parse_geo_header_row("!Sample_title")


@pytest.mark.parametrize(
    ("transform", "count", "message"),
    [
        (lambda text: text.replace("!series_matrix_table_begin\n", ""), 2, "table-begin"),
        (
            lambda text: text.replace('!Sample_source_name_ch1\t"X1001"\t"X1002"\n', ""),
            2,
            "singleton rows",
        ),
        (lambda text: text, 0, "count must be positive"),
        (lambda text: text, 1, "row length"),
        (lambda text: text.replace("age (y): 67", "age 67"), 2, "name: value"),
        (lambda text: text.replace("age (y): 42", "source: 42"), 2, "mixes field names"),
        (
            lambda text: text.replace(
                '!Sample_characteristics_ch1\t"source: UCSD"',
                '!Sample_characteristics_ch1\t"age (y): 67"\t"age (y): 42"\n'
                '!Sample_characteristics_ch1\t"source: UCSD"',
            ),
            2,
            "duplicate sample characteristic",
        ),
        (
            lambda text: text.replace(
                '!Sample_characteristics_ch1\t"ethnicity: A"\t"ethnicity: B"\n', ""
            ),
            2,
            "sample characteristics differ",
        ),
        (lambda text: text.replace("age 67y 1001", "subject 1001"), 2, "frozen grammar"),
        (lambda text: text.replace('"X1001"', '"X9999"'), 2, "source-name"),
        (lambda text: text.replace('"GSM2"', '"GSM1"'), 2, "duplicate GSM"),
        (
            lambda text: text.replace("age 42y 1002", "age 42y 1001").replace('"X1002"', '"X1001"'),
            2,
            "duplicate subject",
        ),
    ],
)
def test_series_parser_rejects_every_header_contract_violation(
    tmp_path: Path,
    transform: Callable[[str], str],
    count: int,
    message: str,
) -> None:
    path = tmp_path / "series.txt.gz"
    _write_gzip(path, transform(_series_text()))
    with pytest.raises(ValueError, match=message):
        parse_series_matrix_metadata(path, expected_sample_count=count)


def test_sample_key_join_rejects_order_mismatch(tmp_path: Path) -> None:
    series_path = tmp_path / "series.txt.gz"
    _series_fixture(series_path)
    series = parse_series_matrix_metadata(series_path, expected_sample_count=2)
    key_path = tmp_path / "key.txt"
    key_path.write_text(
        "2\t1001\t5815284001_R01C01\n1\t1002\t5815284001_R02C01\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="row sequence"):
        join_sample_key(series, key_path)


def test_sample_key_join_rejects_missing_malformed_and_wrong_subject_rows(tmp_path: Path) -> None:
    series_path = tmp_path / "series.txt.gz"
    _series_fixture(series_path)
    series = parse_series_matrix_metadata(series_path, expected_sample_count=2)
    with pytest.raises(ValueError, match="requires series"):
        join_sample_key((), tmp_path / "absent")
    key = tmp_path / "key.txt"
    key.write_text("1\t1001\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly three"):
        join_sample_key(series, key)
    key.write_text("1\t1001\t5815284001_R01C01\n", encoding="utf-8")
    with pytest.raises(ValueError, match="counts differ"):
        join_sample_key(series, key)
    key.write_text("1\t1002\t5815284001_R01C01\n2\t1001\t5815284001_R02C01\n", encoding="utf-8")
    with pytest.raises(ValueError, match="subject mismatch"):
        join_sample_key(series, key)


def _sample_set() -> OrderedSampleSet:
    return OrderedSampleSet(
        (
            SampleMetadata(
                gsm_accession=parse_gsm_accession("GSM1"),
                subject_id=parse_positive_int(1001),
                sentrix_identity=parse_sentrix_identity("5815284001_R01C01"),
                age=parse_age_years(67),
                source="UCSD",
                plate=parse_positive_int(1),
                gender=Gender.FEMALE,
                ethnicity="A",
                tissue="whole blood",
            ),
        )
    )


def _parity_sample_universe() -> OrderedSampleSet:
    samples: list[SampleMetadata] = []
    index = 0
    for row in range(1, 7):
        for column in range(1, 3):
            for replicate in range(2):
                index += 1
                samples.append(
                    SampleMetadata(
                        gsm_accession=parse_gsm_accession(f"GSM{index}"),
                        subject_id=parse_positive_int(index),
                        sentrix_identity=parse_sentrix_identity(
                            f"{5_800_000_000 + index:010d}_R{row:02d}C{column:02d}"
                        ),
                        age=parse_age_years(20 + replicate),
                        source="source",
                        plate=parse_positive_int(1 + (row + column + replicate) % 4),
                        gender=Gender.FEMALE,
                        ethnicity="ethnicity",
                        tissue="whole blood",
                    )
                )
    return OrderedSampleSet(tuple(samples))


def test_parity_subset_is_target_blind_deterministic_and_position_complete() -> None:
    samples = _parity_sample_universe()
    selected = select_parity_sample_set(samples, expected_count=12, seed=550319)
    repeated = select_parity_sample_set(samples, expected_count=12, seed=550319)
    assert selected == repeated
    assert len(selected) == 12
    assert {str(sample.sentrix_identity).split("_")[1] for sample in selected.samples} == {
        f"R{row:02d}C{column:02d}" for row in range(1, 7) for column in range(1, 3)
    }
    changed_ages = OrderedSampleSet(
        tuple(replace(sample, age=parse_age_years(90)) for sample in samples.samples)
    )
    selected_after_age_change = select_parity_sample_set(
        changed_ages, expected_count=12, seed=550319
    )
    assert tuple(sample.sentrix_identity for sample in selected.samples) == tuple(
        sample.sentrix_identity for sample in selected_after_age_change.samples
    )
    with pytest.raises(ValueError, match="exactly 12 arrays"):
        select_parity_sample_set(samples, expected_count=11, seed=550319)
    missing_position = OrderedSampleSet(samples.samples[:-2])
    with pytest.raises(ValueError, match="12 Sentrix positions"):
        select_parity_sample_set(missing_position, expected_count=12, seed=550319)


def test_idat_inventory_requires_exact_pairs_and_magic(tmp_path: Path) -> None:
    red = tmp_path / "5815284001_R01C01_Red.idat"
    green = tmp_path / "5815284001_R01C01_Grn.idat"
    red.write_bytes(b"IDAT-red")
    green.write_bytes(b"IDAT-green")
    pairs = audit_idat_inventory(tmp_path, _sample_set())
    assert len(pairs) == 1
    assert pairs[0].red_sha256 != pairs[0].green_sha256
    extra = tmp_path / "extra.idat"
    extra.write_bytes(b"IDAT-extra")
    with pytest.raises(ValueError, match="inventory differs"):
        audit_idat_inventory(tmp_path, _sample_set())
    extra.unlink()
    green.write_bytes(b"NOPE")
    with pytest.raises(ValueError, match="absent or empty|magic"):
        audit_idat_inventory(tmp_path, _sample_set())
    green.write_bytes(b"NOPE-extra")
    with pytest.raises(ValueError, match="magic"):
        audit_idat_inventory(tmp_path, _sample_set())


MANIFEST_COLUMNS = (
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
)


def _manifest_row(
    ilmn_id: str,
    name: str,
    *,
    chromosome: str = "1",
    position: str = "1000",
    relation: str = "Island",
    island: str = "chr1:900-1100",
    design: str = "II",
    forward_sequence: str = "AAA[CG]TTT",
    genome_build: str = "37",
    strand: str = "F",
) -> tuple[str, ...]:
    return (
        ilmn_id,
        name,
        design,
        forward_sequence,
        genome_build,
        chromosome,
        position,
        strand,
        island,
        relation,
    )


def _write_manifest(path: Path, rows: tuple[tuple[str, ...], ...]) -> None:
    with gzip.open(path, mode="wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("[Heading]",))
        writer.writerow(("[Assay]",))
        writer.writerow(MANIFEST_COLUMNS)
        writer.writerows(rows)
        writer.writerow(("[Controls]", "", ""))
        writer.writerow(("123", "NEGATIVE", "Green"))


def test_manifest_parser_partitions_assays_and_stops_at_controls(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv.gz"
    _write_manifest(
        path,
        (
            _manifest_row("cg00000001", "cg00000001"),
            _manifest_row("cg00000002", "cg00000002", chromosome="X"),
            _manifest_row("rs00000001", "rs00000001"),
            _manifest_row("ch.1.100F", "ch.1.100F"),
            _manifest_row("cg00000003", "cg00000003", chromosome="", position=""),
        ),
    )
    result = parse_gpl13534_manifest(path)
    assert result.total_assay_rows == 5
    assert [probe.probe_id for probe in result.probes.probes] == ["cg00000001"]
    assert {entry.reason for entry in result.exclusions} == {
        ManifestExclusionReason.SEX_CHROMOSOME,
        ManifestExclusionReason.SNP_ASSAY,
        ManifestExclusionReason.NON_CPG_ASSAY,
        ManifestExclusionReason.UNMAPPED,
    }


def test_manifest_parser_rejects_unreviewed_or_contradictory_rows(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv.gz"
    _write_manifest(path, (_manifest_row("mystery", "mystery"),))
    with pytest.raises(ValueError, match="unrecognized"):
        parse_gpl13534_manifest(path)
    _write_manifest(path, (_manifest_row("cg00000001", "cg00000001"),))
    # Reopening with gzip above replaced the file; now a valid parse confirms fixture determinism.
    assert len(parse_gpl13534_manifest(path).probes) == 1


def test_manifest_schema_and_row_contracts_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv.gz"
    _write_gzip(path, "[Heading]\n")
    with pytest.raises(ValueError, match=r"no \[Assay\]"):
        parse_gpl13534_manifest(path)
    _write_gzip(path, "[Assay]\n")
    with pytest.raises(ValueError, match="no header"):
        parse_gpl13534_manifest(path)
    duplicate_header = (*MANIFEST_COLUMNS, MANIFEST_COLUMNS[0])
    with gzip.open(path, mode="wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("[Assay]",))
        writer.writerow(duplicate_header)
    with pytest.raises(ValueError, match="duplicate column"):
        parse_gpl13534_manifest(path)
    with gzip.open(path, mode="wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("[Assay]",))
        writer.writerow(MANIFEST_COLUMNS[:-1])
    with pytest.raises(ValueError, match="lacks required"):
        parse_gpl13534_manifest(path)
    with gzip.open(path, mode="wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("[Assay]",))
        writer.writerow(MANIFEST_COLUMNS)
        writer.writerow(("too", "short"))
    with pytest.raises(ValueError, match="fields"):
        parse_gpl13534_manifest(path)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ((_manifest_row("", "cg00000001"),), "empty IlmnID"),
        (
            (
                _manifest_row("cg00000001", "cg00000001"),
                _manifest_row("cg00000001", "cg00000001"),
            ),
            "duplicate IlmnID",
        ),
        ((_manifest_row("cg00000001", "cg00000002"),), "Name/IlmnID mismatch"),
        (
            (_manifest_row("cg00000001", "cg00000001", genome_build="38"),),
            "expected '37'",
        ),
        (
            (_manifest_row("cg00000001", "cg00000001", forward_sequence="AAACGTT"),),
            "exactly one.*marker",
        ),
    ],
)
def test_manifest_rejects_identity_build_and_marker_drift(
    tmp_path: Path, rows: tuple[tuple[str, ...], ...], message: str
) -> None:
    path = tmp_path / "manifest.csv.gz"
    _write_manifest(path, rows)
    with pytest.raises(ValueError, match=message):
        parse_gpl13534_manifest(path)


def test_manifest_control_and_semantic_envelopes_are_explicit(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv.gz"
    _write_manifest(
        path,
        (
            _manifest_row("cg00000001", "cg00000001"),
            _manifest_row("control-1", "NEGATIVE"),
        ),
    )
    parsed = parse_gpl13534_manifest(path)
    assert parsed.exclusions[0].reason == ManifestExclusionReason.CONTROL
    with pytest.raises(ValueError, match="required"):
        ProbeExclusion("", "reason", "source")
    with pytest.raises(ValueError, match="partitioned"):
        ManifestParseResult(parsed.probes, (), 2)
    with pytest.raises(ValueError, match="metadata must be complete"):
        PublishedExclusionSource("", "v1", "a" * 64, frozenset({parse_probe_id("cg00000001")}))


def test_pinned_chen_and_zhou_loaders_validate_schema_counts_and_hashes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    chen_path = tmp_path / "chen.csv"
    chen_path.write_text(
        "TargetID,47,48,49,50\ncg00000001,1,0,0,0\ncg00000002,0,0,1,0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hash differs"):
        load_chen2013_cross_reactive(chen_path)
    monkeypatch.setattr(manifest_module, "sha256_file", lambda _: "a" * 64)
    monkeypatch.setattr(manifest_module, "CHEN2013_CROSS_REACTIVE_SHA256", "a" * 64)
    monkeypatch.setattr(manifest_module, "CHEN2013_CROSS_REACTIVE_CPGS", 2)
    chen = load_chen2013_cross_reactive(chen_path)
    assert len(chen.probe_ids) == 2

    zhou_path = tmp_path / "zhou.tsv.gz"
    _write_gzip(
        zhou_path,
        "probeID\tMASK_general\ncg00000001\tTRUE\nch.1.10F\tTRUE\ncg00000002\tFALSE\n",
    )
    monkeypatch.setattr(manifest_module, "ZHOU2017_MASK_GENERAL_SHA256", "a" * 64)
    monkeypatch.setattr(manifest_module, "ZHOU2017_MANIFEST_ROWS", 3)
    monkeypatch.setattr(manifest_module, "ZHOU2017_MASK_GENERAL_ROWS", 2)
    monkeypatch.setattr(manifest_module, "ZHOU2017_MASK_GENERAL_CPGS", 1)
    zhou = load_zhou2017_mask_general(zhou_path)
    assert zhou.probe_ids == frozenset({parse_probe_id("cg00000001")})


def test_pinned_exclusion_loaders_reject_duplicate_and_invalid_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(manifest_module, "sha256_file", lambda _: "a" * 64)
    monkeypatch.setattr(manifest_module, "CHEN2013_CROSS_REACTIVE_SHA256", "a" * 64)
    monkeypatch.setattr(manifest_module, "CHEN2013_CROSS_REACTIVE_CPGS", 1)
    chen_path = tmp_path / "chen.csv"
    chen_path.write_text("TargetID,47,48,49,50\ncg00000001,0,0,0,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid match counts"):
        load_chen2013_cross_reactive(chen_path)

    monkeypatch.setattr(manifest_module, "ZHOU2017_MASK_GENERAL_SHA256", "a" * 64)
    zhou_path = tmp_path / "zhou.tsv.gz"
    _write_gzip(zhou_path, "probeID\tMASK_general\ncg00000001\tMAYBE\n")
    with pytest.raises(ValueError, match="invalid MASK_general"):
        load_zhou2017_mask_general(zhou_path)


def test_published_exclusion_union_preserves_overlapping_reasons(
    make_probe: Callable[..., ProbeLocus],
) -> None:
    probes = NonEmptyProbeSet(
        (
            make_probe(1, context=GenomicContext.ISLAND),
            make_probe(2, context=GenomicContext.SHORE),
            make_probe(3, context=GenomicContext.OPEN_SEA),
        )
    )
    source_a = PublishedExclusionSource(
        "Chen",
        "2013",
        "a" * 64,
        frozenset({parse_probe_id("cg00000001"), parse_probe_id("cg99999999")}),
    )
    source_b = PublishedExclusionSource(
        "Zhou",
        "2017-MASK.general",
        "b" * 64,
        frozenset({parse_probe_id("cg00000001"), parse_probe_id("cg00000002")}),
    )
    result = apply_published_exclusions(probes, (source_a, source_b))
    assert [probe.probe_id for probe in result.retained.probes] == ["cg00000003"]
    assert len(result.ledger) == 3
    assert result.unmatched_by_source["Chen"] == ("cg99999999",)
    with pytest.raises(ValueError, match="unique"):
        apply_published_exclusions(probes, (source_a, source_a))
    with pytest.raises(ValueError, match="at least one"):
        apply_published_exclusions(probes, ())
    all_probes = PublishedExclusionSource(
        "all", "v1", "c" * 64, frozenset(probe.probe_id for probe in probes.probes)
    )
    with pytest.raises(ValueError, match="must not be empty"):
        apply_published_exclusions(probes, (all_probes,))


def test_detection_qc_applies_sample_first_then_complete_probe_rule() -> None:
    beta = t.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=t.float64)
    p = t.tensor(
        [
            [0.001, 0.001, 0.5],
            [0.001, 0.02, 0.5],
            [0.001, 0.001, 0.5],
        ],
        dtype=t.float64,
    )
    result = apply_detection_qc(
        beta,
        p,
        detection_threshold=parse_fraction(0.01),
        maximum_sample_failure_fraction=parse_fraction(0.34),
    )
    assert result.retained_sample_indices.tolist() == [0, 1]
    assert result.excluded_sample_indices.tolist() == [2]
    assert result.retained_probe_indices.tolist() == [0, 2]
    assert result.excluded_probe_indices.tolist() == [1]
    assert result.beta.shape == (2, 2)


def _parity_matrices() -> tuple[t.Tensor, t.Tensor]:
    beta = t.linspace(0.1, 0.8, 4, dtype=t.float64).unsqueeze(1) + t.linspace(
        0.0, 0.01, 12, dtype=t.float64
    )
    detection_p = t.zeros_like(beta)
    return beta, detection_p


def test_methylprep_sesame_parity_gate_accepts_only_preregistered_agreement() -> None:
    beta, detection_p = _parity_matrices()
    audit = assert_processor_parity(
        beta,
        beta.clone(),
        detection_p,
        detection_p.clone(),
        detection_threshold=parse_fraction(0.01),
        maximum_sample_failure_fraction=parse_fraction(0.01),
    )
    assert audit.detection_agreement == 1.0
    assert audit.retained_samples == 12
    assert audit.retained_probes == 4
    assert audit.beta_median_absolute_error == 0.0


def test_methylprep_sesame_parity_gate_rejects_decision_and_beta_drift() -> None:
    beta, detection_p = _parity_matrices()
    sample_drift = detection_p.clone()
    sample_drift[0, 0] = 0.5
    with pytest.raises(ValueError, match="sample-retention decisions differ"):
        assert_processor_parity(
            beta,
            beta,
            sample_drift,
            detection_p,
            detection_threshold=parse_fraction(0.01),
            maximum_sample_failure_fraction=parse_fraction(0.01),
        )
    probe_drift = detection_p.clone()
    probe_drift[0, :] = 0.5
    with pytest.raises(ValueError, match="complete-probe sets differ"):
        assert_processor_parity(
            beta,
            beta,
            probe_drift,
            detection_p,
            detection_threshold=parse_fraction(0.01),
            maximum_sample_failure_fraction=parse_fraction(0.5),
        )
    shifted = beta + 0.02
    with pytest.raises(ValueError, match="thresholds failed"):
        assert_processor_parity(
            shifted,
            beta,
            detection_p,
            detection_p,
            detection_threshold=parse_fraction(0.01),
            maximum_sample_failure_fraction=parse_fraction(0.01),
        )
    with pytest.raises(ValueError, match="exactly 12 arrays"):
        assert_processor_parity(
            beta[:, :-1],
            beta[:, :-1],
            detection_p[:, :-1],
            detection_p[:, :-1],
            detection_threshold=parse_fraction(0.01),
            maximum_sample_failure_fraction=parse_fraction(0.01),
        )


def test_detection_qc_rejects_misalignment_and_empty_outcomes() -> None:
    beta = t.tensor([[0.1, 0.2]], dtype=t.float64)
    with pytest.raises(TypeCheckError, match="detection_p"):
        apply_detection_qc(
            beta,
            t.tensor([[0.1]], dtype=t.float64),
            detection_threshold=parse_fraction(0.01),
            maximum_sample_failure_fraction=parse_fraction(0.01),
        )

    with pytest.raises(ValueError, match="every probe"):
        apply_detection_qc(
            t.tensor([[0.1, 0.2]], dtype=t.float64),
            t.ones((1, 2), dtype=t.float64),
            detection_threshold=parse_fraction(0.01),
            maximum_sample_failure_fraction=parse_fraction(1.0),
        )
    with pytest.raises(ValueError, match="fewer than two"):
        apply_detection_qc(
            t.tensor([[0.1]], dtype=t.float64),
            t.zeros((1, 1), dtype=t.float64),
            detection_threshold=parse_fraction(0.01),
            maximum_sample_failure_fraction=parse_fraction(0.0),
        )


def test_detection_probability_and_result_artifacts_reject_invalid_dtypes() -> None:
    beta = t.tensor([[0.1, 0.2]], dtype=t.float64)
    with pytest.raises(ValueError, match="rank-2 float64"):
        _validate_probability_matrix(t.ones((1, 2), dtype=t.float32), "matrix")
    with pytest.raises(ValueError, match="finite"):
        _validate_probability_matrix(t.tensor([[0.1, float("nan")]], dtype=t.float64), "matrix")
    with pytest.raises(ValueError, match="int64 vectors"):
        DetectionQcResult(
            beta=t.ones((1, 2), dtype=t.float64) / 2,
            retained_probe_indices=t.tensor([0], dtype=t.int32),
            retained_sample_indices=t.tensor([0, 1]),
            excluded_probe_indices=t.empty(0, dtype=t.int64),
            excluded_sample_indices=t.empty(0, dtype=t.int64),
            sample_failure_fractions=t.zeros(2, dtype=t.float64),
        )
    with pytest.raises(ValueError, match="failure fractions"):
        DetectionQcResult(
            beta=t.ones((1, 2), dtype=t.float64) / 2,
            retained_probe_indices=t.tensor([0]),
            retained_sample_indices=t.tensor([0, 1]),
            excluded_probe_indices=t.empty(0, dtype=t.int64),
            excluded_sample_indices=t.empty(0, dtype=t.int64),
            sample_failure_fractions=t.zeros(2, dtype=t.float32),
        )
    with pytest.raises(ValueError, match="every sample"):
        apply_detection_qc(
            beta,
            t.ones_like(beta),
            detection_threshold=parse_fraction(0.01),
            maximum_sample_failure_fraction=parse_fraction(0.0),
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        apply_detection_qc(
            t.tensor([[1.2, 0.2]], dtype=t.float64),
            t.zeros_like(beta),
            detection_threshold=parse_fraction(0.01),
            maximum_sample_failure_fraction=parse_fraction(0.01),
        )


def test_qc_result_rejects_axis_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        DetectionQcResult(
            beta=t.ones((1, 2), dtype=t.float64) / 2,
            retained_probe_indices=t.tensor([0, 1]),
            retained_sample_indices=t.tensor([0, 1]),
            excluded_probe_indices=t.empty(0, dtype=t.int64),
            excluded_sample_indices=t.empty(0, dtype=t.int64),
            sample_failure_fractions=t.zeros(2, dtype=t.float64),
        )
