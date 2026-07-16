from __future__ import annotations

import csv
import gzip
from array import array
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
import torch as t
from jaxtyping import TypeCheckError

from methylation_latent import manifest as manifest_module
from methylation_latent import metadata as metadata_module
from methylation_latent.artifacts import sha256_file
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
from methylation_latent.idat import (
    CompressedIdat,
    CompressedIdatInventory,
    audit_gse87571_compressed_idats,
    load_compressed_idat_inventory,
    materialize_idats,
    save_compressed_idat_inventory_exclusive,
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
    Gse87571RawSample,
    Gse87571RawSampleSet,
    OrderedSampleSet,
    SampleMetadata,
    SeriesSample,
    _parse_geo_header_row,
    assert_gzip_integrity,
    audit_idat_inventory,
    join_sample_key,
    parse_gse87571_filelist,
    parse_gse87571_series_metadata,
    parse_series_matrix_metadata,
    select_gse87571_parity_sample_set,
    select_parity_sample_set,
)
from methylation_latent.preprocessing import (
    DetectionQcResult,
    _validate_probability_matrix,
    apply_detection_qc,
    assert_processor_parity,
)
from methylation_latent.processor_io import (
    ProcessorOutput,
    align_processor_output_to_probe_order,
    load_methylprep_output,
    load_sesame_output,
    select_processor_output_probes,
    write_methylprep_sample_sheet_exclusive,
    write_sesame_sample_list_exclusive,
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


def _gse87571_series_text() -> str:
    green = (
        "ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1nnn/GSM1001/suppl/"
        "GSM1001_5815284001_R01C01_Grn.idat.gz"
    )
    red = (
        "ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1nnn/GSM1001/suppl/"
        "GSM1001_5815284001_R01C01_Red.idat.gz"
    )
    green_missing = (
        "ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1nnn/GSM1002/suppl/"
        "GSM1002_5815284002_R02C01_Grn.idat.gz"
    )
    red_missing = (
        "ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1nnn/GSM1002/suppl/"
        "GSM1002_5815284002_R02C01_Red.idat.gz"
    )
    return "".join(
        (
            '!Sample_title\t"X1 genomic DNA from whole blood"\t"X2 genomic DNA from whole blood"\n',
            '!Sample_geo_accession\t"GSM1001"\t"GSM1002"\n',
            '!Sample_source_name_ch1\t"whole blood"\t"whole blood"\n',
            '!Sample_characteristics_ch1\t"gender: Female"\t"gender: NA"\n',
            '!Sample_characteristics_ch1\t"age: 67"\t"age: NA"\n',
            '!Sample_characteristics_ch1\t"tissue: whole blood"\t"tissue: whole blood"\n',
            '!Sample_characteristics_ch1\t"disease state: normal"\t"disease state: normal"\n',
            f'!Sample_supplementary_file\t"{green}"\t"{green_missing}"\n',
            f'!Sample_supplementary_file\t"{red}"\t"{red_missing}"\n',
            "!series_matrix_table_begin\n",
        )
    )


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


def test_gse87571_parser_separates_raw_and_age_eligible_cohorts(tmp_path: Path) -> None:
    path = tmp_path / "series.txt.gz"
    _write_gzip(path, _gse87571_series_text())
    samples = parse_gse87571_series_metadata(
        path,
        expected_sample_count=2,
        expected_age_eligible_count=1,
    )
    assert len(samples) == 2
    assert len(samples.age_eligible) == 1
    assert samples.samples[0].sentrix_identity == "5815284001_R01C01"
    assert samples.samples[1].age is None
    assert samples.samples[1].gender is None
    assert samples.age_eligible.ages.dtype == t.float64
    assert samples.age_eligible.ages.tolist() == [67.0]
    assert len(samples.source_url_sha256) == 64
    assert len(samples.age_eligible.order_sha256) == 64


@pytest.mark.parametrize(
    ("transform", "message"),
    [
        (
            lambda text: text.replace("X1 genomic DNA from whole blood", "subject 1"),
            "title",
        ),
        (
            lambda text: text.replace(
                "GSM1001_5815284001_R01C01_Red",
                "GSM1001_5815284009_R01C01_Red",
            ),
            "disagree",
        ),
        (lambda text: text.replace("gender: Female", "gender: Unknown"), "gender"),
        (lambda text: text.replace("disease state: normal", "disease state: case", 1), "disease"),
        (
            lambda text: text.replace(
                '!Sample_source_name_ch1\t"whole blood"',
                '!Sample_source_name_ch1\t"saliva"',
            ),
            "source-name",
        ),
        (
            lambda text: text.replace(
                "!series_matrix_table_begin\n",
                '!Sample_supplementary_file\t"x"\t"y"\n!series_matrix_table_begin\n',
            ),
            "exactly two",
        ),
    ],
)
def test_gse87571_parser_rejects_identity_and_metadata_drift(
    tmp_path: Path,
    transform: Callable[[str], str],
    message: str,
) -> None:
    path = tmp_path / "series.txt.gz"
    _write_gzip(path, transform(_gse87571_series_text()))
    with pytest.raises(ValueError, match=message):
        parse_gse87571_series_metadata(
            path,
            expected_sample_count=2,
            expected_age_eligible_count=1,
        )


def _gse87571_parity_universe() -> Gse87571RawSampleSet:
    samples: list[Gse87571RawSample] = []
    index = 0
    for row in range(1, 7):
        for column in range(1, 3):
            for replicate in range(2):
                index += 1
                gsm = f"GSM{index}"
                sentrix = f"{5_800_000_000 + replicate * 100 + index:010d}_R{row:02d}C{column:02d}"
                base = f"ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM0nnn/{gsm}/suppl/{gsm}_{sentrix}"
                samples.append(
                    Gse87571RawSample(
                        gsm_accession=parse_gsm_accession(gsm),
                        subject_id=parse_positive_int(index),
                        sentrix_identity=parse_sentrix_identity(sentrix),
                        age=parse_age_years(20 + replicate),
                        gender=Gender.FEMALE,
                        tissue="whole blood",
                        disease_state="normal",
                        green_url=f"{base}_Grn.idat.gz",
                        red_url=f"{base}_Red.idat.gz",
                    )
                )
    return Gse87571RawSampleSet(tuple(samples))


def test_gse87571_parity_subset_is_target_blind_and_position_complete() -> None:
    samples = _gse87571_parity_universe()
    selected = select_gse87571_parity_sample_set(samples, expected_count=12, seed=550319)
    repeated = select_gse87571_parity_sample_set(samples, expected_count=12, seed=550319)
    assert selected == repeated
    assert len(selected) == 12
    assert {str(sample.sentrix_identity).split("_")[1] for sample in selected.samples} == {
        f"R{row:02d}C{column:02d}" for row in range(1, 7) for column in range(1, 3)
    }
    changed = Gse87571RawSampleSet(
        tuple(replace(sample, age=parse_age_years(99)) for sample in samples.samples)
    )
    changed_selected = select_gse87571_parity_sample_set(
        changed,
        expected_count=12,
        seed=550319,
    )
    assert tuple(sample.sentrix_identity for sample in selected.samples) == tuple(
        sample.sentrix_identity for sample in changed_selected.samples
    )
    with pytest.raises(ValueError, match="exactly 12"):
        select_gse87571_parity_sample_set(samples, expected_count=11, seed=1)
    with pytest.raises(ValueError, match="12 Sentrix positions"):
        select_gse87571_parity_sample_set(
            Gse87571RawSampleSet(samples.samples[:-2]),
            expected_count=12,
            seed=1,
        )


def test_gse87571_filelist_is_byte_pinned_and_exactly_partitioned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        "#Archive/File\tName\tTime\tSize\tType\n",
        "Archive\tGSE87571_RAW.tar\ttime\t100\tTAR\n",
        "File\tmanifest-a.gz\ttime\t10\tXLSX\n",
        "File\tmanifest-b.gz\ttime\t10\tTXT\n",
        "File\tmanifest-c.gz\ttime\t10\tCSV\n",
        "File\tmanifest-d.gz\ttime\t10\tBPM\n",
    ]
    records.extend(
        f"File\tGSM{index}_{channel}.idat.gz\ttime\t10\tIDAT\n"
        for index in range(732)
        for channel in ("Grn", "Red")
    )
    path = tmp_path / "filelist.txt"
    path.write_text("".join(records), encoding="utf-8")
    monkeypatch.setattr(metadata_module, "GSE87571_FILELIST_SHA256", sha256_file(path))
    parsed = parse_gse87571_filelist(path)
    assert len(parsed.idat_files) == 1464
    assert len(parsed.idat_sizes) == 1464
    path.write_text(path.read_text(encoding="utf-8").replace("\tIDAT\n", "\tOTHER\n", 1))
    monkeypatch.setattr(metadata_module, "GSE87571_FILELIST_SHA256", sha256_file(path))
    with pytest.raises(ValueError, match="four platform|1,464"):
        parse_gse87571_filelist(path)


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


def test_compressed_gse87571_idats_are_crc_audited_and_exactly_materialized(
    tmp_path: Path,
) -> None:
    sentrix = "5815284001_R01C01"
    gsm = "GSM1"
    base = f"ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM0nnn/{gsm}/suppl/{gsm}_{sentrix}"
    samples = Gse87571RawSampleSet(
        (
            Gse87571RawSample(
                gsm_accession=parse_gsm_accession(gsm),
                subject_id=parse_positive_int(1),
                sentrix_identity=parse_sentrix_identity(sentrix),
                age=parse_age_years(30),
                gender=Gender.FEMALE,
                tissue="whole blood",
                disease_state="normal",
                green_url=f"{base}_Grn.idat.gz",
                red_url=f"{base}_Red.idat.gz",
            ),
        )
    )
    compressed = tmp_path / "compressed"
    compressed.mkdir()
    expected_sizes: dict[str, int] = {}
    for channel in ("Grn", "Red"):
        path = compressed / f"{gsm}_{sentrix}_{channel}.idat.gz"
        with gzip.open(path, mode="wb") as handle:
            handle.write(b"IDAT-" + channel.encode("ascii") + b"-payload")
        expected_sizes[path.name] = path.stat().st_size
    inventory = audit_gse87571_compressed_idats(
        compressed,
        samples,
        expected_sizes=expected_sizes,
    )
    assert len(inventory.records) == 2
    assert len(inventory.fingerprint) == 64
    inventory_path = tmp_path / "inventory.json"
    save_compressed_idat_inventory_exclusive(inventory_path, inventory)
    assert (
        load_compressed_idat_inventory(
            inventory_path,
            compressed_directory=compressed,
        )
        == inventory
    )
    with pytest.raises(FileExistsError):
        save_compressed_idat_inventory_exclusive(inventory_path, inventory)
    materialized = tmp_path / "materialized"
    paths = materialize_idats(inventory, samples, output_directory=materialized)
    assert [path.name for path in paths] == [
        f"{sentrix}_Grn.idat",
        f"{sentrix}_Red.idat",
    ]
    assert all(path.read_bytes().startswith(b"IDAT") for path in paths)
    assert materialize_idats(inventory, samples, output_directory=materialized) == paths
    paths[0].write_bytes(b"IDAT-wrong")
    with pytest.raises(ValueError, match="differs"):
        materialize_idats(inventory, samples, output_directory=materialized)


def test_compressed_gse87571_idat_audit_rejects_inventory_size_and_magic(
    tmp_path: Path,
) -> None:
    samples = _gse87571_parity_universe()
    sample = Gse87571RawSampleSet((samples.samples[0],))
    compressed = tmp_path / "compressed"
    compressed.mkdir()
    names = tuple(
        url.rsplit("/", maxsplit=1)[-1]
        for url in (sample.samples[0].green_url, sample.samples[0].red_url)
    )
    for name in names:
        with gzip.open(compressed / name, mode="wb") as handle:
            handle.write(b"NOPE-payload")
    sizes = {name: (compressed / name).stat().st_size for name in names}
    with pytest.raises(ValueError, match="magic"):
        audit_gse87571_compressed_idats(compressed, sample, expected_sizes=sizes)
    with pytest.raises(ValueError, match="expected-size inventory"):
        audit_gse87571_compressed_idats(
            compressed,
            sample,
            expected_sizes={names[0]: sizes[names[0]]},
        )
    unknown = compressed / "unknown.idat.gz"
    with gzip.open(unknown, mode="wb") as handle:
        handle.write(b"IDAT-extra")
    with pytest.raises(ValueError, match="directory differs"):
        audit_gse87571_compressed_idats(compressed, sample, expected_sizes=sizes)


def test_compressed_idat_records_reject_incomplete_or_invalid_pairs(tmp_path: Path) -> None:
    record = CompressedIdat(
        sentrix_identity=parse_sentrix_identity("5815284001_R01C01"),
        channel="Grn",
        compressed_path=tmp_path / "x.gz",
        compressed_bytes=10,
        compressed_sha256="0" * 64,
        decompressed_bytes=10,
        decompressed_sha256="1" * 64,
    )
    with pytest.raises(ValueError, match="complete"):
        CompressedIdatInventory((record,))
    with pytest.raises(ValueError, match="channel"):
        replace(record, channel="Blue")


def _write_array(path: Path, typecode: str, values: tuple[float | int, ...]) -> None:
    with path.open(mode="wb") as handle:
        array(typecode, values).tofile(handle)


def test_sesame_sample_list_and_binary_output_preserve_exact_axes(tmp_path: Path) -> None:
    samples = Gse87571RawSampleSet((_gse87571_parity_universe().samples[0],))
    idat_directory = tmp_path / "idats"
    idat_directory.mkdir()
    sentrix = str(samples.samples[0].sentrix_identity)
    for channel in ("Grn", "Red"):
        (idat_directory / f"{sentrix}_{channel}.idat").write_bytes(b"IDAT-payload")
    sample_list = tmp_path / "samples.tsv"
    write_sesame_sample_list_exclusive(
        sample_list,
        samples,
        idat_directory=idat_directory,
    )
    assert sample_list.read_text(encoding="utf-8").splitlines() == [
        "sentrix_identity\tprefix",
        f"{sentrix}\t{idat_directory / sentrix}",
    ]
    with pytest.raises(FileExistsError):
        write_sesame_sample_list_exclusive(
            sample_list,
            samples,
            idat_directory=idat_directory,
        )

    output = tmp_path / "sesame"
    output.mkdir()
    (output / "probe_ids.txt").write_text("cg00000001\ncg00000002\n", encoding="utf-8")
    (output / "sample_order.txt").write_text(f"{sentrix}\n", encoding="utf-8")
    (output / "environment.txt").write_text("sesame=1.30.1\n", encoding="utf-8")
    _write_array(output / f"{sentrix}.quality_excluded.u8", "B", (0, 1))
    _write_array(output / f"{sentrix}.beta.f64", "d", (0.1, 0.8))
    _write_array(output / f"{sentrix}.detection_p.f64", "d", (0.001, 0.2))
    loaded = load_sesame_output(output, expected_sample_ids=(sentrix,))
    assert loaded.beta.shape == (2, 1)
    assert loaded.beta.dtype == t.float64
    assert loaded.detection_p[:, 0].tolist() == [0.001, 0.2]
    assert loaded.quality_excluded[:, 0].tolist() == [False, True]
    assert len(loaded.probe_order_sha256) == 64
    assert len(loaded.sample_order_sha256) == 64


def test_methylprep_sheet_loader_and_probe_alignment_are_exact(tmp_path: Path) -> None:
    samples = Gse87571RawSampleSet((_gse87571_parity_universe().samples[0],))
    idat_directory = tmp_path / "idats"
    idat_directory.mkdir()
    sentrix = str(samples.samples[0].sentrix_identity)
    for channel in ("Grn", "Red"):
        (idat_directory / f"{sentrix}_{channel}.idat").write_bytes(b"IDAT-payload")
    sample_sheet = tmp_path / "samples.csv"
    write_methylprep_sample_sheet_exclusive(
        sample_sheet,
        samples,
        idat_directory=idat_directory,
    )
    sentrix_id, sentrix_position = sentrix.split("_", maxsplit=1)
    assert sample_sheet.read_text(encoding="utf-8").splitlines() == [
        "Sample_Name,Sentrix_ID,Sentrix_Position",
        f"{sentrix},{sentrix_id},{sentrix_position}",
    ]

    output = tmp_path / "methylprep"
    output.mkdir()
    (output / "probe_ids.txt").write_text("cg00000002\ncg00000001\n", encoding="utf-8")
    (output / "sample_order.txt").write_text(f"{sentrix}\n", encoding="utf-8")
    (output / "environment.txt").write_text("methylprep=1.7.1\n", encoding="utf-8")
    _write_array(output / f"{sentrix}.quality_excluded.u8", "B", (1, 0))
    _write_array(output / f"{sentrix}.beta.f64", "d", (0.8, 0.1))
    _write_array(output / f"{sentrix}.detection_p.f64", "d", (0.2, 0.001))
    loaded = load_methylprep_output(output, expected_sample_ids=(sentrix,))
    aligned = align_processor_output_to_probe_order(
        loaded,
        ("cg00000001", "cg00000002"),
    )
    assert aligned.beta[:, 0].tolist() == [0.1, 0.8]
    assert aligned.detection_p[:, 0].tolist() == [0.001, 0.2]
    assert aligned.quality_excluded[:, 0].tolist() == [False, True]
    selected = select_processor_output_probes(loaded, ("cg00000001",))
    assert selected.probe_ids == ("cg00000001",)
    assert selected.beta[:, 0].tolist() == [0.1]
    with pytest.raises(ValueError, match="probe sets differ"):
        align_processor_output_to_probe_order(loaded, ("cg00000003",))
    with pytest.raises(ValueError, match="absent"):
        select_processor_output_probes(loaded, ("cg00000003",))


def test_processor_output_and_sesame_loader_reject_invalid_states(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="finite"):
        ProcessorOutput(
            probe_ids=("cg00000001",),
            sample_ids=("sample",),
            beta=t.tensor([[float("nan")]], dtype=t.float64),
            detection_p=t.zeros((1, 1), dtype=t.float64),
            quality_excluded=t.zeros((1, 1), dtype=t.bool),
        )
    output = tmp_path / "sesame"
    output.mkdir()
    (output / "probe_ids.txt").write_text("cg00000001\n", encoding="utf-8")
    (output / "sample_order.txt").write_text("sample\n", encoding="utf-8")
    (output / "environment.txt").write_text("x\n", encoding="utf-8")
    _write_array(output / "sample.quality_excluded.u8", "B", (2,))
    _write_array(output / "sample.beta.f64", "d", (0.1,))
    _write_array(output / "sample.detection_p.f64", "d", (0.1,))
    with pytest.raises(ValueError, match=r"\{0, 1\}"):
        load_sesame_output(output, expected_sample_ids=("sample",))
    with pytest.raises(ValueError, match="sample order differs"):
        load_sesame_output(output, expected_sample_ids=("different",))


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
    quality = t.zeros_like(beta, dtype=t.bool)
    audit = assert_processor_parity(
        beta,
        beta.clone(),
        detection_p,
        detection_p.clone(),
        quality,
        quality.clone(),
        detection_threshold=parse_fraction(0.01),
        maximum_sample_failure_fraction=parse_fraction(0.01),
    )
    assert audit.detection_agreement == 1.0
    assert audit.retained_samples == 12
    assert audit.retained_probes == 4
    assert audit.beta_median_absolute_error == 0.0


def test_methylprep_sesame_parity_gate_rejects_decision_and_beta_drift() -> None:
    beta, detection_p = _parity_matrices()
    quality = t.zeros_like(beta, dtype=t.bool)
    quality_drift = quality.clone()
    quality_drift[0, 0] = True
    with pytest.raises(ValueError, match="quality-mask complete-probe sets differ"):
        assert_processor_parity(
            beta,
            beta,
            detection_p,
            detection_p,
            quality_drift,
            quality,
            detection_threshold=parse_fraction(0.01),
            maximum_sample_failure_fraction=parse_fraction(0.01),
        )
    sample_drift = detection_p.clone()
    sample_drift[0, 0] = 0.5
    with pytest.raises(ValueError, match="sample-retention decisions differ"):
        assert_processor_parity(
            beta,
            beta,
            sample_drift,
            detection_p,
            quality,
            quality,
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
            quality,
            quality,
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
            quality,
            quality,
            detection_threshold=parse_fraction(0.01),
            maximum_sample_failure_fraction=parse_fraction(0.01),
        )
    with pytest.raises(ValueError, match="exactly 12 arrays"):
        assert_processor_parity(
            beta[:, :-1],
            beta[:, :-1],
            detection_p[:, :-1],
            detection_p[:, :-1],
            quality[:, :-1],
            quality[:, :-1],
            detection_threshold=parse_fraction(0.01),
            maximum_sample_failure_fraction=parse_fraction(0.01),
        )
    bad_quality = quality.to(t.uint8)
    with pytest.raises(TypeCheckError, match="methylprep_quality_excluded"):
        assert_processor_parity(
            beta,
            beta,
            detection_p,
            detection_p,
            bad_quality,
            quality,
            detection_threshold=parse_fraction(0.01),
            maximum_sample_failure_fraction=parse_fraction(0.01),
        )
    almost_all_quality_excluded = quality.clone()
    almost_all_quality_excluded[1:] = True
    with pytest.raises(ValueError, match="fewer than two probes"):
        assert_processor_parity(
            beta,
            beta,
            detection_p,
            detection_p,
            almost_all_quality_excluded,
            almost_all_quality_excluded,
            detection_threshold=parse_fraction(0.01),
            maximum_sample_failure_fraction=parse_fraction(0.01),
        )
    almost_all_samples_fail = detection_p.clone()
    almost_all_samples_fail[:, 1:] = 1.0
    with pytest.raises(ValueError, match="fewer than two arrays"):
        assert_processor_parity(
            beta,
            beta,
            almost_all_samples_fail,
            almost_all_samples_fail,
            quality,
            quality,
            detection_threshold=parse_fraction(0.01),
            maximum_sample_failure_fraction=parse_fraction(0.01),
        )
    one_complete_probe = detection_p.clone()
    one_complete_probe[1:] = 1.0
    with pytest.raises(ValueError, match="fewer than two probes"):
        assert_processor_parity(
            beta,
            beta,
            one_complete_probe,
            one_complete_probe,
            quality,
            quality,
            detection_threshold=parse_fraction(0.01),
            maximum_sample_failure_fraction=parse_fraction(1.0),
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
