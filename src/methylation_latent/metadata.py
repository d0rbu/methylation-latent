"""Strict GSE40279 sample metadata parsing and IDAT identity audits."""

from __future__ import annotations

import csv
import gzip
import re
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TextIO

import torch as t
from beartype import beartype

from methylation_latent.artifacts import sha256_file, sha256_ordered_strings
from methylation_latent.domain import (
    AgeYears,
    GsmAccession,
    PositiveInt,
    SentrixIdentity,
    parse_age_years,
    parse_gsm_accession,
    parse_positive_int,
    parse_sentrix_identity,
)

_TITLE = re.compile(r"^age ([0-9]+)y ([0-9]+)$", flags=re.ASCII)
_EXPECTED_CHARACTERISTICS = frozenset(
    {"age (y)", "source", "plate", "gender", "ethnicity", "tissue"}
)
GSE40279_SERIES_MATRIX_SHA256 = "1f6ef8b5ce08f9a5f1c09b047b36d1f3342757b6e8e19dce0398581bb76807eb"
GSE40279_SAMPLE_KEY_SHA256 = "dbb7c510da6a90bd2ff203564288b3b3400298c9d50217377a347f2ef2a9e7f9"
GSE40279_SAMPLE_ORDER_SHA256 = "f0bf579bb37e25dee7a7f9c3c29a5d498a506ec2226514ee1651c833b4707d61"
_SENTRIX_POSITIONS = frozenset(
    f"R{row:02d}C{column:02d}" for row in range(1, 7) for column in range(1, 3)
)


class Gender(StrEnum):
    """Narrow observed GSE40279 gender field; values outside F/M are rejected."""

    FEMALE = "F"
    MALE = "M"


@beartype
@dataclass(frozen=True, slots=True)
class SeriesSample:
    gsm_accession: GsmAccession
    subject_id: PositiveInt
    age: AgeYears
    source: str
    plate: PositiveInt
    gender: Gender
    ethnicity: str
    tissue: str

    def __post_init__(self) -> None:
        if not self.source or not self.ethnicity:
            raise ValueError("sample source and ethnicity must not be empty")
        if self.tissue != "whole blood":
            raise ValueError(f"unexpected sample tissue: {self.tissue!r}")


@beartype
@dataclass(frozen=True, slots=True)
class SampleMetadata:
    gsm_accession: GsmAccession
    subject_id: PositiveInt
    sentrix_identity: SentrixIdentity
    age: AgeYears
    source: str
    plate: PositiveInt
    gender: Gender
    ethnicity: str
    tissue: str


@dataclass(frozen=True, slots=True)
class OrderedSampleSet:
    samples: tuple[SampleMetadata, ...]

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError("sample set must not be empty")
        for name, values in (
            ("GSM accession", tuple(sample.gsm_accession for sample in self.samples)),
            ("subject ID", tuple(sample.subject_id for sample in self.samples)),
            ("Sentrix identity", tuple(sample.sentrix_identity for sample in self.samples)),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"sample set contains duplicate {name} values")

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def order_sha256(self) -> str:
        return sha256_ordered_strings(
            f"{sample.gsm_accession}\t{sample.subject_id}\t{sample.sentrix_identity}"
            for sample in self.samples
        )


@beartype
def select_parity_sample_set(
    samples: OrderedSampleSet,
    *,
    expected_count: PositiveInt | int,
    seed: int,
) -> OrderedSampleSet:
    """Choose one array per physical position while greedily spreading plates."""

    count = int(expected_count)
    if count != len(_SENTRIX_POSITIONS):
        raise ValueError("the frozen parity subset requires exactly 12 arrays")
    indices_by_position: defaultdict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples.samples):
        position = str(sample.sentrix_identity).split("_", maxsplit=1)[1]
        indices_by_position[position].append(index)
    if set(indices_by_position) != _SENTRIX_POSITIONS:
        raise ValueError("sample inventory does not cover exactly the 12 Sentrix positions")

    generator = t.Generator(device="cpu").manual_seed(seed)
    plate_usage: defaultdict[int, int] = defaultdict(int)
    selected_indices: list[int] = []
    for position in sorted(_SENTRIX_POSITIONS):
        candidates = tuple(indices_by_position[position])
        minimum_usage = min(plate_usage[int(samples.samples[index].plate)] for index in candidates)
        least_used = tuple(
            index
            for index in candidates
            if plate_usage[int(samples.samples[index].plate)] == minimum_usage
        )
        offset = int(t.randint(len(least_used), size=(), generator=generator).item())
        selected = least_used[offset]
        selected_indices.append(selected)
        plate_usage[int(samples.samples[selected].plate)] += 1
    ordered = tuple(samples.samples[index] for index in sorted(selected_indices))
    selected = OrderedSampleSet(ordered)
    positions = {
        str(sample.sentrix_identity).split("_", maxsplit=1)[1] for sample in selected.samples
    }
    if positions != _SENTRIX_POSITIONS:
        raise RuntimeError("parity subset selection lost a Sentrix position")
    return selected


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8", newline="")
    return path.open(mode="rt", encoding="utf-8", newline="")


def assert_gzip_integrity(path: Path, *, chunk_size: int = 1 << 20) -> int:
    """Read through the CRC trailer; parsing only an early header is not an integrity check."""

    if path.suffix != ".gz":
        raise ValueError("gzip integrity audit requires a .gz input")
    if chunk_size <= 0:
        raise ValueError("gzip integrity chunk size must be positive")
    uncompressed_bytes = 0
    with gzip.open(path, mode="rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            uncompressed_bytes += len(chunk)
    if uncompressed_bytes == 0:
        raise ValueError("gzip input decompresses to an empty payload")
    return uncompressed_bytes


def _parse_geo_header_row(raw_line: str) -> tuple[str, tuple[str, ...]]:
    fields = next(csv.reader((raw_line,), delimiter="\t", quotechar='"'))
    if len(fields) < 2:
        raise ValueError("GEO sample header row contains no sample values")
    return fields[0], tuple(fields[1:])


@beartype
def parse_series_matrix_metadata(
    path: Path,
    *,
    expected_sample_count: PositiveInt | int = 656,
) -> tuple[SeriesSample, ...]:
    """Read only the series header and reject missing, duplicate, or drifting fields."""

    singleton_rows: dict[str, tuple[str, ...]] = {}
    characteristic_rows: list[tuple[str, ...]] = []
    required_singletons = {
        "!Sample_title",
        "!Sample_geo_accession",
        "!Sample_source_name_ch1",
    }
    found_table = False
    with _open_text(path) as handle:
        for raw_line in handle:
            if raw_line.startswith("!series_matrix_table_begin"):
                found_table = True
                break
            if raw_line.startswith("!Sample_characteristics_ch1"):
                _, values = _parse_geo_header_row(raw_line)
                characteristic_rows.append(values)
            elif any(raw_line.startswith(name) for name in required_singletons):
                name, values = _parse_geo_header_row(raw_line)
                if name in singleton_rows:
                    raise ValueError(f"duplicate GEO metadata row: {name}")
                singleton_rows[name] = values
    if not found_table:
        raise ValueError("series matrix header never reached table-begin marker")
    if set(singleton_rows) != required_singletons:
        raise ValueError(
            f"series metadata singleton rows differ: missing={sorted(required_singletons - set(singleton_rows))}"
        )
    count = int(expected_sample_count)
    if count <= 0:
        raise ValueError("expected sample count must be positive")
    if any(len(values) != count for values in (*singleton_rows.values(), *characteristic_rows)):
        raise ValueError("series metadata row length differs from expected sample count")

    characteristics: dict[str, tuple[str, ...]] = {}
    for row in characteristic_rows:
        parsed = tuple(value.split(": ", maxsplit=1) for value in row)
        if any(len(fields) != 2 for fields in parsed):
            raise ValueError("sample characteristic does not use the expected 'name: value' form")
        names = {fields[0] for fields in parsed}
        if len(names) != 1:
            raise ValueError("one characteristic row mixes field names across samples")
        name = next(iter(names))
        if name in characteristics:
            raise ValueError(f"duplicate sample characteristic: {name}")
        characteristics[name] = tuple(fields[1] for fields in parsed)
    if set(characteristics) != _EXPECTED_CHARACTERISTICS:
        raise ValueError(
            f"sample characteristics differ: missing={sorted(_EXPECTED_CHARACTERISTICS - set(characteristics))}, "
            f"unknown={sorted(set(characteristics) - _EXPECTED_CHARACTERISTICS)}"
        )

    samples: list[SeriesSample] = []
    for index in range(count):
        title = singleton_rows["!Sample_title"][index]
        title_match = _TITLE.fullmatch(title)
        if title_match is None:
            raise ValueError(f"sample title does not match frozen grammar: {title!r}")
        title_age, subject = title_match.groups()
        if title_age != characteristics["age (y)"][index]:
            raise ValueError(f"title/characteristic age mismatch at sample index {index}")
        source_name = singleton_rows["!Sample_source_name_ch1"][index]
        if source_name != f"X{subject}":
            raise ValueError(f"source-name/subject mismatch at sample index {index}")
        samples.append(
            SeriesSample(
                gsm_accession=parse_gsm_accession(singleton_rows["!Sample_geo_accession"][index]),
                subject_id=parse_positive_int(subject),
                age=parse_age_years(title_age),
                source=characteristics["source"][index],
                plate=parse_positive_int(characteristics["plate"][index]),
                gender=Gender(characteristics["gender"][index]),
                ethnicity=characteristics["ethnicity"][index],
                tissue=characteristics["tissue"][index],
            )
        )
    if len({sample.gsm_accession for sample in samples}) != count:
        raise ValueError("series metadata contains duplicate GSM accessions")
    if len({sample.subject_id for sample in samples}) != count:
        raise ValueError("series metadata contains duplicate subject IDs")
    return tuple(samples)


@beartype
def join_sample_key(
    series_samples: tuple[SeriesSample, ...], sample_key_path: Path
) -> OrderedSampleSet:
    """Join by explicit row and subject identity; never by filename sorting."""

    if not series_samples:
        raise ValueError("sample-key join requires series samples")
    key_rows: list[tuple[int, PositiveInt, SentrixIdentity]] = []
    with _open_text(sample_key_path) as handle:
        reader = csv.reader(handle, delimiter="\t", quotechar='"')
        for line_number, fields in enumerate(reader, start=1):
            if len(fields) != 3:
                raise ValueError(f"sample-key line {line_number} must contain exactly three fields")
            row, subject, sentrix = fields
            key_rows.append(
                (int(row), parse_positive_int(subject), parse_sentrix_identity(sentrix))
            )
    if len(key_rows) != len(series_samples):
        raise ValueError("sample-key and series sample counts differ")
    joined: list[SampleMetadata] = []
    for expected_row, (series_sample, key_row) in enumerate(
        zip(series_samples, key_rows, strict=True), start=1
    ):
        row, subject_id, sentrix_identity = key_row
        if row != expected_row:
            raise ValueError(f"sample-key row sequence breaks at {expected_row}: observed {row}")
        if subject_id != series_sample.subject_id:
            raise ValueError(f"sample-key subject mismatch at row {expected_row}")
        joined.append(
            SampleMetadata(
                gsm_accession=series_sample.gsm_accession,
                subject_id=series_sample.subject_id,
                sentrix_identity=sentrix_identity,
                age=series_sample.age,
                source=series_sample.source,
                plate=series_sample.plate,
                gender=series_sample.gender,
                ethnicity=series_sample.ethnicity,
                tissue=series_sample.tissue,
            )
        )
    return OrderedSampleSet(tuple(joined))


@dataclass(frozen=True, slots=True)
class IdatPair:
    sentrix_identity: SentrixIdentity
    red_path: Path
    green_path: Path
    red_sha256: str
    green_sha256: str


def _validate_idat_file(path: Path) -> str:
    if not path.is_file() or path.stat().st_size <= 4:
        raise ValueError(f"IDAT file is absent or empty: {path}")
    with path.open("rb") as handle:
        magic = handle.read(4)
    if magic != b"IDAT":
        raise ValueError(f"file does not have IDAT magic bytes: {path}")
    return sha256_file(path)


@beartype
def audit_idat_inventory(directory: Path, samples: OrderedSampleSet) -> tuple[IdatPair, ...]:
    """Require exactly one readable red/green pair for every ordered sample identity."""

    expected_paths: set[Path] = set()
    pairs: list[IdatPair] = []
    for sample in samples.samples:
        red = directory / f"{sample.sentrix_identity}_Red.idat"
        green = directory / f"{sample.sentrix_identity}_Grn.idat"
        expected_paths.update((red, green))
        pairs.append(
            IdatPair(
                sentrix_identity=sample.sentrix_identity,
                red_path=red,
                green_path=green,
                red_sha256=_validate_idat_file(red),
                green_sha256=_validate_idat_file(green),
            )
        )
    observed_paths = set(directory.glob("*.idat"))
    if observed_paths != expected_paths:
        raise ValueError(
            f"IDAT inventory differs: missing={sorted(map(str, expected_paths - observed_paths))}, "
            f"unknown={sorted(map(str, observed_paths - expected_paths))}"
        )
    return tuple(pairs)
