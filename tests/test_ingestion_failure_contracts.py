from __future__ import annotations

import gzip
import json
import struct
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import pytest
import torch as t

from methylation_latent import idat as idat_module
from methylation_latent import processor_io as processor_module
from methylation_latent.cohort_diagnostics import (
    DuplicateIdentityAudit,
    PublicGenotypePanel,
    _cell_value,
    _shared_strings,
    audit_duplicate_public_genotypes,
)
from methylation_latent.domain import (
    parse_age_years,
    parse_fraction,
    parse_gsm_accession,
    parse_positive_int,
    parse_sentrix_identity,
)
from methylation_latent.idat import (
    CompressedIdat,
    CompressedIdatInventory,
    _audit_gzip_idat,
    _materialize_one,
    _source_name,
    audit_gse87571_compressed_idats,
    load_compressed_idat_inventory,
    materialize_idats,
    save_compressed_idat_inventory_exclusive,
)
from methylation_latent.metadata import Gender, Gse87571RawSample, Gse87571RawSampleSet
from methylation_latent.preprocessing import _column_correlations, assert_processor_parity
from methylation_latent.processor_io import (
    ProcessorOutput,
    _assert_idat_magic,
    _load_exact_vector,
    _read_nonempty_unique_lines,
    compare_sesame_threshold_outputs,
    load_sesame_output,
    select_processor_output_probes,
)


def _raw_samples(sentrix: str = "5815284001_R01C01") -> Gse87571RawSampleSet:
    gsm = "GSM1"
    base = f"ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM0nnn/{gsm}/suppl/{gsm}_{sentrix}"
    return Gse87571RawSampleSet(
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


def _compressed_record(
    tmp_path: Path,
    channel: str,
    *,
    sentrix: str = "5815284001_R01C01",
) -> CompressedIdat:
    return CompressedIdat(
        sentrix_identity=parse_sentrix_identity(sentrix),
        channel=channel,
        compressed_path=tmp_path / f"{channel}.idat.gz",
        compressed_bytes=10,
        compressed_sha256="0" * 64,
        decompressed_bytes=10,
        decompressed_sha256="1" * 64,
    )


def _processor_output() -> ProcessorOutput:
    return ProcessorOutput(
        probe_ids=("cg00000001", "cg00000002"),
        sample_ids=("sample-1", "sample-2"),
        beta=t.tensor(((0.1, 0.2), (0.3, 0.4)), dtype=t.float64),
        detection_p=t.tensor(((0.001, 0.002), (0.003, 0.004)), dtype=t.float64),
        quality_excluded=t.zeros((2, 2), dtype=t.bool),
    )


def _write_sesame_fixture(
    directory: Path,
    *,
    threshold: str,
    beta_columns: tuple[tuple[float, ...], ...],
) -> tuple[str, ...]:
    sample_ids = ("sample-1", "sample-2")
    probe_ids = ("cg00000001", "cg00000002")
    directory.mkdir()
    (directory / "probe_ids.txt").write_text("\n".join(probe_ids) + "\n", encoding="utf-8")
    (directory / "sample_order.txt").write_text(
        "\n".join(sample_ids) + "\n",
        encoding="utf-8",
    )
    environment = (
        "R=4.6.0",
        "Bioconductor=3.23",
        "sesame=1.30.1",
        "sesameData=1.30.0",
        f"pipeline=QCD-pOOBAH@{threshold}-B",
        "probe_filter=^cg[0-9]{8}$",
        "beta_mask=false",
        f"samples={len(sample_ids)}",
        f"probes={len(probe_ids)}",
    )
    (directory / "environment.txt").write_text(
        "\n".join(environment) + "\n",
        encoding="utf-8",
    )
    for sample_id, beta in zip(sample_ids, beta_columns, strict=True):
        (directory / f"{sample_id}.beta.f64").write_bytes(struct.pack(f"<{len(beta)}d", *beta))
        (directory / f"{sample_id}.detection_p.f64").write_bytes(struct.pack("<2d", 0.001, 0.002))
        (directory / f"{sample_id}.quality_excluded.u8").write_bytes(b"\x00\x01")
    return sample_ids


def test_sesame_threshold_comparison_streams_exact_axes_and_payloads(
    tmp_path: Path,
) -> None:
    audit = tmp_path / "p001"
    primary = tmp_path / "p005"
    sample_ids = _write_sesame_fixture(
        audit,
        threshold="0.01",
        beta_columns=((0.1, 0.2), (0.3, 0.4)),
    )
    _write_sesame_fixture(
        primary,
        threshold="0.05",
        beta_columns=((0.1, 0.2), (0.3, 0.5)),
    )
    comparison = compare_sesame_threshold_outputs(
        audit,
        primary,
        expected_sample_ids=sample_ids,
    )
    assert comparison.probe_count == 2
    assert comparison.sample_count == 2
    assert comparison.beta_nonidentical_values == 1
    assert comparison.beta_maximum_absolute_difference == pytest.approx(0.1)
    assert comparison.beta_mean_absolute_difference == pytest.approx(0.025)

    (primary / "sample-1.detection_p.f64").write_bytes(struct.pack("<2d", 0.001, 0.003))
    with pytest.raises(ValueError, match="raw detection-p"):
        compare_sesame_threshold_outputs(
            audit,
            primary,
            expected_sample_ids=sample_ids,
        )


def _public_panel(*, variants: int = 104) -> PublicGenotypePanel:
    return PublicGenotypePanel(
        gsm_accessions=(parse_gsm_accession("GSM1"), parse_gsm_accession("GSM2")),
        variant_names=tuple(f"rs{index}" for index in range(variants)),
        dosages=t.stack(
            (
                t.zeros(variants, dtype=t.float64),
                t.ones(variants, dtype=t.float64),
            )
        ),
        observed=t.ones((2, variants), dtype=t.bool),
    )


def test_xlsx_shared_string_paths_fail_loudly(tmp_path: Path) -> None:
    archive_path = tmp_path / "shared.xlsx"
    shared_xml = (
        '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        "<si><t>first</t></si><si><r><t>sec</t></r><r><t>ond</t></r></si></sst>"
    )
    with ZipFile(archive_path, mode="w") as archive:
        archive.writestr("xl/sharedStrings.xml", shared_xml)
    with ZipFile(archive_path) as archive:
        assert _shared_strings(archive) == ("first", "second")

    namespace = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    valid = ElementTree.fromstring(f'<c xmlns="{namespace}" t="s"><v>1</v></c>')
    assert _cell_value(valid, shared_strings=("first", "second")) == "second"
    invalid = ElementTree.fromstring(f'<c xmlns="{namespace}" t="s"><v>2</v></c>')
    with pytest.raises(ValueError, match="out of bounds"):
        _cell_value(invalid, shared_strings=("first", "second"))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda panel: replace(panel, gsm_accessions=()), "axes"),
        (
            lambda panel: replace(
                panel,
                gsm_accessions=(panel.gsm_accessions[0], panel.gsm_accessions[0]),
            ),
            "duplicate GSM",
        ),
        (
            lambda panel: replace(
                panel,
                variant_names=(panel.variant_names[0],) * len(panel.variant_names),
            ),
            "duplicate variants",
        ),
        (lambda panel: replace(panel, dosages=panel.dosages[:, :-1]), "dosage shape"),
        (
            lambda panel: replace(panel, observed=panel.observed.to(t.int64)),
            "observation mask",
        ),
        (lambda panel: replace(panel, dosages=panel.dosages.to(t.float32)), "float64"),
        (
            lambda panel: replace(
                panel,
                dosages=panel.dosages.clone().index_put_(
                    (t.tensor([0]), t.tensor([0])),
                    t.tensor([3.0], dtype=t.float64),
                ),
            ),
            r"\[0, 2\]",
        ),
    ],
)
def test_public_genotype_panel_rejects_illegal_states(
    mutation: Callable[[PublicGenotypePanel], PublicGenotypePanel],
    message: str,
) -> None:
    panel = _public_panel()
    with pytest.raises(ValueError, match=message):
        mutation(panel)


def test_public_genotype_audit_rejects_pairs_without_shared_variants() -> None:
    panel = _public_panel(variants=200)
    observed = t.zeros((2, 200), dtype=t.bool)
    observed[0, :100] = True
    observed[1, 100:] = True
    dosages = panel.dosages.clone()
    dosages[~observed] = 0.0
    with pytest.raises(ValueError, match="fewer than two observed"):
        audit_duplicate_public_genotypes(replace(panel, dosages=dosages, observed=observed))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"nearest_mad_left": -1}, "indices"),
        ({"maximum_genotype_call_agreement": 1.1}, "agreement"),
        ({"minimum_mean_absolute_dosage_difference": -0.1}, "non-negative"),
        ({"nearest_mad_shared_variants": 0}, "shared observed"),
        ({"excluded_insufficient_samples": (-1,)}, "non-negative"),
    ],
)
def test_duplicate_identity_audit_rejects_illegal_states(
    kwargs: dict[str, object],
    message: str,
) -> None:
    valid = DuplicateIdentityAudit(0, 1, 0.1, 100, 0.9, 0, 1, 100, ())
    with pytest.raises(ValueError, match=message):
        replace(valid, **kwargs)


def test_compressed_idat_value_objects_reject_corrupt_metadata(tmp_path: Path) -> None:
    green = _compressed_record(tmp_path, "Grn")
    red = _compressed_record(tmp_path, "Red")
    with pytest.raises(ValueError, match="non-empty"):
        replace(green, compressed_bytes=4)
    with pytest.raises(ValueError, match="SHA-256"):
        replace(green, compressed_sha256="bad")
    with pytest.raises(ValueError, match="duplicate"):
        CompressedIdatInventory((green, green))
    other_red = _compressed_record(
        tmp_path,
        "Red",
        sentrix="5815284002_R02C01",
    )
    with pytest.raises(ValueError, match="incomplete"):
        CompressedIdatInventory((green, other_red))
    inventory = CompressedIdatInventory((green, red))
    with pytest.raises(ValueError, match="not fully represented"):
        inventory.records_for(_raw_samples("5815284002_R02C01"))


def test_compressed_inventory_loader_rejects_every_envelope_drift(
    tmp_path: Path,
) -> None:
    inventory = CompressedIdatInventory(
        (_compressed_record(tmp_path, "Grn"), _compressed_record(tmp_path, "Red"))
    )
    path = tmp_path / "inventory.json"
    save_compressed_idat_inventory_exclusive(path, inventory)
    valid = json.loads(path.read_text(encoding="utf-8"))

    invalid_payloads = (
        ([], "invalid envelope"),
        ({**valid, "schema": "wrong"}, "schema differs"),
        ({**valid, "records": []}, "non-empty array"),
        ({**valid, "records": [{"channel": "Grn"}]}, "record fields"),
        (
            {
                **valid,
                "records": [
                    {**record, "compressed_name": "../unsafe.idat.gz"}
                    for record in valid["records"]
                ],
            },
            "unsafe member",
        ),
        ({**valid, "fingerprint": "0" * 64}, "fingerprint differs"),
    )
    for index, (payload, message) in enumerate(invalid_payloads):
        invalid_path = tmp_path / f"invalid-{index}.json"
        invalid_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_compressed_idat_inventory(
                invalid_path,
                compressed_directory=tmp_path,
            )


def test_gzip_idat_helpers_reject_invalid_payloads_and_urls(tmp_path: Path) -> None:
    bad = tmp_path / "bad.idat.gz"
    with gzip.open(bad, mode="wb") as handle:
        handle.write(b"NOPE-payload")
    with pytest.raises(ValueError, match="positive"):
        _audit_gzip_idat(bad, chunk_size=0)
    with pytest.raises(ValueError, match="magic"):
        _audit_gzip_idat(bad)
    empty = tmp_path / "empty.idat.gz"
    with gzip.open(empty, mode="wb") as handle:
        handle.write(b"IDAT")
    with pytest.raises(ValueError, match="empty"):
        _audit_gzip_idat(empty)
    with pytest.raises(ValueError, match=r"\.idat\.gz"):
        _source_name("https://example.test/not-an-idat.txt")


def test_compressed_idat_audit_and_materialization_reject_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = _raw_samples()
    compressed = tmp_path / "compressed"
    compressed.mkdir()
    sizes: dict[str, int] = {}
    for url in (samples.samples[0].green_url, samples.samples[0].red_url):
        name = url.rsplit("/", maxsplit=1)[-1]
        with gzip.open(compressed / name, mode="wb") as handle:
            handle.write(b"IDAT-payload")
        sizes[name] = (compressed / name).stat().st_size
    wrong_sizes = dict(sizes)
    wrong_sizes[next(iter(wrong_sizes))] += 1
    with pytest.raises(ValueError, match="size differs"):
        audit_gse87571_compressed_idats(
            compressed,
            samples,
            expected_sizes=wrong_sizes,
        )

    inventory = audit_gse87571_compressed_idats(
        compressed,
        samples,
        expected_sizes=sizes,
    )
    output = tmp_path / "materialized"
    output.mkdir()
    (output / "unknown.idat").write_bytes(b"IDAT-unknown")
    with pytest.raises(ValueError, match="unknown files"):
        materialize_idats(inventory, samples, output_directory=output)

    record = inventory.records[0]
    destination = tmp_path / record.materialized_name
    original_sha256_file = idat_module.sha256_file

    def corrupt_temporary_hash(path: Path) -> str:
        if path.suffix == ".tmp":
            return "0" * 64
        return original_sha256_file(path)

    monkeypatch.setattr(idat_module, "sha256_file", corrupt_temporary_hash)
    with pytest.raises(RuntimeError, match="verification failed"):
        _materialize_one(record, destination)
    assert not destination.exists()


@pytest.mark.parametrize(
    ("mutation", "exception", "message"),
    [
        (lambda output: replace(output, probe_ids=()), ValueError, "axes"),
        (
            lambda output: replace(
                output,
                probe_ids=(output.probe_ids[0], output.probe_ids[0]),
            ),
            ValueError,
            "duplicate probe",
        ),
        (
            lambda output: replace(
                output,
                sample_ids=(output.sample_ids[0], output.sample_ids[0]),
            ),
            ValueError,
            "duplicate sample",
        ),
        (
            lambda output: replace(output, beta=output.beta[:1]),
            ValueError,
            "shapes",
        ),
        (
            lambda output: replace(output, beta=output.beta.to(t.float32)),
            TypeError,
            "float64",
        ),
        (
            lambda output: replace(
                output,
                quality_excluded=output.quality_excluded.to(t.uint8),
            ),
            ValueError,
            "boolean",
        ),
        (
            lambda output: replace(
                output,
                detection_p=output.detection_p.clone().index_put_(
                    (t.tensor([0]), t.tensor([0])),
                    t.tensor([float("nan")], dtype=t.float64),
                ),
            ),
            ValueError,
            "finite",
        ),
        (
            lambda output: replace(output, beta=output.beta + 1.0),
            ValueError,
            r"\[0, 1\]",
        ),
    ],
)
def test_processor_output_rejects_illegal_states(
    mutation: Callable[[ProcessorOutput], ProcessorOutput],
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        mutation(_processor_output())


def test_processor_file_helpers_reject_ambiguous_or_incompatible_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "missing.idat"
    with pytest.raises(ValueError, match="absent or empty"):
        _assert_idat_magic(missing)
    bad = tmp_path / "bad.idat"
    bad.write_bytes(b"NOPE-payload")
    with pytest.raises(ValueError, match="magic"):
        _assert_idat_magic(bad)

    lines = tmp_path / "lines.txt"
    lines.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        _read_nonempty_unique_lines(lines, "values")
    lines.write_text("a\na\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        _read_nonempty_unique_lines(lines, "values")

    vector = tmp_path / "vector.f64"
    vector.write_bytes(b"short")
    with pytest.raises(ValueError, match="binary size differs"):
        _load_exact_vector(vector, count=2, dtype=t.float64)
    monkeypatch.setattr(processor_module.sys, "byteorder", "big")
    with pytest.raises(RuntimeError, match="little-endian"):
        _load_exact_vector(vector, count=2, dtype=t.float64)


def test_processor_loader_and_selection_reject_inventory_and_order_ambiguity(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "processor"
    directory.mkdir()
    (directory / "probe_ids.txt").write_text("cg00000001\n", encoding="utf-8")
    (directory / "sample_order.txt").write_text("sample\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory differs"):
        load_sesame_output(directory, expected_sample_ids=("sample",))
    with pytest.raises(ValueError, match="non-empty and unique"):
        select_processor_output_probes(_processor_output(), ())
    with pytest.raises(ValueError, match="non-empty and unique"):
        select_processor_output_probes(
            _processor_output(),
            ("cg00000001", "cg00000001"),
        )


def test_processor_parity_rejects_constant_beta_columns() -> None:
    beta = t.ones((2, 12), dtype=t.float64) / 2
    detection_p = t.zeros_like(beta)
    quality = t.zeros_like(beta, dtype=t.bool)
    with pytest.raises(ValueError, match="must not be constant"):
        assert_processor_parity(
            beta,
            beta,
            detection_p,
            detection_p,
            quality,
            quality,
            detection_threshold=parse_fraction(0.01),
            maximum_sample_failure_fraction=parse_fraction(0.01),
        )
    with pytest.raises(ValueError, match="must not be constant"):
        _column_correlations(beta, beta)
