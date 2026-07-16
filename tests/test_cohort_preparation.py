from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch as t

from methylation_latent.cohort import (
    CohortQcAudit,
    load_prepared_cohort,
    load_probe_table,
    prepare_gse87571_cohort,
    save_prepared_cohort_exclusive,
    write_cohort_exclusion_ledger_exclusive,
    write_probe_table_exclusive,
)
from methylation_latent.domain import (
    NonEmptyProbeSet,
    ProbeLocus,
    parse_age_years,
    parse_fraction,
    parse_gsm_accession,
    parse_positive_int,
    parse_sentrix_identity,
)
from methylation_latent.metadata import Gender, Gse87571RawSample, Gse87571RawSampleSet
from methylation_latent.processor_io import ProcessorOutput


def _raw_samples() -> Gse87571RawSampleSet:
    rows: list[Gse87571RawSample] = []
    for index in range(4):
        gsm = f"GSM{index + 1}"
        sentrix = f"581528400{index + 1}_R0{index + 1}C01"
        base = f"ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM0nnn/{gsm}/suppl/{gsm}_{sentrix}"
        phenotype_missing = index == 3
        rows.append(
            Gse87571RawSample(
                gsm_accession=parse_gsm_accession(gsm),
                subject_id=parse_positive_int(index + 1),
                sentrix_identity=parse_sentrix_identity(sentrix),
                age=None if phenotype_missing else parse_age_years(20 + index),
                gender=None if phenotype_missing else Gender.FEMALE,
                tissue="whole blood",
                disease_state="normal",
                green_url=f"{base}_Grn.idat.gz",
                red_url=f"{base}_Red.idat.gz",
            )
        )
    return Gse87571RawSampleSet(tuple(rows))


def _processor(
    probes: NonEmptyProbeSet,
    samples: Gse87571RawSampleSet,
) -> ProcessorOutput:
    beta = t.tensor(
        (
            (0.1, 0.2, 0.3, 0.4),
            (0.2, 0.4, 0.5, 0.6),
            (0.5, 0.5, 0.2, 0.7),
            (0.1, 0.9, 0.5, 0.2),
            (0.3, 0.7, 0.6, 0.1),
        ),
        dtype=t.float64,
    )
    detection_p = t.zeros_like(beta)
    detection_p[1, 0] = 0.1
    detection_p[:2, 2] = 0.1
    quality = t.zeros_like(beta, dtype=t.bool)
    quality[0, 1] = True
    return ProcessorOutput(
        probe_ids=tuple(str(probe.probe_id) for probe in probes.probes),
        sample_ids=tuple(str(sample.sentrix_identity) for sample in samples.samples),
        beta=beta,
        detection_p=detection_p,
        quality_excluded=quality,
    )


def test_gse87571_cohort_qc_is_sample_first_complete_case_and_persisted(
    tmp_path: Path,
    make_probe: Callable[..., ProbeLocus],
) -> None:
    probes = NonEmptyProbeSet(tuple(make_probe(index + 1) for index in range(5)))
    samples = _raw_samples()
    cohort = prepare_gse87571_cohort(
        _processor(probes, samples),
        samples,
        probes,
        detection_threshold=parse_fraction(0.05),
        maximum_sample_failure_fraction=parse_fraction(0.2),
    )
    assert cohort.sample_gsm_ids == ("GSM1", "GSM2")
    assert cohort.retained_raw_sample_indices.tolist() == [0, 1]
    assert [str(probe.probe_id) for probe in cohort.probes.probes] == [
        "cg00000004",
        "cg00000005",
    ]
    assert cohort.audit.quality_excluded_probes == 1
    assert cohort.audit.detection_excluded_probes == 1
    assert cohort.audit.constant_excluded_probes == 1
    assert cohort.audit.detection_failed_samples == 1
    assert cohort.audit.phenotype_missing_samples == 1
    assert cohort.build_targets().rank_upper_bound == 1

    tensor_path = tmp_path / "cohort.safetensors"
    metadata_path = tmp_path / "cohort.json"
    save_prepared_cohort_exclusive(tensor_path, metadata_path, cohort)
    loaded = load_prepared_cohort(
        tensor_path,
        metadata_path,
        probe_universe=probes,
    )
    assert loaded.audit == cohort.audit
    assert loaded.probes == cohort.probes
    assert t.equal(loaded.beta, cohort.beta)
    assert t.equal(loaded.age, cohort.age)
    probe_table = tmp_path / "probes.tsv"
    write_probe_table_exclusive(probe_table, cohort.probes)
    assert load_probe_table(probe_table) == cohort.probes


def test_prepared_cohort_rejects_axis_and_partition_drift(
    make_probe: Callable[..., ProbeLocus],
) -> None:
    probes = NonEmptyProbeSet(tuple(make_probe(index + 1) for index in range(5)))
    samples = _raw_samples()
    output = _processor(probes, samples)
    with pytest.raises(ValueError, match="raw GSE87571 sample order"):
        prepare_gse87571_cohort(
            ProcessorOutput(
                probe_ids=output.probe_ids,
                sample_ids=tuple(reversed(output.sample_ids)),
                beta=output.beta,
                detection_p=output.detection_p,
                quality_excluded=output.quality_excluded,
            ),
            samples,
            probes,
            detection_threshold=parse_fraction(0.05),
            maximum_sample_failure_fraction=parse_fraction(0.2),
        )
    audit = CohortQcAudit(
        detection_threshold=parse_fraction(0.05),
        maximum_sample_failure_fraction=parse_fraction(0.2),
        raw_samples=4,
        phenotype_eligible_samples=3,
        retained_samples=2,
        detection_failed_samples=1,
        phenotype_missing_samples=1,
        static_probes=5,
        quality_excluded_probes=1,
        detection_excluded_probes=1,
        constant_excluded_probes=1,
        retained_probes=2,
    )
    with pytest.raises(ValueError, match="sample QC counts"):
        CohortQcAudit(
            detection_threshold=audit.detection_threshold,
            maximum_sample_failure_fraction=audit.maximum_sample_failure_fraction,
            raw_samples=5,
            phenotype_eligible_samples=audit.phenotype_eligible_samples,
            retained_samples=audit.retained_samples,
            detection_failed_samples=audit.detection_failed_samples,
            phenotype_missing_samples=audit.phenotype_missing_samples,
            static_probes=audit.static_probes,
            quality_excluded_probes=audit.quality_excluded_probes,
            detection_excluded_probes=audit.detection_excluded_probes,
            constant_excluded_probes=audit.constant_excluded_probes,
            retained_probes=audit.retained_probes,
        )


def test_cohort_qc_rejects_empty_complete_case_universe(
    make_probe: Callable[..., ProbeLocus],
) -> None:
    probes = NonEmptyProbeSet(tuple(make_probe(index + 1) for index in range(5)))
    samples = _raw_samples()
    output = _processor(probes, samples)
    with pytest.raises(ValueError, match="removed every static probe"):
        prepare_gse87571_cohort(
            ProcessorOutput(
                probe_ids=output.probe_ids,
                sample_ids=output.sample_ids,
                beta=output.beta,
                detection_p=output.detection_p,
                quality_excluded=t.ones_like(output.quality_excluded),
            ),
            samples,
            probes,
            detection_threshold=parse_fraction(0.05),
            maximum_sample_failure_fraction=parse_fraction(0.2),
        )


def _prepared_cohort(
    make_probe: Callable[..., ProbeLocus],
):
    probes = NonEmptyProbeSet(tuple(make_probe(index + 1) for index in range(5)))
    return (
        probes,
        prepare_gse87571_cohort(
            _processor(probes, _raw_samples()),
            _raw_samples(),
            probes,
            detection_threshold=parse_fraction(0.05),
            maximum_sample_failure_fraction=parse_fraction(0.2),
        ),
    )


def test_cohort_domain_objects_reject_every_axis_drift(
    make_probe: Callable[..., ProbeLocus],
) -> None:
    _, cohort = _prepared_cohort(make_probe)
    audit = cohort.audit
    with pytest.raises(ValueError, match="non-empty sample and probe"):
        replace(audit, retained_probes=0, constant_excluded_probes=3)
    with pytest.raises(ValueError, match="phenotype-eligibility"):
        replace(audit, phenotype_eligible_samples=2)
    with pytest.raises(ValueError, match="probe QC counts"):
        replace(audit, static_probes=6)
    with pytest.raises(ValueError, match="sample axes"):
        replace(cohort, sample_gsm_ids=("GSM1",))
    with pytest.raises(ValueError, match="duplicate Sentrix"):
        replace(cohort, sample_sentrix_ids=(cohort.sample_sentrix_ids[0],) * 2)
    with pytest.raises(ValueError, match="duplicate GSM"):
        replace(cohort, sample_gsm_ids=(cohort.sample_gsm_ids[0],) * 2)
    with pytest.raises(ValueError, match="beta shape"):
        replace(cohort, beta=cohort.beta[:1])
    with pytest.raises(TypeError, match="must be float64"):
        replace(cohort, beta=cohort.beta.float())
    with pytest.raises(ValueError, match="age vector"):
        replace(cohort, age=cohort.age[:1])
    nonfinite_age = cohort.age.clone()
    nonfinite_age[0] = t.nan
    with pytest.raises(ValueError, match="must be finite"):
        replace(cohort, age=nonfinite_age)
    unbounded_beta = cohort.beta.clone()
    unbounded_beta[0, 0] = 1.1
    with pytest.raises(ValueError, match=r"lie in \[0, 1\]"):
        replace(cohort, beta=unbounded_beta)
    with pytest.raises(ValueError, match="failure fractions must align"):
        replace(cohort, raw_sample_failure_fractions=t.zeros(3, dtype=t.float64))
    invalid_failure = cohort.raw_sample_failure_fractions.clone()
    invalid_failure[0] = t.inf
    with pytest.raises(ValueError, match=r"finite values in \[0, 1\]"):
        replace(cohort, raw_sample_failure_fractions=invalid_failure)
    with pytest.raises(ValueError, match="indices must align"):
        replace(cohort, retained_raw_sample_indices=t.tensor([0.0, 1.0]))
    with pytest.raises(ValueError, match="strictly increasing"):
        replace(cohort, retained_raw_sample_indices=t.tensor([1, 0]))
    changed_audit = replace(
        audit,
        retained_samples=3,
        detection_failed_samples=0,
    )
    with pytest.raises(ValueError, match="axes differ"):
        replace(cohort, audit=changed_audit)
    with pytest.raises(ValueError, match="ledger differs"):
        replace(
            cohort,
            quality_excluded_probe_ids=(
                cohort.quality_excluded_probe_ids[0],
                cohort.quality_excluded_probe_ids[0],
            ),
        )


def test_cohort_persistence_rejects_metadata_and_table_drift(
    tmp_path: Path,
    make_probe: Callable[..., ProbeLocus],
) -> None:
    probes, cohort = _prepared_cohort(make_probe)
    tensor = tmp_path / "cohort.safetensors"
    metadata = tmp_path / "cohort.json"
    save_prepared_cohort_exclusive(tensor, metadata, cohort)
    with pytest.raises(ValueError, match="share a directory"):
        save_prepared_cohort_exclusive(
            tmp_path / "other" / "cohort.safetensors",
            tmp_path / "elsewhere" / "cohort.json",
            cohort,
        )

    def metadata_copy(name: str) -> tuple[Path, dict[str, Any]]:
        path = tmp_path / name
        return path, json.loads(metadata.read_text(encoding="utf-8"))

    malformed, raw = metadata_copy("malformed.json")
    raw["unknown"] = 1
    malformed.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata envelope"):
        load_prepared_cohort(tensor, malformed, probe_universe=probes)

    schema, raw = metadata_copy("schema.json")
    raw["schema"] = "other"
    schema.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="schema or tensor"):
        load_prepared_cohort(tensor, schema, probe_universe=probes)

    empty_axis, raw = metadata_copy("empty-axis.json")
    raw["sample_gsm_ids"] = []
    empty_axis.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty string array"):
        load_prepared_cohort(tensor, empty_axis, probe_universe=probes)

    invalid_ledger, raw = metadata_copy("invalid-ledger.json")
    raw["quality_excluded_probe_ids"] = [""]
    invalid_ledger.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a string array"):
        load_prepared_cohort(tensor, invalid_ledger, probe_universe=probes)

    missing_probe, raw = metadata_copy("missing-probe.json")
    raw["probe_ids"][0] = "cg99999999"
    missing_probe.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="absent from the static universe"):
        load_prepared_cohort(tensor, missing_probe, probe_universe=probes)

    audit_fields, raw = metadata_copy("audit-fields.json")
    del raw["audit"]["raw_samples"]
    audit_fields.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="QC audit fields"):
        load_prepared_cohort(tensor, audit_fields, probe_universe=probes)

    probe_hash, raw = metadata_copy("probe-hash.json")
    raw["probe_order_sha256"] = "0" * 64
    probe_hash.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="probe-order fingerprint"):
        load_prepared_cohort(tensor, probe_hash, probe_universe=probes)

    sample_hash, raw = metadata_copy("sample-hash.json")
    raw["sample_order_sha256"] = "0" * 64
    sample_hash.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="sample-order fingerprint"):
        load_prepared_cohort(tensor, sample_hash, probe_universe=probes)

    wrong_header = tmp_path / "wrong-header.tsv"
    wrong_header.write_text("wrong\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header differs"):
        load_probe_table(wrong_header)
    wrong_row = tmp_path / "wrong-row.tsv"
    wrong_row.write_text(
        "probe_id\tchromosome\tposition\tcontext\tdesign\tmanifest_strand\ncg00000001\t1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="wrong field count"):
        load_probe_table(wrong_row)

    ledger = tmp_path / "cohort-exclusions.tsv"
    write_cohort_exclusion_ledger_exclusive(ledger, cohort)
    rows = ledger.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1 + sum(
        (
            cohort.audit.quality_excluded_probes,
            cohort.audit.detection_excluded_probes,
            cohort.audit.constant_excluded_probes,
            cohort.audit.detection_failed_samples,
            cohort.audit.phenotype_missing_samples,
        )
    )


def test_cohort_qc_rejects_too_few_samples_and_only_constant_probes(
    make_probe: Callable[..., ProbeLocus],
) -> None:
    probes = NonEmptyProbeSet(tuple(make_probe(index + 1) for index in range(5)))
    samples = _raw_samples()
    output = _processor(probes, samples)
    with pytest.raises(ValueError, match="fewer than two"):
        prepare_gse87571_cohort(
            output,
            samples,
            probes,
            detection_threshold=parse_fraction(0.05),
            maximum_sample_failure_fraction=parse_fraction(0.0),
        )
    constant_beta = t.full_like(output.beta, 0.5)
    with pytest.raises(ValueError, match="every non-constant"):
        prepare_gse87571_cohort(
            ProcessorOutput(
                probe_ids=output.probe_ids,
                sample_ids=output.sample_ids,
                beta=constant_beta,
                detection_p=t.zeros_like(output.detection_p),
                quality_excluded=t.zeros_like(output.quality_excluded),
            ),
            samples,
            probes,
            detection_threshold=parse_fraction(0.05),
            maximum_sample_failure_fraction=parse_fraction(0.2),
        )
