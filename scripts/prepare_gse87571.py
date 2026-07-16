"""End-to-end construction of the frozen primary data artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch as t
from beartype import beartype

from methylation_latent.artifacts import (
    JsonValue,
    sha256_file,
    sha256_ordered_strings,
    write_canonical_json_exclusive,
)
from methylation_latent.cohort import (
    build_static_probe_universe,
    prepare_gse87571_cohort,
    save_prepared_cohort_exclusive,
    write_cohort_exclusion_ledger_exclusive,
    write_probe_table_exclusive,
    write_static_exclusion_ledger_exclusive,
)
from methylation_latent.config import ProtocolConfig, load_protocol_config
from methylation_latent.experiment_data import (
    build_primary_splits,
    compute_sequence_feature_artifact,
    save_sequence_features_exclusive,
    save_split_artifact_exclusive,
)
from methylation_latent.genome import IndexedFasta, chromosome_lengths
from methylation_latent.hashing import assert_gzip_payload_matches_file, md5_file
from methylation_latent.manifest import (
    load_chen2013_cross_reactive,
    load_zhou2017_mask_general,
    parse_gpl13534_manifest,
)
from methylation_latent.metadata import (
    assert_gzip_integrity,
    parse_gse87571_series_metadata,
)
from methylation_latent.processor_io import (
    SesameThresholdSensitivityAudit,
    compare_sesame_threshold_outputs,
    load_sesame_output,
)
from methylation_latent.storage import save_target_geometry
from methylation_latent.targets import (
    assert_correlation_identity,
    validate_latent_dimension,
)

_STATIC_AUDIT_SCHEMA = "methylation-latent.gse87571-static-universe.v1"
_TARGET_AUDIT_SCHEMA = "methylation-latent.target-geometry-audit.v1"
_PRIMARY_SUMMARY_SCHEMA = "methylation-latent.primary-data-summary.v1"
_SENSITIVITY_SCHEMA = "methylation-latent.sesame-threshold-sensitivity.v1"


@dataclass(frozen=True, slots=True)
class PrimaryDataInputs:
    series_matrix: Path
    manifest: Path
    chen: Path
    zhou: Path
    reference_archive: Path
    fasta: Path
    fasta_index: Path
    raw_inventory: Path
    parity_audit: Path
    additional_characteristics: Path
    filelist: Path
    sesame_p001: Path
    sesame_p005: Path

    def __post_init__(self) -> None:
        paths = (
            self.series_matrix,
            self.manifest,
            self.chen,
            self.zhou,
            self.reference_archive,
            self.fasta,
            self.fasta_index,
            self.raw_inventory,
            self.parity_audit,
            self.additional_characteristics,
            self.filelist,
        )
        missing = tuple(path for path in paths if not path.is_file())
        if missing:
            raise FileNotFoundError(f"primary data inputs are absent: {missing}")
        processor_directories = (self.sesame_p001, self.sesame_p005)
        if any(not path.is_dir() for path in processor_directories):
            raise FileNotFoundError(
                f"seSAMe processor directories are absent: {processor_directories}"
            )


@dataclass(frozen=True, slots=True)
class PrimaryDataSummary:
    protocol_id: str
    static_probes: int
    retained_probes: int
    retained_samples: int
    rank_upper_bound: int
    correlation_identity_maximum_error: float
    cohort_payload_sha256: str
    target_payload_sha256: str
    probe_order_sha256: str
    sample_order_sha256: str

    def __post_init__(self) -> None:
        if not self.protocol_id:
            raise ValueError("primary data summary requires a protocol ID")
        if (
            min(
                self.static_probes,
                self.retained_probes,
                self.retained_samples,
                self.rank_upper_bound,
            )
            <= 0
        ):
            raise ValueError("primary data summary counts must be positive")
        if self.rank_upper_bound != self.retained_samples - 1:
            raise ValueError("primary data rank ceiling must equal retained samples minus one")
        if (
            not bool(t.isfinite(t.tensor(self.correlation_identity_maximum_error)).item())
            or self.correlation_identity_maximum_error < 0.0
        ):
            raise ValueError("correlation-identity error must be finite and non-negative")
        hashes = (
            self.cohort_payload_sha256,
            self.target_payload_sha256,
            self.probe_order_sha256,
            self.sample_order_sha256,
        )
        if any(len(value) != 64 for value in hashes):
            raise ValueError("primary data summary fingerprints must be SHA-256 values")


def _assert_hash(path: Path, expected: str, name: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{name} fingerprint differs: expected={expected}, observed={observed}")


def _assert_raw_inventory(path: Path, config: ProtocolConfig) -> None:
    _assert_hash(path, config.data.raw_inventory_sha256, "raw IDAT inventory")
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected_fields = {"schema", "fingerprint", "records"}
    if not isinstance(raw, dict) or set(raw) != expected_fields:
        raise ValueError("raw IDAT inventory envelope differs")
    records = raw["records"]
    if (
        raw["schema"] != "methylation-latent.compressed-idat-inventory.v1"
        or raw["fingerprint"] != config.data.raw_inventory_fingerprint
        or not isinstance(records, list)
        or len(records) != 2 * int(config.data.raw_samples)
    ):
        raise ValueError("raw IDAT inventory identity or record count differs")


def _assert_parity_audit(path: Path, config: ProtocolConfig) -> None:
    _assert_hash(path, config.qc.parity_audit_sha256, "processor parity audit")
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema",
        "decision",
        "parity_seed",
        "probe_count",
        "sample_ids",
        "failure_reasons",
    }
    if not isinstance(raw, dict) or not required <= set(raw):
        raise ValueError("processor parity audit envelope differs")
    sample_ids = raw["sample_ids"]
    failure_reasons = raw["failure_reasons"]
    if (
        raw["schema"] != "methylation-latent.processor-parity.v1"
        or raw["decision"] != "sesame_primary"
        or raw["parity_seed"] != config.qc.parity_seed
        or raw["probe_count"] != 482_421
        or not isinstance(sample_ids, list)
        or len(sample_ids) != int(config.qc.parity_arrays)
        or not isinstance(failure_reasons, list)
        or not failure_reasons
    ):
        raise ValueError("processor parity audit does not justify seSAMe as primary")


def _sensitivity_json(
    audit: SesameThresholdSensitivityAudit,
    *,
    protocol_id: str,
) -> dict[str, JsonValue]:
    values = cast(dict[str, JsonValue], asdict(audit))
    return {
        "schema": _SENSITIVITY_SCHEMA,
        "protocol_id": protocol_id,
        **values,
    }


@beartype
def build_primary_data_artifacts(
    config: ProtocolConfig,
    inputs: PrimaryDataInputs,
    output_directory: Path,
    *,
    protocol_sha256: str,
) -> PrimaryDataSummary:
    """Build target, split, and sequence-feature artifacts exactly once."""

    if config.status != "frozen":
        raise ValueError("primary data artifacts require a frozen protocol")
    if len(protocol_sha256) != 64:
        raise ValueError("protocol fingerprint must be a SHA-256 value")
    if output_directory.exists():
        raise FileExistsError(output_directory)

    _assert_hash(inputs.series_matrix, config.data.series_matrix_sha256, "series matrix")
    _assert_hash(inputs.manifest, config.data.manifest_sha256, "GPL13534 manifest")
    _assert_hash(
        inputs.additional_characteristics,
        config.data.additional_characteristics_sha256,
        "additional-characteristics workbook",
    )
    _assert_hash(inputs.filelist, config.data.filelist_sha256, "GEO file list")
    _assert_raw_inventory(inputs.raw_inventory, config)
    _assert_parity_audit(inputs.parity_audit, config)
    assert_gzip_integrity(inputs.series_matrix)
    observed_reference_md5 = md5_file(inputs.reference_archive)
    if observed_reference_md5 != config.data.reference_archive_md5:
        raise ValueError("hg19 reference archive MD5 differs from the protocol")
    reference_payload_sha256 = assert_gzip_payload_matches_file(
        inputs.reference_archive,
        inputs.fasta,
    )
    if reference_payload_sha256 != config.reference_audit.payload_sha256:
        raise ValueError("decompressed hg19 payload fingerprint differs from the protocol")

    raw_samples = parse_gse87571_series_metadata(
        inputs.series_matrix,
        expected_sample_count=config.data.raw_samples,
        expected_age_eligible_count=config.data.age_eligible_samples,
    )
    expected_sample_ids = tuple(str(sample.sentrix_identity) for sample in raw_samples.samples)
    if sha256_ordered_strings(expected_sample_ids) != config.data.raw_sentrix_order_sha256:
        raise ValueError("raw GSE87571 Sentrix order differs from the frozen protocol")
    if raw_samples.age_eligible.order_sha256 != config.data.age_eligible_order_sha256:
        raise ValueError("age-eligible GSE87571 order differs from the frozen protocol")

    sensitivity = compare_sesame_threshold_outputs(
        inputs.sesame_p001,
        inputs.sesame_p005,
        expected_sample_ids=expected_sample_ids,
    )
    manifest = parse_gpl13534_manifest(inputs.manifest)
    chen = load_chen2013_cross_reactive(inputs.chen)
    zhou = load_zhou2017_mask_general(inputs.zhou)
    with IndexedFasta(inputs.fasta, inputs.fasta_index) as reference:
        static_universe = build_static_probe_universe(
            manifest,
            (chen, zhou),
            reference,
            config.reference_audit.maximum_window_size,
        )
        autosome_lengths = dict(chromosome_lengths(reference))
    reference_audit = static_universe.reference.audit
    observed_reference_audit = (
        reference_audit.coordinate_cpgs_verified,
        reference_audit.eligible_probes,
        reference_audit.boundary_exclusions,
        reference_audit.non_acgt_exclusions,
        reference_audit.eligible_probe_order_sha256,
    )
    expected_reference_audit = (
        int(config.reference_audit.coordinate_cpgs_verified),
        int(config.reference_audit.eligible_probes),
        config.reference_audit.boundary_exclusions,
        config.reference_audit.non_acgt_exclusions,
        config.reference_audit.eligible_probe_order_sha256,
    )
    if observed_reference_audit != expected_reference_audit:
        raise ValueError("static probe universe differs from the exhaustive reference audit")

    output_directory.mkdir(parents=True)
    write_canonical_json_exclusive(
        output_directory / "sesame-threshold-sensitivity.json",
        _sensitivity_json(sensitivity, protocol_id=config.protocol_id),
    )
    write_probe_table_exclusive(
        output_directory / "static-probes.tsv",
        static_universe.probes,
    )
    write_static_exclusion_ledger_exclusive(
        output_directory / "static-exclusions.tsv",
        static_universe,
        maximum_window_size=config.reference_audit.maximum_window_size,
    )
    write_canonical_json_exclusive(
        output_directory / "static-audit.json",
        {
            "schema": _STATIC_AUDIT_SCHEMA,
            "protocol_id": config.protocol_id,
            "protocol_sha256": protocol_sha256,
            "manifest_sha256": config.data.manifest_sha256,
            "chen_sha256": config.masks.chen_sha256,
            "zhou_sha256": config.masks.zhou_sha256,
            "reference_archive_md5": observed_reference_md5,
            "reference_payload_sha256": reference_payload_sha256,
            "coordinate_convention": "GPL13534_MAPINFO_one_based_plus_strand_cytosine",
            "maximum_window_size": int(config.reference_audit.maximum_window_size),
            "manifest_rows": manifest.total_assay_rows,
            "manifest_retained_autosomal_cpgs": len(manifest.probes),
            "published_mask_retained_probes": len(static_universe.published.retained),
            "static_retained_probes": len(static_universe.probes),
            "static_probe_order_sha256": sha256_ordered_strings(
                str(probe.probe_id) for probe in static_universe.probes.probes
            ),
            "boundary_exclusions": reference_audit.boundary_exclusions,
            "non_acgt_exclusions": reference_audit.non_acgt_exclusions,
        },
    )

    processor_output = load_sesame_output(
        inputs.sesame_p005,
        expected_sample_ids=expected_sample_ids,
    )
    cohort = prepare_gse87571_cohort(
        processor_output,
        raw_samples,
        static_universe.probes,
        detection_threshold=config.qc.detection_p_threshold,
        maximum_sample_failure_fraction=config.qc.maximum_sample_failure_fraction,
    )
    cohort_tensor_path = output_directory / "cohort.safetensors"
    save_prepared_cohort_exclusive(
        cohort_tensor_path,
        output_directory / "cohort.json",
        cohort,
    )
    write_probe_table_exclusive(output_directory / "probes.tsv", cohort.probes)
    write_cohort_exclusion_ledger_exclusive(
        output_directory / "cohort-exclusions.tsv",
        cohort,
    )

    targets = cohort.build_targets()
    audit_count = int(config.targets.correlation_audit_probes)
    if len(cohort.probes) < audit_count:
        raise ValueError("retained cohort is smaller than the frozen correlation audit")
    generator = t.Generator(device="cpu").manual_seed(config.targets.correlation_audit_seed)
    audit_indices = t.randperm(len(cohort.probes), generator=generator)[:audit_count]
    correlation_error = assert_correlation_identity(
        cohort.beta,
        targets.methylation,
        audit_indices,
    )
    useful_latent_ceiling = min(
        validate_latent_dimension(
            dimension,
            embedding_dimension=config.model.embedding_width,
            n_samples=cohort.audit.retained_samples,
        )
        for dimension in config.sweep.latent_dimensions
    )
    if useful_latent_ceiling != min(
        int(config.model.embedding_width),
        targets.rank_upper_bound,
    ):
        raise RuntimeError("latent-dimension audit returned inconsistent useful ceilings")
    target_path = output_directory / "targets.safetensors"
    save_target_geometry(target_path, targets)
    write_canonical_json_exclusive(
        output_directory / "targets.json",
        {
            "schema": _TARGET_AUDIT_SCHEMA,
            "protocol_id": config.protocol_id,
            "protocol_sha256": protocol_sha256,
            "tensor_file": target_path.name,
            "tensor_sha256": sha256_file(target_path),
            "probe_order_sha256": cohort.probe_order_sha256,
            "sample_order_sha256": cohort.sample_order_sha256,
            "probe_count": len(cohort.probes),
            "sample_count": cohort.audit.retained_samples,
            "rank_upper_bound": targets.rank_upper_bound,
            "embedding_dimension_ceiling": int(config.model.embedding_width),
            "useful_latent_dimension_ceiling": useful_latent_ceiling,
            "correlation_audit_seed": config.targets.correlation_audit_seed,
            "correlation_audit_probe_ids": [
                str(cohort.probes.probes[index].probe_id) for index in audit_indices.tolist()
            ],
            "correlation_identity_maximum_error": correlation_error,
            "standardization": "subtract_mean_then_divide_by_l2_norm_float64",
        },
    )

    built_splits = build_primary_splits(
        cohort.probes,
        chromosome_lengths=autosome_lengths,
        block_width=config.splits.block_width,
        anchors_per_context=config.splits.anchors_per_context,
        held_out_chromosome=config.splits.held_out_chromosome,
        window_sizes=config.splits.window_sizes,
        primary_seed=config.splits.primary_seed,
        validation_seed=config.splits.validation_seed,
    )
    split_directory = output_directory / "splits"
    split_directory.mkdir()
    for name, split, nested in (
        (
            "diverse-blocks",
            built_splits.diverse,
            built_splits.diverse_nested,
        ),
        (
            "held-out-chromosome",
            built_splits.held_out_chromosome,
            built_splits.held_out_chromosome_nested,
        ),
    ):
        save_split_artifact_exclusive(
            split_directory / f"{name}.safetensors",
            split_directory / f"{name}.json",
            split,
            nested,
            cohort.probes,
        )

    with IndexedFasta(inputs.fasta, inputs.fasta_index) as reference:
        features = compute_sequence_feature_artifact(
            reference,
            cohort.probes,
            config.splits.window_sizes,
        )
    save_sequence_features_exclusive(
        output_directory / "sequence-features.safetensors",
        output_directory / "sequence-features.json",
        features,
        cohort.probes,
    )

    summary = PrimaryDataSummary(
        protocol_id=config.protocol_id,
        static_probes=len(static_universe.probes),
        retained_probes=len(cohort.probes),
        retained_samples=cohort.audit.retained_samples,
        rank_upper_bound=targets.rank_upper_bound,
        correlation_identity_maximum_error=correlation_error,
        cohort_payload_sha256=sha256_file(cohort_tensor_path),
        target_payload_sha256=sha256_file(target_path),
        probe_order_sha256=cohort.probe_order_sha256,
        sample_order_sha256=cohort.sample_order_sha256,
    )
    write_canonical_json_exclusive(
        output_directory / "summary.json",
        {
            "schema": _PRIMARY_SUMMARY_SCHEMA,
            **cast(dict[str, JsonValue], asdict(summary)),
        },
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--series-matrix", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--chen", type=Path, required=True)
    parser.add_argument("--zhou", type=Path, required=True)
    parser.add_argument("--reference-archive", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--fasta-index", type=Path, required=True)
    parser.add_argument("--raw-inventory", type=Path, required=True)
    parser.add_argument("--parity-audit", type=Path, required=True)
    parser.add_argument("--additional-characteristics", type=Path, required=True)
    parser.add_argument("--filelist", type=Path, required=True)
    parser.add_argument("--sesame-p001", type=Path, required=True)
    parser.add_argument("--sesame-p005", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    config = load_protocol_config(arguments.config)
    summary = build_primary_data_artifacts(
        config,
        PrimaryDataInputs(
            series_matrix=arguments.series_matrix,
            manifest=arguments.manifest,
            chen=arguments.chen,
            zhou=arguments.zhou,
            reference_archive=arguments.reference_archive,
            fasta=arguments.fasta,
            fasta_index=arguments.fasta_index,
            raw_inventory=arguments.raw_inventory,
            parity_audit=arguments.parity_audit,
            additional_characteristics=arguments.additional_characteristics,
            filelist=arguments.filelist,
            sesame_p001=arguments.sesame_p001,
            sesame_p005=arguments.sesame_p005,
        ),
        arguments.output,
        protocol_sha256=sha256_file(arguments.config),
    )
    print(
        json.dumps(
            {
                "correlation_identity_maximum_error": (summary.correlation_identity_maximum_error),
                "output": str(arguments.output.resolve()),
                "protocol_id": summary.protocol_id,
                "rank_upper_bound": summary.rank_upper_bound,
                "retained_probes": summary.retained_probes,
                "retained_samples": summary.retained_samples,
            },
            allow_nan=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
