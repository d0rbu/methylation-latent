"""Strict frozen GSE87571 experiment configuration with no ignored fields."""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from methylation_latent.domain import (
    Autosome,
    Fraction,
    LatentDimension,
    NonNegativeWeight,
    PositiveInt,
    WindowSize,
    parse_autosome,
    parse_fraction,
    parse_latent_dimension,
    parse_non_negative_weight,
    parse_positive_int,
    parse_window_size,
)
from methylation_latent.embeddings import (
    CADUCEUS_CHECKPOINT_SHA256,
    CADUCEUS_EMBEDDING_WIDTH,
    CADUCEUS_REPOSITORY,
    CADUCEUS_REVISION,
)
from methylation_latent.genome import (
    PRIMARY_REFERENCE_BOUNDARY_EXCLUSIONS,
    PRIMARY_REFERENCE_ELIGIBLE_ORDER_SHA256,
    PRIMARY_REFERENCE_ELIGIBLE_PROBES,
    PRIMARY_REFERENCE_INPUT_PROBES,
    PRIMARY_REFERENCE_NON_ACGT_EXCLUSIONS,
    UCSC_HG19_FASTA_ARCHIVE_MD5,
    UCSC_HG19_FASTA_PAYLOAD_SHA256,
)
from methylation_latent.manifest import (
    CHEN2013_CROSS_REACTIVE_SHA256,
    CHEN2013_CROSS_REACTIVE_URL,
    GPL13534_V11_SHA256,
    ZHOU2017_MASK_GENERAL_SHA256,
    ZHOU2017_MASK_GENERAL_URL,
)
from methylation_latent.metadata import (
    GSE87571_ADDITIONAL_CHARACTERISTICS_SHA256,
    GSE87571_AGE_ELIGIBLE_SAMPLE_COUNT,
    GSE87571_FILELIST_SHA256,
    GSE87571_RAW_SAMPLE_COUNT,
    GSE87571_SERIES_MATRIX_SHA256,
)

_PRIMARY_WINDOWS = (1_024, 4_096, 16_384, 65_536)
_PRIMARY_INFERENCE_BATCH_SIZES = (512, 128, 32, 8)
_GSE87571_RAW_INVENTORY_SHA256 = "b0037d92fc9f3a5cc50bb940bc0e87d1682f93e747edba16214d9b37b0aa4be6"
_GSE87571_RAW_INVENTORY_FINGERPRINT = (
    "20fccbecf4b7e7b36f3099daf6084d9042fc3d9074bfa4bf0ab154eeca694ae0"
)
_GSE87571_RAW_SENTRIX_ORDER_SHA256 = (
    "c3700636dfdfa111c27bfc88def03cf3d95028d221c0fa63f1f0e4a90a198f74"
)
_GSE87571_AGE_ELIGIBLE_ORDER_SHA256 = (
    "839ea514d3993eadf749246941b1f31dfa5edd513ad12a8cd22695536c498553"
)
_SESAME_CONTAINER_DIGEST = "sha256:b10002b39efa30c3779ad839549806ebdbb29b3266f0d2428478b04426e55929"
_PARITY_AUDIT_SHA256 = "ff8a0606d041e22fd654005b9a0c1f83d780d1b7e1b6d7aa4b112679a149b095"


def _exact_keys(raw: Mapping[str, object], expected: set[str], context: str) -> None:
    observed = set(raw)
    if observed != expected:
        raise ValueError(
            f"{context} fields differ: missing={sorted(expected - observed)}, "
            f"unknown={sorted(observed - expected)}"
        )


def _table(raw: Mapping[str, object], name: str) -> dict[str, object]:
    value = raw[name]
    if not isinstance(value, dict):
        raise TypeError(f"configuration field {name!r} must be a table")
    return cast(dict[str, object], value)


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"configuration field {name!r} must be an integer")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"configuration field {name!r} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"configuration field {name!r} must be finite")
    return parsed


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"configuration field {name!r} must be a non-empty string")
    return value


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list) or not value:
        raise TypeError(f"configuration field {name!r} must be a non-empty array")
    return cast(list[object], value)


@dataclass(frozen=True, slots=True)
class DataConfig:
    accession: str
    raw_samples: PositiveInt
    age_eligible_samples: PositiveInt
    genome_build: str
    manifest_sha256: str
    series_matrix_sha256: str
    additional_characteristics_sha256: str
    filelist_sha256: str
    raw_inventory_sha256: str
    raw_inventory_fingerprint: str
    raw_sentrix_order_sha256: str
    age_eligible_order_sha256: str
    reference_archive_md5: str

    def __post_init__(self) -> None:
        observed = (
            self.accession,
            int(self.raw_samples),
            int(self.age_eligible_samples),
            self.genome_build,
            self.manifest_sha256,
            self.series_matrix_sha256,
            self.additional_characteristics_sha256,
            self.filelist_sha256,
            self.raw_inventory_sha256,
            self.raw_inventory_fingerprint,
            self.raw_sentrix_order_sha256,
            self.age_eligible_order_sha256,
            self.reference_archive_md5,
        )
        expected = (
            "GSE87571",
            GSE87571_RAW_SAMPLE_COUNT,
            GSE87571_AGE_ELIGIBLE_SAMPLE_COUNT,
            "hg19/GRCh37",
            GPL13534_V11_SHA256,
            GSE87571_SERIES_MATRIX_SHA256,
            GSE87571_ADDITIONAL_CHARACTERISTICS_SHA256,
            GSE87571_FILELIST_SHA256,
            _GSE87571_RAW_INVENTORY_SHA256,
            _GSE87571_RAW_INVENTORY_FINGERPRINT,
            _GSE87571_RAW_SENTRIX_ORDER_SHA256,
            _GSE87571_AGE_ELIGIBLE_ORDER_SHA256,
            UCSC_HG19_FASTA_ARCHIVE_MD5,
        )
        if observed != expected:
            raise ValueError("GSE87571 data identity differs from the frozen primary protocol")


@dataclass(frozen=True, slots=True)
class ReferenceAuditConfig:
    payload_sha256: str
    coordinate_cpgs_verified: PositiveInt
    maximum_window_size: WindowSize
    eligible_probes: PositiveInt
    boundary_exclusions: int
    non_acgt_exclusions: int
    eligible_probe_order_sha256: str

    def __post_init__(self) -> None:
        observed = (
            self.payload_sha256,
            int(self.coordinate_cpgs_verified),
            int(self.maximum_window_size),
            int(self.eligible_probes),
            self.boundary_exclusions,
            self.non_acgt_exclusions,
            self.eligible_probe_order_sha256,
        )
        expected = (
            UCSC_HG19_FASTA_PAYLOAD_SHA256,
            PRIMARY_REFERENCE_INPUT_PROBES,
            max(_PRIMARY_WINDOWS),
            PRIMARY_REFERENCE_ELIGIBLE_PROBES,
            PRIMARY_REFERENCE_BOUNDARY_EXCLUSIONS,
            PRIMARY_REFERENCE_NON_ACGT_EXCLUSIONS,
            PRIMARY_REFERENCE_ELIGIBLE_ORDER_SHA256,
        )
        if observed != expected:
            raise ValueError("hg19 exhaustive reference audit differs from protocol v2")


@dataclass(frozen=True, slots=True)
class QcConfig:
    detection_p_threshold: Fraction
    maximum_sample_failure_fraction: Fraction
    processor: str
    container_digest: str
    r_version: str
    bioconductor_release: str
    sesame_version: str
    sesame_data_version: str
    parity_arrays: PositiveInt
    parity_seed: int
    parity_processor: str
    parity_audit_sha256: str

    def __post_init__(self) -> None:
        observed = (
            float(self.detection_p_threshold),
            float(self.maximum_sample_failure_fraction),
            self.processor,
            self.container_digest,
            self.r_version,
            self.bioconductor_release,
            self.sesame_version,
            self.sesame_data_version,
            int(self.parity_arrays),
            self.parity_seed,
            self.parity_processor,
            self.parity_audit_sha256,
        )
        expected = (
            0.05,
            0.05,
            "sesame-1.30.1-QCD-pOOBAH@0.05-noob-B",
            _SESAME_CONTAINER_DIGEST,
            "4.6.0",
            "3.23",
            "1.30.1",
            "1.30.0",
            12,
            550319,
            "methylprep-1.7.1-poobah-noob-nonlinear-dye",
            _PARITY_AUDIT_SHA256,
        )
        if observed != expected:
            raise ValueError("primary seSAMe QC identity differs from protocol v2")


@dataclass(frozen=True, slots=True)
class MaskConfig:
    chen_url: str
    chen_sha256: str
    zhou_url: str
    zhou_sha256: str

    def __post_init__(self) -> None:
        observed = (self.chen_url, self.chen_sha256, self.zhou_url, self.zhou_sha256)
        expected = (
            CHEN2013_CROSS_REACTIVE_URL,
            CHEN2013_CROSS_REACTIVE_SHA256,
            ZHOU2017_MASK_GENERAL_URL,
            ZHOU2017_MASK_GENERAL_SHA256,
        )
        if observed != expected:
            raise ValueError("published probe-mask identity differs from protocol v2")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    repository: str
    revision: str
    checkpoint_sha256: str
    embedding_width: PositiveInt
    precision: str
    inference_batch_sizes: tuple[PositiveInt, ...]
    embedding_shard_size: PositiveInt

    def __post_init__(self) -> None:
        observed = (
            self.repository,
            self.revision,
            self.checkpoint_sha256,
            int(self.embedding_width),
            self.precision,
            tuple(map(int, self.inference_batch_sizes)),
            int(self.embedding_shard_size),
        )
        expected = (
            CADUCEUS_REPOSITORY,
            CADUCEUS_REVISION,
            CADUCEUS_CHECKPOINT_SHA256,
            CADUCEUS_EMBEDDING_WIDTH,
            "float16",
            _PRIMARY_INFERENCE_BATCH_SIZES,
            1_024,
        )
        if observed != expected:
            raise ValueError("Caduceus inference identity differs from protocol v2")


@dataclass(frozen=True, slots=True)
class SplitConfig:
    window_sizes: tuple[WindowSize, ...]
    block_width: PositiveInt
    anchors_per_context: PositiveInt
    held_out_chromosome: Autosome
    primary_seed: int
    validation_seed: int

    def __post_init__(self) -> None:
        if tuple(map(int, self.window_sizes)) != _PRIMARY_WINDOWS:
            raise ValueError(f"primary window sweep must be {_PRIMARY_WINDOWS}")
        if self.primary_seed == self.validation_seed:
            raise ValueError("primary and validation split seeds must differ")


@dataclass(frozen=True, slots=True)
class TargetConfig:
    correlation_audit_probes: PositiveInt
    correlation_audit_seed: int

    def __post_init__(self) -> None:
        if int(self.correlation_audit_probes) != 256 or self.correlation_audit_seed != 161_803:
            raise ValueError("target-correlation audit schedule differs from protocol v2")


@dataclass(frozen=True, slots=True)
class TrainingProtocolConfig:
    batch_size: PositiveInt
    neighbourhood_width: PositiveInt
    optimizer: str
    learning_rate: float
    weight_decay: float
    tuning_steps: PositiveInt
    validation_interval: PositiveInt
    validation_pair_chunk_size: PositiveInt
    training_seed: int
    full_checkpoint_rule: str
    age_checkpoint_rule: str
    final_refit_rule: str

    def __post_init__(self) -> None:
        if (
            int(self.batch_size) != 512
            or int(self.neighbourhood_width) != 1_048_576
            or self.optimizer != "adam"
            or self.learning_rate != 0.001
            or self.weight_decay != 0.0
            or int(self.tuning_steps) != 2_000
            or int(self.validation_interval) != 100
            or int(self.validation_pair_chunk_size) != 512
            or self.training_seed != 851_733
        ):
            raise ValueError("optimization schedule differs from protocol v2")
        rules = (
            self.full_checkpoint_rule,
            self.age_checkpoint_rule,
            self.final_refit_rule,
        )
        expected_rules = (
            "minimum_unweighted_pair_plus_age_validation_mse",
            "minimum_age_validation_mse",
            "selected_validation_step_on_complete_primary_train",
        )
        if rules != expected_rules:
            raise ValueError("checkpoint or final-refit rule differs from protocol v2")


@dataclass(frozen=True, slots=True)
class SweepConfig:
    latent_dimensions: tuple[LatentDimension, ...]
    lambda_age: tuple[NonNegativeWeight, ...]
    maximum_uniform_evaluation_pairs: PositiveInt
    maximum_pairs_per_distance_class: PositiveInt
    uniform_evaluation_seed: int
    distance_evaluation_seed: int
    projection_window_size: WindowSize

    def __post_init__(self) -> None:
        dimensions = tuple(map(int, self.latent_dimensions))
        lambdas = tuple(map(float, self.lambda_age))
        if dimensions != (16, 32, 64, 128, 256):
            raise ValueError("latent-dimension sweep differs from protocol v2")
        if lambdas != (0.1, 1.0, 10.0):
            raise ValueError("age-loss sweep differs from protocol v2")
        if (
            int(self.maximum_uniform_evaluation_pairs) != 1_000_000
            or int(self.maximum_pairs_per_distance_class) != 100_000
            or self.uniform_evaluation_seed != 314_159
            or self.distance_evaluation_seed != 271_828
            or int(self.projection_window_size) != 16_384
        ):
            raise ValueError("evaluation schedule differs from protocol v2")


@dataclass(frozen=True, slots=True)
class ProtocolConfig:
    schema: str
    protocol_id: str
    status: str
    data: DataConfig
    reference_audit: ReferenceAuditConfig
    qc: QcConfig
    masks: MaskConfig
    model: ModelConfig
    splits: SplitConfig
    targets: TargetConfig
    training: TrainingProtocolConfig
    sweep: SweepConfig

    def __post_init__(self) -> None:
        if self.schema != "methylation-latent.protocol.v2":
            raise ValueError("unknown protocol configuration schema")
        if self.status != "frozen":
            raise ValueError("protocol v2 status must be frozen")
        if self.protocol_id != "gse87571-hg19-caduceus-ps-v2":
            raise ValueError("protocol ID differs from the reviewed v2 identity")


def load_protocol_config(path: Path) -> ProtocolConfig:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    _exact_keys(
        raw,
        {
            "schema",
            "protocol_id",
            "status",
            "data",
            "reference_audit",
            "qc",
            "masks",
            "model",
            "splits",
            "targets",
            "training",
            "sweep",
        },
        "protocol",
    )
    data = _table(raw, "data")
    reference_audit = _table(raw, "reference_audit")
    qc = _table(raw, "qc")
    masks = _table(raw, "masks")
    model = _table(raw, "model")
    splits = _table(raw, "splits")
    targets = _table(raw, "targets")
    training = _table(raw, "training")
    sweep = _table(raw, "sweep")
    _exact_keys(
        data,
        {
            "accession",
            "raw_samples",
            "age_eligible_samples",
            "genome_build",
            "manifest_sha256",
            "series_matrix_sha256",
            "additional_characteristics_sha256",
            "filelist_sha256",
            "raw_inventory_sha256",
            "raw_inventory_fingerprint",
            "raw_sentrix_order_sha256",
            "age_eligible_order_sha256",
            "reference_archive_md5",
        },
        "data",
    )
    _exact_keys(
        reference_audit,
        {
            "payload_sha256",
            "coordinate_cpgs_verified",
            "maximum_window_size",
            "eligible_probes",
            "boundary_exclusions",
            "non_acgt_exclusions",
            "eligible_probe_order_sha256",
        },
        "reference_audit",
    )
    _exact_keys(
        qc,
        {
            "detection_p_threshold",
            "maximum_sample_failure_fraction",
            "processor",
            "container_digest",
            "r_version",
            "bioconductor_release",
            "sesame_version",
            "sesame_data_version",
            "parity_arrays",
            "parity_seed",
            "parity_processor",
            "parity_audit_sha256",
        },
        "qc",
    )
    _exact_keys(masks, {"chen_url", "chen_sha256", "zhou_url", "zhou_sha256"}, "masks")
    _exact_keys(
        model,
        {
            "repository",
            "revision",
            "checkpoint_sha256",
            "embedding_width",
            "precision",
            "inference_batch_sizes",
            "embedding_shard_size",
        },
        "model",
    )
    _exact_keys(
        splits,
        {
            "window_sizes",
            "block_width",
            "anchors_per_context",
            "held_out_chromosome",
            "primary_seed",
            "validation_seed",
        },
        "splits",
    )
    _exact_keys(
        targets,
        {
            "correlation_audit_probes",
            "correlation_audit_seed",
        },
        "targets",
    )
    _exact_keys(
        training,
        {
            "batch_size",
            "neighbourhood_width",
            "optimizer",
            "learning_rate",
            "weight_decay",
            "tuning_steps",
            "validation_interval",
            "validation_pair_chunk_size",
            "training_seed",
            "full_checkpoint_rule",
            "age_checkpoint_rule",
            "final_refit_rule",
        },
        "training",
    )
    _exact_keys(
        sweep,
        {
            "latent_dimensions",
            "lambda_age",
            "maximum_uniform_evaluation_pairs",
            "maximum_pairs_per_distance_class",
            "uniform_evaluation_seed",
            "distance_evaluation_seed",
            "projection_window_size",
        },
        "sweep",
    )
    return ProtocolConfig(
        schema=_string(raw["schema"], "schema"),
        protocol_id=_string(raw["protocol_id"], "protocol_id"),
        status=_string(raw["status"], "status"),
        data=DataConfig(
            accession=_string(data["accession"], "data.accession"),
            raw_samples=parse_positive_int(_integer(data["raw_samples"], "data.raw_samples")),
            age_eligible_samples=parse_positive_int(
                _integer(data["age_eligible_samples"], "data.age_eligible_samples")
            ),
            genome_build=_string(data["genome_build"], "data.genome_build"),
            manifest_sha256=_string(data["manifest_sha256"], "data.manifest_sha256"),
            series_matrix_sha256=_string(data["series_matrix_sha256"], "data.series_matrix_sha256"),
            additional_characteristics_sha256=_string(
                data["additional_characteristics_sha256"],
                "data.additional_characteristics_sha256",
            ),
            filelist_sha256=_string(data["filelist_sha256"], "data.filelist_sha256"),
            raw_inventory_sha256=_string(data["raw_inventory_sha256"], "data.raw_inventory_sha256"),
            raw_inventory_fingerprint=_string(
                data["raw_inventory_fingerprint"], "data.raw_inventory_fingerprint"
            ),
            raw_sentrix_order_sha256=_string(
                data["raw_sentrix_order_sha256"], "data.raw_sentrix_order_sha256"
            ),
            age_eligible_order_sha256=_string(
                data["age_eligible_order_sha256"], "data.age_eligible_order_sha256"
            ),
            reference_archive_md5=_string(
                data["reference_archive_md5"], "data.reference_archive_md5"
            ),
        ),
        reference_audit=ReferenceAuditConfig(
            payload_sha256=_string(
                reference_audit["payload_sha256"], "reference_audit.payload_sha256"
            ),
            coordinate_cpgs_verified=parse_positive_int(
                _integer(
                    reference_audit["coordinate_cpgs_verified"],
                    "reference_audit.coordinate_cpgs_verified",
                )
            ),
            maximum_window_size=parse_window_size(
                _integer(
                    reference_audit["maximum_window_size"],
                    "reference_audit.maximum_window_size",
                )
            ),
            eligible_probes=parse_positive_int(
                _integer(reference_audit["eligible_probes"], "reference_audit.eligible_probes")
            ),
            boundary_exclusions=_integer(
                reference_audit["boundary_exclusions"],
                "reference_audit.boundary_exclusions",
            ),
            non_acgt_exclusions=_integer(
                reference_audit["non_acgt_exclusions"],
                "reference_audit.non_acgt_exclusions",
            ),
            eligible_probe_order_sha256=_string(
                reference_audit["eligible_probe_order_sha256"],
                "reference_audit.eligible_probe_order_sha256",
            ),
        ),
        qc=QcConfig(
            detection_p_threshold=parse_fraction(
                _number(qc["detection_p_threshold"], "qc.detection_p_threshold")
            ),
            maximum_sample_failure_fraction=parse_fraction(
                _number(qc["maximum_sample_failure_fraction"], "qc.maximum_sample_failure_fraction")
            ),
            processor=_string(qc["processor"], "qc.processor"),
            container_digest=_string(qc["container_digest"], "qc.container_digest"),
            r_version=_string(qc["r_version"], "qc.r_version"),
            bioconductor_release=_string(qc["bioconductor_release"], "qc.bioconductor_release"),
            sesame_version=_string(qc["sesame_version"], "qc.sesame_version"),
            sesame_data_version=_string(qc["sesame_data_version"], "qc.sesame_data_version"),
            parity_arrays=parse_positive_int(_integer(qc["parity_arrays"], "qc.parity_arrays")),
            parity_seed=_integer(qc["parity_seed"], "qc.parity_seed"),
            parity_processor=_string(qc["parity_processor"], "qc.parity_processor"),
            parity_audit_sha256=_string(qc["parity_audit_sha256"], "qc.parity_audit_sha256"),
        ),
        masks=MaskConfig(
            chen_url=_string(masks["chen_url"], "masks.chen_url"),
            chen_sha256=_string(masks["chen_sha256"], "masks.chen_sha256"),
            zhou_url=_string(masks["zhou_url"], "masks.zhou_url"),
            zhou_sha256=_string(masks["zhou_sha256"], "masks.zhou_sha256"),
        ),
        model=ModelConfig(
            repository=_string(model["repository"], "model.repository"),
            revision=_string(model["revision"], "model.revision"),
            checkpoint_sha256=_string(model["checkpoint_sha256"], "model.checkpoint_sha256"),
            embedding_width=parse_positive_int(
                _integer(model["embedding_width"], "model.embedding_width")
            ),
            precision=_string(model["precision"], "model.precision"),
            inference_batch_sizes=tuple(
                parse_positive_int(_integer(value, "model.inference_batch_sizes[]"))
                for value in _list(
                    model["inference_batch_sizes"],
                    "model.inference_batch_sizes",
                )
            ),
            embedding_shard_size=parse_positive_int(
                _integer(model["embedding_shard_size"], "model.embedding_shard_size")
            ),
        ),
        splits=SplitConfig(
            window_sizes=tuple(
                parse_window_size(_integer(value, "splits.window_sizes[]"))
                for value in _list(splits["window_sizes"], "splits.window_sizes")
            ),
            block_width=parse_positive_int(_integer(splits["block_width"], "splits.block_width")),
            anchors_per_context=parse_positive_int(
                _integer(splits["anchors_per_context"], "splits.anchors_per_context")
            ),
            held_out_chromosome=parse_autosome(
                _integer(splits["held_out_chromosome"], "splits.held_out_chromosome")
            ),
            primary_seed=_integer(splits["primary_seed"], "splits.primary_seed"),
            validation_seed=_integer(splits["validation_seed"], "splits.validation_seed"),
        ),
        targets=TargetConfig(
            correlation_audit_probes=parse_positive_int(
                _integer(
                    targets["correlation_audit_probes"],
                    "targets.correlation_audit_probes",
                )
            ),
            correlation_audit_seed=_integer(
                targets["correlation_audit_seed"],
                "targets.correlation_audit_seed",
            ),
        ),
        training=TrainingProtocolConfig(
            batch_size=parse_positive_int(_integer(training["batch_size"], "training.batch_size")),
            neighbourhood_width=parse_positive_int(
                _integer(training["neighbourhood_width"], "training.neighbourhood_width")
            ),
            optimizer=_string(training["optimizer"], "training.optimizer"),
            learning_rate=_number(training["learning_rate"], "training.learning_rate"),
            weight_decay=_number(training["weight_decay"], "training.weight_decay"),
            tuning_steps=parse_positive_int(
                _integer(training["tuning_steps"], "training.tuning_steps")
            ),
            validation_interval=parse_positive_int(
                _integer(
                    training["validation_interval"],
                    "training.validation_interval",
                )
            ),
            validation_pair_chunk_size=parse_positive_int(
                _integer(
                    training["validation_pair_chunk_size"],
                    "training.validation_pair_chunk_size",
                )
            ),
            training_seed=_integer(training["training_seed"], "training.training_seed"),
            full_checkpoint_rule=_string(
                training["full_checkpoint_rule"], "training.full_checkpoint_rule"
            ),
            age_checkpoint_rule=_string(
                training["age_checkpoint_rule"], "training.age_checkpoint_rule"
            ),
            final_refit_rule=_string(training["final_refit_rule"], "training.final_refit_rule"),
        ),
        sweep=SweepConfig(
            latent_dimensions=tuple(
                parse_latent_dimension(_integer(value, "sweep.latent_dimensions[]"))
                for value in _list(sweep["latent_dimensions"], "sweep.latent_dimensions")
            ),
            lambda_age=tuple(
                parse_non_negative_weight(_number(value, "sweep.lambda_age[]"))
                for value in _list(sweep["lambda_age"], "sweep.lambda_age")
            ),
            maximum_uniform_evaluation_pairs=parse_positive_int(
                _integer(
                    sweep["maximum_uniform_evaluation_pairs"],
                    "sweep.maximum_uniform_evaluation_pairs",
                )
            ),
            maximum_pairs_per_distance_class=parse_positive_int(
                _integer(
                    sweep["maximum_pairs_per_distance_class"],
                    "sweep.maximum_pairs_per_distance_class",
                )
            ),
            uniform_evaluation_seed=_integer(
                sweep["uniform_evaluation_seed"],
                "sweep.uniform_evaluation_seed",
            ),
            distance_evaluation_seed=_integer(
                sweep["distance_evaluation_seed"],
                "sweep.distance_evaluation_seed",
            ),
            projection_window_size=parse_window_size(
                _integer(
                    sweep["projection_window_size"],
                    "sweep.projection_window_size",
                )
            ),
        ),
    )
