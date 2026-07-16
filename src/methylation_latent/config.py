"""Strict, versioned protocol configuration with no ignored fields."""

from __future__ import annotations

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
    GSE40279_SAMPLE_KEY_SHA256,
    GSE40279_SAMPLE_ORDER_SHA256,
    GSE40279_SERIES_MATRIX_SHA256,
)

_PRIMARY_WINDOWS = (1_024, 4_096, 16_384, 65_536)


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
    return float(value)


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
    expected_samples: PositiveInt
    genome_build: str
    manifest_sha256: str
    series_matrix_sha256: str
    sample_key_sha256: str
    sample_order_sha256: str
    reference_archive_md5: str

    def __post_init__(self) -> None:
        if self.accession != "GSE40279" or self.genome_build != "hg19/GRCh37":
            raise ValueError("data accession/build differ from the primary protocol")
        if int(self.expected_samples) != 656:
            raise ValueError("GSE40279 protocol expects exactly 656 pre-QC samples")
        if self.manifest_sha256 != GPL13534_V11_SHA256:
            raise ValueError("manifest fingerprint differs from reviewed GPL13534 v1.1")
        observed_geo = (
            self.series_matrix_sha256,
            self.sample_key_sha256,
            self.sample_order_sha256,
        )
        expected_geo = (
            GSE40279_SERIES_MATRIX_SHA256,
            GSE40279_SAMPLE_KEY_SHA256,
            GSE40279_SAMPLE_ORDER_SHA256,
        )
        if observed_geo != expected_geo:
            raise ValueError("GSE40279 series, sample-key, or sample-order fingerprint differs")
        if self.reference_archive_md5 != UCSC_HG19_FASTA_ARCHIVE_MD5:
            raise ValueError("reference archive fingerprint differs from UCSC hg19")


@dataclass(frozen=True, slots=True)
class QcConfig:
    detection_p_threshold: Fraction
    maximum_sample_failure_fraction: Fraction
    processor: str
    parity_arrays: PositiveInt
    parity_seed: int
    reference_processor: str
    reference_r_version: str
    reference_bioconductor_release: str
    reference_sesame_version: str
    reference_sesame_data_version: str

    def __post_init__(self) -> None:
        if self.processor != "methylprep-1.7.1-poobah-noob-nonlinear-dye":
            raise ValueError("primary preprocessing processor differs from the reviewed protocol")
        if int(self.parity_arrays) < 12:
            raise ValueError("seSAMe parity gate requires at least 12 arrays")
        reference_identity = (
            self.reference_processor,
            self.reference_r_version,
            self.reference_bioconductor_release,
            self.reference_sesame_version,
            self.reference_sesame_data_version,
        )
        if reference_identity != ("sesame-QCDPB", "4.6", "3.23", "1.30.1", "1.30.0"):
            raise ValueError("seSAMe parity-reference identity differs from protocol v1")


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
            raise ValueError("hg19 exhaustive reference-audit result differs from protocol v1")


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
            raise ValueError("published probe-mask identity differs from the frozen protocol")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    repository: str
    revision: str
    checkpoint_sha256: str
    embedding_width: PositiveInt

    def __post_init__(self) -> None:
        expected = (
            CADUCEUS_REPOSITORY,
            CADUCEUS_REVISION,
            CADUCEUS_CHECKPOINT_SHA256,
            CADUCEUS_EMBEDDING_WIDTH,
        )
        observed = (
            self.repository,
            self.revision,
            self.checkpoint_sha256,
            int(self.embedding_width),
        )
        if observed != expected:
            raise ValueError("Caduceus identity differs from the frozen protocol")


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
class SweepConfig:
    latent_dimensions: tuple[LatentDimension, ...]
    lambda_age: tuple[NonNegativeWeight, ...]
    maximum_evaluation_pairs: PositiveInt

    def __post_init__(self) -> None:
        dimensions = tuple(map(int, self.latent_dimensions))
        lambdas = tuple(map(float, self.lambda_age))
        if len(set(dimensions)) != len(dimensions) or tuple(sorted(dimensions)) != dimensions:
            raise ValueError("latent dimensions must be unique and increasing")
        if max(dimensions) > CADUCEUS_EMBEDDING_WIDTH:
            raise ValueError("latent sweep exceeds Caduceus input width")
        if len(set(lambdas)) != len(lambdas) or any(value <= 0.0 for value in lambdas):
            raise ValueError("age-loss weights must be unique and positive")


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
    sweep: SweepConfig

    def __post_init__(self) -> None:
        if self.schema != "methylation-latent.protocol.v1":
            raise ValueError("unknown protocol configuration schema")
        if self.status not in {"draft", "frozen"}:
            raise ValueError("protocol status must be 'draft' or 'frozen'")
        if not self.protocol_id:
            raise ValueError("protocol ID must not be empty")


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
    sweep = _table(raw, "sweep")
    _exact_keys(
        data,
        {
            "accession",
            "expected_samples",
            "genome_build",
            "manifest_sha256",
            "series_matrix_sha256",
            "sample_key_sha256",
            "sample_order_sha256",
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
            "parity_arrays",
            "parity_seed",
            "reference_processor",
            "reference_r_version",
            "reference_bioconductor_release",
            "reference_sesame_version",
            "reference_sesame_data_version",
        },
        "qc",
    )
    _exact_keys(masks, {"chen_url", "chen_sha256", "zhou_url", "zhou_sha256"}, "masks")
    _exact_keys(model, {"repository", "revision", "checkpoint_sha256", "embedding_width"}, "model")
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
    _exact_keys(sweep, {"latent_dimensions", "lambda_age", "maximum_evaluation_pairs"}, "sweep")
    return ProtocolConfig(
        schema=_string(raw["schema"], "schema"),
        protocol_id=_string(raw["protocol_id"], "protocol_id"),
        status=_string(raw["status"], "status"),
        data=DataConfig(
            accession=_string(data["accession"], "data.accession"),
            expected_samples=parse_positive_int(
                _integer(data["expected_samples"], "data.expected_samples")
            ),
            genome_build=_string(data["genome_build"], "data.genome_build"),
            manifest_sha256=_string(data["manifest_sha256"], "data.manifest_sha256"),
            series_matrix_sha256=_string(data["series_matrix_sha256"], "data.series_matrix_sha256"),
            sample_key_sha256=_string(data["sample_key_sha256"], "data.sample_key_sha256"),
            sample_order_sha256=_string(data["sample_order_sha256"], "data.sample_order_sha256"),
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
            parity_arrays=parse_positive_int(_integer(qc["parity_arrays"], "qc.parity_arrays")),
            parity_seed=_integer(qc["parity_seed"], "qc.parity_seed"),
            reference_processor=_string(qc["reference_processor"], "qc.reference_processor"),
            reference_r_version=_string(qc["reference_r_version"], "qc.reference_r_version"),
            reference_bioconductor_release=_string(
                qc["reference_bioconductor_release"],
                "qc.reference_bioconductor_release",
            ),
            reference_sesame_version=_string(
                qc["reference_sesame_version"], "qc.reference_sesame_version"
            ),
            reference_sesame_data_version=_string(
                qc["reference_sesame_data_version"],
                "qc.reference_sesame_data_version",
            ),
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
        sweep=SweepConfig(
            latent_dimensions=tuple(
                parse_latent_dimension(_integer(value, "sweep.latent_dimensions[]"))
                for value in _list(sweep["latent_dimensions"], "sweep.latent_dimensions")
            ),
            lambda_age=tuple(
                parse_non_negative_weight(_number(value, "sweep.lambda_age[]"))
                for value in _list(sweep["lambda_age"], "sweep.lambda_age")
            ),
            maximum_evaluation_pairs=parse_positive_int(
                _integer(sweep["maximum_evaluation_pairs"], "sweep.maximum_evaluation_pairs")
            ),
        ),
    )
