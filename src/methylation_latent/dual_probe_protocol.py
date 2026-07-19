"""Strict frozen configuration for the post-hoc dual-probe latent experiment."""

from __future__ import annotations

import math
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from methylation_latent.domain import (
    Fraction,
    LatentDimension,
    NonNegativeWeight,
    PositiveInt,
    WindowSize,
    parse_fraction,
    parse_latent_dimension,
    parse_non_negative_weight,
    parse_positive_int,
    parse_window_size,
)
from methylation_latent.dual_probe import AlphaSchedule, AlphaScheduleMode

_SHA256 = re.compile(r"^[0-9a-f]{64}$", flags=re.ASCII)
_SPLITS = ("diverse-blocks", "held-out-chromosome")
_SUPPORTED_WINDOWS = (1_024, 4_096, 16_384)
_V1_WINDOWS = (1_024, 4_096)
_V1_PROTOCOL_ID = "gse87571-posthoc-dual-probe-latent-v1"
_V2_PROTOCOL_ID = "gse87571-posthoc-dual-probe-latent-v2"
_PARENT_LATENT_DIMENSIONS = (16, 32, 64, 128, 256)
_PARENT_LAMBDA_AGES = (0.1, 1.0, 10.0)


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
        raise TypeError(f"dual configuration field {name!r} must be a table")
    return cast(dict[str, object], value)


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"dual configuration field {name!r} must be a non-empty string")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"dual configuration field {name!r} must be Boolean")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"dual configuration field {name!r} must be an integer")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"dual configuration field {name!r} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"dual configuration field {name!r} must be finite")
    return parsed


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list) or not value:
        raise TypeError(f"dual configuration field {name!r} must be a non-empty array")
    return cast(list[object], value)


def _sha256(value: object, name: str) -> str:
    parsed = _string(value, name)
    if _SHA256.fullmatch(parsed) is None:
        raise ValueError(f"dual configuration field {name!r} must be a SHA-256")
    return parsed


@dataclass(frozen=True, slots=True)
class DualInitializationConfig:
    condition_ceiling: float
    relative_residual_ceiling: float
    discarded_cross_moment_ceiling: float

    def __post_init__(self) -> None:
        if (
            self.condition_ceiling != 100_000_000.0
            or self.relative_residual_ceiling != 1.0e-5
            or self.discarded_cross_moment_ceiling != 0.005
        ):
            raise ValueError("dual least-squares audit thresholds differ from reviewed versions")


@dataclass(frozen=True, slots=True)
class DualOptimizationConfig:
    catch_weight: NonNegativeWeight
    batch_size: PositiveInt
    neighbourhood_width: PositiveInt
    steps: PositiveInt
    validation_interval: PositiveInt
    validation_pair_chunk_size: PositiveInt
    learning_rate: float
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        observed = (
            float(self.catch_weight),
            int(self.batch_size),
            int(self.neighbourhood_width),
            int(self.steps),
            int(self.validation_interval),
            int(self.validation_pair_chunk_size),
            self.learning_rate,
            self.seeds,
        )
        expected = (0.1, 512, 1_048_576, 2_000, 100, 512, 0.001, (851733, 851734, 851735))
        if observed != expected:
            raise ValueError("dual optimization schedule differs from reviewed versions")


@dataclass(frozen=True, slots=True)
class DualParentCell:
    split: str
    window_size: WindowSize
    latent_dimension: LatentDimension
    lambda_age: NonNegativeWeight
    selection_sha256: str
    evaluation_sha256: str
    embedding_manifest_sha256: str
    pair_cache_metadata_sha256: str

    def __post_init__(self) -> None:
        if self.split not in _SPLITS or int(self.window_size) not in _SUPPORTED_WINDOWS:
            raise ValueError("dual parent cell is outside the supported split/window grid")
        if int(self.latent_dimension) not in _PARENT_LATENT_DIMENSIONS:
            raise ValueError("dual parent latent dimension is outside the parent sweep")
        if float(self.lambda_age) not in _PARENT_LAMBDA_AGES:
            raise ValueError("dual parent age weight is outside the parent sweep")
        hashes = (
            self.selection_sha256,
            self.evaluation_sha256,
            self.embedding_manifest_sha256,
            self.pair_cache_metadata_sha256,
        )
        if any(_SHA256.fullmatch(value) is None for value in hashes):
            raise ValueError("dual parent artifact fingerprints must be SHA-256 values")

    @property
    def key(self) -> tuple[str, int]:
        return self.split, int(self.window_size)


@dataclass(frozen=True, slots=True)
class DualProbeProtocolConfig:
    protocol_id: str
    parent_protocol_id: str
    parent_protocol_sha256: str
    data_bundle_sha256: str
    windows: tuple[WindowSize, ...]
    splits: tuple[str, ...]
    initialization: DualInitializationConfig
    optimization: DualOptimizationConfig
    fixed_alphas: tuple[Fraction, ...]
    schedule_names: tuple[str, ...]
    parents: tuple[DualParentCell, ...]

    def __post_init__(self) -> None:
        window_values = tuple(map(int, self.windows))
        if self.protocol_id == _V1_PROTOCOL_ID:
            if window_values != _V1_WINDOWS:
                raise ValueError("dual protocol v1 windows differ from the reviewed grid")
            if any(float(parent.lambda_age) != 0.1 for parent in self.parents):
                raise ValueError("dual protocol v1 parent age weights differ from reviewed values")
        elif self.protocol_id == _V2_PROTOCOL_ID:
            canonical_windows = tuple(
                window for window in _SUPPORTED_WINDOWS if window in window_values
            )
            if (
                not window_values
                or len(set(window_values)) != len(window_values)
                or window_values != canonical_windows
                or 16_384 not in window_values
            ):
                raise ValueError(
                    "dual protocol v2 windows must be a canonical supported subset containing 16384"
                )
        else:
            raise ValueError("dual protocol ID is not a supported reviewed version")
        if self.splits != _SPLITS:
            raise ValueError("dual protocol split grid differs from reviewed versions")
        if self.parent_protocol_id != "gse87571-hg19-caduceus-ps-v2":
            raise ValueError("dual parent protocol ID differs")
        hashes = (self.parent_protocol_sha256, self.data_bundle_sha256)
        if any(_SHA256.fullmatch(value) is None for value in hashes):
            raise ValueError("dual protocol fingerprints must be SHA-256 values")
        if tuple(map(float, self.fixed_alphas)) != (0.0, 0.25, 0.5, 0.75, 1.0):
            raise ValueError("dual fixed-alpha sweep differs from reviewed versions")
        if self.schedule_names != ("linear_0_to_1", "linear_1_to_0"):
            raise ValueError("dual alpha schedules differ from reviewed versions")
        keys = tuple(parent.key for parent in self.parents)
        expected = tuple((split, window) for split in self.splits for window in window_values)
        if len(set(keys)) != len(keys) or set(keys) != set(expected):
            raise ValueError("dual parent cells must exactly cover the protocol grid")

    @property
    def manifest_schema(self) -> str:
        """Return the manifest schema whose completeness semantics match this protocol."""

        version = "v1" if self.protocol_id == _V1_PROTOCOL_ID else "v2"
        return f"methylation-latent.dual-probe-manifest.{version}"

    def parent(self, split: str, window_size: int) -> DualParentCell:
        matches = tuple(parent for parent in self.parents if parent.key == (split, window_size))
        if len(matches) != 1:
            raise ValueError("requested dual parent cell is absent or duplicated")
        return matches[0]

    def alpha_schedules(self) -> tuple[AlphaSchedule, ...]:
        fixed = tuple(
            AlphaSchedule(
                name=f"fixed_{float(alpha):g}",
                mode=AlphaScheduleMode.FIXED,
                start=alpha,
                end=alpha,
            )
            for alpha in self.fixed_alphas
        )
        linear = (
            AlphaSchedule(
                "linear_0_to_1",
                AlphaScheduleMode.LINEAR,
                parse_fraction(0.0),
                parse_fraction(1.0),
            ),
            AlphaSchedule(
                "linear_1_to_0",
                AlphaScheduleMode.LINEAR,
                parse_fraction(1.0),
                parse_fraction(0.0),
            ),
        )
        return fixed + linear


def _parse_parent(raw: object) -> DualParentCell:
    expected = {
        "split",
        "window",
        "latent_dimension",
        "lambda_age",
        "selection_sha256",
        "evaluation_sha256",
        "embedding_manifest_sha256",
        "pair_cache_metadata_sha256",
    }
    if not isinstance(raw, dict):
        raise TypeError("dual parent entry must be a table")
    record = cast(dict[str, object], raw)
    _exact_keys(record, expected, "dual parent")
    return DualParentCell(
        split=_string(record["split"], "parent.split"),
        window_size=parse_window_size(_integer(record["window"], "parent.window")),
        latent_dimension=parse_latent_dimension(
            _integer(record["latent_dimension"], "parent.latent_dimension")
        ),
        lambda_age=parse_non_negative_weight(_number(record["lambda_age"], "parent.lambda_age")),
        selection_sha256=_sha256(record["selection_sha256"], "parent.selection_sha256"),
        evaluation_sha256=_sha256(record["evaluation_sha256"], "parent.evaluation_sha256"),
        embedding_manifest_sha256=_sha256(
            record["embedding_manifest_sha256"],
            "parent.embedding_manifest_sha256",
        ),
        pair_cache_metadata_sha256=_sha256(
            record["pair_cache_metadata_sha256"],
            "parent.pair_cache_metadata_sha256",
        ),
    )


def load_dual_probe_protocol(path: Path) -> DualProbeProtocolConfig:
    """Parse every field and reject drift from a reviewed post-hoc protocol version."""

    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    _exact_keys(
        raw,
        {"protocol", "objective", "initialization", "training", "alpha", "selection", "parent"},
        "dual configuration",
    )
    protocol = _table(raw, "protocol")
    _exact_keys(
        protocol,
        {
            "id",
            "status",
            "scientific_status",
            "parent_protocol_id",
            "parent_protocol_sha256",
            "data_bundle_sha256",
            "windows",
            "splits",
        },
        "dual protocol",
    )
    if (
        _string(protocol["status"], "protocol.status") != "frozen"
        or _string(protocol["scientific_status"], "protocol.scientific_status")
        != "post_hoc_hypothesis_generating"
    ):
        raise ValueError("dual protocol status differs from reviewed versions")
    windows = tuple(
        parse_window_size(_integer(value, "protocol.windows"))
        for value in _list(protocol["windows"], "protocol.windows")
    )
    splits = tuple(
        _string(value, "protocol.splits") for value in _list(protocol["splits"], "protocol.splits")
    )

    objective = _table(raw, "objective")
    _exact_keys(
        objective,
        {
            "formula",
            "geometry_formula",
            "sequence_direct_target_weight",
            "catch_weight",
            "weight_decay",
        },
        "dual objective",
    )
    if (
        _string(objective["formula"], "objective.formula")
        != "L_geometry(u,a) + catch_weight * (alpha * (1-cos(u,stopgrad(z))) + (1-alpha) * (1-cos(stopgrad(u),z)))"
        or _string(objective["geometry_formula"], "objective.geometry_formula")
        != "mean((u@u.T-Y)^2) + lambda_age * mean((u@a-rho)^2)"
        or _number(
            objective["sequence_direct_target_weight"],
            "objective.sequence_direct_target_weight",
        )
        != 0.0
        or _number(objective["weight_decay"], "objective.weight_decay") != 0.0
    ):
        raise ValueError("dual objective differs from reviewed versions")

    initialization = _table(raw, "initialization")
    _exact_keys(
        initialization,
        {
            "learned_vectors",
            "projection",
            "normal_equation_condition_ceiling",
            "relative_normal_residual_ceiling",
            "maximum_discarded_cross_moment_fraction",
            "validation_or_test_rows_allowed",
        },
        "dual initialization",
    )
    if (
        _string(initialization["learned_vectors"], "initialization.learned_vectors")
        != "iid_standard_normal_then_rowwise_unit_norm"
        or _string(initialization["projection"], "initialization.projection")
        != "bias_free_training_partition_condition_truncated_least_squares_to_random_unit_vectors"
        or _boolean(
            initialization["validation_or_test_rows_allowed"],
            "initialization.validation_or_test_rows_allowed",
        )
    ):
        raise ValueError("dual initialization contract differs from reviewed versions")
    initialization_config = DualInitializationConfig(
        condition_ceiling=_number(
            initialization["normal_equation_condition_ceiling"],
            "initialization.normal_equation_condition_ceiling",
        ),
        relative_residual_ceiling=_number(
            initialization["relative_normal_residual_ceiling"],
            "initialization.relative_normal_residual_ceiling",
        ),
        discarded_cross_moment_ceiling=_number(
            initialization["maximum_discarded_cross_moment_fraction"],
            "initialization.maximum_discarded_cross_moment_fraction",
        ),
    )

    training = _table(raw, "training")
    _exact_keys(
        training,
        {
            "dense_optimizer",
            "learned_table_optimizer",
            "batch_size",
            "neighbourhood_width",
            "steps",
            "validation_interval",
            "validation_pair_chunk_size",
            "learning_rate",
            "seeds",
        },
        "dual training",
    )
    if (
        _string(training["dense_optimizer"], "training.dense_optimizer") != "adam"
        or _string(
            training["learned_table_optimizer"],
            "training.learned_table_optimizer",
        )
        != "sparse_adam"
    ):
        raise ValueError("dual optimizer kinds differ from reviewed versions")
    optimization = DualOptimizationConfig(
        catch_weight=parse_non_negative_weight(
            _number(objective["catch_weight"], "objective.catch_weight")
        ),
        batch_size=parse_positive_int(_integer(training["batch_size"], "training.batch_size")),
        neighbourhood_width=parse_positive_int(
            _integer(training["neighbourhood_width"], "training.neighbourhood_width")
        ),
        steps=parse_positive_int(_integer(training["steps"], "training.steps")),
        validation_interval=parse_positive_int(
            _integer(training["validation_interval"], "training.validation_interval")
        ),
        validation_pair_chunk_size=parse_positive_int(
            _integer(
                training["validation_pair_chunk_size"],
                "training.validation_pair_chunk_size",
            )
        ),
        learning_rate=_number(training["learning_rate"], "training.learning_rate"),
        seeds=tuple(
            _integer(value, "training.seeds")
            for value in _list(training["seeds"], "training.seeds")
        ),
    )

    alpha = _table(raw, "alpha")
    _exact_keys(alpha, {"fixed", "schedules"}, "dual alpha")
    fixed_alphas = tuple(
        parse_fraction(_number(value, "alpha.fixed"))
        for value in _list(alpha["fixed"], "alpha.fixed")
    )
    schedule_names = tuple(
        _string(value, "alpha.schedules") for value in _list(alpha["schedules"], "alpha.schedules")
    )

    selection = _table(raw, "selection")
    _exact_keys(
        selection,
        {
            "candidate_score",
            "checkpoint_score",
            "tie_break",
            "test_policy",
            "refit_partition",
        },
        "dual selection",
    )
    observed_selection = tuple(_string(selection[name], f"selection.{name}") for name in selection)
    expected_selection = (
        "mean_over_seeds_of_each_seed_minimum_validation_sequence_pair_mse_plus_age_mse",
        "validation_sequence_pair_mse_plus_age_mse",
        "config_order_then_smaller_selected_step",
        "evaluate_only_the_validation_selected_alpha_strategy_for_each_split_and_window",
        "complete_primary_train",
    )
    if observed_selection != expected_selection:
        raise ValueError("dual selection policy differs from reviewed versions")

    parents_raw = raw["parent"]
    if not isinstance(parents_raw, list) or not parents_raw:
        raise TypeError("dual parent configuration must be a non-empty table array")
    return DualProbeProtocolConfig(
        protocol_id=_string(protocol["id"], "protocol.id"),
        parent_protocol_id=_string(protocol["parent_protocol_id"], "protocol.parent_protocol_id"),
        parent_protocol_sha256=_sha256(
            protocol["parent_protocol_sha256"], "protocol.parent_protocol_sha256"
        ),
        data_bundle_sha256=_sha256(protocol["data_bundle_sha256"], "protocol.data_bundle_sha256"),
        windows=windows,
        splits=splits,
        initialization=initialization_config,
        optimization=optimization,
        fixed_alphas=fixed_alphas,
        schedule_names=schedule_names,
        parents=tuple(_parse_parent(value) for value in parents_raw),
    )
