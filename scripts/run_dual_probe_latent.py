"""Tune and evaluate train-only learned probe geometry with sequence catching."""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch as t

from methylation_latent.artifacts import (
    JsonValue,
    require_clean_git_commit,
    sha256_file,
    sha256_ordered_strings,
    write_canonical_json_exclusive,
)
from methylation_latent.cohort import load_probe_table
from methylation_latent.config import load_protocol_config
from methylation_latent.data_bundle import PrimaryDataBundle, verify_primary_data_bundle
from methylation_latent.domain import NonEmptyProbeSet
from methylation_latent.dual_probe import (
    AlphaSchedule,
    DualInitialization,
    DualRefitResult,
    DualTrainingConfig,
    DualTuningResult,
    DualValidationData,
    build_dual_initialization,
    refit_dual_probe,
    tune_dual_probe,
)
from methylation_latent.dual_probe_experiment import (
    DualPartitionData,
    assert_dual_split_roles,
    held_out_age_predictions,
    hybrid_seen_by_held_out_predictions,
    sequence_latent_from_projection,
    subset_dual_partition,
)
from methylation_latent.dual_probe_protocol import (
    DualParentCell,
    DualProbeProtocolConfig,
    load_dual_probe_protocol,
)
from methylation_latent.embedding_cache import load_embedding_cache
from methylation_latent.evaluation import (
    DISTANCE_CLASS_LABELS,
    DistanceClass,
    PairPopulation,
    gather_pair_predictions_chunked,
)
from methylation_latent.evaluation_cache import (
    CachedPairSet,
    EvaluationPairCache,
    EvaluationPairCacheIdentity,
    PairSetName,
    load_evaluation_pair_cache,
)
from methylation_latent.experiment_data import SplitArtifact, load_split_artifact
from methylation_latent.model import normalize_vector_strict
from methylation_latent.storage import (
    EmbeddingMatrix,
    load_exact_safetensors,
    load_target_geometry,
    save_safetensors_exclusive,
)
from methylation_latent.targets import TargetGeometry

_TUNING_SCHEMA = "methylation-latent.dual-probe-tuning.v1"
_SELECTION_SCHEMA = "methylation-latent.dual-probe-selection.v1"
_REFIT_SCHEMA = "methylation-latent.dual-probe-refit.v1"
_RESULTS_SCHEMA = "methylation-latent.dual-probe-results.v1"
_PARENT_SELECTION_SCHEMA = "methylation-latent.hyperparameter-selection.v2"
_PARENT_EVALUATION_SCHEMA = "methylation-latent.held-out-evaluation.v2"
_MODEL_KEYS = {
    "learned_latent",
    "projection_weight",
    "age_direction",
    "global_indices",
}
_SPLITS = ("diverse-blocks", "held-out-chromosome")
_SUPPORTED_WINDOWS = (1_024, 4_096, 16_384)
_PAIR_CHUNK_SIZE = 16_384
_PROJECTION_ROW_CHUNK_SIZE = 8_192
_DISPLAY_LIMIT = 5_000


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent-config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--experiments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("audit", "tune", "evaluate", "all"),
        required=True,
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--splits", nargs="+", choices=_SPLITS)
    parser.add_argument("--windows", nargs="+", type=int, choices=_SUPPORTED_WINDOWS)
    return parser


def _requested_splits(
    requested: list[str] | None,
    protocol: DualProbeProtocolConfig,
) -> tuple[str, ...]:
    values = protocol.splits if requested is None else tuple(requested)
    if len(set(values)) != len(values):
        raise ValueError("dual requested splits must not contain duplicates")
    if not set(values).issubset(protocol.splits):
        raise ValueError("dual requested splits must be an exact subset of the loaded protocol")
    return values


def _requested_windows(
    requested: list[int] | None,
    protocol: DualProbeProtocolConfig,
) -> tuple[int, ...]:
    protocol_windows = tuple(map(int, protocol.windows))
    values = protocol_windows if requested is None else tuple(requested)
    if len(set(values)) != len(values):
        raise ValueError("dual requested windows must not contain duplicates")
    if not set(values).issubset(protocol_windows):
        raise ValueError("dual requested windows must be an exact subset of the loaded protocol")
    return values


def _configure_runtime(device: str) -> None:
    t.use_deterministic_algorithms(True)
    if device == "cuda":
        if not t.cuda.is_available():
            raise ValueError("CUDA dual experiment requested but CUDA is unavailable")
        t.backends.cuda.matmul.allow_tf32 = False
        t.backends.cudnn.allow_tf32 = False


def _load_json(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"expected JSON object: {path}")
    return cast(dict[str, object], raw)


def _object(record: dict[str, object], key: str) -> dict[str, object]:
    value = record.get(key)
    if not isinstance(value, dict) or any(not isinstance(name, str) for name in value):
        raise TypeError(f"{key} must be an object")
    return cast(dict[str, object], value)


def _list(record: dict[str, object], key: str) -> list[object]:
    value = record.get(key)
    if not isinstance(value, list):
        raise TypeError(f"{key} must be an array")
    return cast(list[object], value)


def _string(record: dict[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise TypeError(f"{key} must be a non-empty string")
    return value


def _integer(record: dict[str, object], key: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value


def _number(record: dict[str, object], key: str) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{key} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{key} must be finite")
    return parsed


def _index_sha256(indices: t.Tensor) -> str:
    if indices.dtype != t.int64 or indices.ndim != 1 or indices.numel() == 0:
        raise TypeError("index fingerprint requires a non-empty int64 vector")
    return sha256_ordered_strings(map(str, indices.tolist()))


def _publish_or_verify_json(path: Path, record: dict[str, JsonValue]) -> None:
    if path.exists():
        if _load_json(path) != record:
            raise ValueError(f"existing immutable JSON differs: {path}")
        return
    write_canonical_json_exclusive(path, record)


def _cell_directory(output: Path, split_name: str, window: int) -> Path:
    return output / split_name / f"window-{window}"


def _split_sha256(data: Path, split_name: str) -> str:
    return sha256_ordered_strings(
        (
            sha256_file(data / "splits" / f"{split_name}.json"),
            sha256_file(data / "splits" / f"{split_name}.safetensors"),
        )
    )


def _verify_parent_cell(
    *,
    parent: DualParentCell,
    experiments: Path,
    embeddings: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    split_name, window = parent.key
    selection_path = experiments / "selections" / "full" / split_name / f"window-{window}.json"
    evaluation_path = experiments / "evaluation" / split_name / f"window-{window}.json"
    manifest_path = embeddings / f"window-{window}" / "manifest.json"
    pair_metadata_path = experiments / "evaluation-pairs" / split_name / "metadata.json"
    observed = (
        sha256_file(selection_path),
        sha256_file(evaluation_path),
        sha256_file(manifest_path),
        sha256_file(pair_metadata_path),
    )
    expected = (
        parent.selection_sha256,
        parent.evaluation_sha256,
        parent.embedding_manifest_sha256,
        parent.pair_cache_metadata_sha256,
    )
    if observed != expected:
        raise ValueError(f"parent artifact hashes differ for {split_name} window {window}")
    selection = _load_json(selection_path)
    selected_training = _object(selection, "selected_training")
    if (
        selection.get("schema") != _PARENT_SELECTION_SCHEMA
        or _integer(selected_training, "latent_dimension") != int(parent.latent_dimension)
        or _number(selected_training, "lambda_age") != float(parent.lambda_age)
    ):
        raise ValueError("parent selected d or lambda differs from the dual protocol")
    evaluation = _load_json(evaluation_path)
    if evaluation.get("schema") != _PARENT_EVALUATION_SCHEMA:
        raise ValueError("parent evaluation schema differs")
    return selection, evaluation


def _load_split(data: Path, probes: NonEmptyProbeSet, split_name: str) -> SplitArtifact:
    split = load_split_artifact(
        data / "splits" / f"{split_name}.safetensors",
        data / "splits" / f"{split_name}.json",
        probes=probes,
    )
    assert_dual_split_roles(split)
    return split


def _training_config(
    protocol: DualProbeProtocolConfig,
    parent: DualParentCell,
    schedule: AlphaSchedule,
    *,
    seed: int,
    device: str,
) -> DualTrainingConfig:
    optimization = protocol.optimization
    return DualTrainingConfig(
        latent_dimension=parent.latent_dimension,
        lambda_age=parent.lambda_age,
        catch_weight=optimization.catch_weight,
        alpha_schedule=schedule,
        batch_size=optimization.batch_size,
        neighbourhood_width=optimization.neighbourhood_width,
        steps=optimization.steps,
        validation_interval=optimization.validation_interval,
        validation_pair_chunk_size=optimization.validation_pair_chunk_size,
        learning_rate=optimization.learning_rate,
        seed=seed,
        device=device,
    )


def _training_json(config: DualTrainingConfig) -> dict[str, JsonValue]:
    schedule = config.alpha_schedule
    return {
        "architecture": "train_only_learned_probe_table_caught_by_bias_free_sequence_map",
        "latent_dimension": int(config.latent_dimension),
        "lambda_age": float(config.lambda_age),
        "catch_weight": float(config.catch_weight),
        "sequence_direct_target_weight": 0.0,
        "alpha_strategy": schedule.name,
        "alpha_mode": schedule.mode.value,
        "alpha_start": float(schedule.start),
        "alpha_end": float(schedule.end),
        "batch_size": int(config.batch_size),
        "neighbourhood_width": int(config.neighbourhood_width),
        "steps": int(config.steps),
        "validation_interval": int(config.validation_interval),
        "validation_pair_chunk_size": int(config.validation_pair_chunk_size),
        "learning_rate": config.learning_rate,
        "seed": config.seed,
        "device": config.device,
        "dense_optimizer": "adam",
        "learned_table_optimizer": "sparse_adam",
        "weight_decay": 0.0,
    }


def _initialization(
    partition: DualPartitionData,
    protocol: DualProbeProtocolConfig,
    parent: DualParentCell,
    *,
    seed: int,
    device: str,
) -> DualInitialization:
    audit = protocol.initialization
    return build_dual_initialization(
        partition.embeddings,
        latent_dimension=parent.latent_dimension,
        seed=seed,
        device=device,
        condition_ceiling=audit.condition_ceiling,
        relative_residual_ceiling=audit.relative_residual_ceiling,
        discarded_cross_moment_ceiling=audit.discarded_cross_moment_ceiling,
    )


def _tuning_identity(
    *,
    code_git_commit: str,
    dual_config_sha256: str,
    protocol: DualProbeProtocolConfig,
    bundle: PrimaryDataBundle,
    target_sha256: str,
    split_name: str,
    split_sha256: str,
    window: int,
    parent: DualParentCell,
    split: SplitArtifact,
    schedule: AlphaSchedule,
    seed: int,
    device: str,
) -> dict[str, JsonValue]:
    return {
        "code_git_commit": code_git_commit,
        "protocol_id": protocol.protocol_id,
        "protocol_config_sha256": dual_config_sha256,
        "parent_protocol_id": protocol.parent_protocol_id,
        "parent_protocol_sha256": protocol.parent_protocol_sha256,
        "primary_git_commit": bundle.git_commit,
        "data_bundle_sha256": protocol.data_bundle_sha256,
        "target_sha256": target_sha256,
        "split_name": split_name,
        "split_sha256": split_sha256,
        "window_size": window,
        "embedding_manifest_sha256": parent.embedding_manifest_sha256,
        "parent_selection_sha256": parent.selection_sha256,
        "parent_evaluation_sha256": parent.evaluation_sha256,
        "optimization_indices_sha256": _index_sha256(split.optimization_indices),
        "validation_indices_sha256": _index_sha256(split.validation_indices),
        "alpha_strategy": schedule.name,
        "seed": seed,
        "runtime_device": device,
    }


def _tuning_record(
    result: DualTuningResult,
    initialization: DualInitialization,
    config: DualTrainingConfig,
    identity: dict[str, JsonValue],
    partition: DualPartitionData,
    validation: DualPartitionData,
) -> dict[str, JsonValue]:
    selected = next(row for row in result.validation_history if row.step == result.selected_step)
    return {
        "schema": _TUNING_SCHEMA,
        "status": "post_hoc_hypothesis_generating_validation_only",
        "identity": identity,
        "partition": {
            "optimization_probe_count": len(partition.probes),
            "validation_probe_count": len(validation.probes),
            "learned_table_probe_count": len(partition.probes),
            "ols_fit_probe_count": initialization.least_squares.fit_row_count,
            "validation_learned_row_count": 0,
            "validation_ols_row_count": 0,
            "test_target_access_count": 0,
        },
        "initialization": cast(dict[str, JsonValue], asdict(initialization.least_squares)),
        "training": _training_json(config),
        "selected_step": result.selected_step,
        "selected_validation": cast(dict[str, JsonValue], asdict(selected)),
        "training_history": [
            cast(dict[str, JsonValue], asdict(row)) for row in result.training_history
        ],
        "validation_history": [
            cast(dict[str, JsonValue], asdict(row)) for row in result.validation_history
        ],
    }


def _validate_tuning_record(
    record: dict[str, object],
    expected_identity: dict[str, JsonValue],
    expected_training: dict[str, JsonValue],
) -> None:
    expected_keys = {
        "schema",
        "status",
        "identity",
        "partition",
        "initialization",
        "training",
        "selected_step",
        "selected_validation",
        "training_history",
        "validation_history",
    }
    if set(record) != expected_keys:
        raise ValueError("dual tuning record fields differ")
    if (
        record["schema"] != _TUNING_SCHEMA
        or record["status"] != "post_hoc_hypothesis_generating_validation_only"
        or record["identity"] != expected_identity
        or record["training"] != expected_training
    ):
        raise ValueError("dual tuning identity differs")
    partition = _object(record, "partition")
    if (
        _integer(partition, "optimization_probe_count")
        != _integer(partition, "learned_table_probe_count")
        or _integer(partition, "optimization_probe_count")
        != _integer(partition, "ols_fit_probe_count")
        or any(
            _integer(partition, name) != 0
            for name in (
                "validation_learned_row_count",
                "validation_ols_row_count",
                "test_target_access_count",
            )
        )
    ):
        raise ValueError("dual tuning partition leakage checks differ")
    validation = _list(record, "validation_history")
    if not validation or any(not isinstance(row, dict) for row in validation):
        raise ValueError("dual validation history is malformed")
    selected_step = _integer(record, "selected_step")
    validation_records = tuple(cast(dict[str, object], row) for row in validation)
    candidates = tuple(row for row in validation_records if row.get("step") == selected_step)
    if len(candidates) != 1 or record["selected_validation"] != candidates[0]:
        raise ValueError("dual selected checkpoint differs from validation history")
    scores = tuple(_number(cast(dict[str, object], row), "selection_score") for row in validation)
    steps = tuple(_integer(cast(dict[str, object], row), "step") for row in validation)
    best_index = min(range(len(scores)), key=lambda index: (scores[index], steps[index]))
    if steps[best_index] != selected_step:
        raise ValueError("dual selected checkpoint is not the validation minimum")


def _tuning_path(
    output: Path,
    split_name: str,
    window: int,
    schedule: AlphaSchedule,
    seed: int,
) -> Path:
    return (
        _cell_directory(output, split_name, window) / "tuning" / schedule.name / f"seed-{seed}.json"
    )


def _run_tuning_cell(
    *,
    code_git_commit: str,
    dual_config_sha256: str,
    protocol: DualProbeProtocolConfig,
    bundle: PrimaryDataBundle,
    target_sha256: str,
    split_name: str,
    split_sha256: str,
    window: int,
    parent: DualParentCell,
    probes: NonEmptyProbeSet,
    embeddings: EmbeddingMatrix,
    targets: TargetGeometry,
    split: SplitArtifact,
    output: Path,
    device: str,
) -> None:
    optimization = subset_dual_partition(
        probes,
        embeddings,
        targets,
        split.optimization_indices,
    )
    validation = subset_dual_partition(
        probes,
        embeddings,
        targets,
        split.validation_indices,
    )
    schedules = protocol.alpha_schedules()
    for seed in protocol.optimization.seeds:
        identities = {
            schedule.name: _tuning_identity(
                code_git_commit=code_git_commit,
                dual_config_sha256=dual_config_sha256,
                protocol=protocol,
                bundle=bundle,
                target_sha256=target_sha256,
                split_name=split_name,
                split_sha256=split_sha256,
                window=window,
                parent=parent,
                split=split,
                schedule=schedule,
                seed=seed,
                device=device,
            )
            for schedule in schedules
        }
        missing: list[AlphaSchedule] = []
        existing_initializations: list[dict[str, object]] = []
        for schedule in schedules:
            path = _tuning_path(output, split_name, window, schedule, seed)
            if path.exists():
                record = _load_json(path)
                expected_training = _training_json(
                    _training_config(
                        protocol,
                        parent,
                        schedule,
                        seed=seed,
                        device=device,
                    )
                )
                _validate_tuning_record(
                    record,
                    identities[schedule.name],
                    expected_training,
                )
                existing_initializations.append(_object(record, "initialization"))
            else:
                missing.append(schedule)
        if existing_initializations and any(
            audit != existing_initializations[0] for audit in existing_initializations[1:]
        ):
            raise ValueError("alpha candidates did not share one initialization audit")
        if not missing:
            print(
                f"dual tuning reused split={split_name} window={window} seed={seed}",
                flush=True,
            )
            continue
        initialization = _initialization(
            optimization,
            protocol,
            parent,
            seed=seed,
            device=device,
        )
        initialization_json = cast(dict[str, object], asdict(initialization.least_squares))
        if existing_initializations and initialization_json != existing_initializations[0]:
            raise ValueError("recomputed shared dual initialization audit differs")
        for schedule in missing:
            print(
                f"dual tuning start split={split_name} window={window} "
                f"seed={seed} alpha={schedule.name}",
                flush=True,
            )
            config = _training_config(
                protocol,
                parent,
                schedule,
                seed=seed,
                device=device,
            )
            result = tune_dual_probe(
                optimization.probes,
                optimization.embeddings,
                optimization.targets,
                validation=DualValidationData(
                    validation.probes,
                    validation.embeddings,
                    validation.targets,
                ),
                initialization=initialization,
                config=config,
            )
            record = _tuning_record(
                result,
                initialization,
                config,
                identities[schedule.name],
                optimization,
                validation,
            )
            path = _tuning_path(output, split_name, window, schedule, seed)
            _publish_or_verify_json(path, record)
            selected = cast(dict[str, JsonValue], record["selected_validation"])
            print(
                f"dual tuning done split={split_name} window={window} seed={seed} "
                f"alpha={schedule.name} step={result.selected_step} "
                f"score={selected['selection_score']}",
                flush=True,
            )


def _selection_record(
    *,
    code_git_commit: str,
    dual_config_sha256: str,
    protocol: DualProbeProtocolConfig,
    bundle: PrimaryDataBundle,
    target_sha256: str,
    split_name: str,
    split_sha256: str,
    window: int,
    parent: DualParentCell,
    split: SplitArtifact,
    output: Path,
    device: str,
) -> dict[str, JsonValue]:
    candidate_records: list[dict[str, JsonValue]] = []
    for order, schedule in enumerate(protocol.alpha_schedules()):
        seed_records: list[dict[str, JsonValue]] = []
        for seed in protocol.optimization.seeds:
            path = _tuning_path(output, split_name, window, schedule, seed)
            identity = _tuning_identity(
                code_git_commit=code_git_commit,
                dual_config_sha256=dual_config_sha256,
                protocol=protocol,
                bundle=bundle,
                target_sha256=target_sha256,
                split_name=split_name,
                split_sha256=split_sha256,
                window=window,
                parent=parent,
                split=split,
                schedule=schedule,
                seed=seed,
                device=device,
            )
            record = _load_json(path)
            _validate_tuning_record(
                record,
                identity,
                _training_json(
                    _training_config(
                        protocol,
                        parent,
                        schedule,
                        seed=seed,
                        device=device,
                    )
                ),
            )
            selected = _object(record, "selected_validation")
            seed_records.append(
                {
                    "seed": seed,
                    "selected_step": _integer(record, "selected_step"),
                    "selection_score": _number(selected, "selection_score"),
                    "pair_mse": _number(selected, "pair_mse"),
                    "age_mse": _number(selected, "age_mse"),
                    "tuning_record": path.relative_to(output).as_posix(),
                    "tuning_record_sha256": sha256_file(path),
                }
            )
        score_values = t.tensor(
            tuple(cast(float, record["selection_score"]) for record in seed_records),
            dtype=t.float64,
        )
        candidate_records.append(
            {
                "config_order": order,
                "alpha_strategy": schedule.name,
                "alpha_mode": schedule.mode.value,
                "alpha_start": float(schedule.start),
                "alpha_end": float(schedule.end),
                "mean_seed_minimum_validation_score": float(score_values.mean().item()),
                "standard_deviation_seed_minimum_validation_score": float(
                    score_values.std(unbiased=False).item()
                ),
                "seeds": seed_records,
            }
        )
    selected_index = min(
        range(len(candidate_records)),
        key=lambda index: (
            float(candidate_records[index]["mean_seed_minimum_validation_score"]),
            int(candidate_records[index]["config_order"]),
        ),
    )
    selected = candidate_records[selected_index]
    return {
        "schema": _SELECTION_SCHEMA,
        "status": "post_hoc_hypothesis_generating_validation_selected",
        "identity": {
            "code_git_commit": code_git_commit,
            "protocol_id": protocol.protocol_id,
            "protocol_config_sha256": dual_config_sha256,
            "parent_protocol_id": protocol.parent_protocol_id,
            "parent_protocol_sha256": protocol.parent_protocol_sha256,
            "primary_git_commit": bundle.git_commit,
            "data_bundle_sha256": protocol.data_bundle_sha256,
            "target_sha256": target_sha256,
            "split_name": split_name,
            "split_sha256": split_sha256,
            "window_size": window,
            "embedding_manifest_sha256": parent.embedding_manifest_sha256,
            "parent_selection_sha256": parent.selection_sha256,
            "parent_evaluation_sha256": parent.evaluation_sha256,
            "optimization_indices_sha256": _index_sha256(split.optimization_indices),
            "validation_indices_sha256": _index_sha256(split.validation_indices),
            "runtime_device": device,
        },
        "selection_rule": (
            "minimum_mean_over_seeds_of_each_seed_minimum_validation_"
            "sequence_pair_mse_plus_age_mse_then_config_order"
        ),
        "test_metrics_read": False,
        "candidates": candidate_records,
        "selected_alpha_strategy": str(selected["alpha_strategy"]),
        "selected_candidate_index": selected_index,
    }


def _validate_selection(record: dict[str, object]) -> None:
    expected = {
        "schema",
        "status",
        "identity",
        "selection_rule",
        "test_metrics_read",
        "candidates",
        "selected_alpha_strategy",
        "selected_candidate_index",
    }
    if set(record) != expected or record.get("schema") != _SELECTION_SCHEMA:
        raise ValueError("dual selection record fields or schema differ")
    if record.get("test_metrics_read") is not False:
        raise ValueError("dual alpha selection must not read test metrics")
    candidates_raw = _list(record, "candidates")
    if not candidates_raw or any(not isinstance(value, dict) for value in candidates_raw):
        raise ValueError("dual selection candidates are malformed")
    candidates = tuple(cast(dict[str, object], value) for value in candidates_raw)
    indices = tuple(_integer(candidate, "config_order") for candidate in candidates)
    if indices != tuple(range(len(candidates))):
        raise ValueError("dual selection config order differs")
    selected_index = _integer(record, "selected_candidate_index")
    expected_index = min(
        range(len(candidates)),
        key=lambda index: (
            _number(candidates[index], "mean_seed_minimum_validation_score"),
            _integer(candidates[index], "config_order"),
        ),
    )
    if selected_index != expected_index or _string(record, "selected_alpha_strategy") != _string(
        candidates[selected_index], "alpha_strategy"
    ):
        raise ValueError("dual selected alpha is not the validation-only minimum")


def _select_tuning_cell(
    *,
    code_git_commit: str,
    dual_config_sha256: str,
    protocol: DualProbeProtocolConfig,
    bundle: PrimaryDataBundle,
    target_sha256: str,
    split_name: str,
    split_sha256: str,
    window: int,
    parent: DualParentCell,
    split: SplitArtifact,
    output: Path,
    device: str,
) -> dict[str, object]:
    expected = _selection_record(
        code_git_commit=code_git_commit,
        dual_config_sha256=dual_config_sha256,
        protocol=protocol,
        bundle=bundle,
        target_sha256=target_sha256,
        split_name=split_name,
        split_sha256=split_sha256,
        window=window,
        parent=parent,
        split=split,
        output=output,
        device=device,
    )
    path = _cell_directory(output, split_name, window) / "selection.json"
    _publish_or_verify_json(path, expected)
    record = _load_json(path)
    _validate_selection(record)
    print(
        f"dual selected split={split_name} window={window} "
        f"alpha={_string(record, 'selected_alpha_strategy')}",
        flush=True,
    )
    return record


def _selected_schedule(
    protocol: DualProbeProtocolConfig,
    selection: dict[str, object],
) -> AlphaSchedule:
    name = _string(selection, "selected_alpha_strategy")
    matches = tuple(schedule for schedule in protocol.alpha_schedules() if schedule.name == name)
    if len(matches) != 1:
        raise ValueError("selected dual alpha strategy is absent or duplicated")
    return matches[0]


def _selected_seed_record(selection: dict[str, object], seed: int) -> dict[str, object]:
    candidates = tuple(cast(dict[str, object], row) for row in _list(selection, "candidates"))
    selected = candidates[_integer(selection, "selected_candidate_index")]
    seeds = tuple(cast(dict[str, object], row) for row in _list(selected, "seeds"))
    matches = tuple(record for record in seeds if _integer(record, "seed") == seed)
    if len(matches) != 1:
        raise ValueError("selected dual seed record is absent or duplicated")
    return matches[0]


def _refit_directory(output: Path, split_name: str, window: int, seed: int) -> Path:
    return _cell_directory(output, split_name, window) / "refit" / f"seed-{seed}"


def _refit_payload(result: DualRefitResult, global_indices: t.Tensor) -> dict[str, t.Tensor]:
    model = result.model
    with t.inference_mode():
        learned = model.all_learned().detach().cpu().contiguous()
        projection = model.projection.weight.detach().cpu().contiguous()
        age = normalize_vector_strict(model.age_direction).detach().cpu().contiguous()
    payload = {
        "learned_latent": learned,
        "projection_weight": projection,
        "age_direction": age,
        "global_indices": global_indices.detach().cpu().contiguous(),
    }
    if set(payload) != _MODEL_KEYS:
        raise RuntimeError("dual refit payload keys differ")
    return payload


def _publish_refit(
    *,
    directory: Path,
    identity: dict[str, JsonValue],
    result: DualRefitResult,
    initialization: DualInitialization,
    config: DualTrainingConfig,
    partition: DualPartitionData,
    test_indices: t.Tensor,
) -> None:
    if bool(t.isin(partition.global_indices, test_indices).any().item()):
        raise ValueError("dual refit learned table contains held-out test rows")
    payload = _refit_payload(result, partition.global_indices)
    if directory.exists():
        _verify_refit(
            directory,
            identity,
            _training_json(config),
            partition.global_indices,
        )
        return
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = directory.parent / f".{directory.name}.{secrets.token_hex(16)}.tmp"
    temporary.mkdir()
    model_path = temporary / "model.safetensors"
    save_safetensors_exclusive(model_path, payload)
    metadata: dict[str, JsonValue] = {
        "schema": _REFIT_SCHEMA,
        "status": "post_hoc_selected_strategy_complete_primary_train_refit",
        "identity": identity,
        "partition": {
            "name": "complete_primary_train",
            "probe_count": len(partition.probes),
            "learned_table_probe_count": len(partition.probes),
            "ols_fit_probe_count": initialization.least_squares.fit_row_count,
            "held_out_learned_row_count": 0,
            "held_out_ols_row_count": 0,
            "test_target_access_count": 0,
        },
        "initialization": cast(dict[str, JsonValue], asdict(initialization.least_squares)),
        "training": _training_json(config),
        "selected_steps": result.refit_steps,
        "training_history": [
            cast(dict[str, JsonValue], asdict(row)) for row in result.training_history
        ],
        "model_file": model_path.name,
        "model_sha256": sha256_file(model_path),
    }
    write_canonical_json_exclusive(temporary / "metadata.json", metadata)
    os.rename(temporary, directory)


def _verify_refit(
    directory: Path,
    expected_identity: dict[str, JsonValue],
    expected_training: dict[str, JsonValue],
    expected_global_indices: t.Tensor,
) -> tuple[dict[str, object], dict[str, t.Tensor]]:
    if {path.name for path in directory.iterdir()} != {"metadata.json", "model.safetensors"}:
        raise ValueError("dual refit file inventory differs")
    metadata = _load_json(directory / "metadata.json")
    expected_keys = {
        "schema",
        "status",
        "identity",
        "partition",
        "initialization",
        "training",
        "selected_steps",
        "training_history",
        "model_file",
        "model_sha256",
    }
    model_path = directory / "model.safetensors"
    if (
        set(metadata) != expected_keys
        or metadata.get("schema") != _REFIT_SCHEMA
        or metadata.get("identity") != expected_identity
        or metadata.get("training") != expected_training
        or metadata.get("model_file") != model_path.name
        or metadata.get("model_sha256") != sha256_file(model_path)
    ):
        raise ValueError("dual refit metadata identity differs")
    partition = _object(metadata, "partition")
    if (
        _integer(partition, "probe_count") != _integer(partition, "learned_table_probe_count")
        or _integer(partition, "probe_count") != _integer(partition, "ols_fit_probe_count")
        or any(
            _integer(partition, name) != 0
            for name in (
                "held_out_learned_row_count",
                "held_out_ols_row_count",
                "test_target_access_count",
            )
        )
    ):
        raise ValueError("dual refit partition checks differ")
    payload = load_exact_safetensors(model_path, _MODEL_KEYS)
    if not t.equal(payload["global_indices"], expected_global_indices):
        raise ValueError("dual refit learned-table global index identity differs")
    if (
        payload["learned_latent"].dtype != t.float32
        or payload["projection_weight"].dtype != t.float32
        or payload["age_direction"].dtype != t.float32
        or payload["global_indices"].dtype != t.int64
    ):
        raise TypeError("dual refit payload dtypes differ")
    learned = payload["learned_latent"]
    projection = payload["projection_weight"]
    age = payload["age_direction"]
    if (
        learned.ndim != 2
        or learned.shape[0] != expected_global_indices.numel()
        or projection.shape != (learned.shape[1], 256)
        or age.shape != (learned.shape[1],)
    ):
        raise ValueError("dual refit payload shapes differ")
    norms = t.linalg.vector_norm(learned, dim=1)
    if not t.allclose(norms, t.ones_like(norms), atol=2.0e-5, rtol=0.0):
        raise ValueError("dual refit learned rows are not normalized")
    if not t.allclose(
        t.linalg.vector_norm(age),
        t.ones((), dtype=t.float32),
        atol=2.0e-5,
        rtol=0.0,
    ):
        raise ValueError("dual refit age direction is not normalized")
    return metadata, payload


def _refit_identity(
    *,
    code_git_commit: str,
    dual_config_sha256: str,
    protocol: DualProbeProtocolConfig,
    bundle: PrimaryDataBundle,
    target_sha256: str,
    split_name: str,
    split_sha256: str,
    window: int,
    parent: DualParentCell,
    split: SplitArtifact,
    selection_path: Path,
    schedule: AlphaSchedule,
    seed: int,
    selected_steps: int,
    device: str,
) -> dict[str, JsonValue]:
    return {
        "code_git_commit": code_git_commit,
        "protocol_id": protocol.protocol_id,
        "protocol_config_sha256": dual_config_sha256,
        "parent_protocol_id": protocol.parent_protocol_id,
        "parent_protocol_sha256": protocol.parent_protocol_sha256,
        "primary_git_commit": bundle.git_commit,
        "data_bundle_sha256": protocol.data_bundle_sha256,
        "target_sha256": target_sha256,
        "split_name": split_name,
        "split_sha256": split_sha256,
        "window_size": window,
        "embedding_manifest_sha256": parent.embedding_manifest_sha256,
        "parent_selection_sha256": parent.selection_sha256,
        "parent_evaluation_sha256": parent.evaluation_sha256,
        "dual_selection_sha256": sha256_file(selection_path),
        "primary_train_indices_sha256": _index_sha256(split.primary_train_indices),
        "alpha_strategy": schedule.name,
        "seed": seed,
        "selected_steps": selected_steps,
        "runtime_device": device,
    }


def _run_refits(
    *,
    code_git_commit: str,
    dual_config_sha256: str,
    protocol: DualProbeProtocolConfig,
    bundle: PrimaryDataBundle,
    target_sha256: str,
    split_name: str,
    split_sha256: str,
    window: int,
    parent: DualParentCell,
    probes: NonEmptyProbeSet,
    embeddings: EmbeddingMatrix,
    targets: TargetGeometry,
    split: SplitArtifact,
    selection: dict[str, object],
    output: Path,
    device: str,
) -> None:
    schedule = _selected_schedule(protocol, selection)
    selection_path = _cell_directory(output, split_name, window) / "selection.json"
    partition = subset_dual_partition(
        probes,
        embeddings,
        targets,
        split.primary_train_indices,
    )
    for seed in protocol.optimization.seeds:
        selected_steps = _integer(_selected_seed_record(selection, seed), "selected_step")
        identity = _refit_identity(
            code_git_commit=code_git_commit,
            dual_config_sha256=dual_config_sha256,
            protocol=protocol,
            bundle=bundle,
            target_sha256=target_sha256,
            split_name=split_name,
            split_sha256=split_sha256,
            window=window,
            parent=parent,
            split=split,
            selection_path=selection_path,
            schedule=schedule,
            seed=seed,
            selected_steps=selected_steps,
            device=device,
        )
        config = _training_config(
            protocol,
            parent,
            schedule,
            seed=seed,
            device=device,
        )
        directory = _refit_directory(output, split_name, window, seed)
        if directory.exists():
            _verify_refit(
                directory,
                identity,
                _training_json(config),
                split.primary_train_indices,
            )
            print(
                f"dual refit reused split={split_name} window={window} seed={seed}",
                flush=True,
            )
            continue
        print(
            f"dual refit start split={split_name} window={window} "
            f"seed={seed} alpha={schedule.name} steps={selected_steps}",
            flush=True,
        )
        initialization = _initialization(
            partition,
            protocol,
            parent,
            seed=seed,
            device=device,
        )
        result = refit_dual_probe(
            partition.probes,
            partition.embeddings,
            partition.targets,
            initialization,
            config=config,
            selected_steps=selected_steps,
        )
        _publish_refit(
            directory=directory,
            identity=identity,
            result=result,
            initialization=initialization,
            config=config,
            partition=partition,
            test_indices=split.test_indices,
        )
        print(
            f"dual refit done split={split_name} window={window} seed={seed}",
            flush=True,
        )


def _metric_json(target: t.Tensor, prediction: t.Tensor) -> dict[str, JsonValue]:
    if (
        target.dtype != t.float64
        or prediction.dtype != t.float64
        or target.ndim != 1
        or target.shape != prediction.shape
        or target.numel() < 2
    ):
        raise TypeError("dual metric vectors must be aligned nontrivial float64 vectors")
    if not bool(t.isfinite(target).all().item() and t.isfinite(prediction).all().item()):
        raise ValueError("dual metric vectors must be finite")
    target_centered = target - target.mean()
    prediction_centered = prediction - prediction.mean()
    target_ss = t.sum(t.square(target_centered))
    prediction_ss = t.sum(t.square(prediction_centered))
    if float(target_ss.item()) == 0.0:
        raise ValueError("dual metric target is constant")
    residual_ss = t.sum(t.square(target - prediction))
    pearson: float | None = None
    if float(prediction_ss.item()) != 0.0:
        pearson = float(
            (
                t.sum(target_centered * prediction_centered) / t.sqrt(target_ss * prediction_ss)
            ).item()
        )
    return {
        "count": target.numel(),
        "target_mean": float(target.mean().item()),
        "target_standard_deviation": float(target.std(unbiased=False).item()),
        "prediction_mean": float(prediction.mean().item()),
        "prediction_standard_deviation": float(prediction.std(unbiased=False).item()),
        "mse": float(t.mean(t.square(target - prediction)).item()),
        "pearson": pearson,
        "pearson_status": "defined" if pearson is not None else "undefined_constant_prediction",
        "r_squared": float((1.0 - residual_ss / target_ss).item()),
    }


def _distance_rows(
    pair_set: CachedPairSet,
    prediction: t.Tensor,
) -> list[dict[str, JsonValue]]:
    if prediction.shape != pair_set.targets.shape:
        raise ValueError("dual distance prediction and cached target axes differ")
    return [
        {
            "distance_class": DISTANCE_CLASS_LABELS[int(distance_class)],
            "distance_class_index": int(distance_class),
            "metrics": _metric_json(
                pair_set.targets[pair_set.pairs.distance_class == int(distance_class)],
                prediction[pair_set.pairs.distance_class == int(distance_class)],
            ),
        }
        for distance_class in DistanceClass
        if bool((pair_set.pairs.distance_class == int(distance_class)).any().item())
    ]


def _target_blind_offsets(count: int, *, seed: int) -> t.Tensor:
    if count <= 0:
        raise ValueError("dual display sampling requires a positive population size")
    generator = t.Generator(device="cpu").manual_seed(seed)
    selected = t.randperm(count, generator=generator)[: min(count, _DISPLAY_LIMIT)]
    return t.sort(selected).values


def _pair_seed_report(
    *,
    uniform: CachedPairSet,
    stratified: CachedPairSet,
    sequence_latent: t.Tensor,
    learned_latent: t.Tensor,
    learned_global_indices: t.Tensor,
    display_offsets: t.Tensor,
) -> tuple[dict[str, JsonValue], dict[str, t.Tensor]]:
    sequence_uniform = gather_pair_predictions_chunked(
        sequence_latent,
        uniform.pairs,
        chunk_size=_PAIR_CHUNK_SIZE,
    ).to(t.float64)
    sequence_stratified = gather_pair_predictions_chunked(
        sequence_latent,
        stratified.pairs,
        chunk_size=_PAIR_CHUNK_SIZE,
    ).to(t.float64)
    report: dict[str, JsonValue] = {
        "population": uniform.pairs.population.value,
        "uniform": {
            "sequence_inductive": _metric_json(uniform.targets, sequence_uniform),
        },
        "by_distance": {
            "sequence_inductive": _distance_rows(stratified, sequence_stratified),
        },
    }
    display: dict[str, t.Tensor] = {
        "sequence_inductive": sequence_uniform.index_select(0, display_offsets),
    }
    if uniform.pairs.population == PairPopulation.SEEN_BY_HELD_OUT:
        hybrid_uniform = hybrid_seen_by_held_out_predictions(
            learned_latent,
            learned_global_indices,
            sequence_latent,
            uniform.pairs,
            chunk_size=_PAIR_CHUNK_SIZE,
        ).to(t.float64)
        hybrid_stratified = hybrid_seen_by_held_out_predictions(
            learned_latent,
            learned_global_indices,
            sequence_latent,
            stratified.pairs,
            chunk_size=_PAIR_CHUNK_SIZE,
        ).to(t.float64)
        uniform_record = cast(dict[str, JsonValue], report["uniform"])
        distance_record = cast(dict[str, JsonValue], report["by_distance"])
        uniform_record["hybrid_learned_seen"] = _metric_json(
            uniform.targets,
            hybrid_uniform,
        )
        distance_record["hybrid_learned_seen"] = _distance_rows(
            stratified,
            hybrid_stratified,
        )
        display["hybrid_learned_seen"] = hybrid_uniform.index_select(
            0,
            display_offsets,
        )
    return report, display


def _distance_reference(cache: EvaluationPairCache) -> dict[str, JsonValue]:
    population_names = (
        (PairSetName.SEEN_UNIFORM, PairSetName.SEEN_STRATIFIED),
        (PairSetName.HELD_OUT_UNIFORM, PairSetName.HELD_OUT_STRATIFIED),
    )
    populations: dict[str, JsonValue] = {}
    for uniform_name, stratified_name in population_names:
        uniform = cache.pair_sets[uniform_name]
        stratified = cache.pair_sets[stratified_name]
        uniform_prediction = cache.distance_baseline.predict(uniform.pairs.distance_class)
        stratified_prediction = cache.distance_baseline.predict(stratified.pairs.distance_class)
        populations[uniform.pairs.population.value] = {
            "uniform": _metric_json(uniform.targets, uniform_prediction),
            "by_distance": _distance_rows(stratified, stratified_prediction),
        }
    return {
        "fit_partition": "complete_primary_train",
        "labels": list(DISTANCE_CLASS_LABELS),
        "means": cache.distance_baseline.means.tolist(),
        "counts": cache.distance_baseline.counts.tolist(),
        "populations": populations,
    }


def _aggregate_metrics(records: tuple[dict[str, object], ...]) -> dict[str, JsonValue]:
    if not records:
        raise ValueError("dual metric aggregation requires seed records")
    fixed_keys = ("count", "target_mean", "target_standard_deviation")
    for key in fixed_keys:
        if any(record.get(key) != records[0].get(key) for record in records[1:]):
            raise ValueError(f"dual seed metric target field differs: {key}")
    numeric_keys = (
        "prediction_mean",
        "prediction_standard_deviation",
        "mse",
        "pearson",
        "r_squared",
    )
    aggregates: dict[str, JsonValue] = {key: cast(JsonValue, records[0][key]) for key in fixed_keys}
    for key in numeric_keys:
        values = tuple(record.get(key) for record in records)
        if any(isinstance(value, bool) or not isinstance(value, int | float) for value in values):
            raise ValueError(f"dual seed model metric is undefined or non-numeric: {key}")
        tensor = t.tensor(
            tuple(float(cast(int | float, value)) for value in values), dtype=t.float64
        )
        aggregates[key] = {
            "mean": float(tensor.mean().item()),
            "standard_deviation": float(tensor.std(unbiased=False).item()),
            "values": tensor.tolist(),
        }
    return aggregates


def _aggregate_seed_results(seed_results: tuple[dict[str, JsonValue], ...]) -> dict[str, JsonValue]:
    age_records = tuple(cast(dict[str, object], result["age_metrics"]) for result in seed_results)
    pair_aggregates: dict[str, JsonValue] = {}
    for population in (
        PairPopulation.SEEN_BY_HELD_OUT.value,
        PairPopulation.HELD_OUT_BY_HELD_OUT.value,
    ):
        population_records = tuple(
            cast(dict[str, object], cast(dict[str, JsonValue], result["pair_metrics"])[population])
            for result in seed_results
        )
        uniform_records = tuple(_object(record, "uniform") for record in population_records)
        modes = tuple(uniform_records[0])
        if any(tuple(record) != modes for record in uniform_records[1:]):
            raise ValueError("dual seed pair modes differ")
        uniform_aggregate = {
            mode: _aggregate_metrics(tuple(_object(record, mode) for record in uniform_records))
            for mode in modes
        }
        distance_records = tuple(_object(record, "by_distance") for record in population_records)
        by_distance: dict[str, JsonValue] = {}
        for mode in modes:
            per_seed_rows = tuple(_list(record, mode) for record in distance_records)
            labels = tuple(
                _string(cast(dict[str, object], row), "distance_class") for row in per_seed_rows[0]
            )
            if any(
                tuple(_string(cast(dict[str, object], row), "distance_class") for row in rows)
                != labels
                for rows in per_seed_rows[1:]
            ):
                raise ValueError("dual seed distance strata differ")
            by_distance[mode] = [
                {
                    "distance_class": label,
                    "metrics": _aggregate_metrics(
                        tuple(
                            _object(cast(dict[str, object], rows[index]), "metrics")
                            for rows in per_seed_rows
                        )
                    ),
                }
                for index, label in enumerate(labels)
            ]
        pair_aggregates[population] = {
            "uniform": uniform_aggregate,
            "by_distance": by_distance,
        }
    alignment_records = tuple(
        cast(dict[str, object], result["train_alignment"]) for result in seed_results
    )
    alignment: dict[str, JsonValue] = {"count": _integer(alignment_records[0], "count")}
    if any(_integer(record, "count") != alignment["count"] for record in alignment_records):
        raise ValueError("dual seed alignment counts differ")
    for key in ("mean", "standard_deviation", "minimum", "maximum"):
        values = t.tensor(
            tuple(_number(record, key) for record in alignment_records),
            dtype=t.float64,
        )
        alignment[key] = {
            "mean": float(values.mean().item()),
            "standard_deviation": float(values.std(unbiased=False).item()),
            "values": values.tolist(),
        }
    return {
        "age_metrics": _aggregate_metrics(age_records),
        "pair_metrics": pair_aggregates,
        "train_alignment": alignment,
    }


def _pair_cache(
    *,
    protocol: DualProbeProtocolConfig,
    bundle: PrimaryDataBundle,
    data: Path,
    experiments: Path,
    probes: NonEmptyProbeSet,
    split_name: str,
    split_sha256: str,
) -> EvaluationPairCache:
    identity = EvaluationPairCacheIdentity(
        protocol_id=protocol.parent_protocol_id,
        protocol_sha256=protocol.parent_protocol_sha256,
        git_commit=bundle.git_commit,
        data_sha256=protocol.data_bundle_sha256,
        target_sha256=sha256_file(data / "targets.safetensors"),
        split_name=split_name,
        split_sha256=split_sha256,
        probe_order_sha256=sha256_ordered_strings(str(probe.probe_id) for probe in probes.probes),
    )
    return load_evaluation_pair_cache(
        experiments / "evaluation-pairs" / split_name,
        identity,
    )


def _validate_parent_evaluation(
    evaluation: dict[str, object],
    *,
    protocol: DualProbeProtocolConfig,
    parent: DualParentCell,
    split_name: str,
    split_sha256: str,
    window: int,
    target_sha256: str,
) -> None:
    identity = _object(evaluation, "identity")
    if (
        evaluation.get("schema") != _PARENT_EVALUATION_SCHEMA
        or _string(identity, "protocol_id") != protocol.parent_protocol_id
        or _string(identity, "protocol_sha256") != protocol.parent_protocol_sha256
        or _string(identity, "data_sha256") != protocol.data_bundle_sha256
        or _string(identity, "target_sha256") != target_sha256
        or _string(identity, "split_name") != split_name
        or _string(identity, "split_sha256") != split_sha256
        or _integer(identity, "window_size") != window
        or _string(identity, "embedding_manifest_sha256") != parent.embedding_manifest_sha256
        or _string(identity, "pair_cache_metadata_sha256") != parent.pair_cache_metadata_sha256
    ):
        raise ValueError("parent evaluation scientific identity differs")


def _parent_metrics(evaluation: dict[str, object]) -> dict[str, JsonValue]:
    age = _object(_object(evaluation, "age_metrics"), "full_latent_metric")
    pair_metrics = _object(evaluation, "pair_metrics")
    pairs: dict[str, JsonValue] = {}
    for population in (
        PairPopulation.SEEN_BY_HELD_OUT.value,
        PairPopulation.HELD_OUT_BY_HELD_OUT.value,
    ):
        record = _object(pair_metrics, population)
        pairs[population] = {
            "uniform_sequence_inductive": cast(
                dict[str, JsonValue], _object(record, "uniform_model_metrics")
            ),
            "by_distance_sequence_inductive": cast(list[JsonValue], _list(record, "by_distance")),
        }
    return {
        "model": "parent_sequence_only_full_latent_metric",
        "age_metrics": cast(dict[str, JsonValue], age),
        "pair_metrics": pairs,
    }


def _evaluate_seed(
    *,
    seed: int,
    payload: dict[str, t.Tensor],
    refit_metadata_sha256: str,
    embeddings: EmbeddingMatrix,
    targets: TargetGeometry,
    split: SplitArtifact,
    cache: EvaluationPairCache,
    display_offsets: dict[PairPopulation, t.Tensor],
    age_display_offsets: t.Tensor,
    device: str,
) -> tuple[dict[str, JsonValue], dict[str, object]]:
    sequence = sequence_latent_from_projection(
        embeddings,
        payload["projection_weight"],
        device=device,
        row_chunk_size=_PROJECTION_ROW_CHUNK_SIZE,
    )
    learned = payload["learned_latent"]
    learned_indices = payload["global_indices"]
    alignment = t.sum(
        learned * sequence.index_select(0, learned_indices),
        dim=1,
    ).to(t.float64)
    age_prediction = held_out_age_predictions(
        sequence,
        payload["age_direction"],
        split.test_indices,
    ).to(t.float64)
    age_target = targets.rho.tensor.index_select(0, split.test_indices)
    pair_reports: dict[str, JsonValue] = {}
    pair_displays: dict[str, dict[str, t.Tensor]] = {}
    pair_names = (
        (
            PairPopulation.SEEN_BY_HELD_OUT,
            PairSetName.SEEN_UNIFORM,
            PairSetName.SEEN_STRATIFIED,
        ),
        (
            PairPopulation.HELD_OUT_BY_HELD_OUT,
            PairSetName.HELD_OUT_UNIFORM,
            PairSetName.HELD_OUT_STRATIFIED,
        ),
    )
    for population, uniform_name, stratified_name in pair_names:
        report, display = _pair_seed_report(
            uniform=cache.pair_sets[uniform_name],
            stratified=cache.pair_sets[stratified_name],
            sequence_latent=sequence,
            learned_latent=learned,
            learned_global_indices=learned_indices,
            display_offsets=display_offsets[population],
        )
        pair_reports[population.value] = report
        pair_displays[population.value] = display
    return (
        {
            "seed": seed,
            "refit_metadata_sha256": refit_metadata_sha256,
            "age_metrics": _metric_json(age_target, age_prediction),
            "train_alignment": {
                "count": alignment.numel(),
                "mean": float(alignment.mean().item()),
                "standard_deviation": float(alignment.std(unbiased=False).item()),
                "minimum": float(alignment.min().item()),
                "maximum": float(alignment.max().item()),
            },
            "pair_metrics": pair_reports,
        },
        {
            "age_prediction": age_prediction.index_select(0, age_display_offsets),
            "pair_predictions": pair_displays,
        },
    )


def _display_json(
    *,
    seeds: tuple[int, ...],
    seed_displays: tuple[dict[str, object], ...],
    split: SplitArtifact,
    targets: TargetGeometry,
    cache: EvaluationPairCache,
    display_offsets: dict[PairPopulation, t.Tensor],
    age_display_offsets: t.Tensor,
) -> dict[str, JsonValue]:
    age_global = split.test_indices.index_select(0, age_display_offsets)
    age_predictions = tuple(cast(t.Tensor, display["age_prediction"]) for display in seed_displays)
    pairs: dict[str, JsonValue] = {}
    population_names = (
        (PairPopulation.SEEN_BY_HELD_OUT, PairSetName.SEEN_UNIFORM),
        (PairPopulation.HELD_OUT_BY_HELD_OUT, PairSetName.HELD_OUT_UNIFORM),
    )
    for population, pair_name in population_names:
        pair_set = cache.pair_sets[pair_name]
        offsets = display_offsets[population]
        modes = tuple(
            cast(
                dict[str, t.Tensor],
                cast(dict[str, object], seed_displays[0]["pair_predictions"])[population.value],
            )
        )
        predictions_by_mode: dict[str, JsonValue] = {}
        for mode in modes:
            predictions = tuple(
                cast(
                    dict[str, t.Tensor],
                    cast(dict[str, object], display["pair_predictions"])[population.value],
                )[mode]
                for display in seed_displays
            )
            stacked = t.stack(predictions)
            predictions_by_mode[mode] = {
                "by_seed": [
                    {"seed": seed, "prediction": prediction.tolist()}
                    for seed, prediction in zip(seeds, predictions, strict=True)
                ],
                "mean_prediction": stacked.mean(dim=0).tolist(),
            }
        pairs[population.value] = {
            "sampling_seed": 910_000 + int(population == PairPopulation.HELD_OUT_BY_HELD_OUT),
            "source": pair_name.value,
            "pair_offsets": offsets.tolist(),
            "left_global_indices": pair_set.pairs.left.index_select(0, offsets).tolist(),
            "right_global_indices": pair_set.pairs.right.index_select(0, offsets).tolist(),
            "distance_class": pair_set.pairs.distance_class.index_select(0, offsets).tolist(),
            "target": pair_set.targets.index_select(0, offsets).tolist(),
            "predictions": predictions_by_mode,
        }
    age_stack = t.stack(age_predictions)
    return {
        "sampling": "target_blind_uniform_offset_sample_before_target_or_prediction_access",
        "maximum_points": _DISPLAY_LIMIT,
        "age": {
            "sampling_seed": 920_000,
            "test_offsets": age_display_offsets.tolist(),
            "global_indices": age_global.tolist(),
            "target": targets.rho.tensor.index_select(0, age_global).tolist(),
            "by_seed": [
                {"seed": seed, "prediction": prediction.tolist()}
                for seed, prediction in zip(seeds, age_predictions, strict=True)
            ],
            "mean_prediction": age_stack.mean(dim=0).tolist(),
        },
        "pairs": pairs,
    }


def _results_identity(
    *,
    code_git_commit: str,
    dual_config_sha256: str,
    protocol: DualProbeProtocolConfig,
    bundle: PrimaryDataBundle,
    target_sha256: str,
    split_name: str,
    split_sha256: str,
    window: int,
    parent: DualParentCell,
    split: SplitArtifact,
    selection_path: Path,
    refit_hashes: tuple[str, ...],
    device: str,
) -> dict[str, JsonValue]:
    return {
        "code_git_commit": code_git_commit,
        "protocol_id": protocol.protocol_id,
        "protocol_config_sha256": dual_config_sha256,
        "parent_protocol_id": protocol.parent_protocol_id,
        "parent_protocol_sha256": protocol.parent_protocol_sha256,
        "primary_git_commit": bundle.git_commit,
        "data_bundle_sha256": protocol.data_bundle_sha256,
        "target_sha256": target_sha256,
        "split_name": split_name,
        "split_sha256": split_sha256,
        "window_size": window,
        "embedding_manifest_sha256": parent.embedding_manifest_sha256,
        "pair_cache_metadata_sha256": parent.pair_cache_metadata_sha256,
        "parent_selection_sha256": parent.selection_sha256,
        "parent_evaluation_sha256": parent.evaluation_sha256,
        "dual_selection_sha256": sha256_file(selection_path),
        "refit_metadata_sha256": list(refit_hashes),
        "primary_train_indices_sha256": _index_sha256(split.primary_train_indices),
        "test_indices_sha256": _index_sha256(split.test_indices),
        "runtime_device": device,
    }


def _validate_results(
    record: dict[str, object],
    expected_identity: dict[str, JsonValue],
    selected_strategy: str,
) -> None:
    expected_keys = {
        "schema",
        "status",
        "interpretation",
        "identity",
        "selected_alpha_strategy",
        "test_evaluated_strategy_count",
        "seeds",
        "seed_results",
        "aggregates",
        "distance_reference",
        "parent_comparison",
        "display",
    }
    if (
        set(record) != expected_keys
        or record.get("schema") != _RESULTS_SCHEMA
        or record.get("identity") != expected_identity
        or record.get("selected_alpha_strategy") != selected_strategy
        or record.get("test_evaluated_strategy_count") != 1
    ):
        raise ValueError("dual results identity or selected-only test policy differs")
    seed_results = _list(record, "seed_results")
    if any(
        not isinstance(seed, dict) or "alpha_strategy" in seed or "candidate" in seed
        for seed in seed_results
    ):
        raise ValueError("dual test results contain an unselected candidate field")


def _run_evaluation_cell(
    *,
    code_git_commit: str,
    dual_config_sha256: str,
    protocol: DualProbeProtocolConfig,
    bundle: PrimaryDataBundle,
    data: Path,
    experiments: Path,
    target_sha256: str,
    split_name: str,
    split_sha256: str,
    window: int,
    parent: DualParentCell,
    parent_evaluation: dict[str, object],
    probes: NonEmptyProbeSet,
    embeddings: EmbeddingMatrix,
    targets: TargetGeometry,
    split: SplitArtifact,
    selection: dict[str, object],
    output: Path,
    device: str,
) -> None:
    selected_strategy = _string(selection, "selected_alpha_strategy")
    selection_path = _cell_directory(output, split_name, window) / "selection.json"
    schedule = _selected_schedule(protocol, selection)
    refit_metadata: list[dict[str, object]] = []
    refit_payloads: list[dict[str, t.Tensor]] = []
    refit_hashes: list[str] = []
    for seed in protocol.optimization.seeds:
        selected_steps = _integer(_selected_seed_record(selection, seed), "selected_step")
        identity = _refit_identity(
            code_git_commit=code_git_commit,
            dual_config_sha256=dual_config_sha256,
            protocol=protocol,
            bundle=bundle,
            target_sha256=target_sha256,
            split_name=split_name,
            split_sha256=split_sha256,
            window=window,
            parent=parent,
            split=split,
            selection_path=selection_path,
            schedule=schedule,
            seed=seed,
            selected_steps=selected_steps,
            device=device,
        )
        expected_training = _training_json(
            _training_config(
                protocol,
                parent,
                schedule,
                seed=seed,
                device=device,
            )
        )
        directory = _refit_directory(output, split_name, window, seed)
        metadata, payload = _verify_refit(
            directory,
            identity,
            expected_training,
            split.primary_train_indices,
        )
        if bool(t.isin(payload["global_indices"], split.test_indices).any().item()):
            raise ValueError("dual evaluation found test rows in the learned table")
        refit_metadata.append(metadata)
        refit_payloads.append(payload)
        refit_hashes.append(sha256_file(directory / "metadata.json"))
    identity = _results_identity(
        code_git_commit=code_git_commit,
        dual_config_sha256=dual_config_sha256,
        protocol=protocol,
        bundle=bundle,
        target_sha256=target_sha256,
        split_name=split_name,
        split_sha256=split_sha256,
        window=window,
        parent=parent,
        split=split,
        selection_path=selection_path,
        refit_hashes=tuple(refit_hashes),
        device=device,
    )
    results_path = _cell_directory(output, split_name, window) / "results.json"
    if results_path.exists():
        record = _load_json(results_path)
        _validate_results(record, identity, selected_strategy)
        print(
            f"dual evaluation reused split={split_name} window={window} alpha={selected_strategy}",
            flush=True,
        )
        return
    cache = _pair_cache(
        protocol=protocol,
        bundle=bundle,
        data=data,
        experiments=experiments,
        probes=probes,
        split_name=split_name,
        split_sha256=split_sha256,
    )
    distance_reference = _distance_reference(cache)
    parent_distance = _object(parent_evaluation, "distance_baseline")
    if (
        parent_distance.get("means") != distance_reference["means"]
        or parent_distance.get("counts") != distance_reference["counts"]
    ):
        raise ValueError("dual distance reference differs from parent evaluation")
    display_offsets = {
        PairPopulation.SEEN_BY_HELD_OUT: _target_blind_offsets(
            cache.pair_sets[PairSetName.SEEN_UNIFORM].pairs.count,
            seed=910_000,
        ),
        PairPopulation.HELD_OUT_BY_HELD_OUT: _target_blind_offsets(
            cache.pair_sets[PairSetName.HELD_OUT_UNIFORM].pairs.count,
            seed=910_001,
        ),
    }
    age_display_offsets = _target_blind_offsets(
        split.test_indices.numel(),
        seed=920_000,
    )
    seed_results: list[dict[str, JsonValue]] = []
    seed_displays: list[dict[str, object]] = []
    for seed, metadata, payload, refit_hash in zip(
        protocol.optimization.seeds,
        refit_metadata,
        refit_payloads,
        refit_hashes,
        strict=True,
    ):
        print(
            f"dual evaluation seed start split={split_name} window={window} seed={seed}",
            flush=True,
        )
        if _integer(_object(metadata, "identity"), "seed") != seed:
            raise ValueError("dual refit seed identity differs before evaluation")
        seed_result, seed_display = _evaluate_seed(
            seed=seed,
            payload=payload,
            refit_metadata_sha256=refit_hash,
            embeddings=embeddings,
            targets=targets,
            split=split,
            cache=cache,
            display_offsets=display_offsets,
            age_display_offsets=age_display_offsets,
            device=device,
        )
        seed_results.append(seed_result)
        seed_displays.append(seed_display)
    result_tuple = tuple(seed_results)
    record: dict[str, JsonValue] = {
        "schema": _RESULTS_SCHEMA,
        "status": "post_hoc_hypothesis_generating_selected_only_test_evaluation",
        "interpretation": (
            "Hybrid learned-seen by held-out metrics may use complete-train free probe rows; "
            "only sequence-inductive held-out by held-out metrics support new-site geometry claims."
        ),
        "identity": identity,
        "selected_alpha_strategy": selected_strategy,
        "test_evaluated_strategy_count": 1,
        "seeds": list(protocol.optimization.seeds),
        "seed_results": seed_results,
        "aggregates": _aggregate_seed_results(result_tuple),
        "distance_reference": distance_reference,
        "parent_comparison": _parent_metrics(parent_evaluation),
        "display": _display_json(
            seeds=protocol.optimization.seeds,
            seed_displays=tuple(seed_displays),
            split=split,
            targets=targets,
            cache=cache,
            display_offsets=display_offsets,
            age_display_offsets=age_display_offsets,
        ),
    }
    _validate_results(cast(dict[str, object], record), identity, selected_strategy)
    write_canonical_json_exclusive(results_path, record)
    print(
        f"dual evaluation done split={split_name} window={window} "
        f"alpha={selected_strategy} output={results_path}",
        flush=True,
    )


def _publish_manifest(
    *,
    output: Path,
    protocol: DualProbeProtocolConfig,
    dual_config_sha256: str,
    code_git_commit: str,
) -> None:
    cells = tuple(
        (split_name, window, _cell_directory(output, split_name, window) / "results.json")
        for split_name in protocol.splits
        for window in map(int, protocol.windows)
    )
    if any(not path.is_file() for _, _, path in cells):
        return
    record: dict[str, JsonValue] = {
        "schema": protocol.manifest_schema,
        "protocol_id": protocol.protocol_id,
        "protocol_config_sha256": dual_config_sha256,
        "code_git_commit": code_git_commit,
        "cells": [
            {
                "split_name": split_name,
                "window_size": window,
                "results_file": path.relative_to(output).as_posix(),
                "results_sha256": sha256_file(path),
            }
            for split_name, window, path in cells
        ],
    }
    _publish_or_verify_json(output / "manifest.json", record)


def main() -> None:
    arguments = _parser().parse_args()
    _configure_runtime(arguments.device)
    repository = Path(__file__).resolve().parents[1]
    code_git_commit = require_clean_git_commit(repository)
    protocol = load_dual_probe_protocol(arguments.config)
    dual_config_sha256 = sha256_file(arguments.config)
    if sha256_file(arguments.parent_config) != protocol.parent_protocol_sha256:
        raise ValueError("dual parent protocol configuration hash differs")
    parent_protocol = load_protocol_config(arguments.parent_config)
    if parent_protocol.protocol_id != protocol.parent_protocol_id:
        raise ValueError("dual parent protocol ID differs from its configuration")
    if sha256_file(arguments.data / "bundle.json") != protocol.data_bundle_sha256:
        raise ValueError("dual sealed data-bundle hash differs")
    bundle = verify_primary_data_bundle(
        arguments.data,
        protocol_id=protocol.parent_protocol_id,
        protocol_sha256=protocol.parent_protocol_sha256,
    )
    probes = load_probe_table(arguments.data / "probes.tsv")
    targets = load_target_geometry(arguments.data / "targets.safetensors")
    if (
        len(probes) != bundle.retained_probes
        or targets.methylation.n_rows != len(probes)
        or targets.methylation.n_samples != bundle.retained_samples
    ):
        raise ValueError("dual sealed data dimensions differ")
    target_sha256 = sha256_file(arguments.data / "targets.safetensors")
    splits = _requested_splits(arguments.splits, protocol)
    windows = _requested_windows(arguments.windows, protocol)
    for split_name in splits:
        split = _load_split(arguments.data, probes, split_name)
        split_sha256 = _split_sha256(arguments.data, split_name)
        for window in windows:
            parent = protocol.parent(split_name, window)
            _, parent_evaluation = _verify_parent_cell(
                parent=parent,
                experiments=arguments.experiments,
                embeddings=arguments.embeddings,
            )
            _validate_parent_evaluation(
                parent_evaluation,
                protocol=protocol,
                parent=parent,
                split_name=split_name,
                split_sha256=split_sha256,
                window=window,
                target_sha256=target_sha256,
            )
            embeddings = load_embedding_cache(
                arguments.embeddings / f"window-{window}",
                probes,
                window_size=parent.window_size,
                expected_git_commit=bundle.git_commit,
            )
            if arguments.phase == "audit":
                print(
                    f"dual audit passed split={split_name} window={window}",
                    flush=True,
                )
                continue
            if arguments.phase in {"tune", "all"}:
                _run_tuning_cell(
                    code_git_commit=code_git_commit,
                    dual_config_sha256=dual_config_sha256,
                    protocol=protocol,
                    bundle=bundle,
                    target_sha256=target_sha256,
                    split_name=split_name,
                    split_sha256=split_sha256,
                    window=window,
                    parent=parent,
                    probes=probes,
                    embeddings=embeddings,
                    targets=targets,
                    split=split,
                    output=arguments.output,
                    device=arguments.device,
                )
            selection = _select_tuning_cell(
                code_git_commit=code_git_commit,
                dual_config_sha256=dual_config_sha256,
                protocol=protocol,
                bundle=bundle,
                target_sha256=target_sha256,
                split_name=split_name,
                split_sha256=split_sha256,
                window=window,
                parent=parent,
                split=split,
                output=arguments.output,
                device=arguments.device,
            )
            if arguments.phase in {"evaluate", "all"}:
                _run_refits(
                    code_git_commit=code_git_commit,
                    dual_config_sha256=dual_config_sha256,
                    protocol=protocol,
                    bundle=bundle,
                    target_sha256=target_sha256,
                    split_name=split_name,
                    split_sha256=split_sha256,
                    window=window,
                    parent=parent,
                    probes=probes,
                    embeddings=embeddings,
                    targets=targets,
                    split=split,
                    selection=selection,
                    output=arguments.output,
                    device=arguments.device,
                )
                _run_evaluation_cell(
                    code_git_commit=code_git_commit,
                    dual_config_sha256=dual_config_sha256,
                    protocol=protocol,
                    bundle=bundle,
                    data=arguments.data,
                    experiments=arguments.experiments,
                    target_sha256=target_sha256,
                    split_name=split_name,
                    split_sha256=split_sha256,
                    window=window,
                    parent=parent,
                    parent_evaluation=parent_evaluation,
                    probes=probes,
                    embeddings=embeddings,
                    targets=targets,
                    split=split,
                    selection=selection,
                    output=arguments.output,
                    device=arguments.device,
                )
    if arguments.phase in {"evaluate", "all"}:
        _publish_manifest(
            output=arguments.output,
            protocol=protocol,
            dual_config_sha256=dual_config_sha256,
            code_git_commit=code_git_commit,
        )


if __name__ == "__main__":
    main()
