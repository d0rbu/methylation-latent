"""Restart-safe execution of the frozen baseline and latent-model stages."""

from __future__ import annotations

import argparse
import json
import os
import secrets
from dataclasses import asdict, dataclass
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
from methylation_latent.baselines import fit_sequence_age_baseline
from methylation_latent.cohort import load_probe_table
from methylation_latent.config import ProtocolConfig, load_protocol_config
from methylation_latent.data_bundle import verify_primary_data_bundle
from methylation_latent.domain import (
    LatentDimension,
    NonEmptyProbeSet,
    NonNegativeWeight,
    parse_latent_dimension,
    parse_non_negative_weight,
)
from methylation_latent.embedding_cache import load_embedding_cache
from methylation_latent.evaluation import (
    DISTANCE_CLASS_LABELS,
    DistanceClass,
    PairPopulation,
    RegressionMetrics,
    fit_projection_2d,
    gather_pair_predictions_chunked,
    metrics_by_distance,
    regression_metrics,
)
from methylation_latent.evaluation_cache import (
    CachedPairSet,
    EvaluationPairCache,
    EvaluationPairCacheIdentity,
    PairSetName,
    build_evaluation_pair_cache,
    load_evaluation_pair_cache,
    save_evaluation_pair_cache_exclusive,
)
from methylation_latent.experiment_data import (
    SequenceFeatureArtifact,
    SplitArtifact,
    load_sequence_features,
    load_split_artifact,
)
from methylation_latent.model import LatentMetric
from methylation_latent.storage import (
    EmbeddingMatrix,
    load_exact_safetensors,
    load_target_geometry,
    save_safetensors_exclusive,
)
from methylation_latent.targets import (
    CorrelationVector,
    TargetGeometry,
    UnitNormRows,
)
from methylation_latent.training import (
    TrainingConfig,
    TrainingMode,
    ValidationData,
    refit_latent_metric,
    train_latent_metric,
)

_SEQUENCE_SCHEMA = "methylation-latent.sequence-age-baseline.v2"
_TUNING_SCHEMA = "methylation-latent.latent-tuning-run.v2"
_SELECTION_SCHEMA = "methylation-latent.hyperparameter-selection.v2"
_FINAL_SCHEMA = "methylation-latent.final-refit.v2"
_EVALUATION_SCHEMA = "methylation-latent.held-out-evaluation.v2"
_MODEL_KEYS = {"age_direction", "projection.weight"}
_SPLIT_FILES = (
    ("diverse-blocks", "diverse_blocks"),
    ("held-out-chromosome", "held_out_chromosome"),
)


@dataclass(frozen=True, slots=True)
class ExperimentContext:
    git_commit: str
    config: ProtocolConfig
    protocol_sha256: str
    data_directory: Path
    embedding_directory: Path
    output_directory: Path
    probes: NonEmptyProbeSet
    targets: TargetGeometry
    features: SequenceFeatureArtifact
    splits: dict[str, SplitArtifact]
    data_sha256: str
    target_sha256: str
    split_sha256: dict[str, str]
    retained_sample_count: int


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("sequence", "pairs", "age", "full", "evaluate", "all"),
        required=True,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def _load_context(arguments: argparse.Namespace) -> ExperimentContext:
    git_commit = require_clean_git_commit(Path(__file__).resolve().parents[1])
    config = load_protocol_config(arguments.config)
    if config.status != "frozen":
        raise ValueError("experiment execution requires a frozen protocol")
    protocol_sha256 = sha256_file(arguments.config)
    bundle = verify_primary_data_bundle(
        arguments.data,
        protocol_id=config.protocol_id,
        protocol_sha256=protocol_sha256,
    )
    probes = load_probe_table(arguments.data / "probes.tsv")
    targets = load_target_geometry(arguments.data / "targets.safetensors")
    if len(probes) != targets.methylation.n_rows:
        raise ValueError("probe table and target geometry axes differ")
    features = load_sequence_features(
        arguments.data / "sequence-features.safetensors",
        arguments.data / "sequence-features.json",
        probes=probes,
    )
    splits = {
        split_name: load_split_artifact(
            arguments.data / "splits" / f"{split_name}.safetensors",
            arguments.data / "splits" / f"{split_name}.json",
            probes=probes,
        )
        for split_name, _ in _SPLIT_FILES
    }
    split_hashes = {
        split_name: sha256_ordered_strings(
            (
                sha256_file(arguments.data / "splits" / f"{split_name}.json"),
                sha256_file(arguments.data / "splits" / f"{split_name}.safetensors"),
            )
        )
        for split_name in splits
    }
    if len(probes) != bundle.retained_probes:
        raise ValueError("sealed bundle and loaded probe counts differ")
    if targets.methylation.n_samples != bundle.retained_samples:
        raise ValueError("sealed bundle and loaded sample counts differ")
    return ExperimentContext(
        git_commit=git_commit,
        config=config,
        protocol_sha256=protocol_sha256,
        data_directory=arguments.data,
        embedding_directory=arguments.embeddings,
        output_directory=arguments.output,
        probes=probes,
        targets=targets,
        features=features,
        splits=splits,
        data_sha256=sha256_file(arguments.data / "bundle.json"),
        target_sha256=sha256_file(arguments.data / "targets.safetensors"),
        split_sha256=split_hashes,
        retained_sample_count=bundle.retained_samples,
    )


def _subset_probes(
    probes: NonEmptyProbeSet,
    indices: t.Tensor,
) -> NonEmptyProbeSet:
    if indices.dtype != t.int64 or indices.ndim != 1 or indices.numel() == 0:
        raise ValueError("probe-subset indices must be a non-empty int64 vector")
    if bool(t.any((indices < 0) | (indices >= len(probes))).item()):
        raise IndexError("probe-subset index is outside the global universe")
    if t.unique(indices).numel() != indices.numel():
        raise ValueError("probe-subset indices contain duplicates")
    return NonEmptyProbeSet(tuple(probes.probes[index] for index in indices.tolist()))


def _subset_embeddings(
    embeddings: EmbeddingMatrix,
    indices: t.Tensor,
) -> EmbeddingMatrix:
    return EmbeddingMatrix(embeddings.tensor.index_select(0, indices))


def _subset_targets(
    targets: TargetGeometry,
    indices: t.Tensor,
) -> TargetGeometry:
    return TargetGeometry(
        methylation=UnitNormRows(targets.methylation.tensor.index_select(0, indices)),
        age=targets.age,
        rho=CorrelationVector(targets.rho.tensor.index_select(0, indices)),
    )


def _identity(
    context: ExperimentContext,
    split_name: str,
    *,
    window_size: int,
    embedding_sha256: str | None,
) -> dict[str, JsonValue]:
    return {
        "git_commit": context.git_commit,
        "protocol_id": context.config.protocol_id,
        "protocol_sha256": context.protocol_sha256,
        "data_sha256": context.data_sha256,
        "target_sha256": context.target_sha256,
        "split_name": split_name,
        "split_sha256": context.split_sha256[split_name],
        "window_size": window_size,
        "embedding_manifest_sha256": embedding_sha256,
    }


def _metric_json(values: RegressionMetrics) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], asdict(values))


def _load_json(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"experiment record must be a JSON object: {path}")
    return cast(dict[str, object], raw)


def _required_object(record: dict[str, object], key: str) -> dict[str, object]:
    value = record.get(key)
    if not isinstance(value, dict) or any(not isinstance(name, str) for name in value):
        raise TypeError(f"{key} must be a JSON object with string keys")
    return cast(dict[str, object], value)


def _required_int(record: dict[str, object], key: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be a JSON integer")
    return value


def _required_float(record: dict[str, object], key: str) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{key} must be a JSON number")
    parsed = float(value)
    if not bool(t.isfinite(t.tensor(parsed)).item()):
        raise ValueError(f"{key} must be finite")
    return parsed


def _required_str(record: dict[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a JSON string")
    return value


def _require_record(
    path: Path,
    *,
    schema: str,
    identity: dict[str, JsonValue],
) -> dict[str, object]:
    raw = _load_json(path)
    if raw.get("schema") != schema or raw.get("identity") != identity:
        raise ValueError(f"experiment record identity differs: {path}")
    return raw


def _sequence_record_path(
    context: ExperimentContext,
    split_name: str,
    window_size: int,
) -> Path:
    return (
        context.output_directory
        / "baselines"
        / "sequence"
        / split_name
        / f"window-{window_size}.json"
    )


def run_sequence_baselines(context: ExperimentContext) -> None:
    """Fit immutable train-only CpG-density/GC baselines before neural stages."""

    for split_name, _ in _SPLIT_FILES:
        split = context.splits[split_name]
        train_indices = split.primary_train_indices
        test_indices = split.test_indices
        for window in context.config.splits.window_sizes:
            width = int(window)
            identity = _identity(
                context,
                split_name,
                window_size=width,
                embedding_sha256=None,
            )
            path = _sequence_record_path(context, split_name, width)
            if path.is_file():
                _require_record(path, schema=_SEQUENCE_SCHEMA, identity=identity)
                continue
            features = context.features.by_window[width]
            fit = fit_sequence_age_baseline(
                features.index_select(0, train_indices),
                CorrelationVector(context.targets.rho.tensor.index_select(0, train_indices)),
            )
            prediction = fit.predict(features.index_select(0, test_indices))
            target = context.targets.rho.tensor.index_select(0, test_indices)
            metrics = regression_metrics(target, prediction)
            write_canonical_json_exclusive(
                path,
                {
                    "schema": _SEQUENCE_SCHEMA,
                    "identity": identity,
                    "fit_partition": "complete_primary_train",
                    "train_probe_count": train_indices.numel(),
                    "test_probe_count": test_indices.numel(),
                    "columns": ["intercept", "cpg_density", "gc_content"],
                    "coefficients": fit.coefficients.tolist(),
                    "held_out_metrics": _metric_json(metrics),
                },
            )


def _pair_cache_identity(
    context: ExperimentContext,
    split_name: str,
) -> EvaluationPairCacheIdentity:
    return EvaluationPairCacheIdentity(
        protocol_id=context.config.protocol_id,
        protocol_sha256=context.protocol_sha256,
        git_commit=context.git_commit,
        data_sha256=context.data_sha256,
        target_sha256=context.target_sha256,
        split_name=split_name,
        split_sha256=context.split_sha256[split_name],
        probe_order_sha256=sha256_ordered_strings(
            str(probe.probe_id) for probe in context.probes.probes
        ),
    )


def run_pair_caches(context: ExperimentContext) -> None:
    """Build immutable evaluation populations before any model is evaluated."""

    for split_name, _ in _SPLIT_FILES:
        directory = context.output_directory / "evaluation-pairs" / split_name
        identity = _pair_cache_identity(context, split_name)
        if directory.is_dir():
            load_evaluation_pair_cache(directory, identity)
            continue
        if directory.exists():
            raise FileExistsError(directory)
        split = context.splits[split_name]
        cache = build_evaluation_pair_cache(
            context.probes,
            context.targets.methylation,
            split.primary_train_indices,
            split.test_indices,
            maximum_uniform_pairs=int(context.config.sweep.maximum_uniform_evaluation_pairs),
            maximum_pairs_per_distance_class=int(
                context.config.sweep.maximum_pairs_per_distance_class
            ),
            uniform_seed=context.config.sweep.uniform_evaluation_seed,
            distance_seed=context.config.sweep.distance_evaluation_seed,
            target_chunk_size=4_096,
        )
        save_evaluation_pair_cache_exclusive(directory, cache, identity)
        load_evaluation_pair_cache(directory, identity)


def _embedding_manifest_path(
    context: ExperimentContext,
    window_size: int,
) -> Path:
    return context.embedding_directory / f"window-{window_size}" / "manifest.json"


def _training_json(config: TrainingConfig) -> dict[str, JsonValue]:
    return {
        "mode": config.mode.value,
        "latent_dimension": int(config.latent_dimension),
        "lambda_age": float(config.lambda_age),
        "batch_size": int(config.batch_size),
        "neighbourhood_width": int(config.neighbourhood_width),
        "steps": int(config.steps),
        "validation_interval": int(config.validation_interval),
        "validation_pair_chunk_size": int(config.validation_pair_chunk_size),
        "learning_rate": config.learning_rate,
        "seed": config.seed,
        "device": config.device,
        "weight_decay": 0.0,
    }


def _training_config(
    context: ExperimentContext,
    *,
    mode: TrainingMode,
    latent_dimension: LatentDimension,
    lambda_age: NonNegativeWeight,
    device: str,
) -> TrainingConfig:
    return TrainingConfig(
        mode=mode,
        latent_dimension=latent_dimension,
        lambda_age=lambda_age,
        batch_size=context.config.training.batch_size,
        neighbourhood_width=context.config.training.neighbourhood_width,
        steps=context.config.training.tuning_steps,
        validation_interval=context.config.training.validation_interval,
        validation_pair_chunk_size=(context.config.training.validation_pair_chunk_size),
        learning_rate=context.config.training.learning_rate,
        seed=context.config.training.training_seed,
        device=device,
    )


def _run_directory(
    context: ExperimentContext,
    *,
    mode: TrainingMode,
    split_name: str,
    window_size: int,
    latent_dimension: int,
    lambda_age: float,
) -> Path:
    lambda_component = "" if mode == TrainingMode.AGE_ONLY else f"-lambda-{lambda_age:g}"
    return (
        context.output_directory
        / "tuning"
        / mode.value
        / split_name
        / f"window-{window_size}"
        / f"d-{latent_dimension}{lambda_component}"
    )


def _validation_record_for_step(
    validation_history: list[dict[str, JsonValue]],
    selected_step: int,
) -> dict[str, JsonValue]:
    selected = tuple(record for record in validation_history if record["step"] == selected_step)
    if len(selected) != 1:
        raise RuntimeError("selected step is absent or duplicate in validation history")
    return selected[0]


def _load_completed_model_record(
    directory: Path,
    *,
    schema: str,
    identity: dict[str, JsonValue],
    training: dict[str, JsonValue],
) -> dict[str, object]:
    metadata = _require_record(
        directory / "metadata.json",
        schema=schema,
        identity=identity,
    )
    if metadata.get("training") != training:
        raise ValueError(f"model training configuration differs: {directory}")
    model_path = directory / "model.safetensors"
    if metadata.get("model_sha256") != sha256_file(model_path):
        raise ValueError(f"model checkpoint fingerprint differs: {directory}")
    load_exact_safetensors(model_path, _MODEL_KEYS)
    return metadata


def _run_tuning_model(
    context: ExperimentContext,
    *,
    split_name: str,
    window_size: int,
    embeddings: EmbeddingMatrix,
    embedding_sha256: str,
    mode: TrainingMode,
    latent_dimension: LatentDimension,
    lambda_age: NonNegativeWeight,
    device: str,
    prerequisite_sha256: dict[str, JsonValue],
) -> tuple[Path, dict[str, object]]:
    split = context.splits[split_name]
    training_config = _training_config(
        context,
        mode=mode,
        latent_dimension=latent_dimension,
        lambda_age=lambda_age,
        device=device,
    )
    identity = {
        **_identity(
            context,
            split_name,
            window_size=window_size,
            embedding_sha256=embedding_sha256,
        ),
        "prerequisite_sha256": prerequisite_sha256,
        "optimization_indices_sha256": sha256_ordered_strings(
            map(str, split.optimization_indices.tolist())
        ),
        "validation_indices_sha256": sha256_ordered_strings(
            map(str, split.validation_indices.tolist())
        ),
    }
    final = _run_directory(
        context,
        mode=mode,
        split_name=split_name,
        window_size=window_size,
        latent_dimension=int(training_config.latent_dimension),
        lambda_age=float(training_config.lambda_age),
    )
    training_json = _training_json(training_config)
    if final.is_dir():
        return final, _load_completed_model_record(
            final,
            schema=_TUNING_SCHEMA,
            identity=identity,
            training=training_json,
        )
    if final.exists():
        raise FileExistsError(final)
    optimization = split.optimization_indices
    validation = split.validation_indices
    trained = train_latent_metric(
        _subset_probes(context.probes, optimization),
        _subset_embeddings(embeddings, optimization),
        _subset_targets(context.targets, optimization),
        ValidationData(
            probes=_subset_probes(context.probes, validation),
            embeddings=_subset_embeddings(embeddings, validation),
            targets=_subset_targets(context.targets, validation),
        ),
        config=training_config,
    )
    training_history = [cast(dict[str, JsonValue], asdict(record)) for record in trained.history]
    validation_history = [
        cast(dict[str, JsonValue], asdict(record)) for record in trained.validation_history
    ]
    selected_validation = _validation_record_for_step(
        validation_history,
        trained.selected_step,
    )
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = final.parent / f".{final.name}.{secrets.token_hex(16)}.tmp"
    temporary.mkdir()
    model_path = temporary / "model.safetensors"
    trained.model.to("cpu")
    save_safetensors_exclusive(
        model_path,
        {name: value.detach().cpu() for name, value in trained.model.state_dict().items()},
    )
    metadata: dict[str, JsonValue] = {
        "schema": _TUNING_SCHEMA,
        "identity": identity,
        "training": training_json,
        "selected_step": trained.selected_step,
        "selected_validation": selected_validation,
        "training_history": training_history,
        "validation_history": validation_history,
        "model_file": model_path.name,
        "model_sha256": sha256_file(model_path),
    }
    write_canonical_json_exclusive(temporary / "metadata.json", metadata)
    os.rename(temporary, final)
    return final, cast(dict[str, object], metadata)


def _selection_path(
    context: ExperimentContext,
    *,
    mode: TrainingMode,
    split_name: str,
    window_size: int,
) -> Path:
    return (
        context.output_directory
        / "selections"
        / mode.value
        / split_name
        / f"window-{window_size}.json"
    )


def _select_tuning_record(
    context: ExperimentContext,
    *,
    mode: TrainingMode,
    split_name: str,
    window_size: int,
    records: list[tuple[Path, dict[str, object]]],
    identity: dict[str, JsonValue],
) -> tuple[Path, dict[str, object], Path]:
    if not records:
        raise ValueError("hyperparameter selection requires completed tuning runs")

    def key(item: tuple[Path, dict[str, object]]) -> tuple[float, int, float]:
        _, record = item
        selected = _required_object(record, "selected_validation")
        training = _required_object(record, "training")
        return (
            _required_float(selected, "selection_score"),
            _required_int(training, "latent_dimension"),
            _required_float(training, "lambda_age"),
        )

    selected_path, selected_record = min(records, key=key)
    selected_training = _required_object(selected_record, "training")
    selected_validation = _required_object(selected_record, "selected_validation")
    selection_path = _selection_path(
        context,
        mode=mode,
        split_name=split_name,
        window_size=window_size,
    )
    selection_identity = {
        **identity,
        "mode": mode.value,
        "candidate_metadata_sha256": [sha256_file(path / "metadata.json") for path, _ in records],
    }
    selection_record: dict[str, JsonValue] = {
        "schema": _SELECTION_SCHEMA,
        "identity": selection_identity,
        "tie_break": "selection_score_then_smaller_d_then_smaller_lambda",
        "selected_run": str(selected_path.relative_to(context.output_directory)),
        "selected_metadata_sha256": sha256_file(selected_path / "metadata.json"),
        "selected_step": _required_int(selected_record, "selected_step"),
        "selected_training": cast(dict[str, JsonValue], selected_training),
        "selected_validation": cast(dict[str, JsonValue], selected_validation),
    }
    if selection_path.is_file():
        observed = _require_record(
            selection_path,
            schema=_SELECTION_SCHEMA,
            identity=selection_identity,
        )
        if observed != selection_record:
            raise ValueError("existing hyperparameter selection differs from recomputation")
    else:
        write_canonical_json_exclusive(selection_path, selection_record)
    return selected_path, selected_record, selection_path


def _final_directory(
    context: ExperimentContext,
    *,
    mode: TrainingMode,
    split_name: str,
    window_size: int,
) -> Path:
    return context.output_directory / "final" / mode.value / split_name / f"window-{window_size}"


def _final_identity(
    context: ExperimentContext,
    *,
    split_name: str,
    window_size: int,
    embedding_sha256: str,
    selection_path: Path,
    prerequisite_sha256: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    return {
        **_identity(
            context,
            split_name,
            window_size=window_size,
            embedding_sha256=embedding_sha256,
        ),
        "prerequisite_sha256": prerequisite_sha256,
        "selection_sha256": sha256_file(selection_path),
        "primary_train_indices_sha256": sha256_ordered_strings(
            map(
                str,
                context.splits[split_name].primary_train_indices.tolist(),
            )
        ),
    }


def _run_final_refit(
    context: ExperimentContext,
    *,
    mode: TrainingMode,
    split_name: str,
    window_size: int,
    embeddings: EmbeddingMatrix,
    embedding_sha256: str,
    selected_record: dict[str, object],
    selection_path: Path,
    device: str,
    prerequisite_sha256: dict[str, JsonValue],
) -> Path:
    selected_training = _required_object(selected_record, "training")
    selected_dimension = _required_int(selected_training, "latent_dimension")
    training_config = _training_config(
        context,
        mode=mode,
        latent_dimension=context.config.sweep.latent_dimensions[
            tuple(map(int, context.config.sweep.latent_dimensions)).index(selected_dimension)
        ],
        lambda_age=parse_non_negative_weight(_required_float(selected_training, "lambda_age")),
        device=device,
    )
    identity = _final_identity(
        context,
        split_name=split_name,
        window_size=window_size,
        embedding_sha256=embedding_sha256,
        selection_path=selection_path,
        prerequisite_sha256=prerequisite_sha256,
    )
    final = _final_directory(
        context,
        mode=mode,
        split_name=split_name,
        window_size=window_size,
    )
    training_json = _training_json(training_config)
    if final.is_dir():
        _load_completed_model_record(
            final,
            schema=_FINAL_SCHEMA,
            identity=identity,
            training=training_json,
        )
        return final
    if final.exists():
        raise FileExistsError(final)
    selected_steps = _required_int(selected_record, "selected_step")
    train_indices = context.splits[split_name].primary_train_indices
    refit = refit_latent_metric(
        _subset_probes(context.probes, train_indices),
        _subset_embeddings(embeddings, train_indices),
        _subset_targets(context.targets, train_indices),
        config=training_config,
        selected_steps=selected_steps,
    )
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = final.parent / f".{final.name}.{secrets.token_hex(16)}.tmp"
    temporary.mkdir()
    model_path = temporary / "model.safetensors"
    refit.model.to("cpu")
    save_safetensors_exclusive(
        model_path,
        {name: value.detach().cpu() for name, value in refit.model.state_dict().items()},
    )
    write_canonical_json_exclusive(
        temporary / "metadata.json",
        {
            "schema": _FINAL_SCHEMA,
            "identity": identity,
            "training": training_json,
            "selected_steps": selected_steps,
            "fit_partition": "complete_primary_train",
            "fit_probe_count": train_indices.numel(),
            "history": [cast(dict[str, JsonValue], asdict(record)) for record in refit.history],
            "model_file": model_path.name,
            "model_sha256": sha256_file(model_path),
        },
    )
    os.rename(temporary, final)
    return final


def _stage_prerequisites(
    context: ExperimentContext,
    *,
    mode: TrainingMode,
    split_name: str,
    window_size: int,
) -> dict[str, JsonValue]:
    sequence_path = _sequence_record_path(
        context,
        split_name,
        window_size,
    )
    if not sequence_path.is_file():
        raise FileNotFoundError(
            f"neural stages require the matching durable sequence baseline: {sequence_path}"
        )
    prerequisites: dict[str, JsonValue] = {"sequence_baseline": sha256_file(sequence_path)}
    if mode == TrainingMode.FULL:
        age_directory = _final_directory(
            context,
            mode=TrainingMode.AGE_ONLY,
            split_name=split_name,
            window_size=window_size,
        )
        if not age_directory.is_dir():
            raise FileNotFoundError(
                f"full stage requires the matching final age-only baseline: {age_directory}"
            )
        prerequisites["age_only_metadata"] = sha256_file(age_directory / "metadata.json")
    return prerequisites


def run_model_stage(
    context: ExperimentContext,
    *,
    mode: TrainingMode,
    device: str,
) -> None:
    if mode not in {TrainingMode.AGE_ONLY, TrainingMode.FULL}:
        raise ValueError(f"unsupported model stage: {mode}")
    for window in context.config.splits.window_sizes:
        width = int(window)
        manifest_path = _embedding_manifest_path(context, width)
        embedding_sha256 = sha256_file(manifest_path)
        embeddings = load_embedding_cache(
            manifest_path.parent,
            context.probes,
            window_size=window,
        )
        for split_name, _ in _SPLIT_FILES:
            prerequisites = _stage_prerequisites(
                context,
                mode=mode,
                split_name=split_name,
                window_size=width,
            )
            lambdas = (
                (parse_non_negative_weight(1.0),)
                if mode == TrainingMode.AGE_ONLY
                else context.config.sweep.lambda_age
            )
            records: list[tuple[Path, dict[str, object]]] = []
            for dimension in context.config.sweep.latent_dimensions:
                for lambda_age in lambdas:
                    records.append(
                        _run_tuning_model(
                            context,
                            split_name=split_name,
                            window_size=width,
                            embeddings=embeddings,
                            embedding_sha256=embedding_sha256,
                            mode=mode,
                            latent_dimension=dimension,
                            lambda_age=lambda_age,
                            device=device,
                            prerequisite_sha256=prerequisites,
                        )
                    )
            selected_path, selected_record, selection_path = _select_tuning_record(
                context,
                mode=mode,
                split_name=split_name,
                window_size=width,
                records=records,
                identity={
                    **_identity(
                        context,
                        split_name,
                        window_size=width,
                        embedding_sha256=embedding_sha256,
                    ),
                    "prerequisite_sha256": prerequisites,
                },
            )
            if sha256_file(selected_path / "metadata.json") != sha256_file(
                Path(
                    context.output_directory,
                    _required_str(_load_json(selection_path), "selected_run"),
                    "metadata.json",
                )
            ):
                raise RuntimeError("selection record does not resolve to its selected tuning run")
            _run_final_refit(
                context,
                mode=mode,
                split_name=split_name,
                window_size=width,
                embeddings=embeddings,
                embedding_sha256=embedding_sha256,
                selected_record=selected_record,
                selection_path=selection_path,
                device=device,
                prerequisite_sha256=prerequisites,
            )


def _load_final_model(
    directory: Path,
    *,
    device: str,
) -> LatentMetric:
    metadata = _load_json(directory / "metadata.json")
    if (
        metadata.get("schema") != _FINAL_SCHEMA
        or metadata.get("model_file") != "model.safetensors"
        or metadata.get("model_sha256") != sha256_file(directory / "model.safetensors")
    ):
        raise ValueError("final model metadata or payload fingerprint differs")
    training = _required_object(metadata, "training")
    model = LatentMetric(
        embedding_dimension=256,
        latent_dimension=parse_latent_dimension(_required_int(training, "latent_dimension")),
    )
    model.load_state_dict(
        load_exact_safetensors(directory / "model.safetensors", _MODEL_KEYS),
        strict=True,
    )
    model.to(device)
    model.eval()
    return model


def _load_final_model_for_evaluation(
    context: ExperimentContext,
    *,
    mode: TrainingMode,
    split_name: str,
    window_size: int,
    embedding_sha256: str,
    device: str,
) -> tuple[LatentMetric, dict[str, object], Path]:
    selection_path = _selection_path(
        context,
        mode=mode,
        split_name=split_name,
        window_size=window_size,
    )
    selection = _load_json(selection_path)
    if selection.get("schema") != _SELECTION_SCHEMA:
        raise ValueError(f"selection schema differs: {selection_path}")
    selected_training = _required_object(selection, "selected_training")
    if _required_str(selected_training, "mode") != mode.value:
        raise ValueError("selected training mode differs from requested evaluation stage")
    prerequisites = _stage_prerequisites(
        context,
        mode=mode,
        split_name=split_name,
        window_size=window_size,
    )
    identity = _final_identity(
        context,
        split_name=split_name,
        window_size=window_size,
        embedding_sha256=embedding_sha256,
        selection_path=selection_path,
        prerequisite_sha256=prerequisites,
    )
    final = _final_directory(
        context,
        mode=mode,
        split_name=split_name,
        window_size=window_size,
    )
    metadata = _require_record(
        final / "metadata.json",
        schema=_FINAL_SCHEMA,
        identity=identity,
    )
    if (
        metadata.get("training") != selected_training
        or _required_int(metadata, "selected_steps") != _required_int(selection, "selected_step")
        or _required_str(metadata, "fit_partition") != "complete_primary_train"
        or _required_int(metadata, "fit_probe_count")
        != context.splits[split_name].primary_train_indices.numel()
        or metadata.get("model_sha256") != sha256_file(final / "model.safetensors")
    ):
        raise ValueError("final model does not match its frozen selection and refit contract")
    return _load_final_model(final, device=device), selected_training, final


def _full_model_outputs(
    model: LatentMetric,
    embeddings: EmbeddingMatrix,
    *,
    device: str,
) -> tuple[t.Tensor, t.Tensor]:
    with t.inference_mode():
        values = embeddings.training_tensor(device=device)
        latent = model.latent(values)
        age = model.predict_age_from_latent(latent)
        return latent.cpu().contiguous(), age.cpu().to(t.float64).contiguous()


def _age_model_predictions(
    model: LatentMetric,
    embeddings: EmbeddingMatrix,
    *,
    device: str,
) -> t.Tensor:
    with t.inference_mode():
        values = embeddings.training_tensor(device=device)
        return model.predict_age_from_latent(model.latent(values)).cpu().to(t.float64).contiguous()


def _distance_baseline_metrics(
    target: t.Tensor,
    prediction: t.Tensor,
) -> dict[str, JsonValue]:
    if (
        target.dtype != t.float64
        or prediction.dtype != t.float64
        or target.ndim != 1
        or target.shape != prediction.shape
        or target.numel() < 2
        or not bool(t.isfinite(target).all().item() and t.isfinite(prediction).all().item())
    ):
        raise ValueError("distance-baseline metric vectors must be aligned finite float64 values")
    target_centered = target - target.mean()
    prediction_centered = prediction - prediction.mean()
    target_ss = t.sum(t.square(target_centered))
    prediction_ss = t.sum(t.square(prediction_centered))
    if float(target_ss.item()) == 0.0:
        raise ValueError("distance-baseline R-squared is undefined for a constant target")
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
        "mse": float(t.mean(t.square(target - prediction)).item()),
        "pearson": pearson,
        "pearson_status": (
            "defined" if pearson is not None else "undefined_constant_distance_class_prediction"
        ),
        "r_squared": float((1.0 - residual_ss / target_ss).item()),
    }


def _evaluate_pair_population(
    uniform: CachedPairSet,
    stratified: CachedPairSet,
    cache: EvaluationPairCache,
    latent: t.Tensor,
) -> dict[str, JsonValue]:
    if uniform.pairs.population != stratified.pairs.population:
        raise ValueError("uniform and stratified pair populations differ")
    uniform_prediction = gather_pair_predictions_chunked(
        latent,
        uniform.pairs,
        chunk_size=16_384,
    ).to(t.float64)
    uniform_metrics = regression_metrics(uniform.targets, uniform_prediction)
    baseline_uniform_prediction = cache.distance_baseline.predict(uniform.pairs.distance_class)
    stratified_prediction = gather_pair_predictions_chunked(
        latent,
        stratified.pairs,
        chunk_size=16_384,
    ).to(t.float64)
    by_distance = metrics_by_distance(
        stratified.pairs,
        stratified.targets,
        stratified_prediction,
    )
    return {
        "population": uniform.pairs.population.value,
        "uniform_sample": {
            "count": uniform.pairs.count,
            "total_possible_pairs": uniform.pairs.total_possible_pairs,
            "seed": uniform.pairs.seed,
        },
        "uniform_model_metrics": _metric_json(uniform_metrics),
        "uniform_distance_baseline_metrics": _distance_baseline_metrics(
            uniform.targets,
            baseline_uniform_prediction,
        ),
        "distance_stratified_sample": {
            "count": stratified.pairs.count,
            "total_possible_pairs": stratified.pairs.total_possible_pairs,
            "seed": stratified.pairs.seed,
            "maximum_pairs_per_class": int(
                t.bincount(stratified.pairs.distance_class, minlength=8).max().item()
            ),
        },
        "by_distance": [
            {
                "distance_class": DISTANCE_CLASS_LABELS[int(distance_class)],
                "distance_class_index": int(distance_class),
                "distance_baseline_mean": float(
                    cache.distance_baseline.means[int(distance_class)].item()
                ),
                "model_metrics": _metric_json(by_distance[distance_class]),
            }
            for distance_class in DistanceClass
            if distance_class in by_distance
        ],
    }


def _held_out_projection(
    context: ExperimentContext,
    split_name: str,
    latent: t.Tensor,
) -> dict[str, JsonValue]:
    split = context.splits[split_name]
    projection = fit_projection_2d(latent.index_select(0, split.primary_train_indices))
    coordinates = projection.transform(latent.index_select(0, split.test_indices))
    return {
        "fit_partition": "complete_primary_train",
        "fit_probe_count": split.primary_train_indices.numel(),
        "mean": projection.mean.tolist(),
        "axes": projection.axes.tolist(),
        "points": [
            {
                "probe_id": str(context.probes.probes[index].probe_id),
                "x": float(point[0]),
                "y": float(point[1]),
                "context": context.probes.probes[index].context.value,
            }
            for index, point in zip(
                split.test_indices.tolist(),
                coordinates.tolist(),
                strict=True,
            )
        ],
    }


def _evaluation_path(
    context: ExperimentContext,
    split_name: str,
    window_size: int,
) -> Path:
    return context.output_directory / "evaluation" / split_name / f"window-{window_size}.json"


def run_evaluation(context: ExperimentContext, *, device: str) -> None:
    """Evaluate frozen final models on test loci without reopening any tuning choice."""

    for window in context.config.splits.window_sizes:
        width = int(window)
        manifest_path = _embedding_manifest_path(context, width)
        embedding_sha256 = sha256_file(manifest_path)
        embeddings = load_embedding_cache(
            manifest_path.parent,
            context.probes,
            window_size=window,
        )
        for split_name, _ in _SPLIT_FILES:
            pair_directory = context.output_directory / "evaluation-pairs" / split_name
            pair_cache = load_evaluation_pair_cache(
                pair_directory,
                _pair_cache_identity(context, split_name),
            )
            full_model, full_training, full_directory = _load_final_model_for_evaluation(
                context,
                mode=TrainingMode.FULL,
                split_name=split_name,
                window_size=width,
                embedding_sha256=embedding_sha256,
                device=device,
            )
            age_model, age_training, age_directory = _load_final_model_for_evaluation(
                context,
                mode=TrainingMode.AGE_ONLY,
                split_name=split_name,
                window_size=width,
                embedding_sha256=embedding_sha256,
                device=device,
            )
            sequence_path = _sequence_record_path(context, split_name, width)
            sequence_record = _require_record(
                sequence_path,
                schema=_SEQUENCE_SCHEMA,
                identity=_identity(
                    context,
                    split_name,
                    window_size=width,
                    embedding_sha256=None,
                ),
            )
            identity: dict[str, JsonValue] = {
                **_identity(
                    context,
                    split_name,
                    window_size=width,
                    embedding_sha256=embedding_sha256,
                ),
                "pair_cache_metadata_sha256": sha256_file(pair_directory / "metadata.json"),
                "sequence_baseline_sha256": sha256_file(sequence_path),
                "age_only_model_metadata_sha256": sha256_file(age_directory / "metadata.json"),
                "full_model_metadata_sha256": sha256_file(full_directory / "metadata.json"),
            }
            path = _evaluation_path(context, split_name, width)
            if path.is_file():
                _require_record(path, schema=_EVALUATION_SCHEMA, identity=identity)
                continue
            full_latent, full_age_prediction = _full_model_outputs(
                full_model,
                embeddings,
                device=device,
            )
            age_only_prediction = _age_model_predictions(
                age_model,
                embeddings,
                device=device,
            )
            test_indices = context.splits[split_name].test_indices
            age_target = context.targets.rho.tensor.index_select(0, test_indices)
            age_metrics = {
                "sequence_features": cast(
                    dict[str, JsonValue],
                    _required_object(sequence_record, "held_out_metrics"),
                ),
                "caduceus_age_only": _metric_json(
                    regression_metrics(
                        age_target,
                        age_only_prediction.index_select(0, test_indices),
                    )
                ),
                "full_latent_metric": _metric_json(
                    regression_metrics(
                        age_target,
                        full_age_prediction.index_select(0, test_indices),
                    )
                ),
            }
            pair_metrics = {
                PairPopulation.SEEN_BY_HELD_OUT.value: _evaluate_pair_population(
                    pair_cache.pair_sets[PairSetName.SEEN_UNIFORM],
                    pair_cache.pair_sets[PairSetName.SEEN_STRATIFIED],
                    pair_cache,
                    full_latent,
                ),
                PairPopulation.HELD_OUT_BY_HELD_OUT.value: _evaluate_pair_population(
                    pair_cache.pair_sets[PairSetName.HELD_OUT_UNIFORM],
                    pair_cache.pair_sets[PairSetName.HELD_OUT_STRATIFIED],
                    pair_cache,
                    full_latent,
                ),
            }
            projection: dict[str, JsonValue] | None = None
            if width == int(context.config.sweep.projection_window_size):
                projection = _held_out_projection(
                    context,
                    split_name,
                    full_latent,
                )
            write_canonical_json_exclusive(
                path,
                {
                    "schema": _EVALUATION_SCHEMA,
                    "identity": identity,
                    "test_probe_count": test_indices.numel(),
                    "retained_sample_count": context.retained_sample_count,
                    "selected_full_training": cast(
                        dict[str, JsonValue],
                        full_training,
                    ),
                    "selected_age_only_training": cast(
                        dict[str, JsonValue],
                        age_training,
                    ),
                    "distance_baseline": {
                        "fit_partition": "complete_primary_train",
                        "training_pair_count": pair_cache.pair_sets[
                            PairSetName.TRAINING_STRATIFIED
                        ].pairs.count,
                        "means": pair_cache.distance_baseline.means.tolist(),
                        "counts": pair_cache.distance_baseline.counts.tolist(),
                        "labels": list(DISTANCE_CLASS_LABELS),
                    },
                    "age_metrics": age_metrics,
                    "pair_metrics": pair_metrics,
                    "projection": projection,
                },
            )


def main() -> None:
    arguments = _parser().parse_args()
    context = _load_context(arguments)
    context.output_directory.mkdir(parents=True, exist_ok=True)
    if arguments.stage in {"sequence", "all"}:
        run_sequence_baselines(context)
    if arguments.stage in {"pairs", "all"}:
        run_pair_caches(context)
    if arguments.stage in {"age", "all"}:
        run_model_stage(
            context,
            mode=TrainingMode.AGE_ONLY,
            device=arguments.device,
        )
    if arguments.stage in {"full", "all"}:
        run_model_stage(
            context,
            mode=TrainingMode.FULL,
            device=arguments.device,
        )
    if arguments.stage in {"evaluate", "all"}:
        run_evaluation(context, device=arguments.device)


if __name__ == "__main__":
    main()
