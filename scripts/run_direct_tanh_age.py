"""Run the corrected direct-scalar tanh age baseline as a post-hoc analysis."""

from __future__ import annotations

import argparse
import json
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
from methylation_latent.data_bundle import verify_primary_data_bundle
from methylation_latent.direct_age import (
    DirectAgeData,
    DirectAgeTrainingConfig,
    DirectTanhAgeRegressor,
    refit_direct_tanh_age,
    train_direct_tanh_age,
)
from methylation_latent.domain import NonEmptyProbeSet
from methylation_latent.embedding_cache import load_embedding_cache
from methylation_latent.evaluation import regression_metrics
from methylation_latent.experiment_data import SplitArtifact, load_split_artifact
from methylation_latent.storage import (
    EmbeddingMatrix,
    load_exact_safetensors,
    load_target_geometry,
    save_safetensors_exclusive,
)
from methylation_latent.targets import CorrelationVector

_SCHEMA = "methylation-latent.exploratory-direct-tanh-age.v1"
_MODEL_KEYS = {"linear.bias", "linear.weight"}
_PREDICTION_KEYS = {"prediction", "target", "test_indices"}
_SPLITS = ("diverse-blocks", "held-out-chromosome")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", type=int, nargs="+", required=True)
    return parser


def _configure_runtime() -> None:
    t.set_num_threads(1)
    t.set_num_interop_threads(1)
    t.use_deterministic_algorithms(True)


def _load_json(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"expected JSON object: {path}")
    return cast(dict[str, object], raw)


def _subset_data(
    probes: NonEmptyProbeSet,
    embeddings: EmbeddingMatrix,
    rho: CorrelationVector,
    indices: t.Tensor,
) -> DirectAgeData:
    if indices.dtype != t.int64 or indices.ndim != 1 or indices.numel() == 0:
        raise TypeError("direct-age subset indices must be a non-empty int64 vector")
    if bool(t.any((indices < 0) | (indices >= len(probes))).item()):
        raise IndexError("direct-age subset index is outside the global universe")
    if t.unique(indices).numel() != indices.numel():
        raise ValueError("direct-age subset indices contain duplicates")
    return DirectAgeData(
        probes=NonEmptyProbeSet(tuple(probes.probes[index] for index in indices.tolist())),
        embeddings=EmbeddingMatrix(embeddings.tensor.index_select(0, indices)),
        rho=CorrelationVector(rho.tensor.index_select(0, indices)),
    )


def _training_json(config: DirectAgeTrainingConfig) -> dict[str, JsonValue]:
    return {
        "architecture": "affine_256_to_1_then_tanh",
        "output_dimension": 1,
        "latent_dimension": None,
        "lambda_age": None,
        "loss": "mean_squared_error_over_probe_age_correlations",
        "batch_size": int(config.batch_size),
        "neighbourhood_width": int(config.neighbourhood_width),
        "steps": int(config.steps),
        "validation_interval": int(config.validation_interval),
        "learning_rate": config.learning_rate,
        "seed": config.seed,
        "device": config.device,
        "weight_decay": 0.0,
        "initialization": "zero_weight_and_bias",
    }


def _metric_json(target: t.Tensor, prediction: t.Tensor) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], asdict(regression_metrics(target, prediction)))


def _state(model: DirectTanhAgeRegressor) -> dict[str, t.Tensor]:
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    if set(state) != _MODEL_KEYS:
        raise ValueError("direct-age model state keys differ")
    return state


def _verify_existing(directory: Path, expected_identity: dict[str, JsonValue]) -> None:
    metadata_path = directory / "metadata.json"
    record = _load_json(metadata_path)
    if record.get("schema") != _SCHEMA or record.get("identity") != expected_identity:
        raise ValueError(f"existing direct-age identity differs: {directory}")
    files = (
        ("tuning_model_sha256", "tuning-model.safetensors", _MODEL_KEYS),
        ("final_model_sha256", "final-model.safetensors", _MODEL_KEYS),
        ("predictions_sha256", "predictions.safetensors", _PREDICTION_KEYS),
    )
    for hash_key, filename, keys in files:
        path = directory / filename
        if record.get(hash_key) != sha256_file(path):
            raise ValueError(f"existing direct-age artifact hash differs: {path}")
        load_exact_safetensors(path, keys)
    predictions = load_exact_safetensors(directory / "predictions.safetensors", _PREDICTION_KEYS)
    if predictions["prediction"].dtype != t.float64 or predictions["target"].dtype != t.float64:
        raise TypeError("direct-age saved predictions and targets must be float64")
    if predictions["test_indices"].dtype != t.int64:
        raise TypeError("direct-age saved test indices must be int64")
    if record.get("held_out_metrics") != _metric_json(
        predictions["target"], predictions["prediction"]
    ):
        raise ValueError("existing direct-age metrics differ from saved prediction vectors")


def _run_one(
    *,
    code_git_commit: str,
    primary_git_commit: str,
    protocol_id: str,
    protocol_sha256: str,
    data_sha256: str,
    target_sha256: str,
    split_name: str,
    split_sha256: str,
    window: int,
    probes: NonEmptyProbeSet,
    rho: CorrelationVector,
    embeddings: EmbeddingMatrix,
    embedding_manifest_sha256: str,
    split: SplitArtifact,
    training_config: DirectAgeTrainingConfig,
    output: Path,
) -> None:
    identity: dict[str, JsonValue] = {
        "code_git_commit": code_git_commit,
        "primary_git_commit": primary_git_commit,
        "protocol_id": protocol_id,
        "protocol_sha256": protocol_sha256,
        "data_sha256": data_sha256,
        "target_sha256": target_sha256,
        "split_name": split_name,
        "split_sha256": split_sha256,
        "window_size": window,
        "embedding_manifest_sha256": embedding_manifest_sha256,
        "optimization_indices_sha256": sha256_ordered_strings(
            map(str, split.optimization_indices.tolist())
        ),
        "validation_indices_sha256": sha256_ordered_strings(
            map(str, split.validation_indices.tolist())
        ),
        "primary_train_indices_sha256": sha256_ordered_strings(
            map(str, split.primary_train_indices.tolist())
        ),
        "test_indices_sha256": sha256_ordered_strings(map(str, split.test_indices.tolist())),
    }
    final = output / split_name / f"window-{window}"
    if final.is_dir():
        _verify_existing(final, identity)
        print(f"direct-tanh-age reused split={split_name} window={window} output={final}")
        return
    if final.exists():
        raise FileExistsError(final)
    tuned = train_direct_tanh_age(
        _subset_data(probes, embeddings, rho, split.optimization_indices),
        _subset_data(probes, embeddings, rho, split.validation_indices),
        config=training_config,
    )
    refit = refit_direct_tanh_age(
        _subset_data(probes, embeddings, rho, split.primary_train_indices),
        config=training_config,
        selected_steps=tuned.selected_step,
    )
    test = _subset_data(probes, embeddings, rho, split.test_indices)
    refit.model.eval()
    with t.inference_mode():
        prediction = refit.model(test.embeddings.training_tensor(device="cpu")).to(t.float64)
    target = test.rho.tensor.to(dtype=t.float64)
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = final.parent / f".{final.name}.{secrets.token_hex(16)}.tmp"
    temporary.mkdir()
    tuning_model_path = temporary / "tuning-model.safetensors"
    final_model_path = temporary / "final-model.safetensors"
    predictions_path = temporary / "predictions.safetensors"
    save_safetensors_exclusive(tuning_model_path, _state(tuned.model))
    save_safetensors_exclusive(final_model_path, _state(refit.model))
    save_safetensors_exclusive(
        predictions_path,
        {
            "prediction": prediction,
            "target": target,
            "test_indices": split.test_indices,
        },
    )
    record: dict[str, JsonValue] = {
        "schema": _SCHEMA,
        "status": "post_hoc_corrected_baseline_not_confirmatory",
        "reason": (
            "The direct tanh architecture was introduced after the original cosine age-only "
            "test metrics had been inspected; a new holdout is required for confirmation."
        ),
        "identity": identity,
        "runtime": {
            "device": "cpu",
            "torch_deterministic_algorithms": True,
            "torch_num_threads": t.get_num_threads(),
            "torch_num_interop_threads": t.get_num_interop_threads(),
        },
        "training": _training_json(training_config),
        "selected_step": tuned.selected_step,
        "selected_validation": cast(
            dict[str, JsonValue],
            asdict(
                next(
                    record
                    for record in tuned.validation_history
                    if record.step == tuned.selected_step
                )
            ),
        ),
        "training_history": [cast(dict[str, JsonValue], asdict(row)) for row in tuned.history],
        "validation_history": [
            cast(dict[str, JsonValue], asdict(row)) for row in tuned.validation_history
        ],
        "refit": {
            "partition": "complete_primary_train",
            "probe_count": split.primary_train_indices.numel(),
            "steps": refit.refit_steps,
            "history": [cast(dict[str, JsonValue], asdict(row)) for row in refit.history],
        },
        "held_out_metrics": _metric_json(target, prediction),
        "tuning_model_sha256": sha256_file(tuning_model_path),
        "final_model_sha256": sha256_file(final_model_path),
        "predictions_sha256": sha256_file(predictions_path),
    }
    write_canonical_json_exclusive(temporary / "metadata.json", record)
    os.rename(temporary, final)
    print(
        f"direct-tanh-age split={split_name} window={window} "
        f"selected_step={tuned.selected_step} output={final}"
    )


def main() -> None:
    arguments = _parser().parse_args()
    _configure_runtime()
    repository = Path(__file__).resolve().parents[1]
    code_git_commit = require_clean_git_commit(repository)
    config = load_protocol_config(arguments.config)
    if config.status != "frozen":
        raise ValueError("direct tanh analysis requires the frozen primary protocol")
    protocol_sha256 = sha256_file(arguments.config)
    bundle = verify_primary_data_bundle(
        arguments.data,
        protocol_id=config.protocol_id,
        protocol_sha256=protocol_sha256,
    )
    probes = load_probe_table(arguments.data / "probes.tsv")
    targets = load_target_geometry(arguments.data / "targets.safetensors")
    if len(probes) != bundle.retained_probes or targets.rho.tensor.numel() != len(probes):
        raise ValueError("sealed data dimensions differ from direct-age inputs")
    configured_windows = {int(window): window for window in config.splits.window_sizes}
    windows = tuple(arguments.windows)
    if (
        not windows
        or tuple(sorted(set(windows))) != windows
        or any(window not in configured_windows for window in windows)
    ):
        raise ValueError("direct-age windows must be increasing members of the frozen sweep")
    training_config = DirectAgeTrainingConfig(
        batch_size=config.training.batch_size,
        neighbourhood_width=config.training.neighbourhood_width,
        steps=config.training.tuning_steps,
        validation_interval=config.training.validation_interval,
        learning_rate=config.training.learning_rate,
        seed=config.training.training_seed,
        device="cpu",
    )
    for window in windows:
        embedding_directory = arguments.embeddings / f"window-{window}"
        embeddings = load_embedding_cache(
            embedding_directory,
            probes,
            window_size=configured_windows[window],
        )
        for split_name in _SPLITS:
            split_tensor = arguments.data / "splits" / f"{split_name}.safetensors"
            split_metadata = arguments.data / "splits" / f"{split_name}.json"
            split = load_split_artifact(split_tensor, split_metadata, probes=probes)
            _run_one(
                code_git_commit=code_git_commit,
                primary_git_commit=bundle.git_commit,
                protocol_id=config.protocol_id,
                protocol_sha256=protocol_sha256,
                data_sha256=sha256_file(arguments.data / "bundle.json"),
                target_sha256=sha256_file(arguments.data / "targets.safetensors"),
                split_name=split_name,
                split_sha256=sha256_ordered_strings(
                    (sha256_file(split_metadata), sha256_file(split_tensor))
                ),
                window=window,
                probes=probes,
                rho=targets.rho,
                embeddings=embeddings,
                embedding_manifest_sha256=sha256_file(embedding_directory / "manifest.json"),
                split=split,
                training_config=training_config,
                output=arguments.output,
            )


if __name__ == "__main__":
    main()
