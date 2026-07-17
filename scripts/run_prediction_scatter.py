"""Generate deterministic display samples from frozen predictions and pair caches."""

from __future__ import annotations

import argparse
import json
import math
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
from methylation_latent.domain import parse_latent_dimension
from methylation_latent.embedding_cache import load_embedding_cache
from methylation_latent.evaluation import PairPopulation, regression_metrics
from methylation_latent.evaluation_cache import (
    CachedPairSet,
    EvaluationPairCacheIdentity,
    PairSetName,
    load_evaluation_pair_cache,
)
from methylation_latent.experiment_data import load_split_artifact
from methylation_latent.model import LatentMetric
from methylation_latent.scatter import (
    CorrelationScatterSample,
    deterministic_display_indices,
)
from methylation_latent.storage import load_exact_safetensors, load_target_geometry

_SCHEMA = "methylation-latent.prediction-scatter.v1"
_EVALUATION_SCHEMA = "methylation-latent.held-out-evaluation.v2"
_DIRECT_SCHEMA = "methylation-latent.exploratory-direct-tanh-age.v1"
_FINAL_SCHEMA = "methylation-latent.final-refit.v2"
_MODEL_KEYS = {"age_direction", "projection.weight"}
_DIRECT_PREDICTION_KEYS = {"prediction", "target", "test_indices"}
_SPLITS = ("diverse-blocks", "held-out-chromosome")
_MAXIMUM_DISPLAY_POINTS = 5_000
_SEED = 934_771
_MAX_REPRODUCTION_ULPS = 8


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--experiments", type=Path, required=True)
    parser.add_argument("--direct-age", type=Path, required=True)
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


def _object(record: dict[str, object], key: str) -> dict[str, object]:
    value = record.get(key)
    if not isinstance(value, dict) or any(not isinstance(name, str) for name in value):
        raise TypeError(f"{key} must be an object")
    return cast(dict[str, object], value)


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


def _string(record: dict[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise TypeError(f"{key} must be a non-empty string")
    return value


def _load_latent_model(directory: Path) -> tuple[LatentMetric, dict[str, object]]:
    metadata = _load_json(directory / "metadata.json")
    training = _object(metadata, "training")
    model_path = directory / "model.safetensors"
    if metadata.get("schema") != _FINAL_SCHEMA or metadata.get("model_sha256") != sha256_file(
        model_path
    ):
        raise ValueError(f"final latent model identity differs: {directory}")
    model = LatentMetric(
        embedding_dimension=256,
        latent_dimension=parse_latent_dimension(_integer(training, "latent_dimension")),
    )
    model.load_state_dict(load_exact_safetensors(model_path, _MODEL_KEYS), strict=True)
    model.eval()
    return model, metadata


def _assert_metrics_within_ulps(
    expected: dict[str, object],
    target: t.Tensor,
    prediction: t.Tensor,
) -> None:
    observed = regression_metrics(target, prediction)
    fields = {
        "count": observed.count,
        "target_mean": observed.target_mean,
        "target_standard_deviation": observed.target_standard_deviation,
        "prediction_mean": observed.prediction_mean,
        "prediction_standard_deviation": observed.prediction_standard_deviation,
        "mse": observed.mse,
        "pearson": observed.pearson,
        "r_squared": observed.r_squared,
    }
    if _integer(expected, "count") != fields.pop("count"):
        raise ValueError("scatter source count differs from primary age metrics")
    for name, observed_value in fields.items():
        expected_value = _number(expected, name)
        tolerance = _MAX_REPRODUCTION_ULPS * max(math.ulp(observed_value), math.ulp(expected_value))
        if abs(observed_value - expected_value) > tolerance:
            raise ValueError(
                "scatter source exceeds age-metric reproduction bound: "
                f"field={name}, difference={abs(observed_value - expected_value)}, "
                f"tolerance={tolerance}"
            )


def _age_scatter(
    *,
    test_indices: t.Tensor,
    target: t.Tensor,
    cosine_prediction: t.Tensor,
    full_prediction: t.Tensor,
    direct_predictions_path: Path,
    seed: int,
) -> CorrelationScatterSample:
    direct = load_exact_safetensors(direct_predictions_path, _DIRECT_PREDICTION_KEYS)
    if not t.equal(direct["test_indices"], test_indices) or not t.equal(direct["target"], target):
        raise ValueError("direct-age prediction artifact differs from the frozen test target order")
    display = deterministic_display_indices(
        test_indices.numel(), _MAXIMUM_DISPLAY_POINTS, seed=seed
    )
    return CorrelationScatterSample(
        source_count=test_indices.numel(),
        seed=seed,
        sample_indices=display,
        target=target.index_select(0, display),
        predictions={
            "cosine_age_only": cosine_prediction.index_select(0, display),
            "direct_tanh": direct["prediction"].index_select(0, display),
            "full_latent_metric": full_prediction.index_select(0, display),
        },
    )


def _pair_scatter(
    cached: CachedPairSet,
    latent: t.Tensor,
    *,
    seed: int,
) -> CorrelationScatterSample:
    display = deterministic_display_indices(
        cached.pairs.count,
        _MAXIMUM_DISPLAY_POINTS,
        seed=seed,
    )
    left = cached.pairs.left.index_select(0, display)
    right = cached.pairs.right.index_select(0, display)
    prediction = t.sum(
        latent.index_select(0, left) * latent.index_select(0, right),
        dim=1,
    ).to(t.float64)
    return CorrelationScatterSample(
        source_count=cached.pairs.count,
        seed=seed,
        sample_indices=display,
        target=cached.targets.index_select(0, display),
        predictions={"full_latent_metric": prediction},
    )


def main() -> None:
    arguments = _parser().parse_args()
    _configure_runtime()
    code_git_commit = require_clean_git_commit(Path(__file__).resolve().parents[1])
    config = load_protocol_config(arguments.config)
    if config.status != "frozen":
        raise ValueError("scatter generation requires the frozen primary protocol")
    protocol_sha256 = sha256_file(arguments.config)
    bundle = verify_primary_data_bundle(
        arguments.data,
        protocol_id=config.protocol_id,
        protocol_sha256=protocol_sha256,
    )
    probes = load_probe_table(arguments.data / "probes.tsv")
    targets = load_target_geometry(arguments.data / "targets.safetensors")
    data_sha256 = sha256_file(arguments.data / "bundle.json")
    target_sha256 = sha256_file(arguments.data / "targets.safetensors")
    probe_order_sha256 = sha256_ordered_strings(str(probe.probe_id) for probe in probes.probes)
    configured_windows = {int(window): window for window in config.splits.window_sizes}
    windows = tuple(arguments.windows)
    if (
        not windows
        or tuple(sorted(set(windows))) != windows
        or any(window not in configured_windows for window in windows)
    ):
        raise ValueError("scatter windows must be increasing members of the frozen sweep")
    for window in windows:
        embedding_directory = arguments.embeddings / f"window-{window}"
        embeddings = load_embedding_cache(
            embedding_directory,
            probes,
            window_size=configured_windows[window],
        )
        embedding_values = embeddings.training_tensor(device="cpu")
        for split_offset, split_name in enumerate(_SPLITS):
            split_tensor = arguments.data / "splits" / f"{split_name}.safetensors"
            split_metadata = arguments.data / "splits" / f"{split_name}.json"
            split = load_split_artifact(split_tensor, split_metadata, probes=probes)
            split_sha256 = sha256_ordered_strings(
                (sha256_file(split_metadata), sha256_file(split_tensor))
            )
            evaluation_path = (
                arguments.experiments / "evaluation" / split_name / f"window-{window}.json"
            )
            evaluation = _load_json(evaluation_path)
            evaluation_identity = _object(evaluation, "identity")
            expected_identity = (
                config.protocol_id,
                protocol_sha256,
                data_sha256,
                split_name,
                split_sha256,
                window,
            )
            observed_identity = (
                _string(evaluation_identity, "protocol_id"),
                _string(evaluation_identity, "protocol_sha256"),
                _string(evaluation_identity, "data_sha256"),
                _string(evaluation_identity, "split_name"),
                _string(evaluation_identity, "split_sha256"),
                _integer(evaluation_identity, "window_size"),
            )
            if (
                evaluation.get("schema") != _EVALUATION_SCHEMA
                or observed_identity != expected_identity
            ):
                raise ValueError(f"scatter evaluation identity differs: {evaluation_path}")
            full_directory = (
                arguments.experiments / "final" / "full" / split_name / f"window-{window}"
            )
            age_directory = (
                arguments.experiments / "final" / "age_only" / split_name / f"window-{window}"
            )
            full_model, full_metadata = _load_latent_model(full_directory)
            age_model, age_metadata = _load_latent_model(age_directory)
            if _string(evaluation_identity, "full_model_metadata_sha256") != sha256_file(
                full_directory / "metadata.json"
            ) or _string(evaluation_identity, "age_only_model_metadata_sha256") != sha256_file(
                age_directory / "metadata.json"
            ):
                raise ValueError("scatter model metadata differs from the primary evaluation")
            with t.inference_mode():
                full_latent = full_model.latent(embedding_values)
                full_age_all = full_model.predict_age_from_latent(full_latent).to(t.float64)
                age_latent = age_model.latent(embedding_values)
                cosine_age_all = age_model.predict_age_from_latent(age_latent).to(t.float64)
            age_target = targets.rho.tensor.index_select(0, split.test_indices)
            full_age = full_age_all.index_select(0, split.test_indices)
            cosine_age = cosine_age_all.index_select(0, split.test_indices)
            primary_age = _object(evaluation, "age_metrics")
            _assert_metrics_within_ulps(
                _object(primary_age, "caduceus_age_only"), age_target, cosine_age
            )
            _assert_metrics_within_ulps(
                _object(primary_age, "full_latent_metric"), age_target, full_age
            )
            direct_directory = arguments.direct_age / split_name / f"window-{window}"
            direct_metadata_path = direct_directory / "metadata.json"
            direct_metadata = _load_json(direct_metadata_path)
            direct_identity = _object(direct_metadata, "identity")
            if (
                direct_metadata.get("schema") != _DIRECT_SCHEMA
                or _string(direct_identity, "protocol_id") != config.protocol_id
                or _string(direct_identity, "split_name") != split_name
                or _integer(direct_identity, "window_size") != window
                or _string(direct_identity, "data_sha256") != data_sha256
                or _string(direct_identity, "split_sha256") != split_sha256
            ):
                raise ValueError("direct-age scatter source identity differs")
            pair_directory = arguments.experiments / "evaluation-pairs" / split_name
            pair_cache = load_evaluation_pair_cache(
                pair_directory,
                EvaluationPairCacheIdentity(
                    protocol_id=config.protocol_id,
                    protocol_sha256=protocol_sha256,
                    git_commit=bundle.git_commit,
                    data_sha256=data_sha256,
                    target_sha256=target_sha256,
                    split_name=split_name,
                    split_sha256=split_sha256,
                    probe_order_sha256=probe_order_sha256,
                ),
            )
            base_seed = _SEED + window + split_offset * 10_000
            age_scatter = _age_scatter(
                test_indices=split.test_indices,
                target=age_target,
                cosine_prediction=cosine_age,
                full_prediction=full_age,
                direct_predictions_path=direct_directory / "predictions.safetensors",
                seed=base_seed,
            )
            pair_scatter = {
                PairPopulation.SEEN_BY_HELD_OUT.value: _pair_scatter(
                    pair_cache.pair_sets[PairSetName.SEEN_UNIFORM],
                    full_latent,
                    seed=base_seed + 1,
                ).as_json(),
                PairPopulation.HELD_OUT_BY_HELD_OUT.value: _pair_scatter(
                    pair_cache.pair_sets[PairSetName.HELD_OUT_UNIFORM],
                    full_latent,
                    seed=base_seed + 2,
                ).as_json(),
            }
            record: dict[str, JsonValue] = {
                "schema": _SCHEMA,
                "status": "display_only_no_fitting",
                "identity": {
                    "code_git_commit": code_git_commit,
                    "primary_git_commit": bundle.git_commit,
                    "protocol_id": config.protocol_id,
                    "protocol_sha256": protocol_sha256,
                    "data_sha256": data_sha256,
                    "target_sha256": target_sha256,
                    "split_name": split_name,
                    "split_sha256": split_sha256,
                    "window_size": window,
                    "embedding_manifest_sha256": sha256_file(embedding_directory / "manifest.json"),
                    "evaluation_sha256": sha256_file(evaluation_path),
                    "pair_cache_metadata_sha256": sha256_file(pair_directory / "metadata.json"),
                    "full_model_metadata_sha256": sha256_file(full_directory / "metadata.json"),
                    "full_model_sha256": _string(full_metadata, "model_sha256"),
                    "cosine_age_metadata_sha256": sha256_file(age_directory / "metadata.json"),
                    "cosine_age_model_sha256": _string(age_metadata, "model_sha256"),
                    "direct_age_metadata_sha256": sha256_file(direct_metadata_path),
                    "direct_age_predictions_sha256": sha256_file(
                        direct_directory / "predictions.safetensors"
                    ),
                },
                "sampling": {
                    "method": "target_blind_seeded_without_replacement_then_sorted",
                    "maximum_points": _MAXIMUM_DISPLAY_POINTS,
                },
                "age": age_scatter.as_json(),
                "pairs": cast(dict[str, JsonValue], pair_scatter),
            }
            output_path = arguments.output / split_name / f"window-{window}.json"
            if output_path.is_file():
                if _load_json(output_path) != record:
                    raise ValueError(f"existing scatter output differs: {output_path}")
            else:
                write_canonical_json_exclusive(output_path, record)
            print(
                f"scatter split={split_name} window={window} "
                f"pair_points={2 * _MAXIMUM_DISPLAY_POINTS} "
                f"age_points={age_scatter.sample_indices.numel()} output={output_path}"
            )


if __name__ == "__main__":
    main()
