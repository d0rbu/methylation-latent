"""Run a post-hoc PSD distance-plus-sequence analysis without reopening primary tuning."""

from __future__ import annotations

import argparse
import json
import math
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
from methylation_latent.distance_integration import (
    KernelDesign,
    SimplexKernelFit,
    append_sequence_kernel,
    distance_kernel_design,
    fit_simplex_kernel,
    predict_kernel_mixture,
)
from methylation_latent.domain import NonEmptyProbeSet, WindowSize, parse_latent_dimension
from methylation_latent.embedding_cache import load_embedding_cache
from methylation_latent.evaluation import (
    DISTANCE_CLASS_LABELS,
    DistanceClass,
    PairIndices,
    PairPopulation,
    classify_distances,
    gather_pair_predictions_chunked,
    gather_pair_targets_chunked,
    metrics_by_distance,
    regression_metrics,
)
from methylation_latent.evaluation_cache import (
    CachedPairSet,
    EvaluationPairCache,
    EvaluationPairCacheIdentity,
    PairSetName,
    load_evaluation_pair_cache,
)
from methylation_latent.experiment_data import SplitArtifact, load_split_artifact
from methylation_latent.model import LatentMetric
from methylation_latent.storage import EmbeddingMatrix, load_exact_safetensors, load_target_geometry
from methylation_latent.targets import TargetGeometry

_SCHEMA = "methylation-latent.exploratory-distance-integration.v1"
_EVALUATION_SCHEMA = "methylation-latent.held-out-evaluation.v2"
_SELECTION_SCHEMA = "methylation-latent.hyperparameter-selection.v2"
_TUNING_SCHEMA = "methylation-latent.latent-tuning-run.v2"
_FINAL_SCHEMA = "methylation-latent.final-refit.v2"
_MODEL_KEYS = {"age_direction", "projection.weight"}
_LENGTH_SCALES = (1_024, 4_096, 16_384, 65_536, 262_144, 1_048_576, 4_194_304)
_SPLITS = ("diverse-blocks", "held-out-chromosome")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--experiments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", type=int, nargs="+", required=True)
    return parser


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


def _metric_json(target: t.Tensor, prediction: t.Tensor) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], asdict(regression_metrics(target, prediction)))


def _nullable_pearson_metric_json(
    target: t.Tensor,
    prediction: t.Tensor,
) -> dict[str, JsonValue]:
    if target.dtype != t.float64 or prediction.dtype != t.float64 or target.shape != prediction.shape:
        raise TypeError("nullable-Pearson metric vectors must be aligned float64 tensors")
    target_centered = target - target.mean()
    prediction_centered = prediction - prediction.mean()
    target_ss = t.sum(t.square(target_centered))
    prediction_ss = t.sum(t.square(prediction_centered))
    if float(target_ss.item()) == 0.0:
        raise ValueError("nullable-Pearson target must not be constant")
    residual_ss = t.sum(t.square(target - prediction))
    pearson: float | None = None
    if float(prediction_ss.item()) != 0.0:
        pearson = float(
            (
                t.sum(target_centered * prediction_centered)
                / t.sqrt(target_ss * prediction_ss)
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


def _load_model(directory: Path, *, schema: str) -> tuple[LatentMetric, dict[str, object]]:
    metadata = _load_json(directory / "metadata.json")
    training = _object(metadata, "training")
    model_path = directory / "model.safetensors"
    if (
        metadata.get("schema") != schema
        or metadata.get("model_file") != model_path.name
        or metadata.get("model_sha256") != sha256_file(model_path)
        or _string(training, "mode") != "full"
    ):
        raise ValueError(f"full-model artifact identity differs: {directory}")
    model = LatentMetric(
        embedding_dimension=256,
        latent_dimension=parse_latent_dimension(_integer(training, "latent_dimension")),
    )
    model.load_state_dict(load_exact_safetensors(model_path, _MODEL_KEYS), strict=True)
    model.to("cpu")
    model.eval()
    return model, metadata


def _model_latent(model: LatentMetric, embeddings: EmbeddingMatrix, indices: t.Tensor | None) -> t.Tensor:
    values = embeddings.tensor if indices is None else embeddings.tensor.index_select(0, indices)
    with t.inference_mode():
        latent = model.latent(values.to(device="cpu", dtype=t.float32))
    return latent.cpu().contiguous()


def _validation_pairs(
    probes: NonEmptyProbeSet,
    split: SplitArtifact,
    *,
    seed: int,
) -> tuple[PairIndices, PairIndices, t.Tensor]:
    validation_indices = t.sort(split.validation_indices).values
    count = validation_indices.numel()
    local = t.triu_indices(count, count, offset=1)
    global_left = validation_indices.index_select(0, local[0])
    global_right = validation_indices.index_select(0, local[1])
    distance_class = classify_distances(probes, global_left, global_right)
    total = count * (count - 1) // 2
    global_pairs = PairIndices(
        population=PairPopulation.HELD_OUT_BY_HELD_OUT,
        left=global_left,
        right=global_right,
        distance_class=distance_class,
        total_possible_pairs=total,
        seed=seed,
    )
    local_pairs = PairIndices(
        population=PairPopulation.HELD_OUT_BY_HELD_OUT,
        left=local[0],
        right=local[1],
        distance_class=distance_class,
        total_possible_pairs=total,
        seed=seed,
    )
    return global_pairs, local_pairs, validation_indices


def _fit_json(fit: SimplexKernelFit) -> dict[str, JsonValue]:
    return {
        "names": list(fit.names),
        "weights": list(fit.weights),
        "mse": fit.mse,
    }


def _assert_primary_metric_reproduction(
    observed: dict[str, object],
    recomputed: dict[str, JsonValue],
) -> None:
    for key in ("count", "mse", "pearson", "r_squared"):
        left = observed.get(key)
        right = recomputed[key]
        if key == "count":
            if left != right:
                raise RuntimeError("recomputed primary pair count differs")
        elif (
            isinstance(left, bool)
            or not isinstance(left, int | float)
            or not isinstance(right, int | float)
            or not math.isclose(float(left), float(right), rel_tol=1.0e-12, abs_tol=1.0e-12)
        ):
            raise RuntimeError(f"recomputed primary pair metric differs: {key}")


def _pair_set_report(
    pair_set: CachedPairSet,
    cache: EvaluationPairCache,
    probes: NonEmptyProbeSet,
    latent: t.Tensor,
    distance_fit: SimplexKernelFit,
    combined_fit: SimplexKernelFit,
) -> tuple[dict[str, JsonValue], KernelDesign, KernelDesign]:
    sequence = gather_pair_predictions_chunked(latent, pair_set.pairs, chunk_size=16_384).to(
        t.float64
    )
    distance = distance_kernel_design(
        probes,
        pair_set.pairs.left,
        pair_set.pairs.right,
        length_scales=_LENGTH_SCALES,
    )
    combined = append_sequence_kernel(distance, sequence)
    distance_prediction = predict_kernel_mixture(distance, distance_fit)
    combined_prediction = predict_kernel_mixture(combined, combined_fit)
    registered_prediction = cache.distance_baseline.predict(pair_set.pairs.distance_class)
    return (
        {
            "sequence": _metric_json(pair_set.targets, sequence),
            "registered_distance_class": _nullable_pearson_metric_json(
                pair_set.targets,
                registered_prediction,
            ),
            "psd_distance": _nullable_pearson_metric_json(
                pair_set.targets,
                distance_prediction,
            ),
            "psd_distance_plus_sequence": _nullable_pearson_metric_json(
                pair_set.targets,
                combined_prediction,
            ),
        },
        distance,
        combined,
    )


def _stratified_rows(
    pair_set: CachedPairSet,
    cache: EvaluationPairCache,
    distance: KernelDesign,
    combined: KernelDesign,
    distance_fit: SimplexKernelFit,
    combined_fit: SimplexKernelFit,
    latent: t.Tensor,
) -> list[dict[str, JsonValue]]:
    sequence_prediction = gather_pair_predictions_chunked(
        latent,
        pair_set.pairs,
        chunk_size=16_384,
    ).to(t.float64)
    distance_prediction = predict_kernel_mixture(distance, distance_fit)
    combined_prediction = predict_kernel_mixture(combined, combined_fit)
    sequence_metrics = metrics_by_distance(
        pair_set.pairs,
        pair_set.targets,
        sequence_prediction,
    )
    return [
        {
            "distance_class": DISTANCE_CLASS_LABELS[int(distance_class)],
            "count": sequence_metrics[distance_class].count,
            "registered_distance_mean": float(
                cache.distance_baseline.means[int(distance_class)].item()
            ),
            "sequence": cast(dict[str, JsonValue], asdict(sequence_metrics[distance_class])),
            "psd_distance": _nullable_pearson_metric_json(
                pair_set.targets[pair_set.pairs.distance_class == int(distance_class)],
                distance_prediction[pair_set.pairs.distance_class == int(distance_class)],
            ),
            "psd_distance_plus_sequence": _nullable_pearson_metric_json(
                pair_set.targets[pair_set.pairs.distance_class == int(distance_class)],
                combined_prediction[pair_set.pairs.distance_class == int(distance_class)],
            ),
        }
        for distance_class in DistanceClass
        if distance_class in sequence_metrics
    ]


def _expected_pair_identity(
    *,
    protocol_id: str,
    protocol_sha256: str,
    primary_git_commit: str,
    data_sha256: str,
    target_sha256: str,
    split_name: str,
    split_sha256: str,
    probes: NonEmptyProbeSet,
) -> EvaluationPairCacheIdentity:
    return EvaluationPairCacheIdentity(
        protocol_id=protocol_id,
        protocol_sha256=protocol_sha256,
        git_commit=primary_git_commit,
        data_sha256=data_sha256,
        target_sha256=target_sha256,
        split_name=split_name,
        split_sha256=split_sha256,
        probe_order_sha256=sha256_ordered_strings(
            str(probe.probe_id) for probe in probes.probes
        ),
    )


def _run_one(
    *,
    code_git_commit: str,
    protocol_id: str,
    protocol_sha256: str,
    primary_git_commit: str,
    data_directory: Path,
    embedding_directory: Path,
    experiments: Path,
    output: Path,
    probes: NonEmptyProbeSet,
    targets: TargetGeometry,
    split: SplitArtifact,
    split_name: str,
    split_sha256: str,
    window_size: WindowSize,
    validation_seed: int,
) -> None:
    window = int(window_size)
    data_sha256 = sha256_file(data_directory / "bundle.json")
    target_sha256 = sha256_file(data_directory / "targets.safetensors")
    pair_directory = experiments / "evaluation-pairs" / split_name
    pair_identity = _expected_pair_identity(
        protocol_id=protocol_id,
        protocol_sha256=protocol_sha256,
        primary_git_commit=primary_git_commit,
        data_sha256=data_sha256,
        target_sha256=target_sha256,
        split_name=split_name,
        split_sha256=split_sha256,
        probes=probes,
    )
    cache = load_evaluation_pair_cache(pair_directory, pair_identity)

    evaluation_path = experiments / "evaluation" / split_name / f"window-{window}.json"
    evaluation = _load_json(evaluation_path)
    evaluation_identity = _object(evaluation, "identity")
    if (
        evaluation.get("schema") != _EVALUATION_SCHEMA
        or _string(evaluation_identity, "git_commit") != primary_git_commit
        or _string(evaluation_identity, "protocol_id") != protocol_id
        or _string(evaluation_identity, "protocol_sha256") != protocol_sha256
        or _string(evaluation_identity, "data_sha256") != data_sha256
        or _string(evaluation_identity, "target_sha256") != target_sha256
        or _string(evaluation_identity, "split_name") != split_name
        or _string(evaluation_identity, "split_sha256") != split_sha256
        or _integer(evaluation_identity, "window_size") != window
    ):
        raise ValueError("primary evaluation identity differs")

    embeddings_path = embedding_directory / f"window-{window}"
    embeddings = load_embedding_cache(
        embeddings_path,
        probes,
        window_size=window_size,
        expected_git_commit=primary_git_commit,
    )
    selection_path = experiments / "selections" / "full" / split_name / f"window-{window}.json"
    selection = _load_json(selection_path)
    if selection.get("schema") != _SELECTION_SCHEMA:
        raise ValueError("full-model selection schema differs")
    selected_run = (experiments / _string(selection, "selected_run")).resolve()
    if not selected_run.is_relative_to(experiments.resolve() / "tuning" / "full"):
        raise ValueError("selected tuning run escapes the full-tuning directory")
    tuning_model, tuning_metadata = _load_model(selected_run, schema=_TUNING_SCHEMA)
    if (
        sha256_file(selected_run / "metadata.json")
        != _string(selection, "selected_metadata_sha256")
        or tuning_metadata.get("training") != selection.get("selected_training")
        or tuning_metadata.get("selected_step") != selection.get("selected_step")
    ):
        raise ValueError("selected tuning model differs from selection record")

    final_directory = experiments / "final" / "full" / split_name / f"window-{window}"
    final_model, _ = _load_model(final_directory, schema=_FINAL_SCHEMA)
    if sha256_file(final_directory / "metadata.json") != _string(
        evaluation_identity,
        "full_model_metadata_sha256",
    ):
        raise ValueError("final full model differs from primary evaluation")

    global_validation_pairs, local_validation_pairs, validation_indices = _validation_pairs(
        probes,
        split,
        seed=validation_seed,
    )
    validation_target = gather_pair_targets_chunked(
        targets.methylation,
        global_validation_pairs,
        chunk_size=16_384,
    )
    validation_latent = _model_latent(tuning_model, embeddings, validation_indices)
    validation_sequence = gather_pair_predictions_chunked(
        validation_latent,
        local_validation_pairs,
        chunk_size=16_384,
    ).to(t.float64)
    validation_distance = distance_kernel_design(
        probes,
        global_validation_pairs.left,
        global_validation_pairs.right,
        length_scales=_LENGTH_SCALES,
    )
    validation_combined = append_sequence_kernel(validation_distance, validation_sequence)
    distance_fit = fit_simplex_kernel(validation_distance, validation_target)
    combined_fit = fit_simplex_kernel(validation_combined, validation_target)

    final_latent = _model_latent(final_model, embeddings, None)
    pair_sets = (
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
    test_reports: dict[str, JsonValue] = {}
    primary_pair_metrics = _object(evaluation, "pair_metrics")
    for population, uniform_name, stratified_name in pair_sets:
        uniform_report, _, _ = _pair_set_report(
            cache.pair_sets[uniform_name],
            cache,
            probes,
            final_latent,
            distance_fit,
            combined_fit,
        )
        primary_population = _object(primary_pair_metrics, population.value)
        _assert_primary_metric_reproduction(
            _object(primary_population, "uniform_model_metrics"),
            cast(dict[str, JsonValue], uniform_report["sequence"]),
        )
        stratified = cache.pair_sets[stratified_name]
        sequence_stratified = gather_pair_predictions_chunked(
            final_latent,
            stratified.pairs,
            chunk_size=16_384,
        ).to(t.float64)
        distance_stratified = distance_kernel_design(
            probes,
            stratified.pairs.left,
            stratified.pairs.right,
            length_scales=_LENGTH_SCALES,
        )
        combined_stratified = append_sequence_kernel(
            distance_stratified,
            sequence_stratified,
        )
        test_reports[population.value] = {
            "uniform": uniform_report,
            "by_distance": _stratified_rows(
                stratified,
                cache,
                distance_stratified,
                combined_stratified,
                distance_fit,
                combined_fit,
                final_latent,
            ),
        }

    record: dict[str, JsonValue] = {
        "schema": _SCHEMA,
        "status": "post_hoc_exploratory_not_confirmatory",
        "reason": (
            "Designed after inspecting the 1 kb and 4 kb primary test metrics; weights use only "
            "nested validation targets, but test results remain exploratory."
        ),
        "identity": {
            "code_git_commit": code_git_commit,
            "primary_git_commit": primary_git_commit,
            "protocol_id": protocol_id,
            "protocol_sha256": protocol_sha256,
            "data_sha256": data_sha256,
            "target_sha256": target_sha256,
            "split_name": split_name,
            "split_sha256": split_sha256,
            "window_size": window,
            "embedding_manifest_sha256": sha256_file(embeddings_path / "manifest.json"),
            "pair_cache_metadata_sha256": sha256_file(pair_directory / "metadata.json"),
            "selection_sha256": sha256_file(selection_path),
            "selected_tuning_metadata_sha256": sha256_file(selected_run / "metadata.json"),
            "selected_tuning_model_sha256": sha256_file(selected_run / "model.safetensors"),
            "final_model_metadata_sha256": sha256_file(final_directory / "metadata.json"),
            "final_model_sha256": sha256_file(final_directory / "model.safetensors"),
            "primary_evaluation_sha256": sha256_file(evaluation_path),
        },
        "kernel": {
            "family": "convex_direct_sum_psd_gram",
            "distance": "same_chromosome_laplacian_exp_minus_absolute_delta_over_scale",
            "length_scales_bp": list(_LENGTH_SCALES),
            "global_component": True,
            "sequence_component": "selected_full_model_cosine",
            "constraints": "weights_nonnegative_sum_one",
        },
        "validation": {
            "partition": "nested_validation_distinct_unordered_pairs",
            "pair_count": validation_target.numel(),
            "sequence_metrics": _metric_json(validation_target, validation_sequence),
            "distance_only_fit": _fit_json(distance_fit),
            "distance_plus_sequence_fit": _fit_json(combined_fit),
        },
        "test": test_reports,
    }
    output_path = output / split_name / f"window-{window}.json"
    if output_path.is_file():
        if _load_json(output_path) != record:
            raise ValueError(f"existing exploratory output differs: {output_path}")
    else:
        write_canonical_json_exclusive(output_path, record)
    print(
        f"distance-integration split={split_name} window={window} "
        f"validation_pairs={validation_target.numel()} output={output_path}"
    )


def main() -> None:
    arguments = _parser().parse_args()
    code_git_commit = require_clean_git_commit(Path(__file__).resolve().parents[1])
    config = load_protocol_config(arguments.config)
    if config.status != "frozen":
        raise ValueError("distance integration requires the frozen primary protocol")
    protocol_sha256 = sha256_file(arguments.config)
    bundle = verify_primary_data_bundle(
        arguments.data,
        protocol_id=config.protocol_id,
        protocol_sha256=protocol_sha256,
    )
    probes = load_probe_table(arguments.data / "probes.tsv")
    targets = load_target_geometry(arguments.data / "targets.safetensors")
    if len(probes) != bundle.retained_probes or targets.methylation.n_samples != bundle.retained_samples:
        raise ValueError("sealed data dimensions differ from exploratory inputs")
    configured_windows = {int(window): window for window in config.splits.window_sizes}
    requested_windows = tuple(arguments.windows)
    if (
        not requested_windows
        or len(set(requested_windows)) != len(requested_windows)
        or any(window not in configured_windows for window in requested_windows)
    ):
        raise ValueError("requested windows must be unique members of the frozen protocol")

    for split_name in _SPLITS:
        split_tensor = arguments.data / "splits" / f"{split_name}.safetensors"
        split_metadata = arguments.data / "splits" / f"{split_name}.json"
        split = load_split_artifact(
            split_tensor,
            split_metadata,
            probes=probes,
        )
        split_sha256 = sha256_ordered_strings(
            (sha256_file(split_metadata), sha256_file(split_tensor))
        )
        for window in requested_windows:
            _run_one(
                code_git_commit=code_git_commit,
                protocol_id=config.protocol_id,
                protocol_sha256=protocol_sha256,
                primary_git_commit=bundle.git_commit,
                data_directory=arguments.data,
                embedding_directory=arguments.embeddings,
                experiments=arguments.experiments,
                output=arguments.output,
                probes=probes,
                targets=targets,
                split=split,
                split_name=split_name,
                split_sha256=split_sha256,
                window_size=configured_windows[window],
                validation_seed=config.splits.validation_seed,
            )


if __name__ == "__main__":
    main()
