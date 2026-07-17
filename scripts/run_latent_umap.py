"""Project frozen validation latent spaces with deterministic exact Torch UMAP."""

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
from methylation_latent.domain import GenomicContext, NonEmptyProbeSet, parse_latent_dimension
from methylation_latent.embedding_cache import load_embedding_cache
from methylation_latent.experiment_data import load_split_artifact
from methylation_latent.model import LatentMetric, normalize_vector_strict
from methylation_latent.storage import load_exact_safetensors
from methylation_latent.tsne import deterministic_balanced_indices
from methylation_latent.umap import UmapConfig, exact_umap

_SCHEMA = "methylation-latent.validation-latent-umap.v1"
_TUNING_SCHEMA = "methylation-latent.latent-tuning-run.v2"
_MODEL_KEYS = {"age_direction", "projection.weight"}
_SPLITS = ("diverse-blocks", "held-out-chromosome")
_LAMBDA_AGE = 0.1
_MAXIMUM_DISPLAY_POINTS = 300
_SAMPLING_SEED = 711_301
_CONFIG = UmapConfig(
    n_neighbors=15,
    local_connectivity=1.0,
    smooth_knn_search_steps=64,
    min_dist=0.1,
    spread=1.0,
    optimization_steps=750,
    learning_rate=0.05,
    seed=618_437,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--experiments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", type=int, nargs="+", required=True)
    parser.add_argument("--dimensions", type=int, nargs="+", required=True)
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


def _load_tuning_model(
    directory: Path,
    *,
    protocol_id: str,
    protocol_sha256: str,
    primary_git_commit: str,
    data_sha256: str,
    target_sha256: str,
    split_name: str,
    split_sha256: str,
    window_size: int,
    embedding_manifest_sha256: str,
    validation_indices_sha256: str,
    latent_dimension: int,
) -> tuple[LatentMetric, dict[str, object]]:
    metadata_path = directory / "metadata.json"
    model_path = directory / "model.safetensors"
    metadata = _load_json(metadata_path)
    identity = _object(metadata, "identity")
    training = _object(metadata, "training")
    observed_identity = (
        _string(identity, "protocol_id"),
        _string(identity, "protocol_sha256"),
        _string(identity, "git_commit"),
        _string(identity, "data_sha256"),
        _string(identity, "target_sha256"),
        _string(identity, "split_name"),
        _string(identity, "split_sha256"),
        _integer(identity, "window_size"),
        _string(identity, "embedding_manifest_sha256"),
        _string(identity, "validation_indices_sha256"),
    )
    expected_identity = (
        protocol_id,
        protocol_sha256,
        primary_git_commit,
        data_sha256,
        target_sha256,
        split_name,
        split_sha256,
        window_size,
        embedding_manifest_sha256,
        validation_indices_sha256,
    )
    if (
        metadata.get("schema") != _TUNING_SCHEMA
        or observed_identity != expected_identity
        or _string(training, "mode") != "full"
        or _integer(training, "latent_dimension") != latent_dimension
        or _number(training, "lambda_age") != _LAMBDA_AGE
        or metadata.get("model_file") != model_path.name
        or metadata.get("model_sha256") != sha256_file(model_path)
    ):
        raise ValueError(f"UMAP tuning-model identity differs: {directory}")
    model = LatentMetric(
        embedding_dimension=256,
        latent_dimension=parse_latent_dimension(latent_dimension),
    )
    model.load_state_dict(load_exact_safetensors(model_path, _MODEL_KEYS), strict=True)
    model.to("cpu")
    model.eval()
    return model, metadata


def _context_labels(
    validation_indices: t.Tensor,
    contexts: tuple[GenomicContext, ...],
    probes: NonEmptyProbeSet,
) -> t.Tensor:
    by_context = {context: index for index, context in enumerate(contexts)}
    return t.tensor(
        tuple(by_context[probes.probes[index].context] for index in validation_indices.tolist()),
        dtype=t.int64,
    )


def main() -> None:
    arguments = _parser().parse_args()
    _configure_runtime()
    code_git_commit = require_clean_git_commit(Path(__file__).resolve().parents[1])
    config = load_protocol_config(arguments.config)
    if config.status != "frozen":
        raise ValueError("latent UMAP requires the frozen primary protocol")
    protocol_sha256 = sha256_file(arguments.config)
    bundle = verify_primary_data_bundle(
        arguments.data,
        protocol_id=config.protocol_id,
        protocol_sha256=protocol_sha256,
    )
    probes = load_probe_table(arguments.data / "probes.tsv")
    data_sha256 = sha256_file(arguments.data / "bundle.json")
    target_sha256 = sha256_file(arguments.data / "targets.safetensors")
    configured_windows = {int(window): window for window in config.splits.window_sizes}
    windows = tuple(arguments.windows)
    dimensions = tuple(arguments.dimensions)
    if (
        not windows
        or tuple(sorted(set(windows))) != windows
        or any(window not in configured_windows for window in windows)
    ):
        raise ValueError("UMAP windows must be increasing members of the frozen sweep")
    if (
        not dimensions
        or tuple(sorted(set(dimensions))) != dimensions
        or any(dimension > 128 for dimension in dimensions)
    ):
        raise ValueError("UMAP dimensions must be unique, increasing, and at most 128")
    for dimension in dimensions:
        parse_latent_dimension(dimension)
    contexts = tuple(GenomicContext)
    for split_offset, split_name in enumerate(_SPLITS):
        split_tensor = arguments.data / "splits" / f"{split_name}.safetensors"
        split_metadata = arguments.data / "splits" / f"{split_name}.json"
        split = load_split_artifact(split_tensor, split_metadata, probes=probes)
        split_sha256 = sha256_ordered_strings(
            (sha256_file(split_metadata), sha256_file(split_tensor))
        )
        validation_indices_sha256 = sha256_ordered_strings(
            map(str, split.validation_indices.tolist())
        )
        labels = _context_labels(split.validation_indices, contexts, probes)
        local_display = deterministic_balanced_indices(
            labels,
            group_count=len(contexts),
            maximum_points=_MAXIMUM_DISPLAY_POINTS,
            seed=_SAMPLING_SEED + split_offset,
        )
        display_indices = split.validation_indices.index_select(0, local_display)
        display_labels = labels.index_select(0, local_display)
        expected_context_count = _MAXIMUM_DISPLAY_POINTS // len(contexts)
        observed_context_counts = t.bincount(display_labels, minlength=len(contexts))
        if not t.equal(
            observed_context_counts,
            t.full((len(contexts),), expected_context_count, dtype=t.int64),
        ):
            raise RuntimeError("UMAP display sample is not exactly context-balanced")
        display_probes = tuple(probes.probes[index] for index in display_indices.tolist())
        for window in windows:
            embedding_directory = arguments.embeddings / f"window-{window}"
            embedding_manifest_path = embedding_directory / "manifest.json"
            embedding_manifest_sha256 = sha256_file(embedding_manifest_path)
            embeddings = load_embedding_cache(
                embedding_directory,
                probes,
                window_size=configured_windows[window],
            )
            display_embeddings = embeddings.tensor.index_select(0, display_indices).to(
                device="cpu", dtype=t.float32
            )
            for dimension in dimensions:
                tuning_directory = (
                    arguments.experiments
                    / "tuning"
                    / "full"
                    / split_name
                    / f"window-{window}"
                    / f"d-{dimension}-lambda-{_LAMBDA_AGE:g}"
                )
                model, metadata = _load_tuning_model(
                    tuning_directory,
                    protocol_id=config.protocol_id,
                    protocol_sha256=protocol_sha256,
                    primary_git_commit=bundle.git_commit,
                    data_sha256=data_sha256,
                    target_sha256=target_sha256,
                    split_name=split_name,
                    split_sha256=split_sha256,
                    window_size=window,
                    embedding_manifest_sha256=embedding_manifest_sha256,
                    validation_indices_sha256=validation_indices_sha256,
                    latent_dimension=dimension,
                )
                with t.inference_mode():
                    probe_latent = model.latent(display_embeddings).to(t.float64)
                    age_direction = normalize_vector_strict(model.age_direction).to(t.float64)
                    latent = t.cat((probe_latent, age_direction[None, :]), dim=0)
                row_norms = t.linalg.vector_norm(latent, dim=1)
                if not t.allclose(
                    row_norms,
                    t.ones_like(row_norms),
                    atol=2.0e-7,
                    rtol=0.0,
                ):
                    raise RuntimeError("UMAP input rows, including age, are not unit norm")
                projection = exact_umap(latent, config=_CONFIG)
                probe_coordinates = projection.coordinates[:-1]
                age_coordinate = projection.coordinates[-1]
                points: list[dict[str, JsonValue]] = [
                    {
                        "probe_id": str(probe.probe_id),
                        "chromosome": int(probe.chromosome),
                        "position": int(probe.position),
                        "context": probe.context.value,
                        "x": float(coordinate[0].item()),
                        "y": float(coordinate[1].item()),
                    }
                    for probe, coordinate in zip(
                        display_probes,
                        probe_coordinates,
                        strict=True,
                    )
                ]
                record: dict[str, JsonValue] = {
                    "schema": _SCHEMA,
                    "status": "post_hoc_visualization_validation_partition_only",
                    "identity": {
                        "code_git_commit": code_git_commit,
                        "primary_git_commit": bundle.git_commit,
                        "protocol_id": config.protocol_id,
                        "protocol_sha256": protocol_sha256,
                        "data_sha256": data_sha256,
                        "target_sha256": target_sha256,
                        "split_name": split_name,
                        "split_sha256": split_sha256,
                        "validation_indices_sha256": validation_indices_sha256,
                        "window_size": window,
                        "latent_dimension": dimension,
                        "lambda_age": _LAMBDA_AGE,
                        "embedding_manifest_sha256": embedding_manifest_sha256,
                        "tuning_metadata_sha256": sha256_file(tuning_directory / "metadata.json"),
                        "tuning_model_sha256": _string(metadata, "model_sha256"),
                        "display_indices_sha256": sha256_ordered_strings(
                            map(str, display_indices.tolist())
                        ),
                    },
                    "sampling": {
                        "partition": "nested_validation",
                        "target_access": "none",
                        "method": "metadata_context_balanced_seeded_without_replacement",
                        "seed": _SAMPLING_SEED + split_offset,
                        "source_count": split.validation_indices.numel(),
                        "display_probe_count": _MAXIMUM_DISPLAY_POINTS,
                        "count_per_context": expected_context_count,
                        "age_direction_count": 1,
                    },
                    "umap": {
                        "implementation": "exact_torch_fuzzy_cross_entropy",
                        "input_geometry": (
                            "euclidean_on_unit_probe_rows_and_unit_age_direction_"
                            "equivalent_to_cosine"
                        ),
                        "graph": "exact_knn_default_fuzzy_union",
                        "initialization": "deterministic_normalized_laplacian_spectral",
                        "objective": (
                            "complete_bernoulli_fuzzy_set_cross_entropy_without_"
                            "negative_sampling_approximation"
                        ),
                        "config": cast(dict[str, JsonValue], asdict(_CONFIG)),
                        "curve_a": projection.curve_a,
                        "curve_b": projection.curve_b,
                        "initial_cross_entropy": projection.initial_cross_entropy,
                        "final_cross_entropy": projection.final_cross_entropy,
                        "graph_edge_count": projection.graph_edge_count,
                        "spectral_gap": projection.spectral_gap,
                    },
                    "age_point": {
                        "label": "learned age direction",
                        "x": float(age_coordinate[0].item()),
                        "y": float(age_coordinate[1].item()),
                    },
                    "points": cast(list[JsonValue], points),
                }
                output_path = (
                    arguments.output
                    / split_name
                    / f"window-{window}"
                    / f"d-{dimension}-lambda-{_LAMBDA_AGE:g}.json"
                )
                if output_path.is_file():
                    if _load_json(output_path) != record:
                        raise ValueError(f"existing UMAP output differs: {output_path}")
                else:
                    write_canonical_json_exclusive(output_path, record)
                print(
                    f"umap split={split_name} window={window} d={dimension} "
                    f"lambda={_LAMBDA_AGE:g} probes={len(points)} age=1 "
                    f"cross_entropy={projection.final_cross_entropy:.8f} output={output_path}"
                )


if __name__ == "__main__":
    main()
