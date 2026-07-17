"""Compile completed windows without weakening the all-windows primary site contract."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import cast

from methylation_latent.artifacts import (
    JsonValue,
    require_clean_git_commit,
    sha256_file,
    sha256_ordered_strings,
    write_canonical_json_exclusive,
)
from methylation_latent.cohort import load_probe_table
from methylation_latent.config import ProtocolConfig, load_protocol_config
from methylation_latent.data_bundle import verify_primary_data_bundle
from methylation_latent.domain import (
    GenomicContext,
    NonEmptyProbeSet,
    parse_autosome,
    parse_one_based_position,
    parse_probe_id,
)
from methylation_latent.evaluation import PairPopulation
from methylation_latent.experiment_data import (
    SequenceFeatureArtifact,
    load_sequence_features,
    load_split_artifact,
)
from methylation_latent.interim_site import (
    INTERIM_SITE_SCHEMA,
    AgeClusterAgreement,
    AgeClusterDiagnostic,
    ContextClusterCounts,
    CorrelationScatterPanel,
    ExploratoryAgeMetricPoint,
    ExploratoryMetricPoint,
    InterimSiteData,
    KernelWeightPoint,
    LatentSpherePanel,
    LatentSpherePoint,
    ProbeDisplayAnnotation,
    ScatterPredictionSeries,
    TuningSweepPoint,
    build_interim_static_site,
)
from methylation_latent.site import AgeMetricPoint, DistanceMetricPoint, WindowSweepPoint

_EVALUATION_SCHEMA = "methylation-latent.held-out-evaluation.v2"
_TUNING_SCHEMA = "methylation-latent.latent-tuning-run.v2"
_SELECTION_SCHEMA = "methylation-latent.hyperparameter-selection.v2"
_DISTANCE_SCHEMA = "methylation-latent.exploratory-distance-integration.v2"
_DIRECT_AGE_SCHEMA = "methylation-latent.exploratory-direct-tanh-age.v1"
_SCATTER_SCHEMA = "methylation-latent.prediction-scatter.v1"
_CLUSTER_SCHEMA = "methylation-latent.age-scatter-cluster-audit.v1"
_UMAP_SCHEMA = "methylation-latent.validation-latent-umap.v3"
_SPLITS = (
    ("diverse-blocks", "diverse_blocks"),
    ("held-out-chromosome", "held_out_chromosome"),
)
_POPULATIONS = (
    PairPopulation.SEEN_BY_HELD_OUT,
    PairPopulation.HELD_OUT_BY_HELD_OUT,
)
_STAGES = ("sequence_features", "caduceus_age_only", "full_latent_metric")
_PRIMARY_REPRODUCTION_MAX_ULPS = 8


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--series-matrix", type=Path, required=True)
    parser.add_argument("--experiments", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--distance-integration", type=Path, required=True)
    parser.add_argument("--direct-age", type=Path, required=True)
    parser.add_argument("--prediction-scatter", type=Path, required=True)
    parser.add_argument("--age-cluster-audit", type=Path, required=True)
    parser.add_argument("--latent-umap", type=Path, required=True)
    parser.add_argument("--results-output", type=Path, required=True)
    parser.add_argument("--site-template", type=Path, required=True)
    parser.add_argument("--root-template", type=Path, required=True)
    parser.add_argument("--site-output", type=Path, required=True)
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


def _objects(record: dict[str, object], key: str) -> tuple[dict[str, object], ...]:
    value = record.get(key)
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise TypeError(f"{key} must be an array of objects")
    return cast(tuple[dict[str, object], ...], tuple(value))


def _exact_keys(record: dict[str, object], expected: set[str], name: str) -> None:
    observed = set(record)
    if observed != expected:
        raise ValueError(
            f"{name} fields differ: missing={sorted(expected - observed)}, "
            f"unknown={sorted(observed - expected)}"
        )


def _integers(record: dict[str, object], key: str) -> tuple[int, ...]:
    value = record.get(key)
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise TypeError(f"{key} must be an array of integers")
    return cast(tuple[int, ...], tuple(value))


def _numbers(record: dict[str, object], key: str) -> tuple[float, ...]:
    value = record.get(key)
    if not isinstance(value, list):
        raise TypeError(f"{key} must be an array of numbers")
    return tuple(_number({"value": item}, "value") for item in cast(list[object], value))


def _strings(record: dict[str, object], key: str) -> tuple[str, ...]:
    value = record.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise TypeError(f"{key} must be an array of non-empty strings")
    return cast(tuple[str, ...], tuple(value))


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


def _identity_tuple(record: dict[str, object]) -> tuple[str, str, str, str, str, int]:
    identity = _object(record, "identity")
    return (
        _string(identity, "protocol_id"),
        _string(identity, "protocol_sha256"),
        _string(identity, "data_sha256"),
        _string(identity, "split_name"),
        _string(identity, "split_sha256"),
        _integer(identity, "window_size"),
    )


def _expected_identity(
    *,
    config: ProtocolConfig,
    protocol_sha256: str,
    data_sha256: str,
    split_name: str,
    split_sha256: str,
    window: int,
) -> tuple[str, str, str, str, str, int]:
    return (
        config.protocol_id,
        protocol_sha256,
        data_sha256,
        split_name,
        split_sha256,
        window,
    )


def _evaluation_records(
    experiments: Path,
    *,
    expected_prefix: tuple[str, str, str, str, str],
    windows: tuple[int, ...],
) -> tuple[tuple[Path, dict[str, object]], ...]:
    records: list[tuple[Path, dict[str, object]]] = []
    split_name = expected_prefix[3]
    for window in windows:
        path = experiments / "evaluation" / split_name / f"window-{window}.json"
        record = _load_json(path)
        if record.get("schema") != _EVALUATION_SCHEMA or _identity_tuple(record) != (
            *expected_prefix,
            window,
        ):
            raise ValueError(f"evaluation identity differs: {path}")
        records.append((path, record))
    return tuple(records)


def _primary_window_points(
    records: tuple[tuple[Path, dict[str, object]], ...],
) -> tuple[WindowSweepPoint, ...]:
    points: list[WindowSweepPoint] = []
    for _, record in records:
        window = _integer(_object(record, "identity"), "window_size")
        selected = _object(record, "selected_full_training")
        pair_metrics = _object(record, "pair_metrics")
        for population in _POPULATIONS:
            metrics = _object(_object(pair_metrics, population.value), "uniform_model_metrics")
            points.append(
                WindowSweepPoint(
                    window_size=window,
                    population=population,
                    latent_dimension=_integer(selected, "latent_dimension"),
                    lambda_age=_number(selected, "lambda_age"),
                    mse=_number(metrics, "mse"),
                    pearson=_number(metrics, "pearson"),
                    r_squared=_number(metrics, "r_squared"),
                )
            )
    return tuple(points)


def _primary_age_points(
    records: tuple[tuple[Path, dict[str, object]], ...],
) -> tuple[AgeMetricPoint, ...]:
    return tuple(
        AgeMetricPoint(
            window_size=_integer(_object(record, "identity"), "window_size"),
            stage=stage,
            count=_integer(report, "count"),
            mse=_number(report, "mse"),
            pearson=_number(report, "pearson"),
        )
        for _, record in records
        for stage in _STAGES
        for report in (_object(_object(record, "age_metrics"), stage),)
    )


def _primary_distance_points(
    records: tuple[tuple[Path, dict[str, object]], ...],
) -> tuple[DistanceMetricPoint, ...]:
    points: list[DistanceMetricPoint] = []
    for _, record in records:
        window = _integer(_object(record, "identity"), "window_size")
        pair_metrics = _object(record, "pair_metrics")
        for population in _POPULATIONS:
            for row in _objects(_object(pair_metrics, population.value), "by_distance"):
                metrics = _object(row, "model_metrics")
                points.append(
                    DistanceMetricPoint(
                        window_size=window,
                        population=population,
                        distance_class=_string(row, "distance_class"),
                        count=_integer(metrics, "count"),
                        target_mean=_number(metrics, "target_mean"),
                        prediction_mean=_number(metrics, "prediction_mean"),
                        distance_baseline_mean=_number(row, "distance_baseline_mean"),
                        mse=_number(metrics, "mse"),
                        pearson=_number(metrics, "pearson"),
                        r_squared=_number(metrics, "r_squared"),
                    )
                )
    return tuple(points)


def _tuning_points(
    experiments: Path,
    *,
    config: ProtocolConfig,
    expected_prefix: tuple[str, str, str, str, str],
    windows: tuple[int, ...],
) -> tuple[tuple[TuningSweepPoint, ...], tuple[str, ...]]:
    points: list[TuningSweepPoint] = []
    artifact_ids: list[str] = []
    split_name = expected_prefix[3]
    for window in windows:
        selection_path = experiments / "selections" / "full" / split_name / f"window-{window}.json"
        selection = _load_json(selection_path)
        if selection.get("schema") != _SELECTION_SCHEMA or _identity_tuple(selection) != (
            *expected_prefix,
            window,
        ):
            raise ValueError(f"selection identity differs: {selection_path}")
        selected_run = _string(selection, "selected_run")
        selected_hash = _string(selection, "selected_metadata_sha256")
        candidate_hashes: list[str] = []
        for dimension in map(int, config.sweep.latent_dimensions):
            for lambda_age in map(float, config.sweep.lambda_age):
                relative = (
                    Path("tuning")
                    / "full"
                    / split_name
                    / f"window-{window}"
                    / f"d-{dimension}-lambda-{lambda_age:g}"
                )
                metadata_path = experiments / relative / "metadata.json"
                metadata = _load_json(metadata_path)
                metadata_hash = sha256_file(metadata_path)
                candidate_hashes.append(metadata_hash)
                training = _object(metadata, "training")
                validation = _object(metadata, "selected_validation")
                if (
                    metadata.get("schema") != _TUNING_SCHEMA
                    or _identity_tuple(metadata) != (*expected_prefix, window)
                    or _string(training, "mode") != "full"
                    or _integer(training, "latent_dimension") != dimension
                    or _number(training, "lambda_age") != lambda_age
                ):
                    raise ValueError(f"tuning identity differs: {metadata_path}")
                is_selected = selected_run == relative.as_posix()
                if is_selected and selected_hash != metadata_hash:
                    raise ValueError("selection hash differs from its selected tuning metadata")
                points.append(
                    TuningSweepPoint(
                        window_size=window,
                        latent_dimension=dimension,
                        lambda_age=lambda_age,
                        selected_step=_integer(metadata, "selected_step"),
                        pair_mse=_number(validation, "pair_mse"),
                        age_mse=_number(validation, "age_mse"),
                        selection_score=_number(validation, "selection_score"),
                        selected=is_selected,
                    )
                )
                artifact_ids.append(f"tuning:{window}:{dimension}:{lambda_age:g}:{metadata_hash}")
        identity = _object(selection, "identity")
        if _strings(identity, "candidate_metadata_sha256") != tuple(candidate_hashes):
            raise ValueError("selection candidate hashes differ from the complete tuning grid")
        if sum(point.selected for point in points if point.window_size == window) != 1:
            raise ValueError("selection must resolve to exactly one tuning candidate")
        artifact_ids.append(f"selection:{window}:{sha256_file(selection_path)}")
    return tuple(points), tuple(artifact_ids)


def _exploratory_metric(
    record: dict[str, object],
    *,
    window: int,
    population: PairPopulation,
    distance_class: str | None,
    model: str,
) -> ExploratoryMetricPoint:
    pearson_raw = record.get("pearson")
    pearson: float | None
    if pearson_raw is None:
        pearson = None
        pearson_status = _string(record, "pearson_status")
    else:
        pearson = _number(record, "pearson")
        status_raw = record.get("pearson_status", "defined")
        if not isinstance(status_raw, str):
            raise TypeError("pearson_status must be a string")
        pearson_status = status_raw
    return ExploratoryMetricPoint(
        window_size=window,
        population=population,
        distance_class=distance_class,
        model=model,
        count=_integer(record, "count"),
        target_mean=_number(record, "target_mean"),
        prediction_mean=_number(record, "prediction_mean"),
        mse=_number(record, "mse"),
        pearson=pearson,
        pearson_status=pearson_status,
        r_squared=_number(record, "r_squared"),
    )


def _assert_primary_distance_alignment(
    primary: DistanceMetricPoint,
    row: dict[str, object],
) -> None:
    sequence = _object(row, "sequence")
    if _integer(sequence, "count") != primary.count:
        raise ValueError("exploratory distance row count differs from the primary record")
    values = (
        ("target_mean", _number(sequence, "target_mean"), primary.target_mean),
        ("prediction_mean", _number(sequence, "prediction_mean"), primary.prediction_mean),
        ("mse", _number(sequence, "mse"), primary.mse),
        ("pearson", _number(sequence, "pearson"), primary.pearson),
        ("r_squared", _number(sequence, "r_squared"), primary.r_squared),
        (
            "registered_distance_mean",
            _number(row, "registered_distance_mean"),
            primary.distance_baseline_mean,
        ),
    )
    for name, observed, expected in values:
        tolerance = _PRIMARY_REPRODUCTION_MAX_ULPS * max(
            math.ulp(observed),
            math.ulp(expected),
        )
        if abs(observed - expected) > tolerance:
            raise ValueError(
                "exploratory distance row exceeds the primary reproduction bound: "
                f"field={name}, observed={observed}, expected={expected}, "
                f"absolute_difference={abs(observed - expected)}, tolerance={tolerance}"
            )


def _distance_integration_points(
    distance_root: Path,
    *,
    expected_prefix: tuple[str, str, str, str, str],
    windows: tuple[int, ...],
    evaluations: tuple[tuple[Path, dict[str, object]], ...],
    primary_distance: tuple[DistanceMetricPoint, ...],
) -> tuple[
    tuple[ExploratoryMetricPoint, ...],
    tuple[ExploratoryMetricPoint, ...],
    tuple[KernelWeightPoint, ...],
    tuple[str, ...],
]:
    uniform: list[ExploratoryMetricPoint] = []
    by_distance: list[ExploratoryMetricPoint] = []
    weights: list[KernelWeightPoint] = []
    artifact_ids: list[str] = []
    split_name = expected_prefix[3]
    evaluation_by_window = {
        _integer(_object(record, "identity"), "window_size"): path for path, record in evaluations
    }
    primary_by_key = {
        (point.window_size, point.population, point.distance_class): point
        for point in primary_distance
    }
    for window in windows:
        path = distance_root / split_name / f"window-{window}.json"
        record = _load_json(path)
        identity = _object(record, "identity")
        if (
            record.get("schema") != _DISTANCE_SCHEMA
            or record.get("status") != "post_hoc_exploratory_not_confirmatory"
            or _identity_tuple(record) != (*expected_prefix, window)
            or _string(identity, "primary_evaluation_sha256")
            != sha256_file(evaluation_by_window[window])
            or _object(record, "runtime")
            != {
                "device": "cpu",
                "torch_deterministic_algorithms": True,
                "torch_num_interop_threads": 1,
                "torch_num_threads": 1,
            }
        ):
            raise ValueError(f"exploratory distance identity differs: {path}")
        combined_fit = _object(_object(record, "validation"), "distance_plus_sequence_fit")
        names = _strings(combined_fit, "names")
        raw_weights = combined_fit.get("weights")
        if not isinstance(raw_weights, list) or len(raw_weights) != len(names):
            raise TypeError("combined kernel weights must align to component names")
        parsed_weights = tuple(
            _number({"weight": value}, "weight") for value in cast(list[object], raw_weights)
        )
        for name, weight in zip(names, parsed_weights, strict=True):
            weights.append(KernelWeightPoint(window, name, weight))
        test = _object(record, "test")
        for population in _POPULATIONS:
            report = _object(test, population.value)
            uniform_report = _object(report, "uniform")
            for model in (
                "sequence",
                "registered_distance_class",
                "psd_distance",
                "psd_distance_plus_sequence",
            ):
                uniform.append(
                    _exploratory_metric(
                        _object(uniform_report, model),
                        window=window,
                        population=population,
                        distance_class=None,
                        model=model,
                    )
                )
            for row in _objects(report, "by_distance"):
                distance_class = _string(row, "distance_class")
                primary = primary_by_key[(window, population, distance_class)]
                _assert_primary_distance_alignment(primary, row)
                for model in ("sequence", "psd_distance", "psd_distance_plus_sequence"):
                    by_distance.append(
                        _exploratory_metric(
                            _object(row, model),
                            window=window,
                            population=population,
                            distance_class=distance_class,
                            model=model,
                        )
                    )
        artifact_ids.append(f"exploratory-distance:{window}:{sha256_file(path)}")
    return tuple(uniform), tuple(by_distance), tuple(weights), tuple(artifact_ids)


def _direct_age_points(
    direct_root: Path,
    embeddings_root: Path,
    *,
    expected_prefix: tuple[str, str, str, str, str],
    primary_git_commit: str,
    target_sha256: str,
    windows: tuple[int, ...],
) -> tuple[
    tuple[ExploratoryAgeMetricPoint, ...],
    tuple[str, ...],
    dict[int, tuple[Path, dict[str, object]]],
]:
    points: list[ExploratoryAgeMetricPoint] = []
    artifact_ids: list[str] = []
    records: dict[int, tuple[Path, dict[str, object]]] = {}
    split_name = expected_prefix[3]
    for window in windows:
        directory = direct_root / split_name / f"window-{window}"
        path = directory / "metadata.json"
        predictions_path = directory / "predictions.safetensors"
        record = _load_json(path)
        identity = _object(record, "identity")
        training = _object(record, "training")
        runtime = _object(record, "runtime")
        if (
            record.get("schema") != _DIRECT_AGE_SCHEMA
            or record.get("status") != "post_hoc_corrected_baseline_not_confirmatory"
            or _identity_tuple(record) != (*expected_prefix, window)
            or _string(identity, "primary_git_commit") != primary_git_commit
            or _string(identity, "target_sha256") != target_sha256
            or _string(identity, "embedding_manifest_sha256")
            != sha256_file(embeddings_root / f"window-{window}" / "manifest.json")
            or training.get("architecture") != "affine_256_to_1_then_tanh"
            or training.get("output_dimension") != 1
            or training.get("latent_dimension") is not None
            or training.get("lambda_age") is not None
            or training.get("loss") != "mean_squared_error_over_probe_age_correlations"
            or _number(training, "weight_decay") != 0.0
            or runtime
            != {
                "device": "cpu",
                "torch_deterministic_algorithms": True,
                "torch_num_interop_threads": 1,
                "torch_num_threads": 1,
            }
            or record.get("predictions_sha256") != sha256_file(predictions_path)
        ):
            raise ValueError(f"direct tanh age identity differs: {path}")
        held_out = _object(record, "held_out_metrics")
        selected_validation = _object(record, "selected_validation")
        points.append(
            ExploratoryAgeMetricPoint(
                window_size=window,
                count=_integer(held_out, "count"),
                selected_step=_integer(record, "selected_step"),
                validation_mse=_number(selected_validation, "age_mse"),
                mse=_number(held_out, "mse"),
                pearson=_number(held_out, "pearson"),
                r_squared=_number(held_out, "r_squared"),
            )
        )
        records[window] = (path, record)
        artifact_ids.append(f"exploratory-direct-tanh-age:{window}:{sha256_file(path)}")
    return tuple(points), tuple(artifact_ids), records


def _scatter_panel(
    raw: dict[str, object],
    *,
    window: int,
    population: str,
    annotations: tuple[ProbeDisplayAnnotation, ...],
) -> CorrelationScatterPanel:
    _exact_keys(
        raw,
        {
            "source_count",
            "display_count",
            "seed",
            "sample_indices",
            "target",
            "predictions",
        },
        "scatter sample",
    )
    predictions = _object(raw, "predictions")
    series = tuple(
        ScatterPredictionSeries(
            model=model,
            values=_numbers({"values": values}, "values"),
        )
        for model, values in sorted(predictions.items())
    )
    panel = CorrelationScatterPanel(
        window_size=window,
        population=population,
        source_count=_integer(raw, "source_count"),
        seed=_integer(raw, "seed"),
        sample_indices=_integers(raw, "sample_indices"),
        target=_numbers(raw, "target"),
        predictions=series,
        annotations=annotations,
    )
    if _integer(raw, "display_count") != len(panel.target):
        raise ValueError("scatter display count differs from its aligned vectors")
    return panel


def _probe_annotations(
    indices: tuple[int, ...],
    *,
    window: int,
    probes: NonEmptyProbeSet,
    features: SequenceFeatureArtifact,
) -> tuple[ProbeDisplayAnnotation, ...]:
    if window not in features.by_window:
        raise ValueError(f"sequence features do not contain window {window}")
    if any(index < 0 or index >= len(probes) for index in indices):
        raise ValueError("probe annotation indices are outside the frozen probe universe")
    values = features.by_window[window]
    return tuple(
        ProbeDisplayAnnotation(
            probe_id=probes.probes[index].probe_id,
            chromosome=probes.probes[index].chromosome,
            position=probes.probes[index].position,
            context=probes.probes[index].context,
            design=probes.probes[index].design,
            manifest_strand=probes.probes[index].manifest_strand,
            cpg_density=float(values[index, 0].item()),
            gc_content=float(values[index, 1].item()),
        )
        for index in indices
    )


def _scatter_panels(
    scatter_root: Path,
    experiments: Path,
    embeddings_root: Path,
    *,
    expected_prefix: tuple[str, str, str, str, str],
    primary_git_commit: str,
    target_sha256: str,
    windows: tuple[int, ...],
    evaluations: tuple[tuple[Path, dict[str, object]], ...],
    direct_records: dict[int, tuple[Path, dict[str, object]]],
    test_indices: tuple[int, ...],
    probes: NonEmptyProbeSet,
    features: SequenceFeatureArtifact,
) -> tuple[tuple[CorrelationScatterPanel, ...], tuple[str, ...]]:
    panels: list[CorrelationScatterPanel] = []
    artifact_ids: list[str] = []
    split_name = expected_prefix[3]
    evaluation_paths = {
        _integer(_object(record, "identity"), "window_size"): path for path, record in evaluations
    }
    pair_metadata = experiments / "evaluation-pairs" / split_name / "metadata.json"
    for window in windows:
        path = scatter_root / split_name / f"window-{window}.json"
        record = _load_json(path)
        identity = _object(record, "identity")
        direct_path, direct_record = direct_records[window]
        full_metadata = (
            experiments / "final" / "full" / split_name / f"window-{window}" / "metadata.json"
        )
        age_metadata = (
            experiments / "final" / "age_only" / split_name / f"window-{window}" / "metadata.json"
        )
        full_record = _load_json(full_metadata)
        age_record = _load_json(age_metadata)
        if (
            record.get("schema") != _SCATTER_SCHEMA
            or record.get("status") != "display_only_no_fitting"
            or _identity_tuple(record) != (*expected_prefix, window)
            or _string(identity, "primary_git_commit") != primary_git_commit
            or _string(identity, "target_sha256") != target_sha256
            or _string(identity, "embedding_manifest_sha256")
            != sha256_file(embeddings_root / f"window-{window}" / "manifest.json")
            or _string(identity, "evaluation_sha256") != sha256_file(evaluation_paths[window])
            or _string(identity, "pair_cache_metadata_sha256") != sha256_file(pair_metadata)
            or _string(identity, "full_model_metadata_sha256") != sha256_file(full_metadata)
            or _string(identity, "full_model_sha256") != _string(full_record, "model_sha256")
            or _string(identity, "cosine_age_metadata_sha256") != sha256_file(age_metadata)
            or _string(identity, "cosine_age_model_sha256") != _string(age_record, "model_sha256")
            or _string(identity, "direct_age_metadata_sha256") != sha256_file(direct_path)
            or _string(identity, "direct_age_predictions_sha256")
            != _string(direct_record, "predictions_sha256")
            or _object(record, "sampling")
            != {
                "maximum_points": 5_000,
                "method": "target_blind_seeded_without_replacement_then_sorted",
            }
        ):
            raise ValueError(f"prediction scatter identity differs: {path}")
        age_raw = _object(record, "age")
        display_indices = _integers(age_raw, "sample_indices")
        if _integer(age_raw, "source_count") != len(test_indices):
            raise ValueError("age-scatter source count differs from frozen test probe count")
        annotations = _probe_annotations(
            tuple(test_indices[index] for index in display_indices),
            window=window,
            probes=probes,
            features=features,
        )
        panels.append(
            _scatter_panel(
                age_raw,
                window=window,
                population="age",
                annotations=annotations,
            )
        )
        pairs = _object(record, "pairs")
        _exact_keys(
            pairs,
            {population.value for population in _POPULATIONS},
            "scatter pair populations",
        )
        panels.extend(
            _scatter_panel(
                _object(pairs, population.value),
                window=window,
                population=population.value,
                annotations=(),
            )
            for population in _POPULATIONS
        )
        artifact_ids.append(f"prediction-scatter:{window}:{sha256_file(path)}")
    return tuple(panels), tuple(artifact_ids)


def _two_by_two_numbers(record: dict[str, object], key: str) -> tuple[tuple[float, float], ...]:
    value = record.get(key)
    if not isinstance(value, list) or len(value) != 2:
        raise TypeError(f"{key} must be a two-by-two numeric matrix")
    rows: list[tuple[float, float]] = []
    for raw_row in value:
        if not isinstance(raw_row, list) or len(raw_row) != 2:
            raise TypeError(f"{key} must be a two-by-two numeric matrix")
        parsed = tuple(_number({"value": item}, "value") for item in raw_row)
        rows.append(cast(tuple[float, float], parsed))
    return tuple(rows)


def _cluster_diagnostic(
    raw: dict[str, object],
    *,
    window: int,
    model: str,
    source_count: int,
) -> AgeClusterDiagnostic:
    kmeans = _object(raw, "kmeans")
    labels = _integers(kmeans, "labels_in_frozen_test_order")
    if len(labels) != source_count or set(labels) != {0, 1}:
        raise ValueError("age-cluster labels must align to all frozen test probes")
    centers = _two_by_two_numbers(kmeans, "centers_empirical_then_predicted")
    categorical = _object(raw, "categorical_associations")
    context = _object(categorical, "genomic_context")
    design = _object(categorical, "probe_design")
    strand = _object(categorical, "manifest_strand")
    chromosome = _object(categorical, "chromosome")
    context_counts_raw = _object(context, "counts")
    if set(context_counts_raw) != {context.value for context in GenomicContext}:
        raise ValueError("age-cluster context levels differ")
    context_counts = tuple(
        ContextClusterCounts(
            context=context_value,
            negative_count=_integer(
                _object(context_counts_raw, context_value.value), "negative_cluster"
            ),
            positive_count=_integer(
                _object(context_counts_raw, context_value.value), "positive_cluster"
            ),
        )
        for context_value in GenomicContext
    )
    sequence = _object(raw, "sequence_feature_contrasts")
    cpg_density = _object(sequence, "cpg_density")
    gc_content = _object(sequence, "gc_content")
    sex = _object(raw, "sex_stratified_targets")
    female = _object(sex, "female")
    male = _object(sex, "male")
    female_means = _numbers(female, "cluster_target_means")
    male_means = _numbers(male, "cluster_target_means")
    if len(female_means) != 2 or len(male_means) != 2:
        raise ValueError("sex-stratified cluster means must each contain two values")
    chromosome_status = _string(chromosome, "status")
    diagnostic = AgeClusterDiagnostic(
        window_size=window,
        model=model,
        source_count=source_count,
        negative_count=_integer(kmeans, "negative_cluster_count"),
        positive_count=_integer(kmeans, "positive_cluster_count"),
        negative_empirical_center=centers[0][0],
        negative_prediction_center=centers[0][1],
        positive_empirical_center=centers[1][0],
        positive_prediction_center=centers[1][1],
        context_cramer_v=_number(context, "cramer_v"),
        design_cramer_v=_number(design, "cramer_v"),
        strand_cramer_v=_number(strand, "cramer_v"),
        chromosome_cramer_v=(
            _number(chromosome, "cramer_v") if chromosome_status == "defined" else None
        ),
        chromosome_status=chromosome_status,
        cpg_density_cohen_d=_number(cpg_density, "standardized_mean_difference"),
        gc_content_cohen_d=_number(gc_content, "standardized_mean_difference"),
        female_sample_count=_integer(female, "sample_count"),
        male_sample_count=_integer(male, "sample_count"),
        female_negative_mean=female_means[0],
        female_positive_mean=female_means[1],
        female_separation_cohen_d=_number(female, "cluster_separation_cohen_d"),
        male_negative_mean=male_means[0],
        male_positive_mean=male_means[1],
        male_separation_cohen_d=_number(male, "cluster_separation_cohen_d"),
        female_male_target_pearson=_number(sex, "female_vs_male_target_pearson"),
        context_counts=context_counts,
    )
    if labels.count(0) != diagnostic.negative_count or labels.count(1) != diagnostic.positive_count:
        raise ValueError("age-cluster label counts differ from the audit summary")
    return diagnostic


def _age_cluster_diagnostics(
    cluster_root: Path,
    data_root: Path,
    series_matrix: Path,
    experiments: Path,
    embeddings_root: Path,
    direct_root: Path,
    *,
    expected_prefix: tuple[str, str, str, str, str],
    primary_git_commit: str,
    target_sha256: str,
    test_indices_sha256: str,
    test_count: int,
    windows: tuple[int, ...],
) -> tuple[
    tuple[AgeClusterDiagnostic, ...],
    tuple[AgeClusterAgreement, ...],
    int,
    int,
    str,
    tuple[str, ...],
]:
    diagnostics: list[AgeClusterDiagnostic] = []
    agreements: list[AgeClusterAgreement] = []
    artifact_ids: list[str] = []
    phenotype_identity: tuple[int, int, str] | None = None
    split_name = expected_prefix[3]
    model_names = ("cosine_age_only", "full_latent_metric", "direct_tanh")
    local_hashes = {
        "cohort_metadata": sha256_file(data_root / "cohort.json"),
        "cohort_tensor": sha256_file(data_root / "cohort.safetensors"),
        "series_matrix": sha256_file(series_matrix),
        "sequence_features_metadata": sha256_file(data_root / "sequence-features.json"),
        "sequence_features_tensor": sha256_file(data_root / "sequence-features.safetensors"),
    }
    for window in windows:
        path = cluster_root / split_name / f"window-{window}.json"
        record = _load_json(path)
        identity = _object(record, "identity")
        scope = _object(record, "scope")
        phenotype = _object(record, "phenotype_audit")
        cell_composition = _object(phenotype, "cell_composition")
        female_age = _object(phenotype, "female_age")
        male_age = _object(phenotype, "male_age")
        evaluation = experiments / "evaluation" / split_name / f"window-{window}.json"
        direct_metadata = direct_root / split_name / f"window-{window}" / "metadata.json"
        direct_predictions = (
            direct_root / split_name / f"window-{window}" / "predictions.safetensors"
        )
        if (
            record.get("schema") != _CLUSTER_SCHEMA
            or record.get("status") != "post_hoc_explanatory_frozen_test_targets_used_for_diagnosis"
            or _identity_tuple(record) != (*expected_prefix, window)
            or _string(identity, "primary_git_commit") != primary_git_commit
            or _string(identity, "target_sha256") != target_sha256
            or _string(identity, "cohort_metadata_sha256") != local_hashes["cohort_metadata"]
            or _string(identity, "cohort_tensor_sha256") != local_hashes["cohort_tensor"]
            or _string(identity, "series_matrix_sha256") != local_hashes["series_matrix"]
            or _string(identity, "sequence_features_metadata_sha256")
            != local_hashes["sequence_features_metadata"]
            or _string(identity, "sequence_features_tensor_sha256")
            != local_hashes["sequence_features_tensor"]
            or _string(identity, "test_indices_sha256") != test_indices_sha256
            or _string(identity, "embedding_manifest_sha256")
            != sha256_file(embeddings_root / f"window-{window}" / "manifest.json")
            or _string(identity, "evaluation_sha256") != sha256_file(evaluation)
            or _integer(scope, "test_probe_count") != test_count
            or _strings(scope, "models") != model_names
            or scope.get("cluster_input_uses_empirical_target") is not True
            or scope.get("confirmatory_status") != "post_hoc_not_for_model_selection"
            or _strings(phenotype, "observed_series_fields")
            != ("age", "gender", "tissue", "disease state")
            or _strings(phenotype, "tissue_levels") != ("whole blood",)
            or _strings(phenotype, "disease_state_levels") != ("normal",)
            or _string(cell_composition, "status")
            != "not_testable_no_measured_or_precomputed_cell_proportions"
        ):
            raise ValueError(f"age-cluster audit identity differs: {path}")
        model_artifacts = _object(identity, "model_artifacts")
        if set(model_artifacts) != set(model_names):
            raise ValueError("age-cluster model-artifact identities differ")
        for model_name, stage in (
            ("cosine_age_only", "age_only"),
            ("full_latent_metric", "full"),
        ):
            artifact = _object(model_artifacts, model_name)
            directory = experiments / "final" / stage / split_name / f"window-{window}"
            model_metadata = _load_json(directory / "metadata.json")
            if _string(artifact, "metadata_sha256") != sha256_file(
                directory / "metadata.json"
            ) or _string(artifact, "model_sha256") != _string(model_metadata, "model_sha256"):
                raise ValueError("age-cluster final-model identity differs")
        direct_artifact = _object(model_artifacts, "direct_tanh")
        if _string(direct_artifact, "metadata_sha256") != sha256_file(direct_metadata) or _string(
            direct_artifact, "predictions_sha256"
        ) != sha256_file(direct_predictions):
            raise ValueError("age-cluster direct-tanh identity differs")
        observed_phenotype = (
            _integer(female_age, "count"),
            _integer(male_age, "count"),
            _string(cell_composition, "status"),
        )
        if phenotype_identity is None:
            phenotype_identity = observed_phenotype
        elif phenotype_identity != observed_phenotype:
            raise ValueError("age-cluster phenotype audit differs across windows")
        raw_analyses = _object(record, "analyses")
        if set(raw_analyses) != set(model_names):
            raise ValueError("age-cluster analysis models differ")
        for model_name in model_names:
            diagnostics.append(
                _cluster_diagnostic(
                    _object(raw_analyses, model_name),
                    window=window,
                    model=model_name,
                    source_count=test_count,
                )
            )
        raw_agreement = _object(
            _object(record, "cross_model_cluster_agreement"),
            "cosine_age_only__full_latent_metric",
        )
        agreements.append(
            AgeClusterAgreement(
                window_size=window,
                source_count=test_count,
                permutation_invariant_fraction=_number(
                    raw_agreement, "permutation_invariant_fraction"
                ),
                adjusted_rand_index=_number(raw_agreement, "adjusted_rand_index"),
            )
        )
        artifact_ids.append(f"age-scatter-cluster-audit:{window}:{sha256_file(path)}")
    if phenotype_identity is None:
        raise RuntimeError("age-cluster audit did not contain any completed window")
    return (
        tuple(diagnostics),
        tuple(agreements),
        phenotype_identity[0],
        phenotype_identity[1],
        phenotype_identity[2],
        tuple(artifact_ids),
    )


def _sphere_point(raw: dict[str, object], annotation: ProbeDisplayAnnotation) -> LatentSpherePoint:
    _exact_keys(
        raw,
        {"probe_id", "chromosome", "position", "context", "x", "y", "z"},
        "spherical UMAP point",
    )
    raw_identity = (
        parse_probe_id(_string(raw, "probe_id")),
        parse_autosome(_integer(raw, "chromosome")),
        parse_one_based_position(_integer(raw, "position")),
        GenomicContext(_string(raw, "context")),
    )
    if raw_identity != annotation.locus_identity[:4]:
        raise ValueError("spherical UMAP point identity differs from the frozen probe table")
    return LatentSpherePoint(
        annotation=annotation,
        x=_number(raw, "x"),
        y=_number(raw, "y"),
        z=_number(raw, "z"),
    )


def _sphere_panels(
    umap_root: Path,
    experiments: Path,
    embeddings_root: Path,
    *,
    expected_prefix: tuple[str, str, str, str, str],
    primary_git_commit: str,
    target_sha256: str,
    validation_indices_sha256: str,
    validation_count: int,
    windows: tuple[int, ...],
    dimensions: tuple[int, ...],
    probes: NonEmptyProbeSet,
    features: SequenceFeatureArtifact,
) -> tuple[tuple[LatentSpherePanel, ...], tuple[str, ...]]:
    panels: list[LatentSpherePanel] = []
    artifact_ids: list[str] = []
    split_name = expected_prefix[3]
    probe_index_by_id = {probe.probe_id: index for index, probe in enumerate(probes.probes)}
    for window in windows:
        embedding_manifest = embeddings_root / f"window-{window}" / "manifest.json"
        for dimension in dimensions:
            path = umap_root / split_name / f"window-{window}" / f"d-{dimension}-lambda-0.1.json"
            record = _load_json(path)
            identity = _object(record, "identity")
            tuning_metadata = (
                experiments
                / "tuning"
                / "full"
                / split_name
                / f"window-{window}"
                / f"d-{dimension}-lambda-0.1"
                / "metadata.json"
            )
            tuning_record = _load_json(tuning_metadata)
            sampling = _object(record, "sampling")
            umap = _object(record, "umap")
            distortion = _object(record, "distortion")
            umap_config = _object(umap, "config")
            age_point = _object(record, "age_point")
            _exact_keys(
                umap,
                {
                    "implementation",
                    "input_geometry",
                    "output_geometry",
                    "graph",
                    "initialization",
                    "objective",
                    "constraint",
                    "config",
                    "curve_a",
                    "curve_b",
                    "initial_cross_entropy",
                    "final_cross_entropy",
                    "graph_edge_count",
                    "spectral_gap",
                    "selected_step",
                },
                "spherical UMAP method",
            )
            _exact_keys(
                distortion,
                {
                    "scope",
                    "geodesic_stress_1",
                    "geodesic_distance_pearson",
                    "neighbor_count",
                    "mean_neighbor_recall",
                    "interpretation",
                },
                "spherical UMAP distortion",
            )
            _exact_keys(age_point, {"label", "x", "y", "z"}, "spherical UMAP age point")
            if (
                record.get("schema") != _UMAP_SCHEMA
                or record.get("status") != "post_hoc_visualization_validation_partition_only"
                or _identity_tuple(record) != (*expected_prefix, window)
                or _string(identity, "primary_git_commit") != primary_git_commit
                or _string(identity, "target_sha256") != target_sha256
                or _string(identity, "validation_indices_sha256") != validation_indices_sha256
                or _integer(identity, "latent_dimension") != dimension
                or _number(identity, "lambda_age") != 0.1
                or _string(identity, "embedding_manifest_sha256") != sha256_file(embedding_manifest)
                or _string(identity, "tuning_metadata_sha256") != sha256_file(tuning_metadata)
                or _string(identity, "tuning_model_sha256")
                != _string(tuning_record, "model_sha256")
                or sampling.get("partition") != "nested_validation"
                or sampling.get("target_access") != "none"
                or sampling.get("method") != "metadata_context_balanced_seeded_without_replacement"
                or _integer(sampling, "source_count") != validation_count
                or _integer(sampling, "display_probe_count") != 300
                or _integer(sampling, "count_per_context") != 75
                or _integer(sampling, "age_direction_count") != 1
                or umap.get("implementation") != "exact_torch_fuzzy_cross_entropy"
                or umap.get("input_geometry")
                != "intrinsic_unit_hypersphere_geodesic_arccos_clamped_dot_product"
                or umap.get("output_geometry")
                != "unit_two_sphere_S2_in_R3_with_intrinsic_geodesic_distance"
                or umap.get("graph") != "exact_knn_default_fuzzy_union"
                or umap.get("initialization")
                != (
                    "deterministic_three_eigenvector_normalized_laplacian_spectral_"
                    "then_row_normalized"
                )
                or umap.get("objective")
                != (
                    "complete_bernoulli_fuzzy_set_cross_entropy_with_S2_geodesic_"
                    "output_distances_without_negative_sampling_approximation"
                )
                or umap.get("constraint")
                != "rowwise_unit_norm_projection_after_every_optimizer_step"
                or _number(umap, "curve_a") <= 0.0
                or _number(umap, "curve_b") <= 0.0
                or not 0 < _integer(umap, "selected_step") <= 750
                or {
                    "n_neighbors": _integer(umap_config, "n_neighbors"),
                    "local_connectivity": _number(umap_config, "local_connectivity"),
                    "smooth_knn_search_steps": _integer(umap_config, "smooth_knn_search_steps"),
                    "min_dist": _number(umap_config, "min_dist"),
                    "spread": _number(umap_config, "spread"),
                    "optimization_steps": _integer(umap_config, "optimization_steps"),
                    "learning_rate": _number(umap_config, "learning_rate"),
                    "seed": _integer(umap_config, "seed"),
                    "input_metric": _string(umap_config, "input_metric"),
                }
                != {
                    "n_neighbors": 15,
                    "local_connectivity": 1.0,
                    "smooth_knn_search_steps": 64,
                    "min_dist": 0.1,
                    "spread": 1.0,
                    "optimization_steps": 750,
                    "learning_rate": 0.05,
                    "seed": 618_437,
                    "input_metric": "spherical_geodesic",
                }
                or age_point.get("label") != "learned age direction"
                or distortion.get("scope") != "all_display_probes_plus_age_direction"
                or distortion.get("interpretation")
                != "S_d_minus_1_to_S2_is_lossy_metrics_quantify_projection_distortion"
                or _integer(distortion, "neighbor_count") != 14
            ):
                raise ValueError(f"latent spherical UMAP identity differs: {path}")
            raw_points = _objects(record, "points")
            raw_probe_ids = tuple(
                parse_probe_id(_string(point, "probe_id")) for point in raw_points
            )
            if len(set(raw_probe_ids)) != len(raw_probe_ids):
                raise ValueError("spherical UMAP record contains duplicate probe IDs")
            if any(probe_id not in probe_index_by_id for probe_id in raw_probe_ids):
                raise ValueError(
                    "spherical UMAP record contains a probe outside the frozen probe universe"
                )
            annotations = _probe_annotations(
                tuple(probe_index_by_id[probe_id] for probe_id in raw_probe_ids),
                window=window,
                probes=probes,
                features=features,
            )
            panels.append(
                LatentSpherePanel(
                    window_size=window,
                    latent_dimension=dimension,
                    lambda_age=0.1,
                    source_count=validation_count,
                    count_per_context=75,
                    n_neighbors=15,
                    min_dist=0.1,
                    initial_cross_entropy=_number(umap, "initial_cross_entropy"),
                    final_cross_entropy=_number(umap, "final_cross_entropy"),
                    graph_edge_count=_integer(umap, "graph_edge_count"),
                    spectral_gap=_number(umap, "spectral_gap"),
                    age_x=_number(age_point, "x"),
                    age_y=_number(age_point, "y"),
                    age_z=_number(age_point, "z"),
                    geodesic_stress=_number(distortion, "geodesic_stress_1"),
                    geodesic_distance_pearson=_number(distortion, "geodesic_distance_pearson"),
                    neighbor_recall=_number(distortion, "mean_neighbor_recall"),
                    neighbor_count=_integer(distortion, "neighbor_count"),
                    selected_step=_integer(umap, "selected_step"),
                    points=tuple(
                        _sphere_point(point, annotation)
                        for point, annotation in zip(raw_points, annotations, strict=True)
                    ),
                )
            )
            artifact_ids.append(
                f"validation-latent-S2-umap:{window}:{dimension}:{sha256_file(path)}"
            )
    return tuple(panels), tuple(artifact_ids)


def main() -> None:
    arguments = _parser().parse_args()
    git_commit = require_clean_git_commit(Path(__file__).resolve().parents[1])
    config = load_protocol_config(arguments.config)
    if config.status != "frozen":
        raise ValueError("interim compilation requires the frozen primary protocol")
    planned_windows = tuple(map(int, config.splits.window_sizes))
    windows = tuple(arguments.windows)
    if (
        not windows
        or tuple(sorted(set(windows))) != windows
        or not set(windows) < set(planned_windows)
    ):
        raise ValueError("interim windows must be an increasing strict subset of the frozen sweep")
    if arguments.results_output.exists() or arguments.site_output.exists():
        raise FileExistsError("interim result and site output directories must both be absent")
    protocol_sha256 = sha256_file(arguments.config)
    bundle = verify_primary_data_bundle(
        arguments.data,
        protocol_id=config.protocol_id,
        protocol_sha256=protocol_sha256,
    )
    data_sha256 = sha256_file(arguments.data / "bundle.json")
    target_sha256 = sha256_file(arguments.data / "targets.safetensors")
    probes = load_probe_table(arguments.data / "probes.tsv")
    features = load_sequence_features(
        arguments.data / "sequence-features.safetensors",
        arguments.data / "sequence-features.json",
        probes=probes,
    )
    arguments.results_output.mkdir(parents=True)
    arguments.site_output.mkdir(parents=True)
    for asset in ("index.html", "style.css"):
        source = arguments.root_template / asset
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copyfile(source, arguments.site_output / asset)
    for split_name, split_family in _SPLITS:
        split_tensor = arguments.data / "splits" / f"{split_name}.safetensors"
        split_metadata = arguments.data / "splits" / f"{split_name}.json"
        split = load_split_artifact(split_tensor, split_metadata, probes=probes)
        split_sha256 = sha256_ordered_strings(
            (sha256_file(split_metadata), sha256_file(split_tensor))
        )
        expected = _expected_identity(
            config=config,
            protocol_sha256=protocol_sha256,
            data_sha256=data_sha256,
            split_name=split_name,
            split_sha256=split_sha256,
            window=windows[0],
        )[:-1]
        evaluations = _evaluation_records(
            arguments.experiments,
            expected_prefix=expected,
            windows=windows,
        )
        primary_window = _primary_window_points(evaluations)
        primary_age = _primary_age_points(evaluations)
        primary_distance = _primary_distance_points(evaluations)
        tuning, tuning_artifacts = _tuning_points(
            arguments.experiments,
            config=config,
            expected_prefix=expected,
            windows=windows,
        )
        exploratory_uniform, exploratory_distance, weights, distance_artifacts = (
            _distance_integration_points(
                arguments.distance_integration,
                expected_prefix=expected,
                windows=windows,
                evaluations=evaluations,
                primary_distance=primary_distance,
            )
        )
        exploratory_age, direct_age_artifacts, direct_age_records = _direct_age_points(
            arguments.direct_age,
            arguments.embeddings,
            expected_prefix=expected,
            primary_git_commit=bundle.git_commit,
            target_sha256=target_sha256,
            windows=windows,
        )
        (
            cluster_diagnostics,
            cluster_agreements,
            female_sample_count,
            male_sample_count,
            cell_composition_status,
            cluster_artifacts,
        ) = _age_cluster_diagnostics(
            arguments.age_cluster_audit,
            arguments.data,
            arguments.series_matrix,
            arguments.experiments,
            arguments.embeddings,
            arguments.direct_age,
            expected_prefix=expected,
            primary_git_commit=bundle.git_commit,
            target_sha256=target_sha256,
            test_indices_sha256=sha256_ordered_strings(map(str, split.test_indices.tolist())),
            test_count=split.test_indices.numel(),
            windows=windows,
        )
        scatter_panels, scatter_artifacts = _scatter_panels(
            arguments.prediction_scatter,
            arguments.experiments,
            arguments.embeddings,
            expected_prefix=expected,
            primary_git_commit=bundle.git_commit,
            target_sha256=target_sha256,
            windows=windows,
            evaluations=evaluations,
            direct_records=direct_age_records,
            test_indices=tuple(split.test_indices.tolist()),
            probes=probes,
            features=features,
        )
        projection_dimensions = tuple(
            dimension for dimension in map(int, config.sweep.latent_dimensions) if dimension <= 128
        )
        sphere_panels, umap_artifacts = _sphere_panels(
            arguments.latent_umap,
            arguments.experiments,
            arguments.embeddings,
            expected_prefix=expected,
            primary_git_commit=bundle.git_commit,
            target_sha256=target_sha256,
            validation_indices_sha256=sha256_ordered_strings(
                map(str, split.validation_indices.tolist())
            ),
            validation_count=split.validation_indices.numel(),
            windows=windows,
            dimensions=projection_dimensions,
            probes=probes,
            features=features,
        )
        artifact_ids = (
            f"interim-site-compiler-git:{git_commit}",
            f"primary-producer-git:{bundle.git_commit}",
            f"primary-data-bundle:{data_sha256}",
            *(
                f"evaluation:{window}:{sha256_file(path)}"
                for window, (path, _) in zip(windows, evaluations, strict=True)
            ),
            *tuning_artifacts,
            *distance_artifacts,
            *direct_age_artifacts,
            *cluster_artifacts,
            *scatter_artifacts,
            *umap_artifacts,
        )
        data = InterimSiteData(
            schema=INTERIM_SITE_SCHEMA,
            protocol_id=config.protocol_id,
            status=(
                "Interim report: primary-validated 1 kb and 4 kb selected-model results, "
                "nested-validation hyperparameter sweeps, and separately labeled post-hoc "
                "direct-age, distance-integration, metadata-colored scatter, cluster-audit, "
                "and interactive two-sphere latent UMAP analyses."
            ),
            disclaimer=(
                "The planned 16 kb and 64 kb windows and the preregistered 16 kb projection are "
                "not complete. Distance integration was designed after viewing existing test "
                "results; the corrected direct tanh baseline, k=2 scatter audit, and all new "
                "visualizations are also post-hoc. They require a new holdout for confirmation."
            ),
            split_name=split_name,
            split_family=split_family,
            data_sha256=data_sha256,
            split_sha256=split_sha256,
            retained_probe_count=bundle.retained_probes,
            retained_sample_count=bundle.retained_samples,
            female_sample_count=female_sample_count,
            male_sample_count=male_sample_count,
            cell_composition_status=cell_composition_status,
            completed_windows=windows,
            planned_windows=planned_windows,
            latent_dimensions=tuple(map(int, config.sweep.latent_dimensions)),
            lambda_age_values=tuple(map(float, config.sweep.lambda_age)),
            artifact_ids=artifact_ids,
            tuning_sweep=tuning,
            primary_window_sweep=primary_window,
            primary_distance_metrics=primary_distance,
            primary_age_metrics=primary_age,
            exploratory_age_metrics=exploratory_age,
            scatter_panels=scatter_panels,
            age_cluster_diagnostics=cluster_diagnostics,
            age_cluster_agreements=cluster_agreements,
            latent_sphere_panels=sphere_panels,
            exploratory_uniform_metrics=exploratory_uniform,
            exploratory_distance_metrics=exploratory_distance,
            kernel_weights=weights,
        )
        result_path = arguments.results_output / f"{split_name}.json"
        write_canonical_json_exclusive(
            result_path,
            cast(dict[str, JsonValue], data.as_json()),
        )
        build_interim_static_site(
            data=data,
            template_directory=arguments.site_template,
            output_directory=arguments.site_output / split_name,
        )
    print(
        f"compiled interim protocol={config.protocol_id} windows={windows} "
        f"splits={len(_SPLITS)} site={arguments.site_output}"
    )


if __name__ == "__main__":
    main()
