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
    parse_autosome,
    parse_one_based_position,
    parse_probe_id,
)
from methylation_latent.evaluation import PairPopulation
from methylation_latent.experiment_data import load_split_artifact
from methylation_latent.interim_site import (
    INTERIM_SITE_SCHEMA,
    CorrelationScatterPanel,
    ExploratoryAgeMetricPoint,
    ExploratoryMetricPoint,
    InterimSiteData,
    KernelWeightPoint,
    LatentTsnePanel,
    LatentTsnePoint,
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
_TSNE_SCHEMA = "methylation-latent.validation-latent-tsne.v1"
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
    parser.add_argument("--experiments", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--distance-integration", type=Path, required=True)
    parser.add_argument("--direct-age", type=Path, required=True)
    parser.add_argument("--prediction-scatter", type=Path, required=True)
    parser.add_argument("--latent-tsne", type=Path, required=True)
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
    )
    if _integer(raw, "display_count") != len(panel.target):
        raise ValueError("scatter display count differs from its aligned vectors")
    return panel


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
        panels.append(_scatter_panel(_object(record, "age"), window=window, population="age"))
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
            )
            for population in _POPULATIONS
        )
        artifact_ids.append(f"prediction-scatter:{window}:{sha256_file(path)}")
    return tuple(panels), tuple(artifact_ids)


def _tsne_point(raw: dict[str, object]) -> LatentTsnePoint:
    _exact_keys(raw, {"probe_id", "chromosome", "position", "context", "x", "y"}, "t-SNE point")
    return LatentTsnePoint(
        probe_id=parse_probe_id(_string(raw, "probe_id")),
        chromosome=parse_autosome(_integer(raw, "chromosome")),
        position=parse_one_based_position(_integer(raw, "position")),
        context=GenomicContext(_string(raw, "context")),
        x=_number(raw, "x"),
        y=_number(raw, "y"),
    )


def _tsne_panels(
    tsne_root: Path,
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
) -> tuple[tuple[LatentTsnePanel, ...], tuple[str, ...]]:
    panels: list[LatentTsnePanel] = []
    artifact_ids: list[str] = []
    split_name = expected_prefix[3]
    for window in windows:
        embedding_manifest = embeddings_root / f"window-{window}" / "manifest.json"
        for dimension in dimensions:
            path = tsne_root / split_name / f"window-{window}" / f"d-{dimension}-lambda-0.1.json"
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
            tsne = _object(record, "tsne")
            tsne_config = _object(tsne, "config")
            if (
                record.get("schema") != _TSNE_SCHEMA
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
                or _integer(sampling, "display_count") != 300
                or _integer(sampling, "count_per_context") != 75
                or tsne.get("implementation") != "exact_torch_student_t"
                or tsne.get("input_geometry")
                != "euclidean_on_unit_latent_rows_equivalent_to_cosine"
                or tsne.get("shared_initialization_across_panels") is not True
                or {
                    "perplexity": _number(tsne_config, "perplexity"),
                    "probability_search_steps": _integer(tsne_config, "probability_search_steps"),
                    "optimization_steps": _integer(tsne_config, "optimization_steps"),
                    "early_exaggeration_steps": _integer(tsne_config, "early_exaggeration_steps"),
                    "early_exaggeration": _number(tsne_config, "early_exaggeration"),
                    "learning_rate": _number(tsne_config, "learning_rate"),
                    "seed": _integer(tsne_config, "seed"),
                }
                != {
                    "perplexity": 30.0,
                    "probability_search_steps": 60,
                    "optimization_steps": 1_000,
                    "early_exaggeration_steps": 250,
                    "early_exaggeration": 12.0,
                    "learning_rate": 50.0,
                    "seed": 411_807,
                }
            ):
                raise ValueError(f"latent t-SNE identity differs: {path}")
            raw_points = _objects(record, "points")
            panels.append(
                LatentTsnePanel(
                    window_size=window,
                    latent_dimension=dimension,
                    lambda_age=0.1,
                    source_count=validation_count,
                    count_per_context=75,
                    perplexity=30.0,
                    final_kl_divergence=_number(tsne, "final_kl_divergence"),
                    points=tuple(_tsne_point(point) for point in raw_points),
                )
            )
            artifact_ids.append(f"validation-latent-tsne:{window}:{dimension}:{sha256_file(path)}")
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
        )
        projection_dimensions = tuple(
            dimension for dimension in map(int, config.sweep.latent_dimensions) if dimension <= 128
        )
        tsne_panels, tsne_artifacts = _tsne_panels(
            arguments.latent_tsne,
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
            *scatter_artifacts,
            *tsne_artifacts,
        )
        data = InterimSiteData(
            schema=INTERIM_SITE_SCHEMA,
            protocol_id=config.protocol_id,
            status=(
                "Interim report: primary-validated 1 kb and 4 kb selected-model results, "
                "nested-validation hyperparameter sweeps, and separately labeled post-hoc "
                "direct-age, distance-integration, scatter, and latent t-SNE analyses."
            ),
            disclaimer=(
                "The planned 16 kb and 64 kb windows and the preregistered 16 kb projection are "
                "not complete. Distance integration was designed after viewing existing test "
                "results; the corrected direct tanh baseline and all new visualizations are "
                "also post-hoc. They require a new holdout for confirmation."
            ),
            split_name=split_name,
            split_family=split_family,
            data_sha256=data_sha256,
            split_sha256=split_sha256,
            retained_probe_count=bundle.retained_probes,
            retained_sample_count=bundle.retained_samples,
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
            latent_tsne_panels=tsne_panels,
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
