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
from methylation_latent.evaluation import PairPopulation
from methylation_latent.experiment_data import load_split_artifact
from methylation_latent.interim_site import (
    INTERIM_SITE_SCHEMA,
    ExploratoryMetricPoint,
    InterimSiteData,
    KernelWeightPoint,
    TuningSweepPoint,
    build_interim_static_site,
)
from methylation_latent.site import AgeMetricPoint, DistanceMetricPoint, WindowSweepPoint

_EVALUATION_SCHEMA = "methylation-latent.held-out-evaluation.v2"
_TUNING_SCHEMA = "methylation-latent.latent-tuning-run.v2"
_SELECTION_SCHEMA = "methylation-latent.hyperparameter-selection.v2"
_DISTANCE_SCHEMA = "methylation-latent.exploratory-distance-integration.v2"
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
    parser.add_argument("--distance-integration", type=Path, required=True)
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
        load_split_artifact(split_tensor, split_metadata, probes=probes)
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
        )
        data = InterimSiteData(
            schema=INTERIM_SITE_SCHEMA,
            protocol_id=config.protocol_id,
            status=(
                "Interim report: primary-validated 1 kb and 4 kb selected-model results, "
                "nested-validation hyperparameter sweeps, and separately labeled post-hoc "
                "distance integration."
            ),
            disclaimer=(
                "The planned 16 kb and 64 kb windows and the preregistered 16 kb projection are "
                "not complete. Distance integration was designed after viewing existing test "
                "results and requires a new holdout for confirmation."
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
