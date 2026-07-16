"""Compile validated evaluation records into two split-specific static sites."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import cast

from methylation_latent.artifacts import (
    Eligibility,
    JsonValue,
    sha256_file,
    sha256_ordered_strings,
    write_canonical_json_exclusive,
)
from methylation_latent.cohort import load_probe_table
from methylation_latent.config import load_protocol_config
from methylation_latent.data_bundle import verify_primary_data_bundle
from methylation_latent.domain import GenomicContext, parse_probe_id
from methylation_latent.evaluation import PairPopulation
from methylation_latent.experiment_data import load_split_artifact
from methylation_latent.site import (
    AgeMetricPoint,
    DistanceMetricPoint,
    ProjectionPoint,
    SiteData,
    SiteProvenance,
    WindowSweepPoint,
    build_static_site,
)

_EVALUATION_SCHEMA = "methylation-latent.held-out-evaluation.v1"
_SITE_SCHEMA = "methylation-latent.site-data.v1"
_SPLITS = (
    ("diverse-blocks", "diverse_blocks"),
    ("held-out-chromosome", "held_out_chromosome"),
)
_STAGES = ("sequence_features", "caduceus_age_only", "full_latent_metric")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--experiments", type=Path, required=True)
    parser.add_argument("--results-output", type=Path, required=True)
    parser.add_argument("--site-template", type=Path, required=True)
    parser.add_argument("--root-template", type=Path, required=True)
    parser.add_argument("--site-output", type=Path, required=True)
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


def _objects(record: dict[str, object], key: str) -> list[dict[str, object]]:
    value = record.get(key)
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise TypeError(f"{key} must be an array of objects")
    return cast(list[dict[str, object]], value)


def _integer(record: dict[str, object], key: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value


def _number(record: dict[str, object], key: str) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{key} must be numeric")
    return float(value)


def _string(record: dict[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise TypeError(f"{key} must be a non-empty string")
    return value


def _evaluation_records(
    experiments: Path,
    split_name: str,
    windows: tuple[int, ...],
) -> tuple[tuple[Path, dict[str, object]], ...]:
    records: list[tuple[Path, dict[str, object]]] = []
    for window in windows:
        path = experiments / "evaluation" / split_name / f"window-{window}.json"
        record = _load_json(path)
        identity = _object(record, "identity")
        if (
            record.get("schema") != _EVALUATION_SCHEMA
            or _integer(identity, "window_size") != window
            or _string(identity, "split_name") != split_name
        ):
            raise ValueError(f"evaluation identity differs: {path}")
        records.append((path, record))
    return tuple(records)


def _window_points(
    records: tuple[tuple[Path, dict[str, object]], ...],
) -> tuple[WindowSweepPoint, ...]:
    points: list[WindowSweepPoint] = []
    for _, record in records:
        identity = _object(record, "identity")
        window = _integer(identity, "window_size")
        selected = _object(record, "selected_full_training")
        pair_metrics = _object(record, "pair_metrics")
        for population in (
            PairPopulation.SEEN_BY_HELD_OUT,
            PairPopulation.HELD_OUT_BY_HELD_OUT,
        ):
            report = _object(pair_metrics, population.value)
            metrics = _object(report, "uniform_model_metrics")
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


def _distance_points(
    records: tuple[tuple[Path, dict[str, object]], ...],
) -> tuple[DistanceMetricPoint, ...]:
    points: list[DistanceMetricPoint] = []
    for _, record in records:
        window = _integer(_object(record, "identity"), "window_size")
        pair_metrics = _object(record, "pair_metrics")
        for population in (
            PairPopulation.SEEN_BY_HELD_OUT,
            PairPopulation.HELD_OUT_BY_HELD_OUT,
        ):
            report = _object(pair_metrics, population.value)
            for row in _objects(report, "by_distance"):
                metrics = _object(row, "model_metrics")
                points.append(
                    DistanceMetricPoint(
                        window_size=window,
                        population=population,
                        distance_class=_string(row, "distance_class"),
                        count=_integer(metrics, "count"),
                        target_mean=_number(metrics, "target_mean"),
                        prediction_mean=_number(metrics, "prediction_mean"),
                        distance_baseline_mean=_number(
                            row,
                            "distance_baseline_mean",
                        ),
                        mse=_number(metrics, "mse"),
                        pearson=_number(metrics, "pearson"),
                        r_squared=_number(metrics, "r_squared"),
                    )
                )
    return tuple(points)


def _age_points(
    records: tuple[tuple[Path, dict[str, object]], ...],
) -> tuple[AgeMetricPoint, ...]:
    points: list[AgeMetricPoint] = []
    for _, record in records:
        window = _integer(_object(record, "identity"), "window_size")
        metrics = _object(record, "age_metrics")
        for stage in _STAGES:
            report = _object(metrics, stage)
            points.append(
                AgeMetricPoint(
                    window_size=window,
                    stage=stage,
                    count=_integer(report, "count"),
                    mse=_number(report, "mse"),
                    pearson=_number(report, "pearson"),
                )
            )
    return tuple(points)


def _projection_points(
    records: tuple[tuple[Path, dict[str, object]], ...],
    projection_window: int,
) -> tuple[ProjectionPoint, ...]:
    matches = tuple(
        record
        for _, record in records
        if _integer(_object(record, "identity"), "window_size") == projection_window
    )
    if len(matches) != 1:
        raise ValueError("projection window must resolve to exactly one evaluation record")
    projection = _object(matches[0], "projection")
    return tuple(_projection_point(row) for row in _objects(projection, "points"))


def _projection_point(row: dict[str, object]) -> ProjectionPoint:
    return ProjectionPoint(
        probe_id=parse_probe_id(_string(row, "probe_id")),
        x=_number(row, "x"),
        y=_number(row, "y"),
        context=GenomicContext(_string(row, "context")),
    )


def _artifact_ids(
    data: Path,
    experiments: Path,
    split_name: str,
    records: tuple[tuple[Path, dict[str, object]], ...],
) -> tuple[str, ...]:
    identifiers = [
        f"primary-data-bundle:{sha256_file(data / 'bundle.json')}",
        (
            "evaluation-pair-cache:"
            f"{sha256_file(experiments / 'evaluation-pairs' / split_name / 'metadata.json')}"
        ),
    ]
    for path, record in records:
        identity = _object(record, "identity")
        window = _integer(identity, "window_size")
        identifiers.extend(
            (
                f"sequence-baseline:{window}:{_string(identity, 'sequence_baseline_sha256')}",
                (f"age-only-final:{window}:{_string(identity, 'age_only_model_metadata_sha256')}"),
                (f"full-final:{window}:{_string(identity, 'full_model_metadata_sha256')}"),
                f"evaluation:{window}:{sha256_file(path)}",
            )
        )
    return tuple(identifiers)


def main() -> None:
    arguments = _parser().parse_args()
    config = load_protocol_config(arguments.config)
    protocol_sha256 = sha256_file(arguments.config)
    bundle = verify_primary_data_bundle(
        arguments.data,
        protocol_id=config.protocol_id,
        protocol_sha256=protocol_sha256,
    )
    probes = load_probe_table(arguments.data / "probes.tsv")
    windows = tuple(map(int, config.splits.window_sizes))
    if arguments.results_output.exists():
        raise FileExistsError(arguments.results_output)
    if arguments.site_output.exists():
        raise FileExistsError(arguments.site_output)
    arguments.results_output.mkdir(parents=True)
    arguments.site_output.mkdir(parents=True)
    for asset in ("index.html", "style.css"):
        shutil.copyfile(
            arguments.root_template / asset,
            arguments.site_output / asset,
        )
    for split_name, split_family in _SPLITS:
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
        records = _evaluation_records(
            arguments.experiments,
            split_name,
            windows,
        )
        for _, record in records:
            identity = _object(record, "identity")
            expected_identity = (
                config.protocol_id,
                protocol_sha256,
                sha256_file(arguments.data / "bundle.json"),
                split_name,
                split_sha256,
            )
            observed_identity = (
                _string(identity, "protocol_id"),
                _string(identity, "protocol_sha256"),
                _string(identity, "data_sha256"),
                _string(identity, "split_name"),
                _string(identity, "split_sha256"),
            )
            if observed_identity != expected_identity:
                raise ValueError("evaluation record differs from the sealed site identity")
        site_data = SiteData(
            schema=_SITE_SCHEMA,
            protocol_id=config.protocol_id,
            eligibility=Eligibility.PRIMARY_VALIDATED,
            status=(
                "Primary-validated held-out results from one frozen deterministic "
                f"training seed; split family: {split_family}."
            ),
            artifact_ids=_artifact_ids(
                arguments.data,
                arguments.experiments,
                split_name,
                records,
            ),
            provenance=SiteProvenance(
                split_family=split_family,
                data_sha256=sha256_file(arguments.data / "bundle.json"),
                split_sha256=split_sha256,
                retained_probe_count=bundle.retained_probes,
                retained_sample_count=bundle.retained_samples,
            ),
            window_sweep=_window_points(records),
            distance_metrics=_distance_points(records),
            age_metrics=_age_points(records),
            projection=_projection_points(
                records,
                int(config.sweep.projection_window_size),
            ),
        )
        result_path = arguments.results_output / f"{split_name}.json"
        write_canonical_json_exclusive(
            result_path,
            cast(dict[str, JsonValue], site_data.as_json()),
        )
        build_static_site(
            data_path=result_path,
            template_directory=arguments.site_template,
            output_directory=arguments.site_output / split_name,
        )
        if split.test_indices.numel() != len(site_data.projection):
            raise ValueError("site projection does not contain every held-out probe")
    print(
        f"compiled protocol={config.protocol_id} splits={len(_SPLITS)} site={arguments.site_output}"
    )


if __name__ == "__main__":
    main()
