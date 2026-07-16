"""Strict static-site data schema and generator."""

from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from methylation_latent.artifacts import Eligibility, JsonValue, canonical_json_bytes
from methylation_latent.domain import GenomicContext, ProbeId, parse_probe_id
from methylation_latent.evaluation import DISTANCE_CLASS_LABELS, PairPopulation

_SITE_SCHEMA = "methylation-latent.site-data.v1"
_AGE_STAGES = frozenset({"sequence_features", "caduceus_age_only", "full_latent_metric"})
_SPLIT_FAMILIES = frozenset({"diverse_blocks", "held_out_chromosome"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$", flags=re.ASCII)


def _exact_keys(raw: dict[str, object], expected: set[str], context: str) -> None:
    observed = set(raw)
    if observed != expected:
        raise ValueError(
            f"{context} fields differ: missing={sorted(expected - observed)}, "
            f"unknown={sorted(observed - expected)}"
        )


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TypeError(f"{name} must be a positive integer")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or value == "":
        raise TypeError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class WindowSweepPoint:
    window_size: int
    population: PairPopulation
    latent_dimension: int
    lambda_age: float
    mse: float
    pearson: float
    r_squared: float

    def __post_init__(self) -> None:
        if self.window_size <= 0 or self.latent_dimension <= 0 or self.lambda_age <= 0.0:
            raise ValueError("window sweep dimensions and age weight must be positive")
        if self.mse < 0.0 or not -1.0 <= self.pearson <= 1.0:
            raise ValueError("window sweep MSE must be non-negative and Pearson must be bounded")


@dataclass(frozen=True, slots=True)
class DistanceMetricPoint:
    window_size: int
    population: PairPopulation
    distance_class: str
    count: int
    target_mean: float
    prediction_mean: float
    distance_baseline_mean: float
    mse: float
    pearson: float
    r_squared: float

    def __post_init__(self) -> None:
        if self.window_size <= 0 or self.count <= 0:
            raise ValueError("distance metric window and count must be positive")
        if self.distance_class not in DISTANCE_CLASS_LABELS:
            raise ValueError(f"unknown distance class: {self.distance_class!r}")
        if self.mse < 0.0 or not -1.0 <= self.pearson <= 1.0:
            raise ValueError("distance MSE must be non-negative and Pearson must be bounded")
        if any(
            not -1.0 <= value <= 1.0
            for value in (self.target_mean, self.prediction_mean, self.distance_baseline_mean)
        ):
            raise ValueError("pair target, prediction, and baseline means must be bounded")


@dataclass(frozen=True, slots=True)
class AgeMetricPoint:
    window_size: int
    stage: str
    count: int
    mse: float
    pearson: float

    def __post_init__(self) -> None:
        if self.window_size <= 0 or self.count <= 0:
            raise ValueError("age metric window and count must be positive")
        if self.stage not in _AGE_STAGES:
            raise ValueError(f"unknown age stage: {self.stage!r}")
        if self.mse < 0.0 or not -1.0 <= self.pearson <= 1.0:
            raise ValueError("age MSE must be non-negative and Pearson must be bounded")


@dataclass(frozen=True, slots=True)
class ProjectionPoint:
    probe_id: ProbeId
    x: float
    y: float
    context: GenomicContext


@dataclass(frozen=True, slots=True)
class SiteProvenance:
    """Required scientific identity fields, present only for validated results."""

    split_family: str
    data_sha256: str
    split_sha256: str
    retained_probe_count: int
    retained_sample_count: int

    def __post_init__(self) -> None:
        if self.split_family not in _SPLIT_FAMILIES:
            raise ValueError(f"unknown split family: {self.split_family!r}")
        if (
            _SHA256.fullmatch(self.data_sha256) is None
            or _SHA256.fullmatch(self.split_sha256) is None
        ):
            raise ValueError("site provenance requires lowercase SHA-256 fingerprints")
        if self.retained_probe_count <= 0 or self.retained_sample_count <= 1:
            raise ValueError("site provenance requires positive probes and at least two samples")


@dataclass(frozen=True, slots=True)
class SiteData:
    schema: str
    protocol_id: str
    eligibility: Eligibility
    status: str
    artifact_ids: tuple[str, ...]
    provenance: SiteProvenance | None
    window_sweep: tuple[WindowSweepPoint, ...]
    distance_metrics: tuple[DistanceMetricPoint, ...]
    age_metrics: tuple[AgeMetricPoint, ...]
    projection: tuple[ProjectionPoint, ...]

    def __post_init__(self) -> None:
        if self.schema != _SITE_SCHEMA or not self.protocol_id or not self.status:
            raise ValueError("site schema, protocol ID, and status are required")
        if len(set(self.artifact_ids)) != len(self.artifact_ids):
            raise ValueError("site artifact IDs must be unique")
        sections = (
            self.window_sweep,
            self.distance_metrics,
            self.age_metrics,
            self.projection,
        )
        if self.eligibility != Eligibility.PRIMARY_VALIDATED and any(sections):
            raise ValueError("non-validated artifacts cannot populate scientific site panels")
        if self.eligibility != Eligibility.PRIMARY_VALIDATED and self.provenance is not None:
            raise ValueError("non-validated site data cannot claim scientific provenance")
        if self.eligibility == Eligibility.PRIMARY_VALIDATED and (
            not self.artifact_ids
            or self.provenance is None
            or any(not section for section in sections)
        ):
            raise ValueError(
                "primary-validated site data requires provenance, artifacts, and every panel"
            )
        if self.eligibility == Eligibility.PRIMARY_VALIDATED:
            required_populations = {
                PairPopulation.SEEN_BY_HELD_OUT,
                PairPopulation.HELD_OUT_BY_HELD_OUT,
            }
            if {point.population for point in self.window_sweep} != required_populations:
                raise ValueError("window sweep must contain both evaluation populations")
            if {point.population for point in self.distance_metrics} != required_populations:
                raise ValueError("distance metrics must contain both evaluation populations")
            if {point.stage for point in self.age_metrics} != _AGE_STAGES:
                raise ValueError("age metrics must contain all three preregistered stages")
            if {point.context for point in self.projection} != set(GenomicContext):
                raise ValueError("latent projection must contain every genomic context")
            window_keys = tuple(
                (point.window_size, point.population) for point in self.window_sweep
            )
            if len(set(window_keys)) != len(window_keys):
                raise ValueError("window sweep contains duplicate window/population rows")
            windows = {point.window_size for point in self.window_sweep}
            required_window_populations = {
                (window, population) for window in windows for population in required_populations
            }
            if set(window_keys) != required_window_populations:
                raise ValueError("every window must contain both pair populations")
            distance_keys = tuple(
                (point.window_size, point.population, point.distance_class)
                for point in self.distance_metrics
            )
            if len(set(distance_keys)) != len(distance_keys):
                raise ValueError("distance metrics contain duplicate stratum rows")
            if {
                (point.window_size, point.population) for point in self.distance_metrics
            } != required_window_populations:
                raise ValueError("distance metrics must cover both populations for every window")
            age_keys = tuple((point.window_size, point.stage) for point in self.age_metrics)
            if len(set(age_keys)) != len(age_keys):
                raise ValueError("age metrics contain duplicate window/stage rows")
            if set(age_keys) != {(window, stage) for window in windows for stage in _AGE_STAGES}:
                raise ValueError("age metrics must cover every stage for every window")
            probe_ids = tuple(point.probe_id for point in self.projection)
            if len(set(probe_ids)) != len(probe_ids):
                raise ValueError("latent projection contains duplicate probe IDs")

    def as_json(self) -> dict[str, JsonValue]:
        return cast(dict[str, JsonValue], asdict(self))


def _objects(raw: object, name: str) -> tuple[dict[str, object], ...]:
    if not isinstance(raw, list):
        raise TypeError(f"{name} must be an array")
    if any(not isinstance(item, dict) for item in raw):
        raise TypeError(f"{name} entries must be objects")
    return cast(tuple[dict[str, object], ...], tuple(raw))


def load_site_data(path: Path) -> SiteData:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("site data must be a JSON object")
    _exact_keys(
        raw,
        {
            "schema",
            "protocol_id",
            "eligibility",
            "status",
            "artifact_ids",
            "provenance",
            "window_sweep",
            "distance_metrics",
            "age_metrics",
            "projection",
        },
        "site data",
    )
    artifact_ids_raw = raw["artifact_ids"]
    if not isinstance(artifact_ids_raw, list):
        raise TypeError("site artifact IDs must be an array")
    provenance_raw = raw["provenance"]
    if provenance_raw is not None and not isinstance(provenance_raw, dict):
        raise TypeError("site provenance must be an object or null")
    provenance: SiteProvenance | None = None
    if isinstance(provenance_raw, dict):
        _exact_keys(
            provenance_raw,
            {
                "split_family",
                "data_sha256",
                "split_sha256",
                "retained_probe_count",
                "retained_sample_count",
            },
            "site provenance",
        )
        provenance = SiteProvenance(
            split_family=_string(provenance_raw["split_family"], "split_family"),
            data_sha256=_string(provenance_raw["data_sha256"], "data_sha256"),
            split_sha256=_string(provenance_raw["split_sha256"], "split_sha256"),
            retained_probe_count=_positive_integer(
                provenance_raw["retained_probe_count"], "retained_probe_count"
            ),
            retained_sample_count=_positive_integer(
                provenance_raw["retained_sample_count"], "retained_sample_count"
            ),
        )

    window_rows: list[WindowSweepPoint] = []
    for row in _objects(raw["window_sweep"], "window_sweep"):
        _exact_keys(
            row,
            {
                "window_size",
                "population",
                "latent_dimension",
                "lambda_age",
                "mse",
                "pearson",
                "r_squared",
            },
            "window sweep row",
        )
        window_rows.append(
            WindowSweepPoint(
                window_size=_positive_integer(row["window_size"], "window_size"),
                population=PairPopulation(_string(row["population"], "population")),
                latent_dimension=_positive_integer(row["latent_dimension"], "latent_dimension"),
                lambda_age=_finite_number(row["lambda_age"], "lambda_age"),
                mse=_finite_number(row["mse"], "mse"),
                pearson=_finite_number(row["pearson"], "pearson"),
                r_squared=_finite_number(row["r_squared"], "r_squared"),
            )
        )

    distance_rows: list[DistanceMetricPoint] = []
    for row in _objects(raw["distance_metrics"], "distance_metrics"):
        _exact_keys(
            row,
            {
                "window_size",
                "population",
                "distance_class",
                "count",
                "target_mean",
                "prediction_mean",
                "distance_baseline_mean",
                "mse",
                "pearson",
                "r_squared",
            },
            "distance metric row",
        )
        distance_rows.append(
            DistanceMetricPoint(
                window_size=_positive_integer(row["window_size"], "window_size"),
                population=PairPopulation(_string(row["population"], "population")),
                distance_class=_string(row["distance_class"], "distance_class"),
                count=_positive_integer(row["count"], "count"),
                target_mean=_finite_number(row["target_mean"], "target_mean"),
                prediction_mean=_finite_number(row["prediction_mean"], "prediction_mean"),
                distance_baseline_mean=_finite_number(
                    row["distance_baseline_mean"], "distance_baseline_mean"
                ),
                mse=_finite_number(row["mse"], "mse"),
                pearson=_finite_number(row["pearson"], "pearson"),
                r_squared=_finite_number(row["r_squared"], "r_squared"),
            )
        )

    age_rows: list[AgeMetricPoint] = []
    for row in _objects(raw["age_metrics"], "age_metrics"):
        _exact_keys(row, {"window_size", "stage", "count", "mse", "pearson"}, "age metric row")
        age_rows.append(
            AgeMetricPoint(
                window_size=_positive_integer(row["window_size"], "window_size"),
                stage=_string(row["stage"], "stage"),
                count=_positive_integer(row["count"], "count"),
                mse=_finite_number(row["mse"], "mse"),
                pearson=_finite_number(row["pearson"], "pearson"),
            )
        )

    projection_rows: list[ProjectionPoint] = []
    for row in _objects(raw["projection"], "projection"):
        _exact_keys(row, {"probe_id", "x", "y", "context"}, "projection row")
        projection_rows.append(
            ProjectionPoint(
                probe_id=parse_probe_id(_string(row["probe_id"], "probe_id")),
                x=_finite_number(row["x"], "x"),
                y=_finite_number(row["y"], "y"),
                context=GenomicContext(_string(row["context"], "context")),
            )
        )
    return SiteData(
        schema=_string(raw["schema"], "schema"),
        protocol_id=_string(raw["protocol_id"], "protocol_id"),
        eligibility=Eligibility(_string(raw["eligibility"], "eligibility")),
        status=_string(raw["status"], "status"),
        artifact_ids=tuple(_string(value, "artifact_id") for value in artifact_ids_raw),
        provenance=provenance,
        window_sweep=tuple(window_rows),
        distance_metrics=tuple(distance_rows),
        age_metrics=tuple(age_rows),
        projection=tuple(projection_rows),
    )


def build_static_site(
    *, data_path: Path, template_directory: Path, output_directory: Path
) -> SiteData:
    data = load_site_data(data_path)
    if output_directory.exists():
        raise FileExistsError(output_directory)
    expected_assets = ("index.html", "app.js", "style.css")
    for asset in expected_assets:
        if not (template_directory / asset).is_file():
            raise FileNotFoundError(template_directory / asset)
    output_directory.mkdir(parents=True)
    for asset in expected_assets:
        shutil.copyfile(template_directory / asset, output_directory / asset)
    (output_directory / "results.json").write_bytes(canonical_json_bytes(data.as_json()))
    return data
