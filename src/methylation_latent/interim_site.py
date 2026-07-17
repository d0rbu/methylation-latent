"""Strict schema for a partial-results site that cannot impersonate the primary site."""

from __future__ import annotations

import math
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from methylation_latent.artifacts import JsonValue, canonical_json_bytes
from methylation_latent.domain import Autosome, GenomicContext, OneBasedPosition, ProbeId
from methylation_latent.evaluation import DISTANCE_CLASS_LABELS, PairPopulation
from methylation_latent.site import AgeMetricPoint, DistanceMetricPoint, WindowSweepPoint

INTERIM_SITE_SCHEMA = "methylation-latent.interim-site-data.v2"
EXPLORATORY_MODELS = frozenset(
    {
        "sequence",
        "registered_distance_class",
        "psd_distance",
        "psd_distance_plus_sequence",
    }
)
_PAIR_POPULATIONS = frozenset(
    {PairPopulation.SEEN_BY_HELD_OUT, PairPopulation.HELD_OUT_BY_HELD_OUT}
)
_AGE_STAGES = frozenset({"sequence_features", "caduceus_age_only", "full_latent_metric"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$", flags=re.ASCII)
_SCATTER_POPULATIONS = frozenset({"age", *(population.value for population in _PAIR_POPULATIONS)})
_AGE_SCATTER_MODELS = frozenset({"cosine_age_only", "direct_tanh", "full_latent_metric"})
_PAIR_SCATTER_MODELS = frozenset({"full_latent_metric"})


def _finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class TuningSweepPoint:
    window_size: int
    latent_dimension: int
    lambda_age: float
    selected_step: int
    pair_mse: float
    age_mse: float
    selection_score: float
    selected: bool

    def __post_init__(self) -> None:
        if (
            self.window_size <= 0
            or self.latent_dimension <= 0
            or self.lambda_age <= 0.0
            or self.selected_step < 0
        ):
            raise ValueError("tuning dimensions and age weight must be positive")
        for name, value in (
            ("pair_mse", self.pair_mse),
            ("age_mse", self.age_mse),
            ("selection_score", self.selection_score),
        ):
            _finite(value, name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if not math.isclose(
            self.selection_score,
            self.pair_mse + self.age_mse,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("full-model selection score must equal pair MSE plus age MSE")


@dataclass(frozen=True, slots=True)
class ExploratoryMetricPoint:
    window_size: int
    population: PairPopulation
    distance_class: str | None
    model: str
    count: int
    target_mean: float
    prediction_mean: float
    mse: float
    pearson: float | None
    pearson_status: str
    r_squared: float

    def __post_init__(self) -> None:
        if self.window_size <= 0 or self.count <= 0:
            raise ValueError("exploratory metric window and count must be positive")
        if self.population not in _PAIR_POPULATIONS:
            raise ValueError("exploratory metrics require a held-out pair population")
        if self.distance_class is not None and self.distance_class not in DISTANCE_CLASS_LABELS:
            raise ValueError(f"unknown distance class: {self.distance_class!r}")
        if self.model not in EXPLORATORY_MODELS:
            raise ValueError(f"unknown exploratory model: {self.model!r}")
        for name, value in (
            ("target_mean", self.target_mean),
            ("prediction_mean", self.prediction_mean),
            ("mse", self.mse),
            ("r_squared", self.r_squared),
        ):
            _finite(value, name)
        if self.mse < 0.0 or not -1.0 <= self.target_mean <= 1.0:
            raise ValueError("exploratory MSE must be non-negative and target mean bounded")
        if not -1.0 <= self.prediction_mean <= 1.0:
            raise ValueError("exploratory prediction mean must be bounded")
        if self.pearson is None:
            if self.pearson_status != "undefined_constant_prediction":
                raise ValueError("null Pearson requires an explicit constant-prediction status")
        elif self.pearson_status != "defined" or not -1.0 <= self.pearson <= 1.0:
            raise ValueError("defined Pearson must be bounded and explicitly labeled")


@dataclass(frozen=True, slots=True)
class KernelWeightPoint:
    window_size: int
    component: str
    weight: float

    def __post_init__(self) -> None:
        if self.window_size <= 0 or not self.component:
            raise ValueError("kernel weight requires a positive window and component name")
        _finite(self.weight, "kernel weight")
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError("kernel weight must lie in [0,1]")


@dataclass(frozen=True, slots=True)
class ExploratoryAgeMetricPoint:
    window_size: int
    count: int
    selected_step: int
    validation_mse: float
    mse: float
    pearson: float
    r_squared: float

    def __post_init__(self) -> None:
        if self.window_size <= 0 or self.count <= 0 or self.selected_step < 0:
            raise ValueError("direct-age metric counts and selected step must be non-negative")
        for name, value in (
            ("validation_mse", self.validation_mse),
            ("mse", self.mse),
            ("pearson", self.pearson),
            ("r_squared", self.r_squared),
        ):
            _finite(value, name)
        if self.validation_mse < 0.0 or self.mse < 0.0:
            raise ValueError("direct-age MSE values must be non-negative")
        if not -1.0 <= self.pearson <= 1.0:
            raise ValueError("direct-age Pearson must lie in [-1,1]")


@dataclass(frozen=True, slots=True)
class ScatterPredictionSeries:
    model: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.model or not self.values:
            raise ValueError("scatter prediction series must be named and non-empty")
        if any(not math.isfinite(value) or not -1.0 <= value <= 1.0 for value in self.values):
            raise ValueError("scatter predictions must be finite correlations in [-1,1]")


@dataclass(frozen=True, slots=True)
class CorrelationScatterPanel:
    window_size: int
    population: str
    source_count: int
    seed: int
    sample_indices: tuple[int, ...]
    target: tuple[float, ...]
    predictions: tuple[ScatterPredictionSeries, ...]

    def __post_init__(self) -> None:
        if self.window_size <= 0 or self.source_count <= 0:
            raise ValueError("scatter window and source count must be positive")
        if self.population not in _SCATTER_POPULATIONS:
            raise ValueError(f"unknown scatter population: {self.population!r}")
        if (
            not self.sample_indices
            or tuple(sorted(set(self.sample_indices))) != self.sample_indices
            or self.sample_indices[0] < 0
            or self.sample_indices[-1] >= self.source_count
        ):
            raise ValueError("scatter indices must be non-empty, unique, increasing, and in range")
        if len(self.target) != len(self.sample_indices) or any(
            not math.isfinite(value) or not -1.0 <= value <= 1.0 for value in self.target
        ):
            raise ValueError("scatter targets must be aligned finite correlations in [-1,1]")
        expected_models = _AGE_SCATTER_MODELS if self.population == "age" else _PAIR_SCATTER_MODELS
        observed_models = tuple(series.model for series in self.predictions)
        if (
            len(set(observed_models)) != len(observed_models)
            or set(observed_models) != expected_models
            or any(len(series.values) != len(self.target) for series in self.predictions)
        ):
            raise ValueError("scatter prediction series differ from the required aligned models")


@dataclass(frozen=True, slots=True)
class LatentTsnePoint:
    probe_id: ProbeId
    chromosome: Autosome
    position: OneBasedPosition
    context: GenomicContext
    x: float
    y: float

    def __post_init__(self) -> None:
        _finite(self.x, "t-SNE x")
        _finite(self.y, "t-SNE y")


@dataclass(frozen=True, slots=True)
class LatentTsnePanel:
    window_size: int
    latent_dimension: int
    lambda_age: float
    source_count: int
    count_per_context: int
    perplexity: float
    final_kl_divergence: float
    points: tuple[LatentTsnePoint, ...]

    def __post_init__(self) -> None:
        if (
            self.window_size <= 0
            or self.latent_dimension <= 0
            or self.lambda_age <= 0.0
            or self.source_count < len(self.points)
            or self.count_per_context <= 0
            or self.perplexity <= 0.0
            or self.perplexity >= len(self.points)
        ):
            raise ValueError("t-SNE dimensions, counts, and perplexity are inconsistent")
        _finite(self.lambda_age, "t-SNE lambda")
        _finite(self.perplexity, "t-SNE perplexity")
        _finite(self.final_kl_divergence, "t-SNE KL divergence")
        if self.final_kl_divergence < 0.0:
            raise ValueError("t-SNE KL divergence must be non-negative")
        expected_count = self.count_per_context * len(GenomicContext)
        if len(self.points) != expected_count:
            raise ValueError("t-SNE points must have the declared balanced context count")
        contexts = tuple(point.context for point in self.points)
        if any(contexts.count(context) != self.count_per_context for context in GenomicContext):
            raise ValueError("t-SNE points must be exactly balanced across genomic contexts")
        probe_ids = tuple(point.probe_id for point in self.points)
        if len(set(probe_ids)) != len(probe_ids):
            raise ValueError("t-SNE panel contains duplicate probe IDs")


@dataclass(frozen=True, slots=True)
class InterimSiteData:
    schema: str
    protocol_id: str
    status: str
    disclaimer: str
    split_name: str
    split_family: str
    data_sha256: str
    split_sha256: str
    retained_probe_count: int
    retained_sample_count: int
    completed_windows: tuple[int, ...]
    planned_windows: tuple[int, ...]
    latent_dimensions: tuple[int, ...]
    lambda_age_values: tuple[float, ...]
    artifact_ids: tuple[str, ...]
    tuning_sweep: tuple[TuningSweepPoint, ...]
    primary_window_sweep: tuple[WindowSweepPoint, ...]
    primary_distance_metrics: tuple[DistanceMetricPoint, ...]
    primary_age_metrics: tuple[AgeMetricPoint, ...]
    exploratory_age_metrics: tuple[ExploratoryAgeMetricPoint, ...]
    scatter_panels: tuple[CorrelationScatterPanel, ...]
    latent_tsne_panels: tuple[LatentTsnePanel, ...]
    exploratory_uniform_metrics: tuple[ExploratoryMetricPoint, ...]
    exploratory_distance_metrics: tuple[ExploratoryMetricPoint, ...]
    kernel_weights: tuple[KernelWeightPoint, ...]

    def __post_init__(self) -> None:
        if self.schema != INTERIM_SITE_SCHEMA or not all(
            (self.protocol_id, self.status, self.disclaimer, self.split_name, self.split_family)
        ):
            raise ValueError("interim site identity and labels are required")
        if (
            _SHA256.fullmatch(self.data_sha256) is None
            or _SHA256.fullmatch(self.split_sha256) is None
        ):
            raise ValueError("interim site requires lowercase SHA-256 fingerprints")
        if self.retained_probe_count <= 0 or self.retained_sample_count <= 1:
            raise ValueError("interim site requires positive probes and at least two samples")
        axes = (
            self.completed_windows,
            self.planned_windows,
            self.latent_dimensions,
            self.lambda_age_values,
        )
        if any(not axis or tuple(sorted(set(axis))) != axis for axis in axes):
            raise ValueError("interim sweep axes must be non-empty, unique, and increasing")
        if not set(self.completed_windows) < set(self.planned_windows):
            raise ValueError("interim windows must be a strict subset of planned windows")
        if not self.artifact_ids or len(set(self.artifact_ids)) != len(self.artifact_ids):
            raise ValueError("interim artifact IDs must be non-empty and unique")
        sections = (
            self.tuning_sweep,
            self.primary_window_sweep,
            self.primary_distance_metrics,
            self.primary_age_metrics,
            self.exploratory_age_metrics,
            self.scatter_panels,
            self.latent_tsne_panels,
            self.exploratory_uniform_metrics,
            self.exploratory_distance_metrics,
            self.kernel_weights,
        )
        if any(not section for section in sections):
            raise ValueError("interim site requires every completed-results section")
        self._validate_keys()

    def _validate_keys(self) -> None:
        windows = set(self.completed_windows)
        expected_pairs = {
            (window, population) for window in windows for population in _PAIR_POPULATIONS
        }
        window_keys = tuple(
            (point.window_size, point.population) for point in self.primary_window_sweep
        )
        if len(set(window_keys)) != len(window_keys) or set(window_keys) != expected_pairs:
            raise ValueError("primary interim window results must cover every pair population once")
        age_keys = tuple((point.window_size, point.stage) for point in self.primary_age_metrics)
        if len(set(age_keys)) != len(age_keys) or set(age_keys) != {
            (window, stage) for window in windows for stage in _AGE_STAGES
        }:
            raise ValueError("primary interim age results must cover every stage once")
        exploratory_age_keys = tuple(point.window_size for point in self.exploratory_age_metrics)
        if (
            len(set(exploratory_age_keys)) != len(exploratory_age_keys)
            or set(exploratory_age_keys) != windows
        ):
            raise ValueError("direct tanh age metrics must cover every completed window once")
        scatter_keys = tuple((point.window_size, point.population) for point in self.scatter_panels)
        expected_scatter = {
            (window, population) for window in windows for population in _SCATTER_POPULATIONS
        }
        if len(set(scatter_keys)) != len(scatter_keys) or set(scatter_keys) != expected_scatter:
            raise ValueError("scatter panels must cover age and both pair populations")
        projection_dimensions = tuple(
            dimension for dimension in self.latent_dimensions if dimension <= 128
        )
        tsne_keys = tuple(
            (point.window_size, point.latent_dimension, point.lambda_age)
            for point in self.latent_tsne_panels
        )
        expected_tsne = {
            (window, dimension, 0.1) for window in windows for dimension in projection_dimensions
        }
        if len(set(tsne_keys)) != len(tsne_keys) or set(tsne_keys) != expected_tsne:
            raise ValueError("t-SNE panels must cover every dimension through 128 at lambda 0.1")
        point_identity = tuple(
            (point.probe_id, point.chromosome, point.position, point.context)
            for point in self.latent_tsne_panels[0].points
        )
        if any(
            tuple(
                (point.probe_id, point.chromosome, point.position, point.context)
                for point in panel.points
            )
            != point_identity
            for panel in self.latent_tsne_panels[1:]
        ):
            raise ValueError("every t-SNE panel must use the same ordered validation loci")
        tuning_keys = tuple(
            (point.window_size, point.latent_dimension, point.lambda_age)
            for point in self.tuning_sweep
        )
        expected_tuning = {
            (window, dimension, lambda_age)
            for window in windows
            for dimension in self.latent_dimensions
            for lambda_age in self.lambda_age_values
        }
        if len(set(tuning_keys)) != len(tuning_keys) or set(tuning_keys) != expected_tuning:
            raise ValueError("validation tuning sweep must contain the complete Cartesian grid")
        selected = tuple(point for point in self.tuning_sweep if point.selected)
        selected_primary = {
            (point.window_size, point.latent_dimension, point.lambda_age)
            for point in self.primary_window_sweep
        }
        if (
            len(selected) != len(windows)
            or {(point.window_size, point.latent_dimension, point.lambda_age) for point in selected}
            != selected_primary
        ):
            raise ValueError("one validation-selected candidate must match each primary test model")
        distance_keys = tuple(
            (point.window_size, point.population, point.distance_class)
            for point in self.primary_distance_metrics
        )
        if (
            len(set(distance_keys)) != len(distance_keys)
            or {(window, population) for window, population, _ in distance_keys} != expected_pairs
        ):
            raise ValueError("primary distance rows must be unique and cover every pair population")
        uniform_keys = tuple(
            (point.window_size, point.population, point.model)
            for point in self.exploratory_uniform_metrics
        )
        expected_uniform = {
            (window, population, model)
            for window, population in expected_pairs
            for model in EXPLORATORY_MODELS
        }
        if len(set(uniform_keys)) != len(uniform_keys) or set(uniform_keys) != expected_uniform:
            raise ValueError("exploratory uniform metrics must cover all four models exactly")
        exploratory_distance_keys = tuple(
            (point.window_size, point.population, point.distance_class, point.model)
            for point in self.exploratory_distance_metrics
        )
        expected_exploratory_distance = {
            (*key, model)
            for key in set(distance_keys)
            for model in ("sequence", "psd_distance", "psd_distance_plus_sequence")
        }
        if (
            len(set(exploratory_distance_keys)) != len(exploratory_distance_keys)
            or set(exploratory_distance_keys) != expected_exploratory_distance
        ):
            raise ValueError(
                "exploratory distance metrics must align to every primary distance row"
            )
        weight_keys = tuple((point.window_size, point.component) for point in self.kernel_weights)
        if (
            len(set(weight_keys)) != len(weight_keys)
            or {window for window, _ in weight_keys} != windows
        ):
            raise ValueError("kernel weights must be unique and cover every completed window")
        for window in windows:
            selected_weights = tuple(
                point for point in self.kernel_weights if point.window_size == window
            )
            if not any(
                point.component == "sequence_cosine" for point in selected_weights
            ) or not math.isclose(
                sum(point.weight for point in selected_weights), 1.0, rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise ValueError("each combined kernel must contain sequence and sum to one")

    def as_json(self) -> dict[str, JsonValue]:
        return cast(dict[str, JsonValue], asdict(self))


def build_interim_static_site(
    *, data: InterimSiteData, template_directory: Path, output_directory: Path
) -> None:
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
