from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from methylation_latent.domain import (
    GenomicContext,
    parse_autosome,
    parse_one_based_position,
    parse_probe_id,
)
from methylation_latent.evaluation import PairPopulation
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
    LatentUmapPanel,
    LatentUmapPoint,
    ScatterClusterSeries,
    ScatterPredictionSeries,
    TuningSweepPoint,
    build_interim_static_site,
)
from methylation_latent.site import AgeMetricPoint, DistanceMetricPoint, WindowSweepPoint

_POPULATIONS = (
    PairPopulation.SEEN_BY_HELD_OUT,
    PairPopulation.HELD_OUT_BY_HELD_OUT,
)
_MODELS = (
    "sequence",
    "registered_distance_class",
    "psd_distance",
    "psd_distance_plus_sequence",
)


def _exploratory(
    population: PairPopulation,
    model: str,
    distance_class: str | None = None,
) -> ExploratoryMetricPoint:
    return ExploratoryMetricPoint(
        window_size=1024,
        population=population,
        distance_class=distance_class,
        model=model,
        count=10,
        target_mean=0.03,
        prediction_mean=0.04,
        mse=0.05,
        pearson=0.2,
        pearson_status="defined",
        r_squared=-0.1,
    )


def _cluster_diagnostic(model: str) -> AgeClusterDiagnostic:
    return AgeClusterDiagnostic(
        window_size=1024,
        model=model,
        source_count=8,
        negative_count=4,
        positive_count=4,
        negative_empirical_center=-0.2,
        negative_prediction_center=-0.1,
        positive_empirical_center=0.2,
        positive_prediction_center=0.1,
        context_cramer_v=0.6,
        design_cramer_v=0.3,
        strand_cramer_v=0.01,
        chromosome_cramer_v=0.15,
        chromosome_status="defined",
        cpg_density_cohen_d=1.7,
        gc_content_cohen_d=0.9,
        female_sample_count=374,
        male_sample_count=282,
        female_negative_mean=-0.2,
        female_positive_mean=0.2,
        female_separation_cohen_d=1.6,
        male_negative_mean=-0.25,
        male_positive_mean=0.25,
        male_separation_cohen_d=1.9,
        female_male_target_pearson=0.93,
        context_counts=tuple(
            ContextClusterCounts(context=context, negative_count=1, positive_count=1)
            for context in GenomicContext
        ),
    )


def _valid() -> InterimSiteData:
    primary_window = tuple(
        WindowSweepPoint(1024, population, 16, 1.0, 0.05, 0.2, -0.1) for population in _POPULATIONS
    )
    primary_distance = (
        DistanceMetricPoint(
            1024,
            PairPopulation.SEEN_BY_HELD_OUT,
            "trans",
            10,
            0.03,
            0.04,
            0.02,
            0.05,
            0.2,
            -0.1,
        ),
        DistanceMetricPoint(
            1024,
            PairPopulation.HELD_OUT_BY_HELD_OUT,
            "cis_1_4kb",
            10,
            0.03,
            0.04,
            0.02,
            0.05,
            0.2,
            -0.1,
        ),
    )
    primary_age = tuple(
        AgeMetricPoint(1024, stage, 10, 0.05, 0.2)
        for stage in ("sequence_features", "caduceus_age_only", "full_latent_metric")
    )
    uniform = tuple(
        _exploratory(population, model) for population in _POPULATIONS for model in _MODELS
    )
    by_distance = tuple(
        _exploratory(point.population, model, point.distance_class)
        for point in primary_distance
        for model in ("sequence", "psd_distance", "psd_distance_plus_sequence")
    )
    scatter_target = (-0.1, 0.2)
    scatter_indices = (0, 2)
    scatter_panels = (
        CorrelationScatterPanel(
            1024,
            "age",
            8,
            7,
            scatter_indices,
            scatter_target,
            (
                ScatterPredictionSeries("cosine_age_only", (-0.2, 0.1)),
                ScatterPredictionSeries("direct_tanh", (-0.1, 0.3)),
                ScatterPredictionSeries("full_latent_metric", (0.0, 0.4)),
            ),
            tuple(
                ScatterClusterSeries(model, (0, 1))
                for model in ("cosine_age_only", "direct_tanh", "full_latent_metric")
            ),
        ),
        *(
            CorrelationScatterPanel(
                1024,
                population.value,
                8,
                8,
                scatter_indices,
                scatter_target,
                (ScatterPredictionSeries("full_latent_metric", (-0.1, 0.1)),),
                (),
            )
            for population in _POPULATIONS
        ),
    )
    umap_points = tuple(
        LatentUmapPoint(
            probe_id=parse_probe_id(f"cg{index + 1:08d}"),
            chromosome=parse_autosome(1),
            position=parse_one_based_position(index + 1),
            context=context,
            x=float(index),
            y=float(-index),
        )
        for index, context in enumerate(GenomicContext)
    )
    return InterimSiteData(
        schema=INTERIM_SITE_SCHEMA,
        protocol_id="protocol-v2",
        status="interim",
        disclaimer="not complete and post-hoc",
        split_name="diverse-blocks",
        split_family="diverse_blocks",
        data_sha256="a" * 64,
        split_sha256="b" * 64,
        retained_probe_count=100,
        retained_sample_count=656,
        female_sample_count=374,
        male_sample_count=282,
        cell_composition_status="not_testable_no_measured_or_precomputed_cell_proportions",
        completed_windows=(1024,),
        planned_windows=(1024, 4096),
        latent_dimensions=(16,),
        lambda_age_values=(1.0,),
        artifact_ids=("artifact-one",),
        tuning_sweep=(TuningSweepPoint(1024, 16, 1.0, 100, 0.04, 0.03, 0.07, True),),
        primary_window_sweep=primary_window,
        primary_distance_metrics=primary_distance,
        primary_age_metrics=primary_age,
        exploratory_age_metrics=(ExploratoryAgeMetricPoint(1024, 10, 100, 0.04, 0.05, 0.2, -0.1),),
        scatter_panels=scatter_panels,
        age_cluster_diagnostics=tuple(
            _cluster_diagnostic(model)
            for model in ("cosine_age_only", "direct_tanh", "full_latent_metric")
        ),
        age_cluster_agreements=(AgeClusterAgreement(1024, 8, 0.95, 0.82),),
        latent_umap_panels=(
            LatentUmapPanel(
                window_size=1024,
                latent_dimension=16,
                lambda_age=0.1,
                source_count=10,
                count_per_context=1,
                n_neighbors=2,
                min_dist=0.1,
                initial_cross_entropy=2.0,
                final_cross_entropy=0.5,
                graph_edge_count=4,
                spectral_gap=0.2,
                age_x=0.5,
                age_y=-0.5,
                points=umap_points,
            ),
        ),
        exploratory_uniform_metrics=uniform,
        exploratory_distance_metrics=by_distance,
        kernel_weights=(
            KernelWeightPoint(1024, "global", 0.25),
            KernelWeightPoint(1024, "sequence_cosine", 0.75),
        ),
    )


def test_interim_site_serializes_and_builds_exclusive_static_output(tmp_path: Path) -> None:
    data = _valid()
    payload = data.as_json()
    assert payload["schema"] == INTERIM_SITE_SCHEMA
    assert data.tuning_sweep[0].selected is True
    template = tmp_path / "template"
    template.mkdir()
    for name in ("index.html", "app.js", "style.css"):
        (template / name).write_text(name, encoding="utf-8")
    output = tmp_path / "site"
    build_interim_static_site(data=data, template_directory=template, output_directory=output)
    built = json.loads((output / "results.json").read_text())
    assert built["schema"] == INTERIM_SITE_SCHEMA
    assert built["tuning_sweep"][0]["selected"] is True
    with pytest.raises(FileExistsError):
        build_interim_static_site(data=data, template_directory=template, output_directory=output)
    (template / "app.js").unlink()
    with pytest.raises(FileNotFoundError):
        build_interim_static_site(
            data=data,
            template_directory=template,
            output_directory=tmp_path / "missing-asset",
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"window_size": 0}, "positive"),
        ({"latent_dimension": 0}, "positive"),
        ({"lambda_age": 0.0}, "positive"),
        ({"selected_step": -1}, "positive"),
        ({"pair_mse": float("nan")}, "finite"),
        ({"age_mse": -0.1}, "non-negative"),
        ({"selection_score": 0.08}, "must equal"),
    ],
)
def test_tuning_sweep_point_rejects_invalid_state(change: dict[str, object], message: str) -> None:
    point = TuningSweepPoint(1024, 16, 1.0, 100, 0.04, 0.03, 0.07, True)
    with pytest.raises(ValueError, match=message):
        replace(point, **change)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"window_size": 0}, "positive"),
        ({"count": 0}, "positive"),
        ({"population": PairPopulation.TRAINING_BY_TRAINING}, "held-out"),
        ({"distance_class": "near"}, "unknown distance"),
        ({"model": "oracle"}, "unknown exploratory"),
        ({"target_mean": float("inf")}, "finite"),
        ({"mse": -0.1}, "non-negative"),
        ({"target_mean": 1.1}, "bounded"),
        ({"prediction_mean": -1.1}, "prediction mean"),
        ({"pearson": None}, "null Pearson"),
        ({"pearson_status": "missing"}, "explicitly labeled"),
        ({"pearson": 1.1}, "explicitly labeled"),
    ],
)
def test_exploratory_metric_rejects_invalid_state(change: dict[str, object], message: str) -> None:
    point = _exploratory(PairPopulation.SEEN_BY_HELD_OUT, "sequence")
    with pytest.raises(ValueError, match=message):
        replace(point, **change)


def test_exploratory_metric_accepts_explicitly_undefined_pearson() -> None:
    point = replace(
        _exploratory(PairPopulation.SEEN_BY_HELD_OUT, "psd_distance", "trans"),
        pearson=None,
        pearson_status="undefined_constant_prediction",
    )
    assert point.pearson is None


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"window_size": 0}, "positive window"),
        ({"component": ""}, "component name"),
        ({"weight": float("nan")}, "finite"),
        ({"weight": 1.1}, r"\[0,1\]"),
    ],
)
def test_kernel_weight_rejects_invalid_state(change: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        replace(KernelWeightPoint(1024, "sequence_cosine", 1.0), **change)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"window_size": 0}, "counts"),
        ({"count": 0}, "counts"),
        ({"selected_step": -1}, "selected step"),
        ({"validation_mse": float("nan")}, "finite"),
        ({"mse": -0.1}, "non-negative"),
        ({"pearson": 1.1}, "Pearson"),
    ],
)
def test_exploratory_age_metric_rejects_invalid_state(
    change: dict[str, object], message: str
) -> None:
    point = ExploratoryAgeMetricPoint(1024, 10, 100, 0.04, 0.05, 0.2, -0.1)
    with pytest.raises(ValueError, match=message):
        replace(point, **change)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"model": ""}, "named"),
        ({"values": ()}, "non-empty"),
        ({"values": (float("nan"),)}, "correlations"),
        ({"values": (1.1,)}, "correlations"),
    ],
)
def test_scatter_series_rejects_invalid_state(change: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        replace(ScatterPredictionSeries("full_latent_metric", (0.1,)), **change)


def test_scatter_panel_rejects_misalignment_and_unknown_models() -> None:
    valid = _valid().scatter_panels[0]
    cases: tuple[tuple[dict[str, object], str], ...] = (
        ({"window_size": 0}, "positive"),
        ({"source_count": 0}, "positive"),
        ({"population": "training"}, "unknown"),
        ({"sample_indices": (1, 0)}, "increasing"),
        ({"sample_indices": (0, 8)}, "in range"),
        ({"target": (0.1,)}, "aligned"),
        ({"target": (0.1, 1.1)}, "correlations"),
        ({"predictions": valid.predictions[:-1]}, "required aligned models"),
        (
            {
                "predictions": (
                    *valid.predictions[:-1],
                    ScatterPredictionSeries("full_latent_metric", (0.1,)),
                )
            },
            "required aligned models",
        ),
    )
    for change, message in cases:
        with pytest.raises(ValueError, match=message):
            replace(valid, **change)


def test_umap_point_and_panel_reject_invalid_geometry() -> None:
    data = _valid()
    point = data.latent_umap_panels[0].points[0]
    with pytest.raises(ValueError, match="finite"):
        replace(point, x=float("nan"))
    panel = data.latent_umap_panels[0]
    cases: tuple[tuple[dict[str, object], str], ...] = (
        ({"window_size": 0}, "inconsistent"),
        ({"source_count": 3}, "inconsistent"),
        ({"n_neighbors": 5}, "inconsistent"),
        ({"final_cross_entropy": float("nan")}, "finite"),
        ({"final_cross_entropy": -0.1}, "optimization"),
        ({"final_cross_entropy": 2.0}, "optimization"),
        ({"count_per_context": 2}, "balanced context count"),
        (
            {
                "points": (
                    panel.points[0],
                    replace(panel.points[1], context=GenomicContext.ISLAND),
                    *panel.points[2:],
                )
            },
            "balanced across",
        ),
        (
            {
                "points": (
                    panel.points[0],
                    replace(panel.points[1], probe_id=panel.points[0].probe_id),
                    *panel.points[2:],
                )
            },
            "duplicate",
        ),
    )
    for change, message in cases:
        with pytest.raises(ValueError, match=message):
            replace(panel, **change)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schema": "primary"}, "identity"),
        ({"protocol_id": ""}, "identity"),
        ({"data_sha256": "bad"}, "SHA-256"),
        ({"retained_probe_count": 0}, "positive probes"),
        ({"retained_sample_count": 1}, "at least two"),
        ({"female_sample_count": 0}, "phenotype"),
        ({"completed_windows": ()}, "sweep axes"),
        ({"planned_windows": (4096, 1024)}, "sweep axes"),
        ({"latent_dimensions": (16, 16)}, "sweep axes"),
        ({"completed_windows": (1024, 4096)}, "strict subset"),
        ({"artifact_ids": ()}, "artifact IDs"),
        ({"artifact_ids": ("same", "same")}, "artifact IDs"),
        ({"tuning_sweep": ()}, "every completed-results"),
        ({"scatter_panels": ()}, "every completed-results"),
        ({"age_cluster_diagnostics": ()}, "every completed-results"),
        ({"latent_umap_panels": ()}, "every completed-results"),
    ],
)
def test_interim_envelope_rejects_invalid_state(change: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_valid(), **change)


def test_interim_cross_panel_contract_rejects_incomplete_or_duplicate_keys() -> None:
    data = _valid()
    cases: tuple[tuple[str, object, str], ...] = (
        (
            "primary_window_sweep",
            (*data.primary_window_sweep, data.primary_window_sweep[0]),
            "window results",
        ),
        ("primary_age_metrics", data.primary_age_metrics[:-1], "age results"),
        (
            "tuning_sweep",
            (replace(data.tuning_sweep[0], latent_dimension=32),),
            "Cartesian",
        ),
        (
            "tuning_sweep",
            (replace(data.tuning_sweep[0], selected=False),),
            "selected candidate",
        ),
        (
            "primary_distance_metrics",
            (*data.primary_distance_metrics, data.primary_distance_metrics[0]),
            "distance rows",
        ),
        ("primary_distance_metrics", data.primary_distance_metrics[:1], "distance rows"),
        (
            "exploratory_uniform_metrics",
            data.exploratory_uniform_metrics[:-1],
            "uniform metrics",
        ),
        (
            "exploratory_distance_metrics",
            data.exploratory_distance_metrics[:-1],
            "distance metrics",
        ),
        (
            "kernel_weights",
            (*data.kernel_weights, data.kernel_weights[0]),
            "kernel weights",
        ),
        (
            "kernel_weights",
            (replace(data.kernel_weights[0], weight=1.0),),
            "contain sequence",
        ),
        (
            "kernel_weights",
            (data.kernel_weights[0], replace(data.kernel_weights[1], weight=0.5)),
            "sum to one",
        ),
        (
            "exploratory_age_metrics",
            (*data.exploratory_age_metrics, data.exploratory_age_metrics[0]),
            "direct tanh age",
        ),
        (
            "scatter_panels",
            data.scatter_panels[:-1],
            "scatter panels",
        ),
        (
            "latent_umap_panels",
            (replace(data.latent_umap_panels[0], lambda_age=1.0),),
            "through 128",
        ),
    )
    for field, changed, message in cases:
        with pytest.raises(ValueError, match=message):
            replace(data, **{field: changed})


def test_interim_rejects_different_loci_across_umap_panels() -> None:
    data = _valid()
    second = replace(data.latent_umap_panels[0], window_size=2048)
    expanded = replace(
        data,
        completed_windows=(1024, 2048),
        planned_windows=(1024, 2048, 4096),
        primary_window_sweep=(
            *data.primary_window_sweep,
            *(replace(point, window_size=2048) for point in data.primary_window_sweep),
        ),
        primary_distance_metrics=(
            *data.primary_distance_metrics,
            *(replace(point, window_size=2048) for point in data.primary_distance_metrics),
        ),
        primary_age_metrics=(
            *data.primary_age_metrics,
            *(replace(point, window_size=2048) for point in data.primary_age_metrics),
        ),
        exploratory_age_metrics=(
            *data.exploratory_age_metrics,
            replace(data.exploratory_age_metrics[0], window_size=2048),
        ),
        scatter_panels=(
            *data.scatter_panels,
            *(replace(point, window_size=2048) for point in data.scatter_panels),
        ),
        age_cluster_diagnostics=(
            *data.age_cluster_diagnostics,
            *(replace(point, window_size=2048) for point in data.age_cluster_diagnostics),
        ),
        age_cluster_agreements=(
            *data.age_cluster_agreements,
            replace(data.age_cluster_agreements[0], window_size=2048),
        ),
        latent_umap_panels=(data.latent_umap_panels[0], second),
        tuning_sweep=(
            data.tuning_sweep[0],
            replace(data.tuning_sweep[0], window_size=2048),
        ),
        exploratory_uniform_metrics=(
            *data.exploratory_uniform_metrics,
            *(replace(point, window_size=2048) for point in data.exploratory_uniform_metrics),
        ),
        exploratory_distance_metrics=(
            *data.exploratory_distance_metrics,
            *(replace(point, window_size=2048) for point in data.exploratory_distance_metrics),
        ),
        kernel_weights=(
            *data.kernel_weights,
            *(replace(point, window_size=2048) for point in data.kernel_weights),
        ),
    )
    changed_second = replace(
        second,
        points=(
            replace(second.points[0], probe_id=parse_probe_id("cg99999999")),
            *second.points[1:],
        ),
    )
    with pytest.raises(ValueError, match="same ordered validation loci"):
        replace(
            expanded,
            latent_umap_panels=(data.latent_umap_panels[0], changed_second),
        )


def test_selected_tuning_candidate_must_match_primary_configuration() -> None:
    data = _valid()
    changed = tuple(replace(point, latent_dimension=32) for point in data.primary_window_sweep)
    with pytest.raises(ValueError, match="selected candidate"):
        replace(data, primary_window_sweep=changed)


def test_kernel_sum_tolerance_is_exact_enough_for_serialized_weights() -> None:
    data = _valid()
    nearly_one = (
        KernelWeightPoint(1024, "global", math.nextafter(0.25, 0.0)),
        KernelWeightPoint(1024, "sequence_cosine", 0.75),
    )
    assert replace(data, kernel_weights=nearly_one).kernel_weights == nearly_one
