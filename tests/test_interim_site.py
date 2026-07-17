from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from methylation_latent.evaluation import PairPopulation
from methylation_latent.interim_site import (
    INTERIM_SITE_SCHEMA,
    ExploratoryMetricPoint,
    InterimSiteData,
    KernelWeightPoint,
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
        completed_windows=(1024,),
        planned_windows=(1024, 4096),
        latent_dimensions=(16,),
        lambda_age_values=(1.0,),
        artifact_ids=("artifact-one",),
        tuning_sweep=(TuningSweepPoint(1024, 16, 1.0, 100, 0.04, 0.03, 0.07, True),),
        primary_window_sweep=primary_window,
        primary_distance_metrics=primary_distance,
        primary_age_metrics=primary_age,
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
        ({"schema": "primary"}, "identity"),
        ({"protocol_id": ""}, "identity"),
        ({"data_sha256": "bad"}, "SHA-256"),
        ({"retained_probe_count": 0}, "positive probes"),
        ({"retained_sample_count": 1}, "at least two"),
        ({"completed_windows": ()}, "sweep axes"),
        ({"planned_windows": (4096, 1024)}, "sweep axes"),
        ({"latent_dimensions": (16, 16)}, "sweep axes"),
        ({"completed_windows": (1024, 4096)}, "strict subset"),
        ({"artifact_ids": ()}, "artifact IDs"),
        ({"artifact_ids": ("same", "same")}, "artifact IDs"),
        ({"tuning_sweep": ()}, "every completed-results"),
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
    )
    for field, changed, message in cases:
        with pytest.raises(ValueError, match=message):
            replace(data, **{field: changed})


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
