from __future__ import annotations

from dataclasses import replace

import pytest
import torch as t

from methylation_latent.umap import (
    UmapConfig,
    UmapGraph,
    exact_umap,
    fit_default_curve_parameters,
    fuzzy_simplicial_graph,
)


def _config() -> UmapConfig:
    return UmapConfig(
        n_neighbors=4,
        local_connectivity=1.0,
        smooth_knn_search_steps=64,
        min_dist=0.1,
        spread=1.0,
        optimization_steps=80,
        learning_rate=0.05,
        seed=73,
    )


def _values() -> t.Tensor:
    generator = t.Generator().manual_seed(91)
    first = t.randn((6, 4), generator=generator, dtype=t.float64) - 1.0
    second = t.randn((6, 4), generator=generator, dtype=t.float64) + 1.0
    return t.cat((first, second))


def test_fuzzy_graph_is_symmetric_bounded_connected_and_has_unit_local_edges() -> None:
    graph = fuzzy_simplicial_graph(_values(), config=_config())
    assert t.equal(graph.memberships, graph.memberships.mT)
    assert t.equal(graph.memberships.diagonal(), t.zeros(12, dtype=t.float64))
    assert bool(t.all((graph.memberships >= 0.0) & (graph.memberships <= 1.0)).item())
    assert bool(t.all(graph.memberships.max(dim=1).values == 1.0).item())
    assert bool(t.all(graph.sigmas > 0.0).item())
    assert bool(t.all(graph.rhos > 0.0).item())


def test_fuzzy_graph_forces_mathematical_self_distance_to_exact_zero() -> None:
    generator = t.Generator().manual_seed(113)
    values = t.randn((20, 32), generator=generator, dtype=t.float64)
    graph = fuzzy_simplicial_graph(values, config=replace(_config(), n_neighbors=8))
    assert t.equal(graph.memberships.diagonal(), t.zeros(20, dtype=t.float64))


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ({"memberships": t.ones((2, 3), dtype=t.float64)}, "square"),
        ({"sigmas": t.ones(11, dtype=t.float64)}, "align"),
        ({"sigmas": t.full((12,), float("nan"), dtype=t.float64)}, "finite"),
        ({"sigmas": t.zeros(12, dtype=t.float64)}, "bounds"),
        ({"rhos": -t.ones(12, dtype=t.float64)}, "bounds"),
    ),
)
def test_umap_graph_rejects_invalid_scales_or_geometry(
    change: dict[str, t.Tensor], message: str
) -> None:
    valid = fuzzy_simplicial_graph(_values(), config=_config())
    with pytest.raises((TypeError, ValueError), match=message):
        replace(valid, **change)


def test_umap_graph_rejects_diagonal_asymmetry_and_isolated_vertices() -> None:
    valid = fuzzy_simplicial_graph(_values(), config=_config())
    diagonal = valid.memberships.clone()
    diagonal[0, 0] = 1.0
    with pytest.raises(ValueError, match="diagonal"):
        UmapGraph(diagonal, valid.sigmas, valid.rhos)
    asymmetric = valid.memberships.clone()
    asymmetric[0, 1] = 0.123
    with pytest.raises(ValueError, match="symmetric"):
        UmapGraph(asymmetric, valid.sigmas, valid.rhos)
    isolated = valid.memberships.clone()
    isolated[0] = 0.0
    isolated[:, 0] = 0.0
    with pytest.raises(ValueError, match="isolated"):
        UmapGraph(isolated, valid.sigmas, valid.rhos)


def test_default_curve_fit_reproduces_reference_parameters() -> None:
    first = fit_default_curve_parameters(1.0, 0.1)
    second = fit_default_curve_parameters(1.0, 0.1)
    assert first == second
    assert first[0] == pytest.approx(1.57694346, rel=2e-5)
    assert first[1] == pytest.approx(0.89506088, rel=2e-5)


def test_exact_umap_is_deterministic_centered_finite_and_noncollapsed() -> None:
    first = exact_umap(_values(), config=_config())
    second = exact_umap(_values(), config=_config())
    assert t.equal(first.coordinates, second.coordinates)
    assert first.initial_cross_entropy == second.initial_cross_entropy
    assert first.final_cross_entropy == second.final_cross_entropy
    assert first.final_cross_entropy < first.initial_cross_entropy
    assert t.allclose(
        first.coordinates.mean(dim=0),
        t.zeros(2, dtype=t.float64),
        atol=1e-15,
        rtol=0.0,
    )
    assert bool(t.isfinite(first.coordinates).all().item())
    assert bool(t.all(first.coordinates.std(dim=0) > 0.0).item())
    assert first.graph_edge_count > 0
    assert first.spectral_gap > 0.0


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ({"coordinates": t.ones((2, 2), dtype=t.float64)}, "shape"),
        ({"initial_cross_entropy": -1.0}, "positive"),
        ({"final_cross_entropy": 100.0}, "strictly reduce"),
        ({"graph_edge_count": 0}, "edge"),
        ({"coordinates": t.full((12, 2), float("nan"), dtype=t.float64)}, "finite"),
        ({"coordinates": t.zeros((12, 2), dtype=t.float64)}, "duplicate"),
    ),
)
def test_umap_projection_rejects_invalid_state(change: dict[str, object], message: str) -> None:
    valid = exact_umap(_values(), config=_config())
    with pytest.raises((TypeError, ValueError), match=message):
        replace(valid, **change)


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ({"n_neighbors": 1}, "counts"),
        ({"local_connectivity": 2.0}, "local_connectivity"),
        ({"smooth_knn_search_steps": 0}, "counts"),
        ({"min_dist": -0.1}, "min_dist"),
        ({"min_dist": 2.0}, "min_dist"),
        ({"spread": 0.0}, "positive"),
        ({"optimization_steps": 0}, "counts"),
        ({"learning_rate": 0.0}, "positive"),
    ),
)
def test_umap_config_rejects_invalid_state(change: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_config(), **change)


@pytest.mark.parametrize(
    ("spread", "min_dist"),
    ((0.0, 0.0), (1.0, -0.1), (1.0, 2.0), (float("nan"), 0.1)),
)
def test_umap_curve_fit_rejects_invalid_range(spread: float, min_dist: float) -> None:
    with pytest.raises(ValueError, match="requires"):
        fit_default_curve_parameters(spread, min_dist)


@pytest.mark.parametrize(
    ("values", "message"),
    (
        (t.ones((4, 2), dtype=t.int64), "floating"),
        (t.ones((2, 2), dtype=t.float64), "at least three"),
        (t.ones((5, 0), dtype=t.float64), "non-empty"),
        (t.tensor(((0.0, 1.0), (1.0, 2.0), (float("nan"), 0.0), (2.0, 1.0), (3.0, 1.0))), "finite"),
        (t.tensor(((0.0, 1.0), (0.0, 1.0), (1.0, 2.0), (2.0, 1.0), (3.0, 1.0))), "duplicate"),
    ),
)
def test_exact_umap_rejects_invalid_input(values: t.Tensor, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        exact_umap(values, config=_config())


def test_exact_umap_rejects_neighbor_count_equal_to_points() -> None:
    with pytest.raises(ValueError, match="smaller"):
        exact_umap(_values()[:4], config=_config())


def test_exact_umap_rejects_disconnected_fuzzy_graph() -> None:
    first = t.tensor(((0.0, 0.0), (0.0, 0.1), (0.1, 0.0), (0.1, 0.1)), dtype=t.float64)
    second = first + 100.0
    with pytest.raises(ValueError, match="connected"):
        exact_umap(
            t.cat((first, second)),
            config=replace(_config(), n_neighbors=3),
        )
