"""Deterministic exact fuzzy-cross-entropy UMAP implemented only with Torch."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from functools import cache

import torch as t
from beartype import beartype
from jaxtyping import Float64, jaxtyped

Float64FeatureMatrix = Float64[t.Tensor, "points features"]
Float64SquareMatrix = Float64[t.Tensor, "points points"]
Float64Projection = Float64[t.Tensor, "points 2"]
Float64SphereProjection = Float64[t.Tensor, "points 3"]
Float64PointVector = Float64[t.Tensor, "points"]
Float64NeighborMatrix = Float64[t.Tensor, "points neighbors"]

_SMOOTH_K_TOLERANCE = 1.0e-5
_MIN_K_DISTANCE_SCALE = 1.0e-3
_SPHERICAL_NORM_ABSOLUTE_TOLERANCE = 2.0e-7


class UmapInputMetric(StrEnum):
    """Input-space metric used to construct the exact fuzzy-neighborhood graph."""

    EUCLIDEAN = "euclidean"
    SPHERICAL_GEODESIC = "spherical_geodesic"


@beartype
@dataclass(frozen=True, slots=True)
class UmapConfig:
    input_metric: UmapInputMetric
    n_neighbors: int
    local_connectivity: float
    smooth_knn_search_steps: int
    min_dist: float
    spread: float
    optimization_steps: int
    learning_rate: float
    seed: int

    def __post_init__(self) -> None:
        finite_positive = (
            self.local_connectivity,
            self.spread,
            self.learning_rate,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in finite_positive):
            raise ValueError("UMAP connectivity, spread, and learning rate must be positive")
        if (
            self.n_neighbors < 2
            or self.smooth_knn_search_steps <= 0
            or self.optimization_steps <= 0
        ):
            raise ValueError("UMAP neighbor and step counts must be positive")
        if self.local_connectivity != 1.0:
            raise ValueError("this exact Torch UMAP currently requires local_connectivity=1")
        if not math.isfinite(self.min_dist) or self.min_dist < 0.0 or self.min_dist > self.spread:
            raise ValueError("UMAP min_dist must lie in [0, spread]")


@dataclass(frozen=True, slots=True)
class UmapGraph:
    memberships: Float64SquareMatrix
    sigmas: Float64PointVector
    rhos: Float64PointVector

    def __post_init__(self) -> None:
        count = self.memberships.shape[0]
        if (
            self.memberships.dtype != t.float64
            or self.memberships.ndim != 2
            or self.memberships.shape != (count, count)
            or count < 3
        ):
            raise TypeError("UMAP memberships must be a float64 square matrix of size at least 3")
        if self.sigmas.shape != (count,) or self.rhos.shape != (count,):
            raise ValueError("UMAP local scales must align to graph vertices")
        if not bool(
            t.isfinite(self.memberships).all().item()
            and t.isfinite(self.sigmas).all().item()
            and t.isfinite(self.rhos).all().item()
        ):
            raise ValueError("UMAP graph values must be finite")
        if bool(
            t.any((self.memberships < 0.0) | (self.memberships > 1.0)).item()
            or t.any(self.sigmas <= 0.0).item()
            or t.any(self.rhos < 0.0).item()
        ):
            raise ValueError("UMAP memberships and local scales escaped their bounds")
        if not t.equal(self.memberships.diagonal(), t.zeros(count, dtype=t.float64)):
            raise ValueError("UMAP graph diagonal must be exactly zero")
        if not t.equal(self.memberships, self.memberships.mT):
            raise ValueError("UMAP fuzzy-union graph must be exactly symmetric")
        if bool(t.any(self.memberships.sum(dim=1) == 0.0).item()):
            raise ValueError("UMAP graph contains an isolated vertex")


@dataclass(frozen=True, slots=True)
class UmapProjection:
    coordinates: Float64Projection
    initial_cross_entropy: float
    final_cross_entropy: float
    curve_a: float
    curve_b: float
    graph_edge_count: int
    spectral_gap: float

    def __post_init__(self) -> None:
        if (
            self.coordinates.dtype != t.float64
            or self.coordinates.ndim != 2
            or self.coordinates.shape[0] < 3
            or self.coordinates.shape[1] != 2
        ):
            raise TypeError("UMAP coordinates must have shape [at least three,2] in float64")
        numeric = (
            self.initial_cross_entropy,
            self.final_cross_entropy,
            self.curve_a,
            self.curve_b,
            self.spectral_gap,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in numeric):
            raise ValueError("UMAP objective, curve parameters, and spectral gap must be positive")
        if self.final_cross_entropy >= self.initial_cross_entropy:
            raise ValueError("UMAP optimization must strictly reduce fuzzy cross entropy")
        if self.graph_edge_count <= 0:
            raise ValueError("UMAP graph must contain at least one undirected edge")
        if not bool(t.isfinite(self.coordinates).all().item()):
            raise ValueError("UMAP coordinates must be finite")
        distances = t.linalg.vector_norm(
            self.coordinates[:, None] - self.coordinates,
            dim=2,
        )
        off_diagonal = ~t.eye(self.coordinates.shape[0], dtype=t.bool)
        if bool(t.any((distances == 0.0) & off_diagonal).item()):
            raise ValueError("UMAP projection contains duplicate coordinates")


@dataclass(frozen=True, slots=True)
class SphericalUmapProjection:
    """A UMAP embedding constrained to the two-sphere in three Cartesian coordinates."""

    coordinates: Float64SphereProjection
    initial_cross_entropy: float
    final_cross_entropy: float
    curve_a: float
    curve_b: float
    graph_edge_count: int
    spectral_gap: float
    geodesic_stress: float
    geodesic_distance_pearson: float
    neighbor_recall: float
    neighbor_count: int
    selected_step: int

    def __post_init__(self) -> None:
        if (
            self.coordinates.dtype != t.float64
            or self.coordinates.ndim != 2
            or self.coordinates.shape[0] < 4
            or self.coordinates.shape[1] != 3
        ):
            raise TypeError(
                "spherical UMAP coordinates must have shape [at least four,3] in float64"
            )
        numeric_positive = (
            self.initial_cross_entropy,
            self.final_cross_entropy,
            self.curve_a,
            self.curve_b,
            self.spectral_gap,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in numeric_positive):
            raise ValueError(
                "spherical UMAP objective, curve, and spectral values must be positive"
            )
        if self.final_cross_entropy >= self.initial_cross_entropy:
            raise ValueError("spherical UMAP optimization must strictly reduce fuzzy cross entropy")
        if (
            self.graph_edge_count <= 0
            or self.neighbor_count <= 0
            or self.neighbor_count >= self.coordinates.shape[0]
            or self.selected_step <= 0
        ):
            raise ValueError("spherical UMAP graph, neighbor, and step counts are inconsistent")
        if not math.isfinite(self.geodesic_stress) or self.geodesic_stress < 0.0:
            raise ValueError("spherical UMAP geodesic stress must be finite and non-negative")
        if not -1.0 <= self.geodesic_distance_pearson <= 1.0:
            raise ValueError("spherical UMAP geodesic-distance Pearson must lie in [-1,1]")
        if not 0.0 <= self.neighbor_recall <= 1.0:
            raise ValueError("spherical UMAP neighbor recall must lie in [0,1]")
        if not bool(t.isfinite(self.coordinates).all().item()):
            raise ValueError("spherical UMAP coordinates must be finite")
        norms = t.linalg.vector_norm(self.coordinates, dim=1)
        if not t.allclose(norms, t.ones_like(norms), atol=2.0e-15, rtol=0.0):
            raise ValueError("spherical UMAP coordinates must have unit-norm rows")
        if t.unique(self.coordinates, dim=0).shape[0] != self.coordinates.shape[0]:
            raise ValueError("spherical UMAP projection contains duplicate coordinates")


@jaxtyped(typechecker=beartype)
def _validate_input(values: t.Tensor, config: UmapConfig) -> Float64FeatureMatrix:
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] == 0:
        raise ValueError("UMAP input must have shape [at least three, non-empty features]")
    if not values.is_floating_point():
        raise TypeError("UMAP input must have a floating dtype")
    if not bool(t.isfinite(values).all().item()):
        raise ValueError("UMAP input must be finite")
    if config.n_neighbors >= values.shape[0]:
        raise ValueError("UMAP n_neighbors must be smaller than the point count")
    prepared = values.to(device="cpu", dtype=t.float64)
    if t.unique(prepared, dim=0).shape[0] != prepared.shape[0]:
        raise ValueError("UMAP input contains duplicate feature rows")
    if config.input_metric is UmapInputMetric.SPHERICAL_GEODESIC:
        norms = t.linalg.vector_norm(prepared, dim=1)
        if not t.allclose(
            norms,
            t.ones_like(norms),
            atol=_SPHERICAL_NORM_ABSOLUTE_TOLERANCE,
            rtol=0.0,
        ):
            maximum_error = float(t.max(t.abs(norms - 1.0)).item())
            raise ValueError(
                "spherical-geodesic UMAP requires unit-norm rows: "
                f"maximum_absolute_error={maximum_error}"
            )
    return prepared


@jaxtyped(typechecker=beartype)
def _pairwise_input_distances(
    prepared: Float64FeatureMatrix,
    *,
    metric: UmapInputMetric,
) -> Float64SquareMatrix:
    if metric is UmapInputMetric.EUCLIDEAN:
        pairwise = t.cdist(prepared, prepared)
    elif metric is UmapInputMetric.SPHERICAL_GEODESIC:
        similarities = prepared @ prepared.mT
        similarities = (similarities + similarities.mT) / 2.0
        pairwise = t.acos(t.clamp(similarities, min=-1.0, max=1.0))
    else:
        raise AssertionError(f"unhandled UMAP input metric: {metric!r}")
    pairwise.fill_diagonal_(0.0)
    if not t.equal(pairwise, pairwise.mT):
        raise RuntimeError("UMAP input-distance matrix is not exactly symmetric")
    if not bool(t.isfinite(pairwise).all().item()) or bool(t.any(pairwise < 0.0).item()):
        raise RuntimeError("UMAP input distances must be finite and non-negative")
    return pairwise


@jaxtyped(typechecker=beartype)
def _smooth_knn_scales(
    neighbor_distances: Float64NeighborMatrix,
    config: UmapConfig,
) -> tuple[Float64PointVector, Float64PointVector]:
    positive = neighbor_distances[:, 1:]
    positive_mask = positive > 0.0
    if bool(t.any(~positive_mask.any(dim=1)).item()):
        raise ValueError("UMAP vertex has no positive-distance neighbor")
    rhos = t.where(positive_mask, positive, t.full_like(positive, float("inf"))).min(dim=1).values
    target = math.log2(config.n_neighbors)
    lower = t.zeros(neighbor_distances.shape[0], dtype=t.float64)
    upper = t.full_like(lower, float("inf"))
    midpoint = t.ones_like(lower)
    active = t.ones_like(lower, dtype=t.bool)
    for _ in range(config.smooth_knn_search_steps):
        adjusted = t.clamp(positive - rhos[:, None], min=0.0)
        membership_sum = t.exp(-adjusted / midpoint[:, None]).sum(dim=1)
        converged = t.abs(membership_sum - target) < _SMOOTH_K_TOLERANCE
        active &= ~converged
        too_large = (membership_sum > target) & active
        too_small = (~too_large) & active
        upper = t.where(too_large, midpoint, upper)
        lower = t.where(too_small, midpoint, lower)
        bounded = (lower + upper) / 2.0
        next_midpoint = t.where(
            too_large,
            bounded,
            t.where(t.isinf(upper), midpoint * 2.0, bounded),
        )
        midpoint = t.where(active, next_midpoint, midpoint)
    row_means = neighbor_distances.mean(dim=1)
    global_mean = neighbor_distances.mean()
    minimum = t.where(rhos > 0.0, row_means, global_mean) * _MIN_K_DISTANCE_SCALE
    sigmas = t.maximum(midpoint, minimum)
    if bool(t.any(sigmas <= 0.0).item()):
        raise RuntimeError("UMAP smooth-kNN search produced a non-positive sigma")
    return sigmas, rhos


@jaxtyped(typechecker=beartype)
def fuzzy_simplicial_graph(values: t.Tensor, *, config: UmapConfig) -> UmapGraph:
    """Build the exact-kNN default fuzzy-union graph used by UMAP."""

    prepared = _validate_input(values, config)
    pairwise = _pairwise_input_distances(prepared, metric=config.input_metric)
    count = prepared.shape[0]
    masked = pairwise.masked_fill(t.eye(count, dtype=t.bool), float("inf"))
    neighbor_indices = t.argsort(masked, dim=1, stable=True)[:, : config.n_neighbors - 1]
    self_indices = t.arange(count, dtype=t.int64)[:, None]
    neighbor_indices = t.cat((self_indices, neighbor_indices), dim=1)
    neighbor_distances = pairwise.gather(1, neighbor_indices)
    if not t.equal(neighbor_distances[:, 0], t.zeros(count, dtype=t.float64)):
        raise RuntimeError("UMAP exact-kNN graph did not place self at distance zero")
    sigmas, rhos = _smooth_knn_scales(neighbor_distances, config)
    adjusted = neighbor_distances - rhos[:, None]
    strengths = t.where(
        adjusted <= 0.0,
        t.ones_like(adjusted),
        t.exp(-adjusted / sigmas[:, None]),
    )
    strengths[:, 0] = 0.0
    directed = t.zeros((count, count), dtype=t.float64)
    directed.scatter_(1, neighbor_indices, strengths)
    memberships = directed + directed.mT - directed * directed.mT
    memberships.fill_diagonal_(0.0)
    return UmapGraph(memberships, sigmas, rhos)


@cache
def fit_default_curve_parameters(spread: float, min_dist: float) -> tuple[float, float]:
    """Fit UMAP's differentiable low-dimensional membership curve with Torch."""

    if not math.isfinite(spread) or spread <= 0.0 or not 0.0 <= min_dist <= spread:
        raise ValueError("UMAP curve fitting requires 0 <= min_dist <= positive spread")
    x = t.linspace(0.0, spread * 3.0, 300, dtype=t.float64)
    target = t.where(
        x < min_dist,
        t.ones_like(x),
        t.exp(-(x - min_dist) / spread),
    )
    log_parameters = t.nn.Parameter(t.zeros(2, dtype=t.float64))
    optimizer = t.optim.LBFGS(
        (log_parameters,),
        lr=0.25,
        max_iter=500,
        tolerance_grad=1.0e-13,
        tolerance_change=1.0e-15,
        line_search_fn="strong_wolfe",
    )

    def closure() -> t.Tensor:
        optimizer.zero_grad(set_to_none=True)
        a, b = t.exp(log_parameters).unbind()
        prediction = 1.0 / (1.0 + a * t.pow(x, 2.0 * b))
        loss = t.mean(t.square(prediction - target))
        loss.backward()
        return loss

    optimizer.step(closure)
    with t.inference_mode():
        a, b = map(float, t.exp(log_parameters).tolist())
    if not all(math.isfinite(value) and value > 0.0 for value in (a, b)):
        raise RuntimeError("UMAP curve fit produced invalid parameters")
    return a, b


@jaxtyped(typechecker=beartype)
def _spectral_initialization(
    graph: Float64SquareMatrix,
    *,
    seed: int,
) -> tuple[Float64Projection, float]:
    degrees = graph.sum(dim=1)
    if bool(t.any(degrees <= 0.0).item()):
        raise ValueError("UMAP spectral initialization requires positive graph degrees")
    inverse_sqrt = t.rsqrt(degrees)
    laplacian = t.eye(graph.shape[0], dtype=t.float64) - (
        inverse_sqrt[:, None] * graph * inverse_sqrt[None, :]
    )
    eigenvalues, eigenvectors = t.linalg.eigh(laplacian)
    zero_count = int((eigenvalues < 1.0e-10).sum().item())
    if zero_count != 1:
        raise ValueError(
            f"UMAP fuzzy graph must be connected; normalized Laplacian has {zero_count} zeros"
        )
    coordinates = eigenvectors[:, 1:3].clone()
    for axis in range(2):
        pivot = int(t.argmax(t.abs(coordinates[:, axis])).item())
        if float(coordinates[pivot, axis].item()) < 0.0:
            coordinates[:, axis] *= -1.0
    maximum = t.abs(coordinates).max()
    if float(maximum.item()) == 0.0:
        raise RuntimeError("UMAP spectral coordinates collapsed")
    coordinates *= 10.0 / maximum
    generator = t.Generator(device="cpu").manual_seed(seed)
    coordinates += 1.0e-4 * t.randn(coordinates.shape, dtype=t.float64, generator=generator)
    minimum = coordinates.min(dim=0).values
    span = coordinates.max(dim=0).values - minimum
    if bool(t.any(span == 0.0).item()):
        raise RuntimeError("UMAP spectral initialization has a collapsed axis")
    coordinates = 10.0 * (coordinates - minimum) / span
    return coordinates, float(eigenvalues[1].item())


@jaxtyped(typechecker=beartype)
def _spherical_spectral_initialization(
    graph: Float64SquareMatrix,
    *,
    seed: int,
) -> tuple[Float64SphereProjection, float]:
    degrees = graph.sum(dim=1)
    if bool(t.any(degrees <= 0.0).item()):
        raise ValueError("spherical UMAP spectral initialization requires positive graph degrees")
    inverse_sqrt = t.rsqrt(degrees)
    laplacian = t.eye(graph.shape[0], dtype=t.float64) - (
        inverse_sqrt[:, None] * graph * inverse_sqrt[None, :]
    )
    eigenvalues, eigenvectors = t.linalg.eigh(laplacian)
    zero_count = int((eigenvalues < 1.0e-10).sum().item())
    if zero_count != 1:
        raise ValueError(
            "spherical UMAP fuzzy graph must be connected; "
            f"normalized Laplacian has {zero_count} zeros"
        )
    coordinates = eigenvectors[:, 1:4].clone()
    if coordinates.shape[1] != 3:
        raise ValueError("spherical UMAP requires at least four graph vertices")
    for axis in range(3):
        pivot = int(t.argmax(t.abs(coordinates[:, axis])).item())
        if float(coordinates[pivot, axis].item()) < 0.0:
            coordinates[:, axis] *= -1.0
    generator = t.Generator(device="cpu").manual_seed(seed)
    coordinates += 1.0e-6 * t.randn(coordinates.shape, dtype=t.float64, generator=generator)
    norms = t.linalg.vector_norm(coordinates, dim=1)
    if bool(t.any(norms == 0.0).item()):
        raise RuntimeError("spherical UMAP spectral initialization contains a zero vector")
    coordinates /= norms[:, None]
    return coordinates, float(eigenvalues[1].item())


@jaxtyped(typechecker=beartype)
def _fuzzy_cross_entropy(
    coordinates: Float64Projection,
    memberships: Float64SquareMatrix,
    *,
    curve_a: float,
    curve_b: float,
) -> t.Tensor:
    row, column = t.triu_indices(coordinates.shape[0], coordinates.shape[0], offset=1)
    squared_distance = t.sum(t.square(coordinates[row] - coordinates[column]), dim=1)
    distance_power = t.pow(t.clamp(squared_distance, min=t.finfo(t.float64).tiny), curve_b)
    odds = curve_a * distance_power
    log_denominator = t.log1p(odds)
    high = memberships[row, column]
    return t.mean(high * log_denominator + (1.0 - high) * (log_denominator - t.log(odds)))


@jaxtyped(typechecker=beartype)
def _spherical_fuzzy_cross_entropy(
    coordinates: Float64SphereProjection,
    memberships: Float64SquareMatrix,
    *,
    curve_a: float,
    curve_b: float,
) -> t.Tensor:
    row, column = t.triu_indices(coordinates.shape[0], coordinates.shape[0], offset=1)
    cosine = t.sum(coordinates[row] * coordinates[column], dim=1)
    angular_distance = t.acos(t.clamp(cosine, min=-1.0 + 1.0e-12, max=1.0 - 1.0e-12))
    distance_power = t.pow(t.square(angular_distance), curve_b)
    odds = curve_a * t.clamp(distance_power, min=t.finfo(t.float64).tiny)
    log_denominator = t.log1p(odds)
    high = memberships[row, column]
    return t.mean(high * log_denominator + (1.0 - high) * (log_denominator - t.log(odds)))


@jaxtyped(typechecker=beartype)
def _spherical_projection_diagnostics(
    input_distances: Float64SquareMatrix,
    output_coordinates: Float64SphereProjection,
    *,
    neighbor_count: int,
) -> tuple[float, float, float]:
    count = input_distances.shape[0]
    if output_coordinates.shape[0] != count or not 0 < neighbor_count < count:
        raise ValueError("spherical UMAP diagnostics require aligned points and neighbor count")
    output_distances = _pairwise_input_distances(
        output_coordinates,
        metric=UmapInputMetric.SPHERICAL_GEODESIC,
    )
    row, column = t.triu_indices(count, count, offset=1)
    target = input_distances[row, column]
    observed = output_distances[row, column]
    denominator = t.sum(t.square(target))
    if float(denominator.item()) == 0.0:
        raise ValueError("spherical UMAP input distances have zero stress denominator")
    stress = float(t.sqrt(t.sum(t.square(observed - target)) / denominator).item())
    target_centered = target - target.mean()
    observed_centered = observed - observed.mean()
    pearson_denominator = t.linalg.vector_norm(target_centered) * t.linalg.vector_norm(
        observed_centered
    )
    if float(pearson_denominator.item()) == 0.0:
        raise ValueError("spherical UMAP distance Pearson is undefined for a constant vector")
    pearson = float(t.dot(target_centered, observed_centered).item() / pearson_denominator.item())
    diagonal_mask = t.eye(count, dtype=t.bool)
    input_neighbors = t.argsort(
        input_distances.masked_fill(diagonal_mask, float("inf")),
        dim=1,
        stable=True,
    )[:, :neighbor_count]
    output_neighbors = t.argsort(
        output_distances.masked_fill(diagonal_mask, float("inf")),
        dim=1,
        stable=True,
    )[:, :neighbor_count]
    overlap = (input_neighbors[:, :, None] == output_neighbors[:, None, :]).any(dim=2).sum()
    recall = float(overlap.item() / (count * neighbor_count))
    return stress, pearson, recall


@jaxtyped(typechecker=beartype)
def exact_umap(values: t.Tensor, *, config: UmapConfig) -> UmapProjection:
    """Optimize the complete UMAP fuzzy-set cross entropy without negative sampling."""

    prepared = _validate_input(values, config)
    graph = fuzzy_simplicial_graph(prepared, config=config)
    curve_a, curve_b = fit_default_curve_parameters(config.spread, config.min_dist)
    initial, spectral_gap = _spectral_initialization(graph.memberships, seed=config.seed)
    initial_cross_entropy = float(
        _fuzzy_cross_entropy(
            initial,
            graph.memberships,
            curve_a=curve_a,
            curve_b=curve_b,
        ).item()
    )
    coordinates = t.nn.Parameter(initial)
    optimizer = t.optim.Adam((coordinates,), lr=config.learning_rate)
    for step in range(config.optimization_steps):
        loss = _fuzzy_cross_entropy(
            coordinates,
            graph.memberships,
            curve_a=curve_a,
            curve_b=curve_b,
        )
        if not bool(t.isfinite(loss).item()):
            raise RuntimeError(f"UMAP loss became non-finite at optimization step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with t.no_grad():
            coordinates -= coordinates.mean(dim=0, keepdim=True)
            if not bool(t.isfinite(coordinates).all().item()):
                raise RuntimeError(f"UMAP coordinates became non-finite at step {step}")
    with t.inference_mode():
        final = coordinates.detach().clone()
        final_cross_entropy = float(
            _fuzzy_cross_entropy(
                final,
                graph.memberships,
                curve_a=curve_a,
                curve_b=curve_b,
            ).item()
        )
    edge_count = int(t.count_nonzero(t.triu(graph.memberships, diagonal=1)).item())
    return UmapProjection(
        coordinates=final,
        initial_cross_entropy=initial_cross_entropy,
        final_cross_entropy=final_cross_entropy,
        curve_a=curve_a,
        curve_b=curve_b,
        graph_edge_count=edge_count,
        spectral_gap=spectral_gap,
    )


@jaxtyped(typechecker=beartype)
def exact_spherical_umap(values: t.Tensor, *, config: UmapConfig) -> SphericalUmapProjection:
    """Optimize exact UMAP cross entropy with output points constrained to the unit two-sphere."""

    if config.input_metric is not UmapInputMetric.SPHERICAL_GEODESIC:
        raise ValueError("spherical-output UMAP requires spherical-geodesic input distances")
    prepared = _validate_input(values, config)
    input_distances = _pairwise_input_distances(
        prepared,
        metric=UmapInputMetric.SPHERICAL_GEODESIC,
    )
    graph = fuzzy_simplicial_graph(prepared, config=config)
    curve_a, curve_b = fit_default_curve_parameters(config.spread, config.min_dist)
    initial, spectral_gap = _spherical_spectral_initialization(
        graph.memberships,
        seed=config.seed,
    )
    initial_cross_entropy = float(
        _spherical_fuzzy_cross_entropy(
            initial,
            graph.memberships,
            curve_a=curve_a,
            curve_b=curve_b,
        ).item()
    )
    coordinates = t.nn.Parameter(initial)
    optimizer = t.optim.Adam((coordinates,), lr=config.learning_rate)
    best_cross_entropy = initial_cross_entropy
    best_coordinates = initial.clone()
    selected_step = 0
    for step in range(config.optimization_steps):
        loss = _spherical_fuzzy_cross_entropy(
            coordinates,
            graph.memberships,
            curve_a=curve_a,
            curve_b=curve_b,
        )
        if not bool(t.isfinite(loss).item()):
            raise RuntimeError(f"spherical UMAP loss became non-finite at optimization step {step}")
        loss_value = float(loss.item())
        if loss_value < best_cross_entropy:
            best_cross_entropy = loss_value
            best_coordinates = coordinates.detach().clone()
            selected_step = step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with t.no_grad():
            coordinates /= t.linalg.vector_norm(coordinates, dim=1, keepdim=True)
            if not bool(t.isfinite(coordinates).all().item()):
                raise RuntimeError(f"spherical UMAP coordinates became non-finite at step {step}")
    with t.inference_mode():
        terminal_cross_entropy = float(
            _spherical_fuzzy_cross_entropy(
                coordinates,
                graph.memberships,
                curve_a=curve_a,
                curve_b=curve_b,
            ).item()
        )
        if terminal_cross_entropy < best_cross_entropy:
            best_cross_entropy = terminal_cross_entropy
            best_coordinates = coordinates.detach().clone()
            selected_step = config.optimization_steps
        if selected_step == 0:
            raise RuntimeError("spherical UMAP optimization did not improve its initialization")
        final = best_coordinates
        final_cross_entropy = best_cross_entropy
        neighbor_count = config.n_neighbors - 1
        stress, pearson, recall = _spherical_projection_diagnostics(
            input_distances,
            final,
            neighbor_count=neighbor_count,
        )
    edge_count = int(t.count_nonzero(t.triu(graph.memberships, diagonal=1)).item())
    return SphericalUmapProjection(
        coordinates=final,
        initial_cross_entropy=initial_cross_entropy,
        final_cross_entropy=final_cross_entropy,
        curve_a=curve_a,
        curve_b=curve_b,
        graph_edge_count=edge_count,
        spectral_gap=spectral_gap,
        geodesic_stress=stress,
        geodesic_distance_pearson=pearson,
        neighbor_recall=recall,
        neighbor_count=neighbor_count,
        selected_step=selected_step,
    )
