"""Small deterministic exact t-SNE implemented only with Torch."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch as t
from beartype import beartype
from jaxtyping import Float64, jaxtyped

Float64FeatureMatrix = Float64[t.Tensor, "points features"]
ProbabilityMatrix = Float64[t.Tensor, "points points"]
ProjectionMatrix = Float64[t.Tensor, "points 2"]


@beartype
def deterministic_balanced_indices(
    group_labels: t.Tensor,
    *,
    group_count: int,
    maximum_points: int,
    seed: int,
) -> t.Tensor:
    """Sample the same positive count per metadata group without target access."""

    if group_labels.dtype != t.int64 or group_labels.ndim != 1 or group_labels.numel() == 0:
        raise TypeError("t-SNE group labels must be a non-empty int64 vector")
    if group_count <= 0 or maximum_points <= 0:
        raise ValueError("t-SNE group and display counts must be positive")
    if maximum_points > group_labels.numel():
        raise ValueError("t-SNE display count cannot exceed its source population")
    if maximum_points % group_count != 0:
        raise ValueError("t-SNE display count must divide evenly across metadata groups")
    if bool(t.any((group_labels < 0) | (group_labels >= group_count)).item()):
        raise ValueError("t-SNE group label is outside the declared group range")
    per_group = maximum_points // group_count
    generator = t.Generator(device="cpu").manual_seed(seed)
    selected: list[t.Tensor] = []
    for group in range(group_count):
        candidates = t.nonzero(group_labels == group).flatten()
        if candidates.numel() < per_group:
            raise ValueError(
                f"t-SNE group {group} has {candidates.numel()} points; need {per_group}"
            )
        order = t.randperm(candidates.numel(), generator=generator)[:per_group]
        selected.append(candidates.index_select(0, order))
    result = t.sort(t.cat(selected)).values
    if result.numel() != maximum_points or t.unique(result).numel() != maximum_points:
        raise RuntimeError("t-SNE balanced sampling did not produce the requested unique count")
    return result


@beartype
@dataclass(frozen=True, slots=True)
class TsneConfig:
    perplexity: float
    probability_search_steps: int
    optimization_steps: int
    early_exaggeration_steps: int
    early_exaggeration: float
    learning_rate: float
    seed: int

    def __post_init__(self) -> None:
        numeric = (self.perplexity, self.early_exaggeration, self.learning_rate)
        if any(not math.isfinite(value) or value <= 0.0 for value in numeric):
            raise ValueError("t-SNE perplexity, exaggeration, and learning rate must be positive")
        if (
            self.probability_search_steps <= 0
            or self.optimization_steps <= 0
            or self.early_exaggeration_steps < 0
            or self.early_exaggeration_steps > self.optimization_steps
        ):
            raise ValueError("t-SNE step counts are inconsistent")


@dataclass(frozen=True, slots=True)
class TsneProjection:
    coordinates: ProjectionMatrix
    final_kl_divergence: float

    def __post_init__(self) -> None:
        if (
            self.coordinates.dtype != t.float64
            or self.coordinates.ndim != 2
            or self.coordinates.shape[0] < 2
            or self.coordinates.shape[1] != 2
        ):
            raise TypeError("t-SNE coordinates must have shape [at least two, 2] in float64")
        if not bool(t.isfinite(self.coordinates).all().item()):
            raise ValueError("t-SNE coordinates must be finite")
        if not math.isfinite(self.final_kl_divergence) or self.final_kl_divergence < 0.0:
            raise ValueError("t-SNE KL divergence must be finite and non-negative")


@jaxtyped(typechecker=beartype)
def _validate_input(values: t.Tensor, config: TsneConfig) -> Float64FeatureMatrix:
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] == 0:
        raise ValueError("t-SNE input must have shape [at least two, non-empty features]")
    if not values.is_floating_point():
        raise TypeError("t-SNE input must have a floating dtype")
    if not bool(t.isfinite(values).all().item()):
        raise ValueError("t-SNE input must be finite")
    if config.perplexity >= values.shape[0]:
        raise ValueError("t-SNE perplexity must be smaller than the point count")
    return values.to(device="cpu", dtype=t.float64)


@jaxtyped(typechecker=beartype)
def _conditional_probabilities(
    values: Float64FeatureMatrix,
    config: TsneConfig,
) -> ProbabilityMatrix:
    squared_distances = t.square(t.cdist(values, values))
    count = values.shape[0]
    diagonal = t.eye(count, dtype=t.bool)
    target_entropy = math.log(config.perplexity)
    beta = t.ones(count, dtype=t.float64)
    lower = t.zeros(count, dtype=t.float64)
    upper = t.full((count,), float("inf"), dtype=t.float64)
    probabilities = t.empty_like(squared_distances)
    for _ in range(config.probability_search_steps):
        logits = -squared_distances * beta[:, None]
        logits.masked_fill_(diagonal, float("-inf"))
        probabilities = t.softmax(logits, dim=1)
        entropy = t.logsumexp(logits, dim=1) + beta * t.sum(
            probabilities * squared_distances,
            dim=1,
        )
        too_diffuse = entropy > target_entropy
        lower = t.where(too_diffuse, beta, lower)
        upper = t.where(too_diffuse, upper, beta)
        doubled = beta * 2.0
        halved = beta / 2.0
        bounded_midpoint = (lower + upper) / 2.0
        beta = t.where(
            too_diffuse,
            t.where(t.isinf(upper), doubled, bounded_midpoint),
            t.where(lower == 0.0, halved, bounded_midpoint),
        )
    if not bool(t.isfinite(probabilities).all().item()):
        raise RuntimeError("t-SNE conditional probabilities are non-finite")
    return probabilities


@jaxtyped(typechecker=beartype)
def _joint_probabilities(
    values: Float64FeatureMatrix,
    config: TsneConfig,
) -> ProbabilityMatrix:
    conditional = _conditional_probabilities(values, config)
    joint = (conditional + conditional.mT) / (2.0 * values.shape[0])
    joint.fill_diagonal_(0.0)
    if not bool(
        t.isclose(
            joint.sum(),
            t.tensor(1.0, dtype=t.float64),
            atol=1.0e-12,
            rtol=0.0,
        ).item()
    ):
        raise RuntimeError("t-SNE joint probabilities do not sum to one")
    return joint


@jaxtyped(typechecker=beartype)
def _low_dimensional_probabilities(
    coordinates: ProjectionMatrix,
) -> ProbabilityMatrix:
    numerator = 1.0 / (1.0 + t.square(t.cdist(coordinates, coordinates)))
    diagonal = t.eye(coordinates.shape[0], dtype=t.bool)
    numerator = numerator.masked_fill(diagonal, 0.0)
    denominator = numerator.sum()
    if not bool(t.isfinite(denominator).item()) or float(denominator.item()) <= 0.0:
        raise RuntimeError("t-SNE low-dimensional normalization is invalid")
    return numerator / denominator


@jaxtyped(typechecker=beartype)
def _kl_divergence(
    joint: ProbabilityMatrix,
    low_dimensional: ProbabilityMatrix,
) -> t.Tensor:
    positive = joint > 0.0
    return t.sum(joint[positive] * (t.log(joint[positive]) - t.log(low_dimensional[positive])))


@jaxtyped(typechecker=beartype)
def exact_tsne(values: t.Tensor, *, config: TsneConfig) -> TsneProjection:
    """Fit exact t-SNE with a target-independent deterministic initialization."""

    prepared = _validate_input(values, config)
    joint = _joint_probabilities(prepared, config)
    generator = t.Generator(device="cpu").manual_seed(config.seed)
    coordinates = t.nn.Parameter(
        1.0e-4 * t.randn((prepared.shape[0], 2), generator=generator, dtype=t.float64)
    )
    optimizer = t.optim.Adam((coordinates,), lr=config.learning_rate)
    for step in range(config.optimization_steps):
        low_dimensional = _low_dimensional_probabilities(coordinates)
        weight = (
            joint * config.early_exaggeration if step < config.early_exaggeration_steps else joint
        )
        positive = weight > 0.0
        loss = t.sum(weight[positive] * (t.log(joint[positive]) - t.log(low_dimensional[positive])))
        if not bool(t.isfinite(loss).item()):
            raise RuntimeError(f"t-SNE loss became non-finite at optimization step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with t.no_grad():
            coordinates -= coordinates.mean(dim=0, keepdim=True)
            if not bool(t.isfinite(coordinates).all().item()):
                raise RuntimeError(
                    f"t-SNE coordinates became non-finite at optimization step {step}"
                )
    with t.inference_mode():
        final_coordinates = coordinates.detach().clone()
        final_kl = float(
            _kl_divergence(
                joint,
                _low_dimensional_probabilities(final_coordinates),
            ).item()
        )
    return TsneProjection(final_coordinates, final_kl)
