"""Linear sequence metric and its explicitly separated objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch as t
from beartype import beartype
from jaxtyping import Float, jaxtyped

from methylation_latent.domain import LatentDimension, NonNegativeWeight, PositiveInt
from methylation_latent.targets import CorrelationMatrix, CorrelationVector

EmbeddingBatch = Float[t.Tensor, "batch embedding"]
LatentBatch = Float[t.Tensor, "batch latent"]
PredictionVector = Float[t.Tensor, "batch"]


def _require_floating_finite(tensor: t.Tensor, name: str) -> None:
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating dtype, got {tensor.dtype}")
    if not bool(t.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must contain only finite values")


@jaxtyped(typechecker=beartype)
def normalize_rows_strict(values: LatentBatch) -> LatentBatch:
    """Normalize rows without converting a zero vector into plausible output."""

    if values.ndim != 2 or 0 in values.shape:
        raise ValueError("latent values must be a non-empty rank-2 tensor")
    _require_floating_finite(values, "latent values")
    norms = t.linalg.vector_norm(values, dim=1, keepdim=True)
    if bool(t.any(norms == 0).item()):
        indices = t.nonzero(norms.squeeze(1) == 0).flatten().tolist()
        raise ValueError(f"zero latent rows cannot define cosine similarity: {indices[:10]}")
    return values / norms


@jaxtyped(typechecker=beartype)
def normalize_vector_strict(values: PredictionVector) -> PredictionVector:
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("latent direction must be a non-empty vector")
    _require_floating_finite(values, "latent direction")
    norm = t.linalg.vector_norm(values)
    if float(norm.item()) == 0.0:
        raise ValueError("zero latent direction cannot define cosine similarity")
    return values / norm


class LatentMetric(t.nn.Module):
    """Bias-free linear Mahalanobis metric with an independent age direction."""

    def __init__(
        self,
        *,
        embedding_dimension: PositiveInt | int,
        latent_dimension: LatentDimension,
    ) -> None:
        super().__init__()
        embedding_width = int(embedding_dimension)
        latent_width = int(latent_dimension)
        if embedding_width <= 0:
            raise ValueError("embedding dimension must be positive")
        if latent_width > embedding_width:
            raise ValueError("latent dimension cannot exceed embedding dimension")
        self.embedding_dimension = embedding_width
        self.latent_dimension = latent_width
        self.projection = t.nn.Linear(embedding_width, latent_width, bias=False)
        t.nn.init.orthogonal_(self.projection.weight)
        self.age_direction = t.nn.Parameter(t.empty(latent_width))
        t.nn.init.normal_(self.age_direction)

    @jaxtyped(typechecker=beartype)
    def latent(self, embeddings: EmbeddingBatch) -> LatentBatch:
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError("embedding batch must be a non-empty rank-2 tensor")
        if embeddings.shape[1] != self.embedding_dimension:
            raise ValueError(
                f"expected embedding width {self.embedding_dimension}, got {embeddings.shape[1]}"
            )
        _require_floating_finite(embeddings, "embeddings")
        if embeddings.dtype != self.projection.weight.dtype:
            raise TypeError(
                f"embedding dtype {embeddings.dtype} does not match model dtype "
                f"{self.projection.weight.dtype}"
            )
        return normalize_rows_strict(self.projection(embeddings))

    @jaxtyped(typechecker=beartype)
    def predict(self, embeddings: EmbeddingBatch) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
        latent = self.latent(embeddings)
        pair_prediction = self.predict_pairs_from_latent(latent)
        age_prediction = self.predict_age_from_latent(latent)
        return latent, pair_prediction, age_prediction

    @jaxtyped(typechecker=beartype)
    def predict_pairs_from_latent(self, latent: LatentBatch) -> t.Tensor:
        normalized = normalize_rows_strict(latent)
        return normalized @ normalized.mT

    @jaxtyped(typechecker=beartype)
    def predict_age_from_latent(self, latent: LatentBatch) -> t.Tensor:
        normalized = normalize_rows_strict(latent)
        return normalized @ normalize_vector_strict(self.age_direction)


@dataclass(frozen=True, slots=True)
class PairObjectiveInput:
    """Pair prediction and target that are either both present or both absent."""

    prediction: t.Tensor
    target: CorrelationMatrix

    def __post_init__(self) -> None:
        if self.prediction.shape != self.target.tensor.shape:
            raise ValueError("pair prediction and target shapes differ")
        _require_floating_finite(self.prediction, "pair predictions")
        if self.prediction.dtype != self.target.tensor.dtype:
            raise TypeError("pair prediction and target dtypes differ")


@dataclass(frozen=True, slots=True)
class ObjectiveTerms:
    """Differentiable scalar terms kept separate for truthful logging."""

    total: t.Tensor
    pair_mse: t.Tensor | None
    age_mse: t.Tensor

    def __post_init__(self) -> None:
        for name, value in (("total", self.total), ("age_mse", self.age_mse)):
            if value.ndim != 0 or not value.is_floating_point():
                raise ValueError(f"{name} must be a floating scalar tensor")
        if self.pair_mse is not None and (
            self.pair_mse.ndim != 0 or not self.pair_mse.is_floating_point()
        ):
            raise ValueError("pair_mse must be a floating scalar tensor when present")


@jaxtyped(typechecker=beartype)
def metric_objective(
    age_prediction: PredictionVector,
    age_target: CorrelationVector,
    *,
    lambda_age: NonNegativeWeight,
    pair: PairObjectiveInput | None,
) -> ObjectiveTerms:
    """Compute age-only or full loss; age is never inserted as a Gram row."""

    if age_prediction.shape != age_target.tensor.shape:
        raise ValueError("age prediction and target shapes differ")
    _require_floating_finite(age_prediction, "age predictions")
    if age_prediction.dtype != age_target.tensor.dtype:
        raise TypeError("age prediction and target dtypes differ")
    age_mse = t.mean(t.square(age_prediction - age_target.tensor))
    if pair is None:
        return ObjectiveTerms(total=float(lambda_age) * age_mse, pair_mse=None, age_mse=age_mse)
    pair_mse = t.mean(t.square(pair.prediction - pair.target.tensor))
    return ObjectiveTerms(
        total=pair_mse + float(lambda_age) * age_mse,
        pair_mse=pair_mse,
        age_mse=age_mse,
    )


@beartype
def make_optimizer(model: LatentMetric, *, learning_rate: float) -> t.optim.Optimizer:
    """Construct the preregistered zero-weight-decay optimizer."""

    if not t.isfinite(t.tensor(learning_rate)).item() or learning_rate <= 0.0:
        raise ValueError("learning rate must be finite and positive")
    optimizer = t.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)
    if any(float(group["weight_decay"]) != 0.0 for group in optimizer.param_groups):
        raise RuntimeError("latent metric optimizer must not use weight decay")
    return optimizer
