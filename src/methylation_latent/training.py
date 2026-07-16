"""Fixed-step training that consumes only cached embeddings and target geometry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch as t
from beartype import beartype

from methylation_latent.batching import GenomicBatchSampler
from methylation_latent.domain import (
    LatentDimension,
    NonEmptyProbeSet,
    NonNegativeWeight,
    PositiveInt,
)
from methylation_latent.model import (
    LatentMetric,
    PairObjectiveInput,
    make_optimizer,
    metric_objective,
)
from methylation_latent.storage import EmbeddingMatrix
from methylation_latent.targets import (
    CorrelationMatrix,
    CorrelationVector,
    TargetGeometry,
    validate_latent_dimension,
)


class TrainingMode(StrEnum):
    AGE_ONLY = "age_only"
    FULL = "full"


@beartype
@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Frozen optimization choices for one age-only or full run."""

    mode: TrainingMode
    latent_dimension: LatentDimension
    lambda_age: NonNegativeWeight
    batch_size: PositiveInt
    neighbourhood_width: PositiveInt
    steps: PositiveInt
    learning_rate: float
    seed: int
    device: str

    def __post_init__(self) -> None:
        learning_rate_tensor = t.tensor(self.learning_rate)
        if not bool(t.isfinite(learning_rate_tensor).item()) or self.learning_rate <= 0.0:
            raise ValueError("training learning rate must be finite and positive")
        if self.mode == TrainingMode.AGE_ONLY and float(self.lambda_age) == 0.0:
            raise ValueError("age-only training requires a positive age-loss weight")


@dataclass(frozen=True, slots=True)
class TrainingStep:
    step: int
    total_loss: float
    age_mse: float
    pair_mse: float | None

    def __post_init__(self) -> None:
        values = (self.total_loss, self.age_mse)
        if any(not bool(t.isfinite(t.tensor(value)).item()) or value < 0.0 for value in values):
            raise ValueError("training losses must be finite and non-negative")
        if self.pair_mse is not None and (
            not bool(t.isfinite(t.tensor(self.pair_mse)).item()) or self.pair_mse < 0.0
        ):
            raise ValueError("pair loss must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class TrainedMetric:
    model: LatentMetric
    history: tuple[TrainingStep, ...]

    def __post_init__(self) -> None:
        if not self.history:
            raise ValueError("trained metric must retain a non-empty loss history")


def _cuda_fork_devices(device: t.device) -> list[int]:
    if device.type != "cuda":
        return []
    return [device.index if device.index is not None else t.cuda.current_device()]


@beartype
def train_latent_metric(
    probes: NonEmptyProbeSet,
    embeddings: EmbeddingMatrix,
    targets: TargetGeometry,
    *,
    config: TrainingConfig,
) -> TrainedMetric:
    """Train on an already subsetted/aligned training universe with no re-embedding path."""

    if len(probes) != embeddings.tensor.shape[0] or len(probes) != targets.methylation.n_rows:
        raise ValueError("training probe, embedding, and target axes must align exactly")
    validate_latent_dimension(
        config.latent_dimension,
        embedding_dimension=embeddings.tensor.shape[1],
        n_samples=targets.methylation.n_samples,
    )
    if int(config.batch_size) > len(probes):
        raise ValueError("training batch size cannot exceed the aligned training universe")
    device = t.device(config.device)
    if device.type == "cuda" and not t.cuda.is_available():
        raise ValueError("CUDA training was requested but CUDA is unavailable")
    sampler = GenomicBatchSampler(
        probes=probes,
        batch_size=config.batch_size,
        neighbourhood_width=config.neighbourhood_width,
        seed=config.seed,
    )
    embedding_values = embeddings.training_tensor(device=device)
    target_rows = targets.methylation.tensor.to(device=device, dtype=t.float32)
    target_rho = targets.rho.tensor.to(device=device, dtype=t.float32)

    with t.random.fork_rng(devices=_cuda_fork_devices(device)):
        t.manual_seed(config.seed)
        model = LatentMetric(
            embedding_dimension=embedding_values.shape[1],
            latent_dimension=config.latent_dimension,
        ).to(device)
        optimizer = make_optimizer(model, learning_rate=config.learning_rate)
        history: list[TrainingStep] = []
        for step in range(int(config.steps)):
            batch = sampler.sample()
            batch_indices = batch.indices.to(device)
            batch_embeddings = embedding_values.index_select(0, batch_indices)
            batch_rho = CorrelationVector(target_rho.index_select(0, batch_indices))
            latent = model.latent(batch_embeddings)
            age_prediction = model.predict_age_from_latent(latent)
            pair: PairObjectiveInput | None = None
            if config.mode == TrainingMode.FULL:
                selected_rows = target_rows.index_select(0, batch_indices)
                pair_target = CorrelationMatrix(selected_rows @ selected_rows.mT)
                pair = PairObjectiveInput(
                    prediction=model.predict_pairs_from_latent(latent),
                    target=pair_target,
                )
            terms = metric_objective(
                age_prediction,
                batch_rho,
                lambda_age=config.lambda_age,
                pair=pair,
            )
            optimizer.zero_grad(set_to_none=True)
            terms.total.backward()
            optimizer.step()
            history.append(
                TrainingStep(
                    step=step,
                    total_loss=float(terms.total.detach().cpu().item()),
                    age_mse=float(terms.age_mse.detach().cpu().item()),
                    pair_mse=(
                        None
                        if terms.pair_mse is None
                        else float(terms.pair_mse.detach().cpu().item())
                    ),
                )
            )
    return TrainedMetric(model=model, history=tuple(history))
