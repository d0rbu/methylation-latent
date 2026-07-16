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
    validation_interval: PositiveInt
    validation_pair_chunk_size: PositiveInt
    learning_rate: float
    seed: int
    device: str

    def __post_init__(self) -> None:
        learning_rate_tensor = t.tensor(self.learning_rate)
        if not bool(t.isfinite(learning_rate_tensor).item()) or self.learning_rate <= 0.0:
            raise ValueError("training learning rate must be finite and positive")
        if self.mode == TrainingMode.AGE_ONLY and float(self.lambda_age) == 0.0:
            raise ValueError("age-only training requires a positive age-loss weight")
        if int(self.validation_interval) > int(self.steps):
            raise ValueError("validation interval cannot exceed the training step budget")


@dataclass(frozen=True, slots=True)
class TrainingStep:
    step: int
    total_loss: float
    age_mse: float
    pair_mse: float | None

    def __post_init__(self) -> None:
        if self.step <= 0:
            raise ValueError("training step numbers must be positive")
        values = (self.total_loss, self.age_mse)
        if any(not bool(t.isfinite(t.tensor(value)).item()) or value < 0.0 for value in values):
            raise ValueError("training losses must be finite and non-negative")
        if self.pair_mse is not None and (
            not bool(t.isfinite(t.tensor(self.pair_mse)).item()) or self.pair_mse < 0.0
        ):
            raise ValueError("pair loss must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class ValidationData:
    """Aligned, target-sealed validation inputs kept separate from optimization."""

    probes: NonEmptyProbeSet
    embeddings: EmbeddingMatrix
    targets: TargetGeometry

    def __post_init__(self) -> None:
        if (
            len(self.probes) != self.embeddings.tensor.shape[0]
            or len(self.probes) != self.targets.methylation.n_rows
        ):
            raise ValueError("validation probe, embedding, and target axes must align exactly")


@dataclass(frozen=True, slots=True)
class ValidationStep:
    """Exact full-validation losses and the lambda-independent selection score."""

    step: int
    training_objective: float
    selection_score: float
    age_mse: float
    pair_mse: float | None

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("validation step cannot be negative")
        values = (self.training_objective, self.selection_score, self.age_mse)
        if any(not bool(t.isfinite(t.tensor(value)).item()) or value < 0.0 for value in values):
            raise ValueError("validation losses must be finite and non-negative")
        if self.pair_mse is not None and (
            not bool(t.isfinite(t.tensor(self.pair_mse)).item()) or self.pair_mse < 0.0
        ):
            raise ValueError("validation pair loss must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class TrainedMetric:
    model: LatentMetric
    history: tuple[TrainingStep, ...]
    validation_history: tuple[ValidationStep, ...]
    selected_step: int

    def __post_init__(self) -> None:
        if not self.history or not self.validation_history:
            raise ValueError("trained metric must retain non-empty train and validation histories")
        validation_steps = tuple(record.step for record in self.validation_history)
        if tuple(sorted(set(validation_steps))) != validation_steps:
            raise ValueError("validation checkpoints must be unique and increasing")
        if self.selected_step not in validation_steps:
            raise ValueError("selected checkpoint is absent from validation history")


def _cuda_fork_devices(device: t.device) -> list[int]:
    if device.type != "cuda":
        return []
    return [device.index if device.index is not None else t.cuda.current_device()]


def _validate_training_and_validation_axes(
    probes: NonEmptyProbeSet,
    embeddings: EmbeddingMatrix,
    targets: TargetGeometry,
    validation: ValidationData,
) -> None:
    if len(probes) != embeddings.tensor.shape[0] or len(probes) != targets.methylation.n_rows:
        raise ValueError("training probe, embedding, and target axes must align exactly")
    train_ids = {probe.probe_id for probe in probes.probes}
    validation_ids = {probe.probe_id for probe in validation.probes.probes}
    if train_ids & validation_ids:
        raise ValueError("training and validation probe IDs must be disjoint")
    if embeddings.tensor.shape[1] != validation.embeddings.tensor.shape[1]:
        raise ValueError("training and validation embedding widths differ")
    if targets.methylation.n_samples != validation.targets.methylation.n_samples:
        raise ValueError("training and validation target sample axes differ")


def _exact_validation(
    model: LatentMetric,
    validation: ValidationData,
    *,
    mode: TrainingMode,
    lambda_age: NonNegativeWeight,
    step: int,
    pair_chunk_size: int,
    device: t.device,
) -> ValidationStep:
    model.eval()
    with t.inference_mode():
        embeddings = validation.embeddings.training_tensor(device=device)
        target_rows = validation.targets.methylation.tensor.to(
            device=device,
            dtype=t.float32,
        )
        target_rho = validation.targets.rho.tensor.to(
            device=device,
            dtype=t.float32,
        )
        latent = model.latent(embeddings)
        age_prediction = model.predict_age_from_latent(latent)
        age_mse_tensor = t.mean(t.square(age_prediction - target_rho))
        pair_mse_tensor: t.Tensor | None = None
        if mode == TrainingMode.FULL:
            squared_error_sum = t.zeros((), dtype=t.float64, device=device)
            for start in range(0, latent.shape[0], pair_chunk_size):
                stop = min(start + pair_chunk_size, latent.shape[0])
                prediction = latent[start:stop] @ latent.mT
                target = target_rows[start:stop] @ target_rows.mT
                squared_error_sum += t.square(prediction - target).sum(dtype=t.float64)
            pair_mse_tensor = (squared_error_sum / (latent.shape[0] * latent.shape[0])).to(
                t.float32
            )
        age_mse = float(age_mse_tensor.cpu().item())
        pair_mse = None if pair_mse_tensor is None else float(pair_mse_tensor.cpu().item())
        training_objective = (
            float(lambda_age) * age_mse
            if pair_mse is None
            else pair_mse + float(lambda_age) * age_mse
        )
        selection_score = age_mse if pair_mse is None else pair_mse + age_mse
    model.train()
    return ValidationStep(
        step=step,
        training_objective=training_objective,
        selection_score=selection_score,
        age_mse=age_mse,
        pair_mse=pair_mse,
    )


@beartype
def train_latent_metric(
    probes: NonEmptyProbeSet,
    embeddings: EmbeddingMatrix,
    targets: TargetGeometry,
    validation: ValidationData,
    *,
    config: TrainingConfig,
) -> TrainedMetric:
    """Train on an already subsetted/aligned training universe with no re-embedding path."""

    _validate_training_and_validation_axes(probes, embeddings, targets, validation)
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
        validation_history = [
            _exact_validation(
                model,
                validation,
                mode=config.mode,
                lambda_age=config.lambda_age,
                step=0,
                pair_chunk_size=int(config.validation_pair_chunk_size),
                device=device,
            )
        ]
        best_record = validation_history[0]
        best_state = {
            name: value.detach().cpu().clone() for name, value in model.state_dict().items()
        }
        for step_index in range(int(config.steps)):
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
                    step=step_index + 1,
                    total_loss=float(terms.total.detach().cpu().item()),
                    age_mse=float(terms.age_mse.detach().cpu().item()),
                    pair_mse=(
                        None
                        if terms.pair_mse is None
                        else float(terms.pair_mse.detach().cpu().item())
                    ),
                )
            )
            completed_steps = step_index + 1
            should_validate = completed_steps % int(
                config.validation_interval
            ) == 0 or completed_steps == int(config.steps)
            if should_validate:
                record = _exact_validation(
                    model,
                    validation,
                    mode=config.mode,
                    lambda_age=config.lambda_age,
                    step=completed_steps,
                    pair_chunk_size=int(config.validation_pair_chunk_size),
                    device=device,
                )
                validation_history.append(record)
                if record.selection_score < best_record.selection_score:
                    best_record = record
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
        model.load_state_dict(best_state, strict=True)
    return TrainedMetric(
        model=model,
        history=tuple(history),
        validation_history=tuple(validation_history),
        selected_step=best_record.step,
    )
