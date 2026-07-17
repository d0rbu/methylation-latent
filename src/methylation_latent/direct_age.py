"""Direct scalar tanh baseline for locus-level age correlation."""

from __future__ import annotations

from dataclasses import dataclass

import torch as t
from beartype import beartype
from jaxtyping import Float, jaxtyped

from methylation_latent.batching import GenomicBatchSampler
from methylation_latent.domain import NonEmptyProbeSet, PositiveInt
from methylation_latent.storage import EmbeddingMatrix
from methylation_latent.targets import CorrelationVector

EmbeddingBatch = Float[t.Tensor, "batch embedding"]
PredictionVector = Float[t.Tensor, "batch"]


def _require_floating_finite(values: t.Tensor, name: str) -> None:
    if not values.is_floating_point():
        raise TypeError(f"{name} must have a floating dtype")
    if not bool(t.isfinite(values).all().item()):
        raise ValueError(f"{name} must be finite")


class DirectTanhAgeRegressor(t.nn.Module):
    """One affine scalar followed by tanh; no latent dimension or age-loss weight."""

    def __init__(self, embedding_dimension: PositiveInt | int) -> None:
        super().__init__()
        width = int(embedding_dimension)
        if width <= 0:
            raise ValueError("embedding dimension must be positive")
        self.embedding_dimension = width
        self.linear = t.nn.Linear(width, 1, bias=True)
        t.nn.init.zeros_(self.linear.weight)
        t.nn.init.zeros_(self.linear.bias)

    @jaxtyped(typechecker=beartype)
    def forward(self, embeddings: EmbeddingBatch) -> PredictionVector:
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError("embedding batch must be a non-empty rank-2 tensor")
        if embeddings.shape[1] != self.embedding_dimension:
            raise ValueError(
                f"expected embedding width {self.embedding_dimension}, got {embeddings.shape[1]}"
            )
        _require_floating_finite(embeddings, "embeddings")
        if embeddings.dtype != self.linear.weight.dtype:
            raise TypeError("embedding and direct-age model dtypes differ")
        prediction = t.tanh(self.linear(embeddings).squeeze(1))
        if prediction.shape != (embeddings.shape[0],):
            raise RuntimeError("direct-age prediction shape differs from its batch axis")
        return prediction


@jaxtyped(typechecker=beartype)
def direct_age_mse(
    prediction: PredictionVector,
    target: CorrelationVector,
) -> t.Tensor:
    if prediction.shape != target.tensor.shape:
        raise ValueError("direct-age prediction and target shapes differ")
    _require_floating_finite(prediction, "direct-age prediction")
    if prediction.dtype != target.tensor.dtype:
        raise TypeError("direct-age prediction and target dtypes differ")
    return t.mean(t.square(prediction - target.tensor))


@beartype
@dataclass(frozen=True, slots=True)
class DirectAgeData:
    probes: NonEmptyProbeSet
    embeddings: EmbeddingMatrix
    rho: CorrelationVector

    def __post_init__(self) -> None:
        if (
            len(self.probes) != self.embeddings.tensor.shape[0]
            or len(self.probes) != self.rho.tensor.numel()
        ):
            raise ValueError("direct-age probe, embedding, and target axes must align")


@beartype
@dataclass(frozen=True, slots=True)
class DirectAgeTrainingConfig:
    batch_size: PositiveInt
    neighbourhood_width: PositiveInt
    steps: PositiveInt
    validation_interval: PositiveInt
    learning_rate: float
    seed: int
    device: str

    def __post_init__(self) -> None:
        if not bool(t.isfinite(t.tensor(self.learning_rate)).item()) or self.learning_rate <= 0.0:
            raise ValueError("direct-age learning rate must be finite and positive")
        if int(self.validation_interval) > int(self.steps):
            raise ValueError("validation interval cannot exceed direct-age training steps")


@dataclass(frozen=True, slots=True)
class DirectAgeStep:
    step: int
    age_mse: float

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("direct-age step cannot be negative")
        if not bool(t.isfinite(t.tensor(self.age_mse)).item()) or self.age_mse < 0.0:
            raise ValueError("direct-age MSE must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class TrainedDirectAge:
    model: DirectTanhAgeRegressor
    history: tuple[DirectAgeStep, ...]
    validation_history: tuple[DirectAgeStep, ...]
    selected_step: int

    def __post_init__(self) -> None:
        if not self.history or not self.validation_history:
            raise ValueError("trained direct-age model requires non-empty histories")
        validation_steps = tuple(record.step for record in self.validation_history)
        if tuple(sorted(set(validation_steps))) != validation_steps:
            raise ValueError("direct-age validation steps must be unique and increasing")
        if self.selected_step not in validation_steps:
            raise ValueError("selected direct-age step is absent from validation history")


@dataclass(frozen=True, slots=True)
class RefitDirectAge:
    model: DirectTanhAgeRegressor
    history: tuple[DirectAgeStep, ...]
    refit_steps: int

    def __post_init__(self) -> None:
        if self.refit_steps < 0 or len(self.history) != self.refit_steps:
            raise ValueError("direct-age refit history must exactly cover its step count")
        if self.history and self.history[-1].step != self.refit_steps:
            raise ValueError("direct-age refit history does not end at its frozen step count")


def _cuda_fork_devices(device: t.device) -> list[int]:
    if device.type != "cuda":
        return []
    return [device.index if device.index is not None else t.cuda.current_device()]


def _validate_partitions(training: DirectAgeData, validation: DirectAgeData) -> None:
    training_ids = {probe.probe_id for probe in training.probes.probes}
    validation_ids = {probe.probe_id for probe in validation.probes.probes}
    if training_ids & validation_ids:
        raise ValueError("direct-age training and validation probes must be disjoint")


def _device(config: DirectAgeTrainingConfig) -> t.device:
    device = t.device(config.device)
    if device.type == "cuda" and not t.cuda.is_available():
        raise ValueError("CUDA direct-age training was requested but CUDA is unavailable")
    return device


def _make_optimizer(
    model: DirectTanhAgeRegressor,
    *,
    learning_rate: float,
) -> t.optim.Optimizer:
    optimizer = t.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)
    if any(float(group["weight_decay"]) != 0.0 for group in optimizer.param_groups):
        raise RuntimeError("direct-age optimizer must not use weight decay")
    return optimizer


def _validation_step(
    model: DirectTanhAgeRegressor,
    validation: DirectAgeData,
    *,
    step: int,
    device: t.device,
) -> DirectAgeStep:
    model.eval()
    with t.inference_mode():
        prediction = model(validation.embeddings.training_tensor(device=device))
        target = CorrelationVector(validation.rho.tensor.to(device=device, dtype=t.float32))
        mse = direct_age_mse(prediction, target)
    model.train()
    return DirectAgeStep(step=step, age_mse=float(mse.cpu().item()))


def _optimization_step(
    model: DirectTanhAgeRegressor,
    optimizer: t.optim.Optimizer,
    sampler: GenomicBatchSampler,
    embeddings: t.Tensor,
    rho: t.Tensor,
    *,
    step: int,
) -> DirectAgeStep:
    indices = sampler.sample().indices
    prediction = model(embeddings.index_select(0, indices))
    target = CorrelationVector(rho.index_select(0, indices))
    mse = direct_age_mse(prediction, target)
    optimizer.zero_grad(set_to_none=True)
    mse.backward()
    optimizer.step()
    return DirectAgeStep(step=step, age_mse=float(mse.detach().cpu().item()))


@beartype
def train_direct_tanh_age(
    training: DirectAgeData,
    validation: DirectAgeData,
    *,
    config: DirectAgeTrainingConfig,
) -> TrainedDirectAge:
    """Tune one direct scalar head with exact full-validation step selection."""

    _validate_partitions(training, validation)
    if int(config.batch_size) > len(training.probes):
        raise ValueError("direct-age batch size cannot exceed the training universe")
    device = _device(config)
    sampler = GenomicBatchSampler(
        probes=training.probes,
        batch_size=config.batch_size,
        neighbourhood_width=config.neighbourhood_width,
        seed=config.seed,
    )
    embeddings = training.embeddings.training_tensor(device=device)
    rho = training.rho.tensor.to(device=device, dtype=t.float32)
    with t.random.fork_rng(devices=_cuda_fork_devices(device)):
        t.manual_seed(config.seed)
        model = DirectTanhAgeRegressor(embeddings.shape[1]).to(device)
        optimizer = _make_optimizer(model, learning_rate=config.learning_rate)
        history: list[DirectAgeStep] = []
        validation_history = [_validation_step(model, validation, step=0, device=device)]
        best = validation_history[0]
        best_state = {
            name: value.detach().cpu().clone() for name, value in model.state_dict().items()
        }
        for step_index in range(int(config.steps)):
            step = step_index + 1
            history.append(
                _optimization_step(
                    model,
                    optimizer,
                    sampler,
                    embeddings,
                    rho,
                    step=step,
                )
            )
            if step % int(config.validation_interval) == 0 or step == int(config.steps):
                record = _validation_step(model, validation, step=step, device=device)
                validation_history.append(record)
                if record.age_mse < best.age_mse:
                    best = record
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
        model.load_state_dict(best_state, strict=True)
    return TrainedDirectAge(
        model=model,
        history=tuple(history),
        validation_history=tuple(validation_history),
        selected_step=best.step,
    )


@beartype
def refit_direct_tanh_age(
    training: DirectAgeData,
    *,
    config: DirectAgeTrainingConfig,
    selected_steps: int,
) -> RefitDirectAge:
    """Refit the direct head on complete primary train for a frozen step count."""

    if selected_steps < 0 or selected_steps > int(config.steps):
        raise ValueError("direct-age refit steps must lie within the tuning budget")
    if int(config.batch_size) > len(training.probes):
        raise ValueError("direct-age refit batch size cannot exceed the training universe")
    device = _device(config)
    sampler = GenomicBatchSampler(
        probes=training.probes,
        batch_size=config.batch_size,
        neighbourhood_width=config.neighbourhood_width,
        seed=config.seed,
    )
    embeddings = training.embeddings.training_tensor(device=device)
    rho = training.rho.tensor.to(device=device, dtype=t.float32)
    with t.random.fork_rng(devices=_cuda_fork_devices(device)):
        t.manual_seed(config.seed)
        model = DirectTanhAgeRegressor(embeddings.shape[1]).to(device)
        optimizer = _make_optimizer(model, learning_rate=config.learning_rate)
        history = tuple(
            _optimization_step(
                model,
                optimizer,
                sampler,
                embeddings,
                rho,
                step=step,
            )
            for step in range(1, selected_steps + 1)
        )
    return RefitDirectAge(model=model, history=history, refit_steps=selected_steps)
