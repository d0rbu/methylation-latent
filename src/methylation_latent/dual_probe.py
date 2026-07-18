"""Train-only learned probe geometry with gradient-routed sequence catching."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch as t
from beartype import beartype
from jaxtyping import Float, Int64, jaxtyped

from methylation_latent.batching import GenomicBatchSampler
from methylation_latent.domain import (
    Fraction,
    LatentDimension,
    NonEmptyProbeSet,
    NonNegativeWeight,
    PositiveInt,
)
from methylation_latent.model import (
    PairObjectiveInput,
    metric_objective,
    normalize_rows_strict,
    normalize_vector_strict,
)
from methylation_latent.storage import EmbeddingMatrix
from methylation_latent.targets import (
    CorrelationMatrix,
    CorrelationVector,
    TargetGeometry,
    validate_latent_dimension,
)

EmbeddingRows = Float[t.Tensor, "probes embedding"]
LearnedRows = Float[t.Tensor, "probes latent"]
ProjectionWeight = Float[t.Tensor, "latent embedding"]
LatentRows = Float[t.Tensor, "probes latent"]
LatentVector = Float[t.Tensor, "latent"]
LocalIndices = Int64[t.Tensor, "selected"]


def _require_finite_float(values: t.Tensor, name: str) -> None:
    if not values.is_floating_point():
        raise TypeError(f"{name} must have a floating dtype")
    if not bool(t.isfinite(values).all().item()):
        raise ValueError(f"{name} must contain only finite values")


class AlphaScheduleMode(StrEnum):
    FIXED = "fixed"
    LINEAR = "linear"


@beartype
@dataclass(frozen=True, slots=True)
class AlphaSchedule:
    """A closed-interval catch-gradient schedule with exact endpoints."""

    name: str
    mode: AlphaScheduleMode
    start: Fraction
    end: Fraction

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("alpha schedule name must not be empty")
        if self.mode == AlphaScheduleMode.FIXED and self.start != self.end:
            raise ValueError("fixed alpha schedule requires equal endpoints")
        if self.mode == AlphaScheduleMode.LINEAR and self.start == self.end:
            raise ValueError("linear alpha schedule requires distinct endpoints")

    def value(self, step: int, total_steps: int) -> Fraction:
        if total_steps <= 0 or not 1 <= step <= total_steps:
            raise ValueError("alpha schedule step must lie inside the positive step budget")
        if self.mode == AlphaScheduleMode.FIXED:
            return self.start
        progress = 1.0 if total_steps == 1 else (step - 1) / (total_steps - 1)
        value = float(self.start) + progress * (float(self.end) - float(self.start))
        return Fraction.parse(value)


@dataclass(frozen=True, slots=True)
class LeastSquaresAudit:
    fit_row_count: int
    embedding_dimension: int
    latent_dimension: int
    effective_rank: int
    raw_condition_number: float
    retained_condition_number: float
    discarded_spectral_mass_fraction: float
    discarded_cross_moment_fraction: float
    relative_retained_normal_residual: float
    initial_mean_cosine: float

    def __post_init__(self) -> None:
        if (
            self.fit_row_count < self.embedding_dimension
            or self.embedding_dimension <= 0
            or self.latent_dimension <= 0
        ):
            raise ValueError("least-squares audit dimensions are invalid")
        values = (
            self.raw_condition_number,
            self.retained_condition_number,
            self.discarded_spectral_mass_fraction,
            self.discarded_cross_moment_fraction,
            self.relative_retained_normal_residual,
            self.initial_mean_cosine,
        )
        if any(not bool(t.isfinite(t.tensor(value)).item()) for value in values):
            raise ValueError("least-squares audit values must be finite")
        if not 0 < self.effective_rank <= self.embedding_dimension:
            raise ValueError("least-squares effective rank is invalid")
        if (
            self.raw_condition_number < self.retained_condition_number
            or self.retained_condition_number < 1.0
        ):
            raise ValueError("least-squares condition numbers are invalid")
        fractions = (
            self.discarded_spectral_mass_fraction,
            self.discarded_cross_moment_fraction,
        )
        if any(not 0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("least-squares discarded fractions must lie in [0, 1]")
        if self.relative_retained_normal_residual < 0.0:
            raise ValueError("least-squares retained residual must be non-negative")
        if not -1.0 <= self.initial_mean_cosine <= 1.0:
            raise ValueError("initial mean cosine must lie in [-1, 1]")


@jaxtyped(typechecker=beartype)
def normal_equation_projection(
    embeddings: EmbeddingRows,
    learned_vectors: LearnedRows,
    *,
    condition_ceiling: float,
    relative_residual_ceiling: float,
    discarded_cross_moment_ceiling: float,
) -> tuple[ProjectionWeight, LeastSquaresAudit]:
    """Solve a condition-truncated bias-free least-squares map on fitting rows only."""

    if embeddings.ndim != 2 or learned_vectors.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("least-squares inputs must be non-empty matrices")
    if embeddings.shape[0] != learned_vectors.shape[0]:
        raise ValueError("least-squares embedding and learned-vector rows differ")
    if embeddings.shape[0] < embeddings.shape[1]:
        raise ValueError("least-squares fit requires at least as many rows as embedding columns")
    if embeddings.dtype != learned_vectors.dtype:
        raise TypeError("least-squares inputs must share a dtype")
    _require_finite_float(embeddings, "least-squares embeddings")
    _require_finite_float(learned_vectors, "least-squares learned vectors")
    thresholds = t.tensor(
        (
            condition_ceiling,
            relative_residual_ceiling,
            discarded_cross_moment_ceiling,
        ),
        dtype=t.float64,
    )
    if (
        not bool(t.isfinite(thresholds).all().item())
        or condition_ceiling < 1.0
        or relative_residual_ceiling <= 0.0
        or not 0.0 < discarded_cross_moment_ceiling <= 1.0
    ):
        raise ValueError("least-squares audit thresholds are invalid")
    row_count = embeddings.shape[0]
    gram = (embeddings.mT @ embeddings / row_count).to(device="cpu", dtype=t.float64)
    cross = (embeddings.mT @ learned_vectors / row_count).to(device="cpu", dtype=t.float64)
    eigenvalues, eigenvectors = t.linalg.eigh(gram)
    if float(eigenvalues[0].item()) <= 0.0:
        raise ValueError("embedding normal matrix is not positive definite")
    raw_condition = float((eigenvalues[-1] / eigenvalues[0]).item())
    retained = eigenvalues > eigenvalues[-1] / condition_ceiling
    if not bool(retained.any().item()):  # pragma: no cover - largest value always survives
        raise RuntimeError("condition truncation removed every embedding direction")
    retained_values = eigenvalues[retained]
    retained_vectors = eigenvectors[:, retained]
    retained_condition = float((retained_values[-1] / retained_values[0]).item())
    if retained_condition > condition_ceiling:
        raise RuntimeError("retained least-squares system exceeds its condition ceiling")
    retained_cross = retained_vectors.mT @ cross
    solution = retained_vectors @ (retained_cross / retained_values[:, None])
    cross_norm = t.linalg.vector_norm(cross)
    if float(cross_norm.item()) == 0.0:
        raise ValueError("least-squares cross moment is exactly zero")
    relative_residual = float(
        (
            t.linalg.vector_norm(retained_vectors.mT @ (gram @ solution - cross))
            / t.linalg.vector_norm(retained_cross)
        ).item()
    )
    if relative_residual > relative_residual_ceiling:
        raise ValueError(
            f"least-squares relative normal residual {relative_residual} exceeds "
            f"{relative_residual_ceiling}"
        )
    discarded = ~retained
    discarded_spectral_mass = float((eigenvalues[discarded].sum() / eigenvalues.sum()).item())
    discarded_cross_fraction = float(
        (t.linalg.vector_norm(eigenvectors[:, discarded].mT @ cross) / cross_norm).item()
    )
    if discarded_cross_fraction > discarded_cross_moment_ceiling:
        raise ValueError(
            "least-squares discarded cross-moment fraction "
            f"{discarded_cross_fraction} exceeds {discarded_cross_moment_ceiling}"
        )
    weight = solution.mT.to(device=embeddings.device, dtype=embeddings.dtype).contiguous()
    predicted = normalize_rows_strict(embeddings @ weight.mT)
    targets = normalize_rows_strict(learned_vectors)
    mean_cosine = float(t.mean(t.sum(predicted * targets, dim=1)).detach().cpu().item())
    audit = LeastSquaresAudit(
        fit_row_count=row_count,
        embedding_dimension=embeddings.shape[1],
        latent_dimension=learned_vectors.shape[1],
        effective_rank=int(retained.sum().item()),
        raw_condition_number=raw_condition,
        retained_condition_number=retained_condition,
        discarded_spectral_mass_fraction=discarded_spectral_mass,
        discarded_cross_moment_fraction=discarded_cross_fraction,
        relative_retained_normal_residual=relative_residual,
        initial_mean_cosine=mean_cosine,
    )
    return weight, audit


@dataclass(frozen=True, slots=True)
class DualInitialization:
    learned_vectors: t.Tensor
    projection_weight: t.Tensor
    age_direction: t.Tensor
    least_squares: LeastSquaresAudit

    def __post_init__(self) -> None:
        tensors = (self.learned_vectors, self.projection_weight, self.age_direction)
        if any(tensor.dtype != t.float32 for tensor in tensors):
            raise TypeError("dual initialization tensors must be float32")
        if any(not bool(t.isfinite(tensor).all().item()) for tensor in tensors):
            raise ValueError("dual initialization tensors must be finite")
        if self.learned_vectors.ndim != 2 or self.projection_weight.ndim != 2:
            raise ValueError("dual initialization matrices must be rank two")
        if self.age_direction.ndim != 1:
            raise ValueError("dual initialization age direction must be rank one")
        if (
            self.learned_vectors.shape[0] != self.least_squares.fit_row_count
            or self.learned_vectors.shape[1] != self.least_squares.latent_dimension
            or self.projection_weight.shape
            != (
                self.least_squares.latent_dimension,
                self.least_squares.embedding_dimension,
            )
            or self.age_direction.shape[0] != self.least_squares.latent_dimension
        ):
            raise ValueError("dual initialization tensors do not match their audit dimensions")
        learned_norms = t.linalg.vector_norm(self.learned_vectors, dim=1)
        if not t.allclose(learned_norms, t.ones_like(learned_norms), atol=2.0e-6, rtol=0.0):
            raise ValueError("initial learned probe vectors must have unit norm")
        if float(t.linalg.vector_norm(self.age_direction).item()) == 0.0:
            raise ValueError("initial age direction must not be zero")


@beartype
def build_dual_initialization(
    embeddings: EmbeddingMatrix,
    *,
    latent_dimension: LatentDimension,
    seed: int,
    device: str,
    condition_ceiling: float,
    relative_residual_ceiling: float,
    discarded_cross_moment_ceiling: float,
) -> DualInitialization:
    """Create identical reusable state for every alpha candidate in one fitting cell."""

    if int(latent_dimension) > embeddings.tensor.shape[1]:
        raise ValueError("dual latent dimension cannot exceed the embedding width")
    resolved = t.device(device)
    if resolved.type == "cuda" and not t.cuda.is_available():
        raise ValueError("CUDA initialization was requested but CUDA is unavailable")
    values = embeddings.training_tensor(device=resolved)
    devices = (
        []
        if resolved.type != "cuda"
        else [resolved.index if resolved.index is not None else t.cuda.current_device()]
    )
    with t.random.fork_rng(devices=devices):
        t.manual_seed(seed)
        learned = normalize_rows_strict(
            t.randn(
                (values.shape[0], int(latent_dimension)),
                dtype=t.float32,
                device=resolved,
            )
        )
        age_direction = t.randn(int(latent_dimension), dtype=t.float32, device=resolved)
        projection, audit = normal_equation_projection(
            values,
            learned,
            condition_ceiling=condition_ceiling,
            relative_residual_ceiling=relative_residual_ceiling,
            discarded_cross_moment_ceiling=discarded_cross_moment_ceiling,
        )
    return DualInitialization(
        learned_vectors=learned.detach().cpu().contiguous(),
        projection_weight=projection.detach().cpu().contiguous(),
        age_direction=age_direction.detach().cpu().contiguous(),
        least_squares=audit,
    )


class DualProbeLatent(t.nn.Module):
    """A fitting-only learned table coupled to a bias-free sequence map."""

    def __init__(self, initialization: DualInitialization) -> None:
        super().__init__()
        self.embedding_dimension = initialization.least_squares.embedding_dimension
        self.latent_dimension = initialization.least_squares.latent_dimension
        self.fit_probe_count = initialization.least_squares.fit_row_count
        self.learned_vectors = t.nn.Embedding.from_pretrained(
            initialization.learned_vectors.clone(),
            freeze=False,
            sparse=True,
        )
        self.projection = t.nn.Linear(
            self.embedding_dimension,
            self.latent_dimension,
            bias=False,
        )
        with t.no_grad():
            self.projection.weight.copy_(initialization.projection_weight)
        self.age_direction = t.nn.Parameter(initialization.age_direction.clone())

    @jaxtyped(typechecker=beartype)
    def learned(self, indices: LocalIndices) -> LatentRows:
        if indices.ndim != 1 or indices.numel() == 0:
            raise ValueError("learned-vector indices must be a non-empty vector")
        if bool(t.any((indices < 0) | (indices >= self.fit_probe_count)).item()):
            raise IndexError("learned-vector index is outside the fitting table")
        return normalize_rows_strict(self.learned_vectors(indices))

    def all_learned(self) -> t.Tensor:
        return normalize_rows_strict(self.learned_vectors.weight)

    @jaxtyped(typechecker=beartype)
    def sequence(self, embeddings: EmbeddingRows) -> LatentRows:
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError("sequence embeddings must be a non-empty matrix")
        if embeddings.shape[1] != self.embedding_dimension:
            raise ValueError("sequence embedding width differs from the initialized map")
        if embeddings.dtype != self.projection.weight.dtype:
            raise TypeError("sequence embeddings and projection must share a dtype")
        _require_finite_float(embeddings, "sequence embeddings")
        return normalize_rows_strict(self.projection(embeddings))

    @jaxtyped(typechecker=beartype)
    def age(self, latent: LatentRows) -> t.Tensor:
        return normalize_rows_strict(latent) @ normalize_vector_strict(self.age_direction)


@dataclass(frozen=True, slots=True)
class DualObjectiveTerms:
    total: t.Tensor
    geometry_total: t.Tensor
    pair_mse: t.Tensor
    age_mse: t.Tensor
    catch: t.Tensor
    catch_to_learned: t.Tensor
    catch_to_sequence: t.Tensor
    mean_alignment: t.Tensor

    def __post_init__(self) -> None:
        values = (
            ("total", self.total),
            ("geometry_total", self.geometry_total),
            ("pair_mse", self.pair_mse),
            ("age_mse", self.age_mse),
            ("catch", self.catch),
            ("catch_to_learned", self.catch_to_learned),
            ("catch_to_sequence", self.catch_to_sequence),
            ("mean_alignment", self.mean_alignment),
        )
        for name, value in values:
            if value.ndim != 0 or not value.is_floating_point():
                raise ValueError(f"{name} must be a floating scalar tensor")


@jaxtyped(typechecker=beartype)
def dual_probe_objective(
    learned_latent: LearnedRows,
    sequence_latent: LatentRows,
    age_direction: LatentVector,
    pair_target: CorrelationMatrix,
    age_target: CorrelationVector,
    *,
    lambda_age: NonNegativeWeight,
    catch_weight: NonNegativeWeight,
    alpha: Fraction,
) -> DualObjectiveTerms:
    """Apply empirical geometry to learned rows and route only the catch gradient."""

    if learned_latent.shape != sequence_latent.shape:
        raise ValueError("learned and sequence latent batches must have identical shapes")
    learned = normalize_rows_strict(learned_latent)
    sequence = normalize_rows_strict(sequence_latent)
    pair = PairObjectiveInput(
        prediction=learned @ learned.mT,
        target=pair_target,
    )
    geometry = metric_objective(
        learned @ normalize_vector_strict(age_direction),
        age_target,
        lambda_age=lambda_age,
        pair=pair,
    )
    if geometry.pair_mse is None:  # pragma: no cover - pair is required above
        raise RuntimeError("dual geometry unexpectedly omitted its pair term")
    catch_to_learned = 0.5 * t.mean(t.sum(t.square(learned - sequence.detach()), dim=1))
    catch_to_sequence = 0.5 * t.mean(t.sum(t.square(learned.detach() - sequence), dim=1))
    if not t.allclose(
        catch_to_learned.detach(),
        catch_to_sequence.detach(),
        atol=2.0e-6,
        rtol=0.0,
    ):
        raise RuntimeError("routed catch terms differ numerically")
    catch = float(alpha) * catch_to_learned + (1.0 - float(alpha)) * catch_to_sequence
    mean_alignment = t.mean(t.sum(learned * sequence, dim=1))
    return DualObjectiveTerms(
        total=geometry.total + float(catch_weight) * catch,
        geometry_total=geometry.total,
        pair_mse=geometry.pair_mse,
        age_mse=geometry.age_mse,
        catch=catch,
        catch_to_learned=catch_to_learned,
        catch_to_sequence=catch_to_sequence,
        mean_alignment=mean_alignment,
    )


@beartype
@dataclass(frozen=True, slots=True)
class DualTrainingConfig:
    latent_dimension: LatentDimension
    lambda_age: NonNegativeWeight
    catch_weight: NonNegativeWeight
    alpha_schedule: AlphaSchedule
    batch_size: PositiveInt
    neighbourhood_width: PositiveInt
    steps: PositiveInt
    validation_interval: PositiveInt
    validation_pair_chunk_size: PositiveInt
    learning_rate: float
    seed: int
    device: str

    def __post_init__(self) -> None:
        if not bool(t.isfinite(t.tensor(self.learning_rate)).item()) or self.learning_rate <= 0.0:
            raise ValueError("dual training learning rate must be finite and positive")
        if int(self.validation_interval) > int(self.steps):
            raise ValueError("dual validation interval cannot exceed the step budget")
        if float(self.catch_weight) <= 0.0:
            raise ValueError("dual training requires a positive catch weight")


@dataclass(frozen=True, slots=True)
class DualTrainingStep:
    step: int
    alpha: float
    total_loss: float
    geometry_total: float
    pair_mse: float
    age_mse: float
    catch: float
    mean_alignment: float

    def __post_init__(self) -> None:
        if self.step <= 0 or not 0.0 <= self.alpha <= 1.0:
            raise ValueError("dual training step or alpha is invalid")
        nonnegative = (
            self.total_loss,
            self.geometry_total,
            self.pair_mse,
            self.age_mse,
            self.catch,
        )
        if any(
            not bool(t.isfinite(t.tensor(value)).item()) or value < 0.0 for value in nonnegative
        ):
            raise ValueError("dual training losses must be finite and non-negative")
        if not bool(t.isfinite(t.tensor(self.mean_alignment)).item()) or not (
            -1.0 <= self.mean_alignment <= 1.0
        ):
            raise ValueError("dual mean alignment must lie in [-1, 1]")


@dataclass(frozen=True, slots=True)
class DualValidationStep:
    step: int
    selection_score: float
    pair_mse: float
    age_mse: float

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("dual validation step cannot be negative")
        values = (self.selection_score, self.pair_mse, self.age_mse)
        if any(not bool(t.isfinite(t.tensor(value)).item()) or value < 0.0 for value in values):
            raise ValueError("dual validation losses must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class DualValidationData:
    """Validation rows that deliberately have no learned-vector table."""

    probes: NonEmptyProbeSet
    embeddings: EmbeddingMatrix
    targets: TargetGeometry

    def __post_init__(self) -> None:
        if (
            len(self.probes) != self.embeddings.tensor.shape[0]
            or len(self.probes) != self.targets.methylation.n_rows
        ):
            raise ValueError("dual validation probe, embedding, and target axes differ")


@dataclass(frozen=True, slots=True)
class DualTuningResult:
    training_history: tuple[DualTrainingStep, ...]
    validation_history: tuple[DualValidationStep, ...]
    selected_step: int

    def __post_init__(self) -> None:
        if not self.training_history or not self.validation_history:
            raise ValueError("dual tuning histories must be non-empty")
        validation_steps = tuple(record.step for record in self.validation_history)
        if tuple(sorted(set(validation_steps))) != validation_steps:
            raise ValueError("dual validation steps must be unique and increasing")
        if self.selected_step not in validation_steps:
            raise ValueError("selected dual checkpoint is absent from validation history")


@dataclass(frozen=True, slots=True)
class DualRefitResult:
    model: DualProbeLatent
    training_history: tuple[DualTrainingStep, ...]
    refit_steps: int

    def __post_init__(self) -> None:
        if self.refit_steps < 0:
            raise ValueError("dual refit step count cannot be negative")
        if self.refit_steps == 0 and self.training_history:
            raise ValueError("zero-step dual refit must have empty history")
        if self.refit_steps > 0 and (
            not self.training_history or self.training_history[-1].step != self.refit_steps
        ):
            raise ValueError("dual refit history must end at the selected step")
        steps = tuple(record.step for record in self.training_history)
        if tuple(sorted(set(steps))) != steps:
            raise ValueError("dual refit checkpoints must be unique and increasing")


def _make_optimizers(
    model: DualProbeLatent,
    learning_rate: float,
) -> tuple[t.optim.Optimizer, t.optim.Optimizer]:
    if not bool(t.isfinite(t.tensor(learning_rate)).item()) or learning_rate <= 0.0:
        raise ValueError("dual optimizer learning rate must be finite and positive")
    dense_optimizer = t.optim.Adam(
        (model.projection.weight, model.age_direction),
        lr=learning_rate,
        weight_decay=0.0,
    )
    sparse_optimizer = t.optim.SparseAdam(
        (model.learned_vectors.weight,),
        lr=learning_rate,
    )
    if any(float(group["weight_decay"]) != 0.0 for group in dense_optimizer.param_groups):
        raise RuntimeError("dual optimizer must use zero weight decay")
    if any("weight_decay" in group for group in sparse_optimizer.param_groups):
        raise RuntimeError("SparseAdam unexpectedly exposes a weight-decay option")
    return dense_optimizer, sparse_optimizer


def _validate_axes(
    probes: NonEmptyProbeSet,
    embeddings: EmbeddingMatrix,
    targets: TargetGeometry,
    initialization: DualInitialization,
) -> None:
    row_count = len(probes)
    if row_count != embeddings.tensor.shape[0] or row_count != targets.methylation.n_rows:
        raise ValueError("dual probe, embedding, and target axes must align exactly")
    if row_count != initialization.least_squares.fit_row_count:
        raise ValueError("dual initialization row axis differs from fitting probes")
    if embeddings.tensor.shape[1] != initialization.least_squares.embedding_dimension:
        raise ValueError("dual initialization embedding width differs")


def _validate_config_initialization(
    config: DualTrainingConfig,
    initialization: DualInitialization,
) -> None:
    if int(config.latent_dimension) != initialization.least_squares.latent_dimension:
        raise ValueError("dual training latent dimension differs from initialization")


def _exact_sequence_validation(
    model: DualProbeLatent,
    embeddings: EmbeddingMatrix,
    pair_target: CorrelationMatrix,
    age_target: CorrelationVector,
    *,
    step: int,
    pair_chunk_size: int,
    device: t.device,
) -> DualValidationStep:
    model.eval()
    with t.inference_mode():
        values = embeddings.training_tensor(device=device)
        target = pair_target.tensor
        rho = age_target.tensor
        if target.device != device or rho.device != device:
            raise ValueError("precomputed dual validation targets are on the wrong device")
        latent = model.sequence(values)
        age_mse_tensor = t.mean(t.square(model.age(latent) - rho))
        squared_error_sum = t.zeros((), dtype=t.float64, device=device)
        for start in range(0, latent.shape[0], pair_chunk_size):
            stop = min(start + pair_chunk_size, latent.shape[0])
            prediction = latent[start:stop] @ latent.mT
            squared_error_sum += t.square(prediction - target[start:stop]).sum(dtype=t.float64)
        pair_mse_tensor = (squared_error_sum / (latent.shape[0] * latent.shape[0])).to(t.float32)
    model.train()
    pair_mse = float(pair_mse_tensor.detach().cpu().item())
    age_mse = float(age_mse_tensor.detach().cpu().item())
    return DualValidationStep(
        step=step,
        selection_score=pair_mse + age_mse,
        pair_mse=pair_mse,
        age_mse=age_mse,
    )


def _optimization_step(
    model: DualProbeLatent,
    optimizers: tuple[t.optim.Optimizer, t.optim.Optimizer],
    sampler: GenomicBatchSampler,
    embeddings: t.Tensor,
    target_rows: t.Tensor,
    target_rho: t.Tensor,
    *,
    config: DualTrainingConfig,
    step: int,
) -> DualTrainingStep:
    local_indices = sampler.sample().indices.to(embeddings.device)
    selected_rows = target_rows.index_select(0, local_indices)
    learned = model.learned(local_indices)
    sequence = model.sequence(embeddings.index_select(0, local_indices))
    alpha = config.alpha_schedule.value(step, int(config.steps))
    terms = dual_probe_objective(
        learned,
        sequence,
        model.age_direction,
        CorrelationMatrix(selected_rows @ selected_rows.mT),
        CorrelationVector(target_rho.index_select(0, local_indices)),
        lambda_age=config.lambda_age,
        catch_weight=config.catch_weight,
        alpha=alpha,
    )
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    terms.total.backward()
    for optimizer in optimizers:
        optimizer.step()
    return DualTrainingStep(
        step=step,
        alpha=float(alpha),
        total_loss=float(terms.total.detach().cpu().item()),
        geometry_total=float(terms.geometry_total.detach().cpu().item()),
        pair_mse=float(terms.pair_mse.detach().cpu().item()),
        age_mse=float(terms.age_mse.detach().cpu().item()),
        catch=float(terms.catch.detach().cpu().item()),
        mean_alignment=float(terms.mean_alignment.detach().cpu().item()),
    )


@beartype
def tune_dual_probe(
    probes: NonEmptyProbeSet,
    embeddings: EmbeddingMatrix,
    targets: TargetGeometry,
    validation: DualValidationData,
    initialization: DualInitialization,
    *,
    config: DualTrainingConfig,
) -> DualTuningResult:
    """Tune alpha using sequence-only validation; learned validation rows do not exist."""

    _validate_axes(probes, embeddings, targets, initialization)
    _validate_config_initialization(config, initialization)
    train_ids = {probe.probe_id for probe in probes.probes}
    validation_ids = {probe.probe_id for probe in validation.probes.probes}
    if train_ids & validation_ids:
        raise ValueError("dual fitting and validation probe IDs must be disjoint")
    if embeddings.tensor.shape[1] != validation.embeddings.tensor.shape[1]:
        raise ValueError("dual fitting and validation embedding widths differ")
    if targets.methylation.n_samples != validation.targets.methylation.n_samples:
        raise ValueError("dual fitting and validation sample axes differ")
    validate_latent_dimension(
        config.latent_dimension,
        embedding_dimension=embeddings.tensor.shape[1],
        n_samples=targets.methylation.n_samples,
    )
    if int(config.batch_size) > len(probes):
        raise ValueError("dual batch size exceeds the fitting probe count")
    device = t.device(config.device)
    if device.type == "cuda" and not t.cuda.is_available():
        raise ValueError("CUDA dual tuning was requested but CUDA is unavailable")
    sampler = GenomicBatchSampler(
        probes=probes,
        batch_size=config.batch_size,
        neighbourhood_width=config.neighbourhood_width,
        seed=config.seed,
    )
    values = embeddings.training_tensor(device=device)
    rows = targets.methylation.tensor.to(device=device, dtype=t.float32)
    rho = targets.rho.tensor.to(device=device, dtype=t.float32)
    model = DualProbeLatent(initialization).to(device)
    optimizers = _make_optimizers(model, config.learning_rate)
    validation_rows = validation.targets.methylation.tensor.to(
        device=device,
        dtype=t.float32,
    )
    validation_pair_target = CorrelationMatrix(validation_rows @ validation_rows.mT)
    validation_age_target = CorrelationVector(
        validation.targets.rho.tensor.to(device=device, dtype=t.float32)
    )
    validation_history = [
        _exact_sequence_validation(
            model,
            validation.embeddings,
            validation_pair_target,
            validation_age_target,
            step=0,
            pair_chunk_size=int(config.validation_pair_chunk_size),
            device=device,
        )
    ]
    training_history: list[DualTrainingStep] = []
    for step in range(1, int(config.steps) + 1):
        record = _optimization_step(
            model,
            optimizers,
            sampler,
            values,
            rows,
            rho,
            config=config,
            step=step,
        )
        should_validate = step % int(config.validation_interval) == 0 or step == int(config.steps)
        if should_validate:
            training_history.append(record)
            validation_history.append(
                _exact_sequence_validation(
                    model,
                    validation.embeddings,
                    validation_pair_target,
                    validation_age_target,
                    step=step,
                    pair_chunk_size=int(config.validation_pair_chunk_size),
                    device=device,
                )
            )
    best = min(validation_history, key=lambda record: (record.selection_score, record.step))
    return DualTuningResult(
        training_history=tuple(training_history),
        validation_history=tuple(validation_history),
        selected_step=best.step,
    )


@beartype
def refit_dual_probe(
    probes: NonEmptyProbeSet,
    embeddings: EmbeddingMatrix,
    targets: TargetGeometry,
    initialization: DualInitialization,
    *,
    config: DualTrainingConfig,
    selected_steps: int,
) -> DualRefitResult:
    """Refit the selected strategy on complete primary training only."""

    _validate_axes(probes, embeddings, targets, initialization)
    _validate_config_initialization(config, initialization)
    if not 0 <= selected_steps <= int(config.steps):
        raise ValueError("dual refit steps must lie inside the tuning budget")
    if int(config.batch_size) > len(probes):
        raise ValueError("dual refit batch size exceeds the fitting probe count")
    validate_latent_dimension(
        config.latent_dimension,
        embedding_dimension=embeddings.tensor.shape[1],
        n_samples=targets.methylation.n_samples,
    )
    device = t.device(config.device)
    if device.type == "cuda" and not t.cuda.is_available():
        raise ValueError("CUDA dual refit was requested but CUDA is unavailable")
    sampler = GenomicBatchSampler(
        probes=probes,
        batch_size=config.batch_size,
        neighbourhood_width=config.neighbourhood_width,
        seed=config.seed,
    )
    values = embeddings.training_tensor(device=device)
    rows = targets.methylation.tensor.to(device=device, dtype=t.float32)
    rho = targets.rho.tensor.to(device=device, dtype=t.float32)
    model = DualProbeLatent(initialization).to(device)
    optimizers = _make_optimizers(model, config.learning_rate)
    history: list[DualTrainingStep] = []
    for step in range(1, selected_steps + 1):
        record = _optimization_step(
            model,
            optimizers,
            sampler,
            values,
            rows,
            rho,
            config=config,
            step=step,
        )
        if step % int(config.validation_interval) == 0 or step == selected_steps:
            history.append(record)
    return DualRefitResult(
        model=model,
        training_history=tuple(history),
        refit_steps=selected_steps,
    )
