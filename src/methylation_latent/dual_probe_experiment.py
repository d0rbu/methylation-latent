"""Leakage-safe data and evaluation helpers for dual-probe latent experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch as t
from beartype import beartype

from methylation_latent.domain import NonEmptyProbeSet
from methylation_latent.evaluation import PairIndices, PairPopulation
from methylation_latent.experiment_data import SplitArtifact
from methylation_latent.model import normalize_rows_strict, normalize_vector_strict
from methylation_latent.storage import EmbeddingMatrix
from methylation_latent.targets import (
    CorrelationVector,
    TargetGeometry,
    UnitNormRows,
)


@dataclass(frozen=True, slots=True)
class DualPartitionData:
    """One explicitly indexed and axis-aligned dual-model partition."""

    global_indices: t.Tensor
    probes: NonEmptyProbeSet
    embeddings: EmbeddingMatrix
    targets: TargetGeometry

    def __post_init__(self) -> None:
        if (
            self.global_indices.dtype != t.int64
            or self.global_indices.ndim != 1
            or self.global_indices.numel() == 0
        ):
            raise TypeError("dual partition global indices must be a non-empty int64 vector")
        if t.unique(self.global_indices).numel() != self.global_indices.numel():
            raise ValueError("dual partition global indices must be unique")
        count = self.global_indices.numel()
        if (
            len(self.probes) != count
            or self.embeddings.tensor.shape[0] != count
            or self.targets.methylation.n_rows != count
        ):
            raise ValueError("dual partition axes differ")


@beartype
def subset_dual_partition(
    probes: NonEmptyProbeSet,
    embeddings: EmbeddingMatrix,
    targets: TargetGeometry,
    indices: t.Tensor,
) -> DualPartitionData:
    """Subset every probe axis with one checked global index vector."""

    if len(probes) != embeddings.tensor.shape[0] or len(probes) != targets.methylation.n_rows:
        raise ValueError("global dual inputs have different probe axes")
    if indices.dtype != t.int64 or indices.ndim != 1 or indices.numel() == 0:
        raise TypeError("dual subset indices must be a non-empty int64 vector")
    if bool(t.any((indices < 0) | (indices >= len(probes))).item()):
        raise IndexError("dual subset index is outside the global probe universe")
    if t.unique(indices).numel() != indices.numel():
        raise ValueError("dual subset indices must be unique")
    selected = indices.clone()
    return DualPartitionData(
        global_indices=selected,
        probes=NonEmptyProbeSet(tuple(probes.probes[index] for index in selected.tolist())),
        embeddings=EmbeddingMatrix(embeddings.tensor.index_select(0, selected)),
        targets=TargetGeometry(
            methylation=UnitNormRows(targets.methylation.tensor.index_select(0, selected)),
            age=targets.age,
            rho=CorrelationVector(targets.rho.tensor.index_select(0, selected)),
        ),
    )


@beartype
def assert_dual_split_roles(split: SplitArtifact) -> None:
    """Prove that learned-table fitting partitions exclude every test row."""

    test = split.test_indices
    fitting_partitions = (split.optimization_indices, split.primary_train_indices)
    if any(bool(t.isin(partition, test).any().item()) for partition in fitting_partitions):
        raise ValueError("dual learned-table fitting partition contains test probes")
    if bool(t.isin(split.optimization_indices, split.validation_indices).any().item()):
        raise ValueError("dual optimization and validation partitions overlap")
    expected_primary = t.sort(
        t.cat(
            (
                split.optimization_indices,
                split.validation_indices,
                split.validation_buffer_indices,
            )
        )
    ).values
    if not t.equal(split.primary_train_indices, expected_primary):
        raise ValueError("dual complete-primary-train partition identity differs")


@beartype
def sequence_latent_from_projection(
    embeddings: EmbeddingMatrix,
    projection_weight: t.Tensor,
    *,
    device: str,
    row_chunk_size: int,
) -> t.Tensor:
    """Apply one frozen sequence map in bounded chunks and return CPU unit rows."""

    if (
        projection_weight.dtype != t.float32
        or projection_weight.ndim != 2
        or projection_weight.shape[1] != embeddings.tensor.shape[1]
    ):
        raise ValueError("dual projection weight is not aligned float32 [latent, embedding]")
    if not bool(t.isfinite(projection_weight).all().item()):
        raise ValueError("dual projection weight must be finite")
    if row_chunk_size <= 0:
        raise ValueError("dual sequence projection chunk size must be positive")
    resolved = t.device(device)
    if resolved.type == "cuda" and not t.cuda.is_available():
        raise ValueError("CUDA sequence projection was requested but CUDA is unavailable")
    weight = projection_weight.to(device=resolved)
    chunks = tuple(
        normalize_rows_strict(
            embeddings.tensor[start:stop].to(device=resolved, dtype=t.float32) @ weight.mT
        )
        .cpu()
        .contiguous()
        for start in range(0, embeddings.tensor.shape[0], row_chunk_size)
        for stop in (min(start + row_chunk_size, embeddings.tensor.shape[0]),)
    )
    latent = t.cat(chunks)
    norms = t.linalg.vector_norm(latent, dim=1)
    if not t.allclose(norms, t.ones_like(norms), atol=2.0e-5, rtol=0.0):
        raise RuntimeError("chunked dual sequence projection produced non-unit rows")
    return latent


@beartype
def hybrid_seen_by_held_out_predictions(
    learned_latent: t.Tensor,
    learned_global_indices: t.Tensor,
    sequence_latent: t.Tensor,
    pairs: PairIndices,
    *,
    chunk_size: int,
) -> t.Tensor:
    """Predict learned-seen by sequence-held-out pairs with an exact global lookup."""

    if pairs.population != PairPopulation.SEEN_BY_HELD_OUT:
        raise ValueError("hybrid dual predictions require seen-by-held-out pairs")
    if chunk_size <= 0:
        raise ValueError("hybrid dual pair chunk size must be positive")
    if (
        learned_latent.ndim != 2
        or sequence_latent.ndim != 2
        or learned_latent.shape[1] != sequence_latent.shape[1]
        or learned_latent.shape[0] != learned_global_indices.numel()
    ):
        raise ValueError("hybrid dual latent axes differ")
    if (
        learned_global_indices.dtype != t.int64
        or learned_global_indices.ndim != 1
        or t.unique(learned_global_indices).numel() != learned_global_indices.numel()
    ):
        raise ValueError("hybrid dual learned global indices are invalid")
    if not (learned_latent.dtype == sequence_latent.dtype == t.float32):
        raise TypeError("hybrid dual latent matrices must both be float32")
    if any(
        not bool(t.isfinite(values).all().item()) for values in (learned_latent, sequence_latent)
    ):
        raise ValueError("hybrid dual latent matrices must be finite")
    for values in (learned_latent, sequence_latent):
        norms = t.linalg.vector_norm(values, dim=1)
        if not t.allclose(norms, t.ones_like(norms), atol=2.0e-5, rtol=0.0):
            raise ValueError("hybrid dual latent rows must have unit norm")
    if bool(
        t.any(
            (learned_global_indices < 0) | (learned_global_indices >= sequence_latent.shape[0])
        ).item()
    ):
        raise IndexError("hybrid dual learned global index is outside the sequence universe")
    lookup = t.full((sequence_latent.shape[0],), -1, dtype=t.int64)
    lookup[learned_global_indices] = t.arange(learned_global_indices.numel(), dtype=t.int64)
    local_left = lookup.index_select(0, pairs.left)
    if bool(t.any(local_left < 0).item()):
        raise ValueError("hybrid pair has a seen-side probe absent from the learned table")
    if bool(t.any(lookup.index_select(0, pairs.right) >= 0).item()):
        raise ValueError("hybrid held-out side unexpectedly has a learned-table row")
    return t.cat(
        tuple(
            t.sum(
                learned_latent.index_select(0, local_left[start:stop])
                * sequence_latent.index_select(0, pairs.right[start:stop]),
                dim=1,
            )
            for start in range(0, pairs.count, chunk_size)
            for stop in (min(start + chunk_size, pairs.count),)
        )
    )


@beartype
def held_out_age_predictions(
    sequence_latent: t.Tensor,
    age_direction: t.Tensor,
    held_out_indices: t.Tensor,
) -> t.Tensor:
    if sequence_latent.ndim != 2 or sequence_latent.dtype != t.float32:
        raise TypeError("dual age sequence latent must be a float32 matrix")
    if age_direction.shape != (sequence_latent.shape[1],) or age_direction.dtype != t.float32:
        raise TypeError("dual age direction must be an aligned float32 vector")
    if held_out_indices.dtype != t.int64 or held_out_indices.ndim != 1:
        raise TypeError("dual held-out indices must be an int64 vector")
    if bool(t.any((held_out_indices < 0) | (held_out_indices >= sequence_latent.shape[0])).item()):
        raise IndexError("dual held-out age index is outside the sequence universe")
    return sequence_latent.index_select(0, held_out_indices) @ normalize_vector_strict(
        age_direction
    )
