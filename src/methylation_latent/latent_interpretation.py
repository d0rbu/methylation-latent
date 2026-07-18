"""Invariant and reliability analyses for already fitted latent metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch as t
from beartype import beartype
from jaxtyping import Float, Int64, jaxtyped

from methylation_latent.artifacts import sha256_file
from methylation_latent.domain import (
    InfiniumDesign,
    ManifestStrand,
    NonEmptyProbeSet,
    parse_autosome,
    parse_genomic_context,
    parse_one_based_position,
)
from methylation_latent.manifest import iter_gpl13534_manifest_rows
from methylation_latent.model import normalize_rows_strict, normalize_vector_strict

ValueVector = Float[t.Tensor, "items"]
ValueMatrix = Float[t.Tensor, "items dimensions"]
IndexVector = Int64[t.Tensor, "items"]


def _require_vector(values: t.Tensor, name: str) -> None:
    if values.ndim != 1 or values.numel() == 0 or not values.is_floating_point():
        raise TypeError(f"{name} must be a non-empty floating vector")
    if not bool(t.isfinite(values).all().item()):
        raise ValueError(f"{name} must be finite")


def _require_indices(values: t.Tensor, upper: int, name: str) -> None:
    if values.dtype != t.int64 or values.ndim != 1 or values.numel() == 0:
        raise TypeError(f"{name} must be a non-empty int64 vector")
    if bool(t.any((values < 0) | (values >= upper)).item()):
        raise IndexError(f"{name} is outside its axis")
    if t.unique(values).numel() != values.numel():
        raise ValueError(f"{name} contains duplicates")


def _require_pair_axis(values: t.Tensor, upper: int, name: str) -> None:
    if values.dtype != t.int64 or values.ndim != 1 or values.numel() == 0:
        raise TypeError(f"{name} must be a non-empty int64 vector")
    if bool(t.any((values < 0) | (values >= upper)).item()):
        raise IndexError(f"{name} is outside its axis")


@jaxtyped(typechecker=beartype)
def strict_pearson(left: ValueVector, right: ValueVector) -> float:
    """Return Pearson correlation and reject constant or invalid vectors."""

    if left.shape != right.shape or left.numel() < 2:
        raise ValueError("Pearson inputs must be aligned and contain at least two values")
    _require_vector(left, "left Pearson input")
    _require_vector(right, "right Pearson input")
    left_working = left.to(t.float64)
    right_working = right.to(t.float64)
    left_centered = left_working - left_working.mean()
    right_centered = right_working - right_working.mean()
    denominator = t.linalg.vector_norm(left_centered) * t.linalg.vector_norm(right_centered)
    if float(denominator.item()) == 0.0:
        raise ValueError("Pearson correlation is undefined for a constant vector")
    value = float(t.dot(left_centered, right_centered).div(denominator).item())
    if not math.isfinite(value) or abs(value) > 1.0 + 1.0e-12:
        raise RuntimeError("Pearson correlation escaped its mathematical bounds")
    return value


@jaxtyped(typechecker=beartype)
def standardize_rows(values: ValueMatrix) -> t.Tensor:
    """Center and L2-normalize rows, rejecting constant inputs."""

    if values.ndim != 2 or 0 in values.shape or not values.is_floating_point():
        raise TypeError("standardization input must be a non-empty floating matrix")
    if not bool(t.isfinite(values).all().item()):
        raise ValueError("standardization input must be finite")
    centered = values - values.mean(dim=1, keepdim=True)
    norms = t.linalg.vector_norm(centered, dim=1, keepdim=True)
    if bool(t.any(norms == 0.0).item()):
        raise ValueError("cannot standardize a constant row")
    result = centered / norms
    tolerance = 1.0e-10 if result.dtype == t.float64 else 2.0e-5
    if not bool(
        t.allclose(result.mean(dim=1), t.zeros(result.shape[0], dtype=result.dtype), atol=tolerance)
        and t.allclose(
            t.linalg.vector_norm(result, dim=1),
            t.ones(result.shape[0], dtype=result.dtype),
            atol=tolerance,
            rtol=0.0,
        )
    ):
        raise RuntimeError("row standardization invariant failed")
    return result


@jaxtyped(typechecker=beartype)
def standardize_vector(values: ValueVector) -> t.Tensor:
    _require_vector(values, "standardization vector")
    centered = values - values.mean()
    norm = t.linalg.vector_norm(centered)
    if float(norm.item()) == 0.0:
        raise ValueError("cannot standardize a constant vector")
    result = centered / norm
    tolerance = 1.0e-10 if result.dtype == t.float64 else 2.0e-5
    if not bool(
        t.isclose(result.mean(), t.zeros((), dtype=result.dtype), atol=tolerance)
        and t.isclose(t.linalg.vector_norm(result), t.ones((), dtype=result.dtype), atol=tolerance)
    ):
        raise RuntimeError("vector standardization invariant failed")
    return result


@jaxtyped(typechecker=beartype)
def age_stratified_half_indices(age: ValueVector, *, seed: int) -> tuple[t.Tensor, t.Tensor]:
    """Pair adjacent sorted ages and randomly assign one subject to each equal half."""

    _require_vector(age, "age")
    if age.numel() % 2 != 0 or seed <= 0:
        raise ValueError("age-stratified halves require an even subject count and positive seed")
    ordered = t.argsort(age, stable=True).reshape(-1, 2)
    generator = t.Generator(device="cpu").manual_seed(seed)
    choices = t.randint(2, (ordered.shape[0],), generator=generator)
    rows = t.arange(ordered.shape[0], dtype=t.int64)
    first = t.sort(ordered[rows, choices]).values
    second = t.sort(ordered[rows, 1 - choices]).values
    expected = t.arange(age.numel(), dtype=t.int64)
    if first.numel() != second.numel() or not t.equal(
        t.sort(t.cat((first, second))).values, expected
    ):
        raise RuntimeError("age-stratified halves do not partition subjects exactly")
    if bool(t.isin(first, second).any().item()):
        raise RuntimeError("age-stratified halves overlap")
    return first, second


@beartype
def stratified_sample_indices(
    distance_class: t.Tensor,
    *,
    maximum_per_class: int,
    seed: int,
) -> t.Tensor:
    """Select a target-blind deterministic cap independently within each distance class."""

    if (
        distance_class.dtype != t.int64
        or distance_class.ndim != 1
        or distance_class.numel() == 0
        or maximum_per_class <= 0
        or seed <= 0
    ):
        raise ValueError("distance sampling inputs are invalid")
    if bool(t.any((distance_class < 0) | (distance_class > 7)).item()):
        raise ValueError("distance sampling contains an unknown class")
    generator = t.Generator(device="cpu").manual_seed(seed)
    selected: list[t.Tensor] = []
    for class_index in range(8):
        candidates = t.nonzero(distance_class == class_index).flatten()
        if candidates.numel() == 0:
            continue
        if candidates.numel() > maximum_per_class:
            local = t.randperm(candidates.numel(), generator=generator)[:maximum_per_class]
            candidates = candidates.index_select(0, local)
        selected.append(candidates)
    if not selected:
        raise RuntimeError("distance-stratified sample is unexpectedly empty")
    result = t.sort(t.cat(selected)).values
    if t.unique(result).numel() != result.numel():
        raise RuntimeError("distance-stratified sampling produced duplicates")
    return result


@beartype
def gather_pair_products(
    rows: t.Tensor,
    left: t.Tensor,
    right: t.Tensor,
    *,
    chunk_size: int,
) -> t.Tensor:
    if rows.ndim != 2 or 0 in rows.shape or not rows.is_floating_point():
        raise TypeError("pair-product rows must be a non-empty floating matrix")
    _require_pair_axis(left, rows.shape[0], "left pair indices")
    _require_pair_axis(right, rows.shape[0], "right pair indices")
    if left.shape != right.shape or chunk_size <= 0:
        raise ValueError("pair-product indices or chunk size differ")
    return t.cat(
        tuple(
            t.sum(
                rows.index_select(0, left[start:stop]) * rows.index_select(0, right[start:stop]),
                dim=1,
            )
            for start in range(0, left.numel(), chunk_size)
            for stop in (min(start + chunk_size, left.numel()),)
        )
    )


@dataclass(frozen=True, slots=True)
class PairDecomposition:
    target_total: t.Tensor
    target_age: t.Tensor
    target_residual: t.Tensor
    prediction_total: t.Tensor
    prediction_age: t.Tensor
    prediction_residual: t.Tensor

    def __post_init__(self) -> None:
        vectors = (
            self.target_total,
            self.target_age,
            self.target_residual,
            self.prediction_total,
            self.prediction_age,
            self.prediction_residual,
        )
        if any(
            vector.dtype != t.float64
            or vector.ndim != 1
            or vector.shape != self.target_total.shape
            or not bool(t.isfinite(vector).all().item())
            for vector in vectors
        ):
            raise ValueError("pair-decomposition vectors must be aligned finite float64 values")
        if not bool(
            t.allclose(
                self.target_total,
                self.target_age + self.target_residual,
                atol=2.0e-12,
                rtol=0.0,
            )
            and t.allclose(
                self.prediction_total,
                self.prediction_age + self.prediction_residual,
                atol=2.0e-6,
                rtol=0.0,
            )
        ):
            raise ValueError("pair decomposition does not reconstruct its total")


@beartype
def exact_pair_decomposition(
    latent: t.Tensor,
    age_direction: t.Tensor,
    empirical_rows: t.Tensor,
    empirical_age: t.Tensor,
    empirical_rho: t.Tensor,
    target_total: t.Tensor,
    left: t.Tensor,
    right: t.Tensor,
    *,
    chunk_size: int,
) -> PairDecomposition:
    """Decompose predicted and empirical pairs and independently assert both identities."""

    if latent.dtype != t.float32 or empirical_rows.dtype != t.float64:
        raise TypeError("decomposition requires float32 latents and float64 empirical rows")
    if latent.shape[0] != empirical_rows.shape[0] or empirical_rho.shape != (latent.shape[0],):
        raise ValueError("decomposition probe axes differ")
    if empirical_age.dtype != t.float64 or empirical_age.shape != (empirical_rows.shape[1],):
        raise ValueError("decomposition empirical age axis differs")
    _require_vector(target_total, "pair targets")
    if target_total.dtype != t.float64 or target_total.shape != left.shape:
        raise ValueError("pair targets must be aligned float64 values")
    normalized_latent = normalize_rows_strict(latent)
    normalized_age = normalize_vector_strict(age_direction)
    predicted_rho = normalized_latent @ normalized_age
    latent_residual = normalized_latent - predicted_rho[:, None] * normalized_age[None, :]
    prediction_total = gather_pair_products(
        normalized_latent, left, right, chunk_size=chunk_size
    ).to(t.float64)
    prediction_age = (
        predicted_rho.index_select(0, left) * predicted_rho.index_select(0, right)
    ).to(t.float64)
    prediction_residual = gather_pair_products(
        latent_residual, left, right, chunk_size=chunk_size
    ).to(t.float64)
    empirical_residual_chunks: list[t.Tensor] = []
    for start in range(0, left.numel(), chunk_size):
        stop = min(start + chunk_size, left.numel())
        local_left = left[start:stop]
        local_right = right[start:stop]
        left_residual = (
            empirical_rows.index_select(0, local_left)
            - empirical_rho.index_select(0, local_left)[:, None] * empirical_age[None, :]
        )
        right_residual = (
            empirical_rows.index_select(0, local_right)
            - empirical_rho.index_select(0, local_right)[:, None] * empirical_age[None, :]
        )
        empirical_residual_chunks.append(t.sum(left_residual * right_residual, dim=1))
    target_residual = t.cat(empirical_residual_chunks)
    target_age = empirical_rho.index_select(0, left) * empirical_rho.index_select(0, right)
    decomposition = PairDecomposition(
        target_total=target_total,
        target_age=target_age,
        target_residual=target_residual,
        prediction_total=prediction_total,
        prediction_age=prediction_age,
        prediction_residual=prediction_residual,
    )
    if not bool(t.allclose(target_total, target_age + target_residual, atol=2.0e-12, rtol=0.0)):
        raise RuntimeError("independent empirical age decomposition failed")
    return decomposition


@beartype
def split_half_targets(
    beta: t.Tensor,
    age: t.Tensor,
    probe_indices: t.Tensor,
    sample_indices: t.Tensor,
) -> tuple[t.Tensor, t.Tensor]:
    if beta.dtype != t.float64 or beta.ndim != 2 or age.dtype != t.float64:
        raise TypeError("split-half targets require float64 beta and age")
    if age.shape != (beta.shape[1],):
        raise ValueError("split-half age and beta sample axes differ")
    _require_indices(probe_indices, beta.shape[0], "split-half probe indices")
    _require_indices(sample_indices, beta.shape[1], "split-half sample indices")
    rows = standardize_rows(beta.index_select(0, probe_indices).index_select(1, sample_indices))
    standardized_age = standardize_vector(age.index_select(0, sample_indices))
    rho = rows @ standardized_age
    if bool(t.any(rho.abs() > 1.0 + 1.0e-12).item()):
        raise RuntimeError("split-half age correlations escaped [-1,1]")
    return rows, rho


@beartype
def spearman_brown(split_half_correlation: float) -> float:
    if not math.isfinite(split_half_correlation) or not -1.0 <= split_half_correlation <= 1.0:
        raise ValueError("split-half correlation must lie in [-1,1]")
    denominator = 1.0 + split_half_correlation
    if denominator == 0.0:
        raise ValueError("Spearman-Brown is undefined at correlation -1")
    return 2.0 * split_half_correlation / denominator


@beartype
def participation_rank(eigenvalues: t.Tensor) -> float:
    _require_vector(eigenvalues, "eigenvalues")
    if bool(t.any(eigenvalues < -1.0e-10).item()):
        raise ValueError("participation rank requires non-negative eigenvalues")
    values = eigenvalues.clamp_min(0.0)
    denominator = t.sum(values.square())
    if float(denominator.item()) == 0.0:
        raise ValueError("zero spectrum has no participation rank")
    return float(t.sum(values).square().div(denominator).item())


@beartype
def normalized_frobenius_cosine(left: t.Tensor, right: t.Tensor) -> float:
    if left.shape != right.shape or left.numel() == 0 or not left.is_floating_point():
        raise ValueError("Frobenius inputs must be aligned non-empty floating tensors")
    if right.dtype != left.dtype or not bool(t.isfinite(left).all() and t.isfinite(right).all()):
        raise ValueError("Frobenius inputs must have matching finite dtypes")
    denominator = t.linalg.vector_norm(left) * t.linalg.vector_norm(right)
    if float(denominator.item()) == 0.0:
        raise ValueError("zero tensor has undefined Frobenius cosine")
    return float(t.sum(left * right).div(denominator).item())


@beartype
def linear_cka(left: t.Tensor, right: t.Tensor) -> float:
    """Compute rotation-invariant linear CKA without materializing probe Gram matrices."""

    if (
        left.ndim != 2
        or right.ndim != 2
        or left.shape[0] != right.shape[0]
        or left.shape[0] < 2
        or left.dtype != right.dtype
        or not left.is_floating_point()
    ):
        raise ValueError("CKA inputs must be aligned floating matrices")
    left_centered = left - left.mean(dim=0, keepdim=True)
    right_centered = right - right.mean(dim=0, keepdim=True)
    cross = left_centered.mT @ right_centered
    left_covariance = left_centered.mT @ left_centered
    right_covariance = right_centered.mT @ right_centered
    denominator = t.linalg.matrix_norm(left_covariance) * t.linalg.matrix_norm(right_covariance)
    if float(denominator.item()) == 0.0:
        raise ValueError("constant representation has undefined CKA")
    value = float(t.linalg.matrix_norm(cross).square().div(denominator).item())
    if not -1.0e-12 <= value <= 1.0 + 1.0e-8:
        raise RuntimeError("linear CKA escaped [0,1]")
    return value


@beartype
def mean_neighbour_overlap(left: t.Tensor, right: t.Tensor, *, neighbours: int) -> float:
    if (
        left.ndim != 2
        or right.ndim != 2
        or left.shape[0] != right.shape[0]
        or left.shape[0] <= neighbours
        or neighbours <= 0
    ):
        raise ValueError("neighbour-overlap inputs or count differ")
    left_normalized = normalize_rows_strict(left)
    right_normalized = normalize_rows_strict(right)
    left_similarity = left_normalized @ left_normalized.mT
    right_similarity = right_normalized @ right_normalized.mT
    diagonal = t.arange(left.shape[0])
    left_similarity[diagonal, diagonal] = -t.inf
    right_similarity[diagonal, diagonal] = -t.inf
    left_neighbours = t.topk(left_similarity, k=neighbours, dim=1).indices
    right_neighbours = t.topk(right_similarity, k=neighbours, dim=1).indices
    overlap = (left_neighbours[:, :, None] == right_neighbours[:, None, :]).any(dim=2).sum(dim=1)
    return float(overlap.to(t.float64).mean().div(neighbours).item())


@beartype
def deterministic_indices(size: int, maximum: int, *, seed: int) -> t.Tensor:
    if size <= 0 or maximum <= 0 or seed <= 0:
        raise ValueError("deterministic sample inputs must be positive")
    if size <= maximum:
        return t.arange(size, dtype=t.int64)
    generator = t.Generator(device="cpu").manual_seed(seed)
    return t.sort(t.randperm(size, generator=generator)[:maximum]).values


@beartype
def stable_extreme_indices(
    first: t.Tensor,
    second: t.Tensor,
    *,
    positive: bool,
    maximum: int,
) -> t.Tensor:
    """Rank same-sign cross-window predictions without reading empirical targets."""

    _require_vector(first, "first candidate prediction")
    _require_vector(second, "second candidate prediction")
    if first.shape != second.shape or maximum <= 0:
        raise ValueError("candidate predictions or maximum differ")
    eligible = (first > 0.0) & (second > 0.0) if positive else (first < 0.0) & (second < 0.0)
    indices = t.nonzero(eligible).flatten()
    if indices.numel() < maximum:
        raise ValueError("not enough same-sign predictions for the candidate table")
    score = t.minimum(first.abs(), second.abs()).index_select(0, indices)
    order = t.argsort(score, descending=True, stable=True)[:maximum]
    return indices.index_select(0, order)


@beartype
def fit_linear_surrogate(train_features: t.Tensor, train_target: t.Tensor) -> t.Tensor:
    if (
        train_features.dtype != t.float64
        or train_features.ndim != 2
        or train_target.dtype != t.float64
        or train_target.shape != (train_features.shape[0],)
        or train_features.shape[0] <= train_features.shape[1]
    ):
        raise ValueError("linear-surrogate training axes or dtypes differ")
    if not bool(t.isfinite(train_features).all() and t.isfinite(train_target).all()):
        raise ValueError("linear-surrogate training values must be finite")
    rank = int(t.linalg.matrix_rank(train_features).item())
    if rank != train_features.shape[1]:
        raise ValueError("linear-surrogate design is not full column rank")
    coefficients = t.linalg.lstsq(train_features, train_target[:, None]).solution[:, 0]
    if not bool(t.isfinite(coefficients).all().item()):
        raise RuntimeError("linear-surrogate coefficients are not finite")
    return coefficients


@dataclass(frozen=True, slots=True)
class ManifestInterpretationAnnotation:
    refgene_names: tuple[str, ...]
    refgene_groups: tuple[str, ...]
    island_relation: str
    enhancer: bool
    regulatory_feature_group: str
    dhs: bool

    @property
    def regulatory(self) -> bool:
        return self.regulatory_feature_group != ""


@dataclass(frozen=True, slots=True)
class ManifestAnnotationAudit:
    source_sha256: str
    requested_probes: int
    matched_probes: int

    def __post_init__(self) -> None:
        if self.requested_probes <= 0 or self.matched_probes != self.requested_probes:
            raise ValueError("manifest annotation join is incomplete")
        if len(self.source_sha256) != 64:
            raise ValueError("manifest annotation source hash is invalid")


def _semicolon_values(raw: str) -> tuple[str, ...]:
    return tuple(value for value in raw.split(";") if value)


def _manifest_boolean(raw: str, name: str) -> bool:
    if raw not in {"", "TRUE"}:
        raise ValueError(f"unexpected GPL13534 {name} value: {raw!r}")
    return raw == "TRUE"


@beartype
def load_manifest_interpretation_annotations(
    path: Path,
    probes: NonEmptyProbeSet,
    *,
    expected_sha256: str,
) -> tuple[dict[str, ManifestInterpretationAnnotation], ManifestAnnotationAudit]:
    """Strictly join retained loci to the exact GPL13534 build-37 annotation rows."""

    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise ValueError("GPL13534 annotation source hash differs")
    requested = {str(probe.probe_id): probe for probe in probes.probes}
    annotations: dict[str, ManifestInterpretationAnnotation] = {}
    for row in iter_gpl13534_manifest_rows(path):
        probe_id = row["IlmnID"]
        probe = requested.get(probe_id)
        if probe is None:
            continue
        if probe_id in annotations:
            raise ValueError(f"duplicate retained annotation row: {probe_id}")
        if (
            row["Name"] != probe_id
            or row["Genome_Build"] != "37"
            or parse_autosome(row["CHR"]) != probe.chromosome
            or parse_one_based_position(row["MAPINFO"]) != probe.position
            or parse_genomic_context(
                row["Relation_to_UCSC_CpG_Island"], row["UCSC_CpG_Islands_Name"]
            )
            != probe.context
            or InfiniumDesign(row["Infinium_Design_Type"]) != probe.design
            or ManifestStrand(row["Strand"]) != probe.manifest_strand
        ):
            raise ValueError(f"GPL13534 annotation identity differs for {probe_id}")
        annotations[probe_id] = ManifestInterpretationAnnotation(
            refgene_names=_semicolon_values(row["UCSC_RefGene_Name"]),
            refgene_groups=_semicolon_values(row["UCSC_RefGene_Group"]),
            island_relation=row["Relation_to_UCSC_CpG_Island"],
            enhancer=_manifest_boolean(row["Enhancer"], "Enhancer"),
            regulatory_feature_group=row["Regulatory_Feature_Group"],
            dhs=_manifest_boolean(row["DHS"], "DHS"),
        )
    missing = set(requested) - set(annotations)
    unknown = set(annotations) - set(requested)
    if missing or unknown:
        raise ValueError(
            "GPL13534 annotation join differs: "
            f"missing={sorted(missing)[:10]}, unknown={sorted(unknown)[:10]}"
        )
    audit = ManifestAnnotationAudit(
        source_sha256=observed_sha256,
        requested_probes=len(requested),
        matched_probes=len(annotations),
    )
    return annotations, audit
