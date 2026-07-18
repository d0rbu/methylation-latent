"""Strict geometry for the post-hoc protein extension."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch as t
from beartype import beartype
from jaxtyping import Bool, Float, Float64, Int64, jaxtyped

from methylation_latent.domain import NonNegativeWeight, ProbeLocus, WindowSize
from methylation_latent.genome import GenomicInterval
from methylation_latent.model import normalize_rows_strict
from methylation_latent.targets import (
    CorrelationMatrix,
    CorrelationVector,
    UnitNormRows,
    UnitNormVector,
)

MeasurementMatrix = Float64[t.Tensor, "measurements samples"]
ProteinFeatures = Float[t.Tensor, "proteins features"]
ProbeLatentRows = Float[t.Tensor, "probes latent"]
ProteinLatentRows = Float[t.Tensor, "proteins latent"]
IndexVector = Int64[t.Tensor, "selected"]
UniquePairMask = Bool[t.Tensor, "proteins proteins"]


def _require_finite(values: t.Tensor, name: str) -> None:
    if not bool(t.isfinite(values).all().item()):
        raise ValueError(f"{name} must contain only finite values")


@jaxtyped(typechecker=beartype)
def unique_unordered_pair_mask(
    protein_count: int, *, device: str | t.device = "cpu"
) -> UniquePairMask:
    """Select each unordered off-diagonal protein pair exactly once."""

    if protein_count < 2:
        raise ValueError("at least two proteins are required to form a pair")
    return t.triu(
        t.ones((protein_count, protein_count), dtype=t.bool, device=device),
        diagonal=1,
    )


@dataclass(frozen=True, slots=True)
class ProteinPanel:
    """A complete, aligned protein-by-subject matrix with explicit identities."""

    display_names: tuple[str, ...]
    gene_symbols: tuple[str, ...]
    uniprot_accessions: tuple[str, ...]
    gsm_accessions: tuple[str, ...]
    values: t.Tensor

    def __post_init__(self) -> None:
        protein_count = len(self.display_names)
        axes = (self.display_names, self.gene_symbols, self.uniprot_accessions)
        if protein_count < 2 or any(len(axis) != protein_count for axis in axes):
            raise ValueError("protein identity axes must align and contain at least two proteins")
        if any(len(set(axis)) != len(axis) or any(not value for value in axis) for axis in axes):
            raise ValueError("protein identity axes must be non-empty and unique")
        if len(self.gsm_accessions) < 3 or len(set(self.gsm_accessions)) != len(
            self.gsm_accessions
        ):
            raise ValueError("protein panel requires at least three unique GSM accessions")
        if self.values.shape != (protein_count, len(self.gsm_accessions)):
            raise ValueError("protein values do not align to their protein and subject axes")
        if self.values.dtype != t.float64:
            raise TypeError(f"protein values must be float64, got {self.values.dtype}")
        _require_finite(self.values, "protein values")


@dataclass(frozen=True, slots=True)
class CorrelationRectangle:
    """A finite non-empty rectangular collection of bounded correlations."""

    tensor: t.Tensor

    def __post_init__(self) -> None:
        if self.tensor.ndim != 2 or 0 in self.tensor.shape:
            raise ValueError("correlation rectangle must be non-empty and rank two")
        if self.tensor.dtype not in {t.float32, t.float64}:
            raise TypeError("correlation rectangle must be float32 or float64")
        _require_finite(self.tensor, "correlation rectangle")
        tolerance = 1.0e-10 if self.tensor.dtype == t.float64 else 2.0e-5
        if not bool(t.all(self.tensor.abs() <= 1.0 + tolerance).item()):
            maximum = float(self.tensor.abs().max().item())
            raise ValueError(f"correlation rectangle magnitude exceeds one: {maximum}")


@dataclass(frozen=True, slots=True)
class ProteinTargets:
    """All empirical edges on one common subject axis."""

    standardized_proteins: UnitNormRows
    protein_gram: CorrelationMatrix
    direct_age: CorrelationVector
    probe_protein: CorrelationRectangle

    def __post_init__(self) -> None:
        protein_count = self.standardized_proteins.n_rows
        if self.protein_gram.tensor.shape != (protein_count, protein_count):
            raise ValueError("protein Gram matrix does not align to standardized proteins")
        if self.direct_age.tensor.shape != (protein_count,):
            raise ValueError("direct protein-age correlations do not align to proteins")
        if self.probe_protein.tensor.shape[1] != protein_count:
            raise ValueError("probe-protein correlations do not align to proteins")


@jaxtyped(typechecker=beartype)
def standardize_measurement_rows(values: MeasurementMatrix) -> UnitNormRows:
    """Standardize unbounded continuous measurements without beta-value assumptions."""

    if values.ndim != 2 or 0 in values.shape:
        raise ValueError("measurement matrix must have non-empty axes")
    _require_finite(values, "measurement matrix")
    centered = values - values.mean(dim=1, keepdim=True)
    norms = t.linalg.vector_norm(centered, dim=1, keepdim=True)
    if bool(t.any(norms == 0.0).item()):
        indices = t.nonzero(norms.squeeze(1) == 0.0).flatten().tolist()
        raise ValueError(f"constant measurement rows cannot be standardized: {indices[:10]}")
    return UnitNormRows(centered / norms)


@beartype
def build_protein_targets(
    probe_rows: UnitNormRows,
    protein_values: t.Tensor,
    age: UnitNormVector,
) -> ProteinTargets:
    """Build each correlation by a dot product on one asserted common subject axis."""

    proteins = standardize_measurement_rows(protein_values)
    if probe_rows.n_samples != proteins.n_samples or proteins.n_samples != age.tensor.numel():
        raise ValueError("probe, protein, and age subject axes do not match")
    protein_gram = CorrelationMatrix(proteins.tensor @ proteins.tensor.mT)
    direct_age = CorrelationVector(proteins.tensor @ age.tensor)
    probe_protein = CorrelationRectangle(probe_rows.tensor @ proteins.tensor.mT)
    return ProteinTargets(
        standardized_proteins=proteins,
        protein_gram=protein_gram,
        direct_age=direct_age,
        probe_protein=probe_protein,
    )


@dataclass(frozen=True, slots=True)
class ProteinSplit:
    """A target-blind exhaustive protein partition."""

    protein_count: int
    train_indices: t.Tensor
    validation_indices: t.Tensor
    test_indices: t.Tensor

    def __post_init__(self) -> None:
        partitions = (self.train_indices, self.validation_indices, self.test_indices)
        if self.protein_count < 3:
            raise ValueError("protein split requires at least three proteins")
        if any(values.dtype != t.int64 or values.ndim != 1 for values in partitions):
            raise TypeError("protein split partitions must be int64 vectors")
        if any(values.numel() == 0 for values in partitions):
            raise ValueError("protein split partitions must be non-empty")
        joined = t.cat(partitions)
        if joined.numel() != self.protein_count or not t.equal(
            t.sort(joined).values,
            t.arange(self.protein_count, dtype=t.int64),
        ):
            raise ValueError("protein split must exactly and disjointly cover its universe")

    @property
    def refit_indices(self) -> t.Tensor:
        return t.cat((self.train_indices, self.validation_indices))


@beartype
def target_blind_protein_split(
    gene_symbols: tuple[str, ...],
    *,
    validation_count: int,
    test_count: int,
    seed: int,
) -> ProteinSplit:
    """Partition only from gene identity and a frozen seed, never target values."""

    protein_count = len(gene_symbols)
    if len(set(gene_symbols)) != protein_count or any(not symbol for symbol in gene_symbols):
        raise ValueError("gene symbols must be non-empty and unique")
    if min(validation_count, test_count) <= 0 or validation_count + test_count >= protein_count:
        raise ValueError("validation and test counts must leave a non-empty train partition")
    ranked = sorted(
        range(protein_count),
        key=lambda index: hashlib.sha256(
            f"{seed}:{gene_symbols[index]}".encode()
        ).digest(),
    )
    test = ranked[:test_count]
    validation = ranked[test_count : test_count + validation_count]
    train = ranked[test_count + validation_count :]
    return ProteinSplit(
        protein_count=protein_count,
        train_indices=t.tensor(train, dtype=t.int64),
        validation_indices=t.tensor(validation, dtype=t.int64),
        test_indices=t.tensor(test, dtype=t.int64),
    )


class LinearProteinMapper(t.nn.Module):
    """A bias-free shared map from fixed protein features into probe latent space."""

    def __init__(self, *, feature_dimension: int, latent_dimension: int) -> None:
        super().__init__()
        if min(feature_dimension, latent_dimension) <= 0:
            raise ValueError("feature and latent dimensions must be positive")
        self.feature_dimension = feature_dimension
        self.latent_dimension = latent_dimension
        self.projection = t.nn.Linear(feature_dimension, latent_dimension, bias=False)
        t.nn.init.orthogonal_(self.projection.weight)

    @jaxtyped(typechecker=beartype)
    def latent(self, features: ProteinFeatures) -> t.Tensor:
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("protein features must be a non-empty matrix")
        if features.shape[1] != self.feature_dimension:
            raise ValueError(
                f"expected protein feature width {self.feature_dimension}, got {features.shape[1]}"
            )
        if features.dtype != self.projection.weight.dtype:
            raise TypeError("protein feature and mapper dtypes differ")
        _require_finite(features, "protein features")
        return normalize_rows_strict(self.projection(features))


class FreeProteinVectors(t.nn.Module):
    """Transductive protein vectors: an upper bound, not an inductive sequence model."""

    def __init__(self, *, protein_count: int, latent_dimension: int) -> None:
        super().__init__()
        if min(protein_count, latent_dimension) <= 0:
            raise ValueError("protein count and latent dimension must be positive")
        self.vectors = t.nn.Parameter(t.empty((protein_count, latent_dimension)))
        t.nn.init.normal_(self.vectors)

    def latent(self) -> t.Tensor:
        return normalize_rows_strict(self.vectors)


@dataclass(frozen=True, slots=True)
class ProteinObjectiveTerms:
    total: t.Tensor
    cross_mse: t.Tensor
    protein_pair_mse: t.Tensor

    def __post_init__(self) -> None:
        for name, value in (
            ("total", self.total),
            ("cross_mse", self.cross_mse),
            ("protein_pair_mse", self.protein_pair_mse),
        ):
            if value.ndim != 0 or not value.is_floating_point() or not bool(
                t.isfinite(value).item()
            ):
                raise ValueError(f"{name} must be a finite floating scalar")


@jaxtyped(typechecker=beartype)
def protein_objective(
    probe_latent: ProbeLatentRows,
    protein_latent: ProteinLatentRows,
    cross_target: t.Tensor,
    protein_gram_target: t.Tensor,
    *,
    lambda_protein_pairs: NonNegativeWeight,
) -> ProteinObjectiveTerms:
    """Average cross and off-diagonal protein-pair blocks separately."""

    probes = normalize_rows_strict(probe_latent)
    proteins = normalize_rows_strict(protein_latent)
    expected_cross_shape = (probes.shape[0], proteins.shape[0])
    expected_gram_shape = (proteins.shape[0], proteins.shape[0])
    if cross_target.shape != expected_cross_shape:
        raise ValueError("cross target shape differs from latent axes")
    if protein_gram_target.shape != expected_gram_shape:
        raise ValueError("protein Gram target shape differs from protein axis")
    if proteins.shape[0] < 2:
        raise ValueError("off-diagonal protein-pair loss requires at least two proteins")
    if cross_target.dtype != probes.dtype or protein_gram_target.dtype != probes.dtype:
        raise TypeError("latent predictions and protein targets must share one dtype")
    _require_finite(cross_target, "cross target")
    _require_finite(protein_gram_target, "protein Gram target")
    cross_mse = t.square(probes @ proteins.mT - cross_target).mean()
    mask = ~t.eye(proteins.shape[0], dtype=t.bool, device=proteins.device)
    pair_mse = t.square(proteins @ proteins.mT - protein_gram_target)[mask].mean()
    return ProteinObjectiveTerms(
        total=cross_mse + float(lambda_protein_pairs) * pair_mse,
        cross_mse=cross_mse,
        protein_pair_mse=pair_mse,
    )


@beartype
@dataclass(frozen=True, slots=True)
class GeneLocus:
    """One Ensembl GRCh37 gene interval with its transcriptional strand."""

    gene_symbol: str
    chromosome: str
    start: int
    end: int
    strand: int

    def __post_init__(self) -> None:
        if not self.gene_symbol or not self.chromosome:
            raise ValueError("gene symbol and chromosome must not be empty")
        if self.start <= 0 or self.end < self.start:
            raise ValueError("gene coordinates must be one-based inclusive and ordered")
        if self.strand not in {-1, 1}:
            raise ValueError("gene strand must be -1 or 1")

    @property
    def transcription_start(self) -> int:
        return self.start if self.strand == 1 else self.end


@beartype
def tss_window_interval(
    gene: GeneLocus,
    window_size: WindowSize,
    *,
    chromosome_length: int,
) -> GenomicInterval:
    """Return a plus-reference-strand window centred at the GRCh37 gene TSS."""

    tss_zero = gene.transcription_start - 1
    start = tss_zero - int(window_size) // 2
    end = start + int(window_size)
    if start < 0 or end > chromosome_length:
        raise ValueError(f"gene {gene.gene_symbol} cannot provide a complete TSS window")
    return GenomicInterval(f"chr{gene.chromosome}", start, end)


@beartype
def cpg_protein_windows_overlap(
    probe: ProbeLocus,
    gene: GeneLocus,
    window_size: WindowSize,
    *,
    chromosome_length: int,
) -> bool:
    cytosine_zero = int(probe.position) - 1
    start = cytosine_zero - int(window_size) // 2
    probe_interval = GenomicInterval(
        f"chr{int(probe.chromosome)}",
        start,
        start + int(window_size),
    )
    return probe_interval.overlaps(
        tss_window_interval(gene, window_size, chromosome_length=chromosome_length)
    )


@beartype
def protein_chunk_ranges(residue_count: int, *, maximum_residues: int) -> tuple[range, ...]:
    """Partition every residue exactly once; long proteins are never truncated."""

    if residue_count <= 0 or maximum_residues <= 0:
        raise ValueError("residue count and maximum chunk width must be positive")
    chunks = tuple(
        range(start, min(start + maximum_residues, residue_count))
        for start in range(0, residue_count, maximum_residues)
    )
    flattened = tuple(index for chunk in chunks for index in chunk)
    if flattened != tuple(range(residue_count)):  # pragma: no cover - construction proof
        raise RuntimeError("protein chunking did not cover every residue exactly once")
    return chunks
