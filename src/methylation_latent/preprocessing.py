"""Detection-p/sample completeness gates for externally normalized HM450 data."""

from __future__ import annotations

from dataclasses import dataclass

import torch as t
from beartype import beartype
from jaxtyping import Bool, Float64, jaxtyped

from methylation_latent.domain import Fraction

BetaMatrix = Float64[t.Tensor, "probes samples"]
DetectionPMatrix = Float64[t.Tensor, "probes samples"]
QualityExclusionMatrix = Bool[t.Tensor, "probes samples"]
PARITY_ARRAY_COUNT = 12
PARITY_DETECTION_AGREEMENT_MINIMUM = 0.999
PARITY_PER_ARRAY_DETECTION_AGREEMENT_MINIMUM = 0.995
PARITY_BETA_MEDIAN_ABSOLUTE_ERROR_MAXIMUM = 0.001
PARITY_BETA_999_QUANTILE_ERROR_MAXIMUM = 0.01
PARITY_PER_ARRAY_BETA_CORRELATION_MINIMUM = 0.999


def _validate_probability_matrix(values: t.Tensor, name: str) -> None:
    if values.dtype != t.float64 or values.ndim != 2 or 0 in values.shape:
        raise ValueError(f"{name} must be a non-empty rank-2 float64 tensor")
    if not bool(t.isfinite(values).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    if not bool(t.all((values >= 0.0) & (values <= 1.0)).item()):
        raise ValueError(f"{name} values must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class DetectionQcResult:
    """Complete beta matrix after sample-first then all-sample probe QC."""

    beta: t.Tensor
    retained_probe_indices: t.Tensor
    retained_sample_indices: t.Tensor
    excluded_probe_indices: t.Tensor
    excluded_sample_indices: t.Tensor
    sample_failure_fractions: t.Tensor

    def __post_init__(self) -> None:
        index_vectors = (
            self.retained_probe_indices,
            self.retained_sample_indices,
            self.excluded_probe_indices,
            self.excluded_sample_indices,
        )
        if any(vector.dtype != t.int64 or vector.ndim != 1 for vector in index_vectors):
            raise ValueError("QC index artifacts must be int64 vectors")
        if self.beta.shape != (
            self.retained_probe_indices.numel(),
            self.retained_sample_indices.numel(),
        ):
            raise ValueError("QC beta shape does not match retained axes")
        _validate_probability_matrix(self.beta, "QC beta matrix")
        if (
            self.sample_failure_fractions.dtype != t.float64
            or self.sample_failure_fractions.ndim != 1
        ):
            raise ValueError("sample failure fractions must be a float64 vector")


@dataclass(frozen=True, slots=True)
class ProcessorParityAudit:
    """Pre-registered methylprep-versus-seSAMe parity measurements."""

    detection_agreement: float
    minimum_per_array_detection_agreement: float
    beta_median_absolute_error: float
    beta_999_quantile_absolute_error: float
    minimum_per_array_beta_correlation: float
    retained_samples: int
    retained_probes: int


def _column_correlations(left: t.Tensor, right: t.Tensor) -> t.Tensor:
    left_centered = left - left.mean(dim=0, keepdim=True)
    right_centered = right - right.mean(dim=0, keepdim=True)
    left_norms = t.linalg.vector_norm(left_centered, dim=0)
    right_norms = t.linalg.vector_norm(right_centered, dim=0)
    if bool(t.any((left_norms == 0.0) | (right_norms == 0.0)).item()):
        raise ValueError("processor parity beta columns must not be constant")
    return t.sum(left_centered * right_centered, dim=0) / (left_norms * right_norms)


@jaxtyped(typechecker=beartype)
def assert_processor_parity(
    methylprep_beta: BetaMatrix,
    sesame_beta: BetaMatrix,
    methylprep_detection_p: DetectionPMatrix,
    sesame_detection_p: DetectionPMatrix,
    methylprep_quality_excluded: QualityExclusionMatrix,
    sesame_quality_excluded: QualityExclusionMatrix,
    *,
    detection_threshold: Fraction,
    maximum_sample_failure_fraction: Fraction,
) -> ProcessorParityAudit:
    """Require the frozen cross-implementation gate before methylprep can be primary."""

    matrices = (
        (methylprep_beta, "methylprep beta"),
        (sesame_beta, "seSAMe beta"),
        (methylprep_detection_p, "methylprep detection-p"),
        (sesame_detection_p, "seSAMe detection-p"),
    )
    for values, name in matrices:
        _validate_probability_matrix(values, name)
    shapes = {tuple(values.shape) for values, _ in matrices}
    if len(shapes) != 1:
        raise ValueError("processor parity matrices must have identical probe/sample axes")
    expected_shape = methylprep_beta.shape
    quality_matrices = (
        (methylprep_quality_excluded, "methylprep quality exclusions"),
        (sesame_quality_excluded, "seSAMe quality exclusions"),
    )
    for values, name in quality_matrices:
        if values.dtype != t.bool or values.shape != expected_shape:
            raise ValueError(f"{name} must be a probe/sample-aligned boolean matrix")
    if methylprep_beta.shape[1] != PARITY_ARRAY_COUNT:
        raise ValueError(f"processor parity requires exactly {PARITY_ARRAY_COUNT} arrays")

    methylprep_quality_retained = ~methylprep_quality_excluded.any(dim=1)
    sesame_quality_retained = ~sesame_quality_excluded.any(dim=1)
    if not t.equal(methylprep_quality_retained, sesame_quality_retained):
        raise ValueError("methylprep and seSAMe quality-mask complete-probe sets differ")
    quality_indices = t.nonzero(methylprep_quality_retained).flatten()
    if quality_indices.numel() < 2:
        raise ValueError("processor parity quality masks retained fewer than two probes")
    methylprep_passed = methylprep_detection_p.index_select(0, quality_indices) < float(
        detection_threshold
    )
    sesame_passed = sesame_detection_p.index_select(0, quality_indices) < float(detection_threshold)
    agreement = methylprep_passed == sesame_passed
    overall_agreement = float(agreement.to(t.float64).mean().item())
    per_array_agreement = agreement.to(t.float64).mean(dim=0)
    minimum_array_agreement = float(per_array_agreement.min().item())

    sample_limit = float(maximum_sample_failure_fraction)
    methylprep_samples = (~methylprep_passed).to(t.float64).mean(dim=0) <= sample_limit
    sesame_samples = (~sesame_passed).to(t.float64).mean(dim=0) <= sample_limit
    if not t.equal(methylprep_samples, sesame_samples):
        raise ValueError("methylprep and seSAMe parity sample-retention decisions differ")
    retained_sample_indices = t.nonzero(methylprep_samples).flatten()
    if retained_sample_indices.numel() < 2:
        raise ValueError("processor parity retained fewer than two arrays")
    methylprep_probes = methylprep_passed.index_select(1, retained_sample_indices).all(dim=1)
    sesame_probes = sesame_passed.index_select(1, retained_sample_indices).all(dim=1)
    if not t.equal(methylprep_probes, sesame_probes):
        raise ValueError("methylprep and seSAMe parity complete-probe sets differ")
    retained_probe_indices = quality_indices.index_select(
        0,
        t.nonzero(methylprep_probes).flatten(),
    )
    if retained_probe_indices.numel() < 2:
        raise ValueError("processor parity retained fewer than two probes")

    methylprep_retained = methylprep_beta.index_select(0, retained_probe_indices).index_select(
        1, retained_sample_indices
    )
    sesame_retained = sesame_beta.index_select(0, retained_probe_indices).index_select(
        1, retained_sample_indices
    )
    absolute_error = (methylprep_retained - sesame_retained).abs()
    median_error = float(absolute_error.median().item())
    quantile_error = float(t.quantile(absolute_error.flatten(), 0.999).item())
    minimum_beta_correlation = float(
        _column_correlations(methylprep_retained, sesame_retained).min().item()
    )
    threshold_checks = (
        (
            overall_agreement >= PARITY_DETECTION_AGREEMENT_MINIMUM,
            "overall detection-call agreement",
        ),
        (
            minimum_array_agreement >= PARITY_PER_ARRAY_DETECTION_AGREEMENT_MINIMUM,
            "minimum per-array detection-call agreement",
        ),
        (
            median_error <= PARITY_BETA_MEDIAN_ABSOLUTE_ERROR_MAXIMUM,
            "beta median absolute-error ceiling",
        ),
        (
            quantile_error <= PARITY_BETA_999_QUANTILE_ERROR_MAXIMUM,
            "beta 99.9% absolute-error ceiling",
        ),
        (
            minimum_beta_correlation >= PARITY_PER_ARRAY_BETA_CORRELATION_MINIMUM,
            "minimum per-array beta correlation",
        ),
    )
    failures = tuple(name for passed, name in threshold_checks if not passed)
    if failures:
        raise ValueError(f"methylprep/seSAMe parity thresholds failed: {failures}")
    return ProcessorParityAudit(
        detection_agreement=overall_agreement,
        minimum_per_array_detection_agreement=minimum_array_agreement,
        beta_median_absolute_error=median_error,
        beta_999_quantile_absolute_error=quantile_error,
        minimum_per_array_beta_correlation=minimum_beta_correlation,
        retained_samples=retained_sample_indices.numel(),
        retained_probes=retained_probe_indices.numel(),
    )


@jaxtyped(typechecker=beartype)
def apply_detection_qc(
    normalized_beta: BetaMatrix,
    detection_p: DetectionPMatrix,
    *,
    detection_threshold: Fraction,
    maximum_sample_failure_fraction: Fraction,
) -> DetectionQcResult:
    """Retain p<threshold calls; remove samples first, then require complete probes."""

    _validate_probability_matrix(normalized_beta, "normalized beta matrix")
    _validate_probability_matrix(detection_p, "detection-p matrix")
    if normalized_beta.shape != detection_p.shape:
        raise ValueError("beta and detection-p matrices must have identical axes")
    passed = detection_p < float(detection_threshold)
    failure_fractions = (~passed).to(t.float64).mean(dim=0)
    retained_sample_mask = failure_fractions <= float(maximum_sample_failure_fraction)
    if not bool(retained_sample_mask.any().item()):
        raise ValueError("detection QC removed every sample")
    retained_sample_indices = t.nonzero(retained_sample_mask).flatten()
    excluded_sample_indices = t.nonzero(~retained_sample_mask).flatten()
    retained_probe_mask = passed.index_select(1, retained_sample_indices).all(dim=1)
    if not bool(retained_probe_mask.any().item()):
        raise ValueError("detection QC removed every probe")
    retained_probe_indices = t.nonzero(retained_probe_mask).flatten()
    excluded_probe_indices = t.nonzero(~retained_probe_mask).flatten()
    beta = normalized_beta.index_select(0, retained_probe_indices).index_select(
        1, retained_sample_indices
    )
    if beta.shape[1] < 2:
        raise ValueError("detection QC retained fewer than two samples")
    return DetectionQcResult(
        beta=beta,
        retained_probe_indices=retained_probe_indices,
        retained_sample_indices=retained_sample_indices,
        excluded_probe_indices=excluded_probe_indices,
        excluded_sample_indices=excluded_sample_indices,
        sample_failure_fractions=failure_fractions,
    )
