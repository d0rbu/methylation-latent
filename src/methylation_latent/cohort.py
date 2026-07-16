"""Target-blind construction and persistence of the primary GSE87571 cohort."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch as t
from beartype import beartype

from methylation_latent.artifacts import (
    JsonValue,
    sha256_ordered_strings,
    write_canonical_json_exclusive,
)
from methylation_latent.domain import (
    Fraction,
    GenomicContext,
    InfiniumDesign,
    ManifestStrand,
    NonEmptyProbeSet,
    ProbeLocus,
    WindowSize,
    parse_autosome,
    parse_one_based_position,
    parse_probe_id,
)
from methylation_latent.genome import (
    IndexedFasta,
    ReferenceWindowFilterResult,
    filter_reference_windows,
)
from methylation_latent.manifest import (
    ManifestParseResult,
    PublishedExclusionResult,
    PublishedExclusionSource,
    apply_published_exclusions,
)
from methylation_latent.metadata import Gse87571RawSampleSet
from methylation_latent.processor_io import ProcessorOutput, select_processor_output_probes
from methylation_latent.storage import load_exact_safetensors, save_safetensors_exclusive
from methylation_latent.targets import TargetGeometry, build_target_geometry

_PREPARED_COHORT_SCHEMA = "methylation-latent.gse87571-prepared-cohort.v1"
_PROBE_TABLE_HEADER = (
    "probe_id",
    "chromosome",
    "position",
    "context",
    "design",
    "manifest_strand",
)


@dataclass(frozen=True, slots=True)
class StaticProbeUniverse:
    """Probe universe determined without beta values, detection calls, or age."""

    probes: NonEmptyProbeSet
    manifest: ManifestParseResult
    published: PublishedExclusionResult
    reference: ReferenceWindowFilterResult

    def __post_init__(self) -> None:
        if self.probes != self.reference.eligible:
            raise ValueError("static probe universe must equal the reference-eligible universe")
        if self.reference.audit.input_probes != len(self.published.retained):
            raise ValueError("reference audit does not consume the published-mask output")


@beartype
def build_static_probe_universe(
    manifest: ManifestParseResult,
    published_sources: tuple[PublishedExclusionSource, ...],
    reference: IndexedFasta,
    maximum_window_size: WindowSize,
) -> StaticProbeUniverse:
    """Apply only manifest, published-list, and reference-sequence exclusions."""

    published = apply_published_exclusions(manifest.probes, published_sources)
    reference_result = filter_reference_windows(
        reference,
        published.retained,
        maximum_window_size,
    )
    return StaticProbeUniverse(
        probes=reference_result.eligible,
        manifest=manifest,
        published=published,
        reference=reference_result,
    )


@dataclass(frozen=True, slots=True)
class CohortQcAudit:
    """Complete counts and thresholds for target-blind sample/probe filtering."""

    detection_threshold: Fraction
    maximum_sample_failure_fraction: Fraction
    raw_samples: int
    phenotype_eligible_samples: int
    retained_samples: int
    detection_failed_samples: int
    phenotype_missing_samples: int
    static_probes: int
    quality_excluded_probes: int
    detection_excluded_probes: int
    constant_excluded_probes: int
    retained_probes: int

    def __post_init__(self) -> None:
        if (
            min(
                self.raw_samples,
                self.phenotype_eligible_samples,
                self.retained_samples,
                self.static_probes,
                self.retained_probes,
            )
            <= 0
        ):
            raise ValueError("cohort QC must retain non-empty sample and probe axes")
        if (
            self.detection_failed_samples + self.phenotype_missing_samples + self.retained_samples
            != self.raw_samples
        ):
            raise ValueError("sample QC counts do not partition the raw cohort")
        if self.phenotype_eligible_samples != self.raw_samples - self.phenotype_missing_samples:
            raise ValueError("phenotype-eligibility count differs from its exclusion count")
        if (
            self.quality_excluded_probes
            + self.detection_excluded_probes
            + self.constant_excluded_probes
            + self.retained_probes
            != self.static_probes
        ):
            raise ValueError("probe QC counts do not partition the static universe")


@dataclass(frozen=True, slots=True)
class PreparedCohort:
    """Finite complete beta matrix aligned to exact loci, samples, and ages."""

    probes: NonEmptyProbeSet
    sample_sentrix_ids: tuple[str, ...]
    sample_gsm_ids: tuple[str, ...]
    beta: t.Tensor
    age: t.Tensor
    raw_sample_failure_fractions: t.Tensor
    retained_raw_sample_indices: t.Tensor
    quality_excluded_probe_ids: tuple[str, ...]
    detection_excluded_probe_ids: tuple[str, ...]
    constant_excluded_probe_ids: tuple[str, ...]
    detection_excluded_sample_sentrix_ids: tuple[str, ...]
    phenotype_excluded_sample_sentrix_ids: tuple[str, ...]
    audit: CohortQcAudit

    def __post_init__(self) -> None:
        sample_count = len(self.sample_sentrix_ids)
        if sample_count == 0 or len(self.sample_gsm_ids) != sample_count:
            raise ValueError("prepared cohort sample axes must be non-empty and aligned")
        if len(set(self.sample_sentrix_ids)) != sample_count:
            raise ValueError("prepared cohort contains duplicate Sentrix identities")
        if len(set(self.sample_gsm_ids)) != sample_count:
            raise ValueError("prepared cohort contains duplicate GSM accessions")
        if self.beta.shape != (len(self.probes), sample_count):
            raise ValueError("prepared beta shape differs from recorded probe/sample axes")
        if self.beta.dtype != t.float64 or self.age.dtype != t.float64:
            raise TypeError("prepared beta and age tensors must be float64")
        if self.age.shape != (sample_count,):
            raise ValueError("prepared age vector differs from the sample axis")
        if not bool(t.isfinite(self.beta).all().item() and t.isfinite(self.age).all().item()):
            raise ValueError("prepared beta and age tensors must be finite")
        if not bool(t.all((self.beta >= 0.0) & (self.beta <= 1.0)).item()):
            raise ValueError("prepared beta values must lie in [0, 1]")
        if (
            self.raw_sample_failure_fractions.shape != (self.audit.raw_samples,)
            or self.raw_sample_failure_fractions.dtype != t.float64
        ):
            raise ValueError("raw sample-failure fractions must align to all raw samples")
        if not bool(
            t.isfinite(self.raw_sample_failure_fractions).all().item()
            and t.all(
                (self.raw_sample_failure_fractions >= 0.0)
                & (self.raw_sample_failure_fractions <= 1.0)
            ).item()
        ):
            raise ValueError("raw sample-failure fractions must be finite values in [0, 1]")
        if (
            self.retained_raw_sample_indices.shape != (sample_count,)
            or self.retained_raw_sample_indices.dtype != t.int64
        ):
            raise ValueError("retained raw sample indices must align to prepared samples")
        if not bool(
            t.all(
                self.retained_raw_sample_indices[1:] > self.retained_raw_sample_indices[:-1]
            ).item()
        ):
            raise ValueError("retained raw sample indices must be strictly increasing")
        if self.audit.retained_samples != sample_count or self.audit.retained_probes != len(
            self.probes
        ):
            raise ValueError("prepared cohort axes differ from the QC audit")
        exclusion_axes = (
            (
                self.quality_excluded_probe_ids,
                self.audit.quality_excluded_probes,
                "quality-excluded probes",
            ),
            (
                self.detection_excluded_probe_ids,
                self.audit.detection_excluded_probes,
                "detection-excluded probes",
            ),
            (
                self.constant_excluded_probe_ids,
                self.audit.constant_excluded_probes,
                "constant-excluded probes",
            ),
            (
                self.detection_excluded_sample_sentrix_ids,
                self.audit.detection_failed_samples,
                "detection-excluded samples",
            ),
            (
                self.phenotype_excluded_sample_sentrix_ids,
                self.audit.phenotype_missing_samples,
                "phenotype-excluded samples",
            ),
        )
        for values, expected_count, name in exclusion_axes:
            if len(values) != expected_count or len(set(values)) != len(values):
                raise ValueError(f"{name} ledger differs from its audited count or is duplicate")

    @property
    def probe_order_sha256(self) -> str:
        return sha256_ordered_strings(str(probe.probe_id) for probe in self.probes.probes)

    @property
    def sample_order_sha256(self) -> str:
        return sha256_ordered_strings(self.sample_sentrix_ids)

    def build_targets(self) -> TargetGeometry:
        return build_target_geometry(self.beta, self.age)


@beartype
def prepare_gse87571_cohort(
    output: ProcessorOutput,
    raw_samples: Gse87571RawSampleSet,
    static_probes: NonEmptyProbeSet,
    *,
    detection_threshold: Fraction,
    maximum_sample_failure_fraction: Fraction,
) -> PreparedCohort:
    """Apply sample-first pOOBAH QC, then complete-case probe QC without imputation."""

    expected_sample_ids = tuple(str(sample.sentrix_identity) for sample in raw_samples.samples)
    if output.sample_ids != expected_sample_ids:
        raise ValueError("processor output does not follow the raw GSE87571 sample order")
    selected = select_processor_output_probes(
        output,
        tuple(str(probe.probe_id) for probe in static_probes.probes),
    )
    detection_passed = selected.detection_p <= float(detection_threshold)
    sample_failure_fractions = (~detection_passed).to(t.float64).mean(dim=0)
    detection_eligible = sample_failure_fractions <= float(maximum_sample_failure_fraction)
    phenotype_eligible = t.tensor(
        tuple(
            sample.age is not None and sample.gender is not None for sample in raw_samples.samples
        ),
        dtype=t.bool,
    )
    retained_sample_mask = detection_eligible & phenotype_eligible
    if int(retained_sample_mask.sum().item()) < 2:
        raise ValueError("GSE87571 cohort QC retained fewer than two age-eligible samples")
    retained_sample_indices = t.nonzero(retained_sample_mask).flatten()

    quality_failed = selected.quality_excluded.index_select(1, retained_sample_indices).any(dim=1)
    detection_failed = ~detection_passed.index_select(1, retained_sample_indices).all(dim=1)
    quality_excluded_indices = t.nonzero(quality_failed).flatten()
    detection_only_excluded_indices = t.nonzero(~quality_failed & detection_failed).flatten()
    complete_probe_mask = ~quality_failed & ~detection_failed
    if not bool(complete_probe_mask.any().item()):
        raise ValueError("GSE87571 complete-case QC removed every static probe")
    complete_indices = t.nonzero(complete_probe_mask).flatten()
    complete_beta = selected.beta.index_select(0, complete_indices).index_select(
        1, retained_sample_indices
    )
    centered = complete_beta - complete_beta.mean(dim=1, keepdim=True)
    constant_mask = t.linalg.vector_norm(centered, dim=1) == 0.0
    constant_excluded_indices = complete_indices.index_select(0, t.nonzero(constant_mask).flatten())
    retained_complete_indices = complete_indices.index_select(
        0, t.nonzero(~constant_mask).flatten()
    )
    if retained_complete_indices.numel() == 0:
        raise ValueError("GSE87571 cohort QC removed every non-constant probe")

    retained_probe_ids = tuple(
        selected.probe_ids[index] for index in retained_complete_indices.tolist()
    )
    by_probe: dict[str, ProbeLocus] = {str(probe.probe_id): probe for probe in static_probes.probes}
    retained_probes = NonEmptyProbeSet(tuple(by_probe[probe_id] for probe_id in retained_probe_ids))
    retained_samples = tuple(
        raw_samples.samples[index] for index in retained_sample_indices.tolist()
    )
    if any(sample.age is None for sample in retained_samples):
        raise RuntimeError("retained phenotype mask admitted a sample without age")
    ages = t.tensor(tuple(float(sample.age) for sample in retained_samples), dtype=t.float64)
    phenotype_missing_mask = ~phenotype_eligible
    detection_failed_only_mask = phenotype_eligible & ~detection_eligible
    detection_excluded_samples = tuple(
        raw_samples.samples[index]
        for index in t.nonzero(detection_failed_only_mask).flatten().tolist()
    )
    phenotype_excluded_samples = tuple(
        raw_samples.samples[index] for index in t.nonzero(phenotype_missing_mask).flatten().tolist()
    )
    audit = CohortQcAudit(
        detection_threshold=detection_threshold,
        maximum_sample_failure_fraction=maximum_sample_failure_fraction,
        raw_samples=len(raw_samples),
        phenotype_eligible_samples=int(phenotype_eligible.sum().item()),
        retained_samples=len(retained_samples),
        detection_failed_samples=int(detection_failed_only_mask.sum().item()),
        phenotype_missing_samples=int(phenotype_missing_mask.sum().item()),
        static_probes=len(static_probes),
        quality_excluded_probes=quality_excluded_indices.numel(),
        detection_excluded_probes=detection_only_excluded_indices.numel(),
        constant_excluded_probes=constant_excluded_indices.numel(),
        retained_probes=len(retained_probes),
    )
    return PreparedCohort(
        probes=retained_probes,
        sample_sentrix_ids=tuple(str(sample.sentrix_identity) for sample in retained_samples),
        sample_gsm_ids=tuple(str(sample.gsm_accession) for sample in retained_samples),
        beta=selected.beta.index_select(0, retained_complete_indices).index_select(
            1, retained_sample_indices
        ),
        age=ages,
        raw_sample_failure_fractions=sample_failure_fractions,
        retained_raw_sample_indices=retained_sample_indices,
        quality_excluded_probe_ids=tuple(
            selected.probe_ids[index] for index in quality_excluded_indices.tolist()
        ),
        detection_excluded_probe_ids=tuple(
            selected.probe_ids[index] for index in detection_only_excluded_indices.tolist()
        ),
        constant_excluded_probe_ids=tuple(
            selected.probe_ids[index] for index in constant_excluded_indices.tolist()
        ),
        detection_excluded_sample_sentrix_ids=tuple(
            str(sample.sentrix_identity) for sample in detection_excluded_samples
        ),
        phenotype_excluded_sample_sentrix_ids=tuple(
            str(sample.sentrix_identity) for sample in phenotype_excluded_samples
        ),
        audit=audit,
    )


def _audit_json(audit: CohortQcAudit) -> dict[str, JsonValue]:
    return {
        "detection_threshold": float(audit.detection_threshold),
        "maximum_sample_failure_fraction": float(audit.maximum_sample_failure_fraction),
        "raw_samples": audit.raw_samples,
        "phenotype_eligible_samples": audit.phenotype_eligible_samples,
        "retained_samples": audit.retained_samples,
        "detection_failed_samples": audit.detection_failed_samples,
        "phenotype_missing_samples": audit.phenotype_missing_samples,
        "static_probes": audit.static_probes,
        "quality_excluded_probes": audit.quality_excluded_probes,
        "detection_excluded_probes": audit.detection_excluded_probes,
        "constant_excluded_probes": audit.constant_excluded_probes,
        "retained_probes": audit.retained_probes,
    }


@beartype
def save_prepared_cohort_exclusive(
    tensor_path: Path,
    metadata_path: Path,
    cohort: PreparedCohort,
) -> None:
    if tensor_path.parent.resolve() != metadata_path.parent.resolve():
        raise ValueError("prepared cohort tensor and metadata files must share a directory")
    save_safetensors_exclusive(
        tensor_path,
        {
            "beta": cohort.beta,
            "age": cohort.age,
            "raw_sample_failure_fractions": cohort.raw_sample_failure_fractions,
            "retained_raw_sample_indices": cohort.retained_raw_sample_indices,
        },
    )
    write_canonical_json_exclusive(
        metadata_path,
        {
            "schema": _PREPARED_COHORT_SCHEMA,
            "tensor_file": tensor_path.name,
            "probe_ids": [str(probe.probe_id) for probe in cohort.probes.probes],
            "sample_sentrix_ids": list(cohort.sample_sentrix_ids),
            "sample_gsm_ids": list(cohort.sample_gsm_ids),
            "quality_excluded_probe_ids": list(cohort.quality_excluded_probe_ids),
            "detection_excluded_probe_ids": list(cohort.detection_excluded_probe_ids),
            "constant_excluded_probe_ids": list(cohort.constant_excluded_probe_ids),
            "detection_excluded_sample_sentrix_ids": list(
                cohort.detection_excluded_sample_sentrix_ids
            ),
            "phenotype_excluded_sample_sentrix_ids": list(
                cohort.phenotype_excluded_sample_sentrix_ids
            ),
            "probe_order_sha256": cohort.probe_order_sha256,
            "sample_order_sha256": cohort.sample_order_sha256,
            "audit": _audit_json(cohort.audit),
        },
    )


def _require_string_array(value: object, name: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(f"prepared cohort {name} must be a non-empty string array")
    return tuple(str(item) for item in value)


def _require_string_array_allow_empty(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"prepared cohort {name} must be a string array")
    return tuple(str(item) for item in value)


@beartype
def load_prepared_cohort(
    tensor_path: Path,
    metadata_path: Path,
    *,
    probe_universe: NonEmptyProbeSet,
) -> PreparedCohort:
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_fields = {
        "schema",
        "tensor_file",
        "probe_ids",
        "sample_sentrix_ids",
        "sample_gsm_ids",
        "quality_excluded_probe_ids",
        "detection_excluded_probe_ids",
        "constant_excluded_probe_ids",
        "detection_excluded_sample_sentrix_ids",
        "phenotype_excluded_sample_sentrix_ids",
        "probe_order_sha256",
        "sample_order_sha256",
        "audit",
    }
    if not isinstance(raw, dict) or set(raw) != expected_fields:
        raise ValueError("prepared cohort metadata envelope differs")
    if raw["schema"] != _PREPARED_COHORT_SCHEMA or raw["tensor_file"] != tensor_path.name:
        raise ValueError("prepared cohort schema or tensor filename differs")
    probe_ids = _require_string_array(raw["probe_ids"], "probe_ids")
    sentrix_ids = _require_string_array(raw["sample_sentrix_ids"], "sample_sentrix_ids")
    gsm_ids = _require_string_array(raw["sample_gsm_ids"], "sample_gsm_ids")
    quality_excluded_probe_ids = _require_string_array_allow_empty(
        raw["quality_excluded_probe_ids"],
        "quality_excluded_probe_ids",
    )
    detection_excluded_probe_ids = _require_string_array_allow_empty(
        raw["detection_excluded_probe_ids"],
        "detection_excluded_probe_ids",
    )
    constant_excluded_probe_ids = _require_string_array_allow_empty(
        raw["constant_excluded_probe_ids"],
        "constant_excluded_probe_ids",
    )
    detection_excluded_sample_sentrix_ids = _require_string_array_allow_empty(
        raw["detection_excluded_sample_sentrix_ids"],
        "detection_excluded_sample_sentrix_ids",
    )
    phenotype_excluded_sample_sentrix_ids = _require_string_array_allow_empty(
        raw["phenotype_excluded_sample_sentrix_ids"],
        "phenotype_excluded_sample_sentrix_ids",
    )
    by_probe = {str(probe.probe_id): probe for probe in probe_universe.probes}
    missing = set(probe_ids) - set(by_probe)
    if missing:
        raise ValueError(
            f"prepared cohort probes are absent from the static universe: {sorted(missing)[:10]}"
        )
    audit_raw = raw["audit"]
    audit_fields = {
        "detection_threshold",
        "maximum_sample_failure_fraction",
        "raw_samples",
        "phenotype_eligible_samples",
        "retained_samples",
        "detection_failed_samples",
        "phenotype_missing_samples",
        "static_probes",
        "quality_excluded_probes",
        "detection_excluded_probes",
        "constant_excluded_probes",
        "retained_probes",
    }
    if not isinstance(audit_raw, dict) or set(audit_raw) != audit_fields:
        raise ValueError("prepared cohort QC audit fields differ")
    tensors = load_exact_safetensors(
        tensor_path,
        {
            "beta",
            "age",
            "raw_sample_failure_fractions",
            "retained_raw_sample_indices",
        },
    )
    cohort = PreparedCohort(
        probes=NonEmptyProbeSet(tuple(by_probe[probe_id] for probe_id in probe_ids)),
        sample_sentrix_ids=sentrix_ids,
        sample_gsm_ids=gsm_ids,
        beta=tensors["beta"],
        age=tensors["age"],
        raw_sample_failure_fractions=tensors["raw_sample_failure_fractions"],
        retained_raw_sample_indices=tensors["retained_raw_sample_indices"],
        quality_excluded_probe_ids=quality_excluded_probe_ids,
        detection_excluded_probe_ids=detection_excluded_probe_ids,
        constant_excluded_probe_ids=constant_excluded_probe_ids,
        detection_excluded_sample_sentrix_ids=detection_excluded_sample_sentrix_ids,
        phenotype_excluded_sample_sentrix_ids=phenotype_excluded_sample_sentrix_ids,
        audit=CohortQcAudit(
            detection_threshold=Fraction.parse(float(audit_raw["detection_threshold"])),
            maximum_sample_failure_fraction=Fraction.parse(
                float(audit_raw["maximum_sample_failure_fraction"])
            ),
            raw_samples=int(audit_raw["raw_samples"]),
            phenotype_eligible_samples=int(audit_raw["phenotype_eligible_samples"]),
            retained_samples=int(audit_raw["retained_samples"]),
            detection_failed_samples=int(audit_raw["detection_failed_samples"]),
            phenotype_missing_samples=int(audit_raw["phenotype_missing_samples"]),
            static_probes=int(audit_raw["static_probes"]),
            quality_excluded_probes=int(audit_raw["quality_excluded_probes"]),
            detection_excluded_probes=int(audit_raw["detection_excluded_probes"]),
            constant_excluded_probes=int(audit_raw["constant_excluded_probes"]),
            retained_probes=int(audit_raw["retained_probes"]),
        ),
    )
    if raw["probe_order_sha256"] != cohort.probe_order_sha256:
        raise ValueError("prepared cohort probe-order fingerprint differs")
    if raw["sample_order_sha256"] != cohort.sample_order_sha256:
        raise ValueError("prepared cohort sample-order fingerprint differs")
    return cohort


@beartype
def write_probe_table_exclusive(path: Path, probes: NonEmptyProbeSet) -> None:
    """Persist the exact downstream locus order without re-reading source manifests."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode="x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(_PROBE_TABLE_HEADER)
        writer.writerows(
            (
                str(probe.probe_id),
                int(probe.chromosome),
                int(probe.position),
                probe.context.value,
                probe.design.value,
                probe.manifest_strand.value,
            )
            for probe in probes.probes
        )
        handle.flush()


@beartype
def load_probe_table(path: Path) -> NonEmptyProbeSet:
    with path.open(mode="r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = tuple(next(reader, ()))
        if header != _PROBE_TABLE_HEADER:
            raise ValueError(f"probe-table header differs: observed={header}")
        probes: list[ProbeLocus] = []
        for line_number, fields in enumerate(reader, start=2):
            if len(fields) != len(_PROBE_TABLE_HEADER):
                raise ValueError(f"probe-table row {line_number} has the wrong field count")
            probe_id, chromosome, position, context, design, manifest_strand = fields
            probes.append(
                ProbeLocus(
                    probe_id=parse_probe_id(probe_id),
                    chromosome=parse_autosome(chromosome),
                    position=parse_one_based_position(position),
                    context=GenomicContext(context),
                    design=InfiniumDesign(design),
                    manifest_strand=ManifestStrand(manifest_strand),
                )
            )
    return NonEmptyProbeSet(tuple(probes))


@beartype
def write_static_exclusion_ledger_exclusive(
    path: Path,
    universe: StaticProbeUniverse,
    *,
    maximum_window_size: WindowSize,
) -> None:
    """Write every manifest, published-list, and reference exclusion source row."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode="x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("probe_id", "reason", "source"))
        writer.writerows(
            (entry.raw_identifier, entry.reason, entry.source)
            for entry in universe.manifest.exclusions
        )
        writer.writerows(
            (entry.raw_identifier, entry.reason, entry.source)
            for entry in universe.published.ledger
        )
        writer.writerows(
            (
                str(entry.probe.probe_id),
                entry.reason,
                f"UCSC_hg19_plus_strand_window_{int(maximum_window_size)}",
            )
            for entry in universe.reference.exclusions
        )
        handle.flush()


@beartype
def write_cohort_exclusion_ledger_exclusive(
    path: Path,
    cohort: PreparedCohort,
) -> None:
    """Write every dynamic sample/probe removal without target-derived fields."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode="x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("entity_id", "entity_type", "reason", "source"))
        writer.writerows(
            (probe_id, "probe", "quality_mask", "seSAMe_QCD")
            for probe_id in cohort.quality_excluded_probe_ids
        )
        writer.writerows(
            (probe_id, "probe", "detection_p_complete_case", "seSAMe_pOOBAH")
            for probe_id in cohort.detection_excluded_probe_ids
        )
        writer.writerows(
            (probe_id, "probe", "constant_beta", "normalized_beta")
            for probe_id in cohort.constant_excluded_probe_ids
        )
        writer.writerows(
            (sample_id, "sample", "detection_p_failure_fraction", "seSAMe_pOOBAH")
            for sample_id in cohort.detection_excluded_sample_sentrix_ids
        )
        writer.writerows(
            (sample_id, "sample", "missing_age_or_gender", "GSE87571_series_matrix")
            for sample_id in cohort.phenotype_excluded_sample_sentrix_ids
        )
        handle.flush()
