"""Run the frozen CPU-only post-hoc latent interpretation analysis."""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch as t

from methylation_latent.artifacts import (
    JsonValue,
    require_clean_git_commit,
    sha256_file,
    sha256_ordered_strings,
    write_canonical_json_exclusive,
)
from methylation_latent.cohort import load_probe_table
from methylation_latent.config import load_protocol_config
from methylation_latent.data_bundle import verify_primary_data_bundle
from methylation_latent.domain import (
    GenomicContext,
    InfiniumDesign,
    ManifestStrand,
    NonEmptyProbeSet,
    parse_latent_dimension,
)
from methylation_latent.embedding_cache import load_embedding_cache
from methylation_latent.evaluation import (
    DISTANCE_CLASS_LABELS,
    DistanceClass,
    regression_metrics,
)
from methylation_latent.evaluation_cache import (
    EvaluationPairCache,
    EvaluationPairCacheIdentity,
    PairSetName,
    load_evaluation_pair_cache,
)
from methylation_latent.experiment_data import (
    SequenceFeatureArtifact,
    SplitArtifact,
    load_sequence_features,
    load_split_artifact,
)
from methylation_latent.latent_interpretation import (
    ManifestInterpretationAnnotation,
    PairDecomposition,
    age_stratified_half_indices,
    deterministic_indices,
    exact_pair_decomposition,
    fit_linear_surrogate,
    gather_pair_products,
    linear_cka,
    load_manifest_interpretation_annotations,
    mean_neighbour_overlap,
    normalized_frobenius_cosine,
    participation_rank,
    spearman_brown,
    split_half_targets,
    stable_extreme_indices,
    stratified_sample_indices,
    strict_pearson,
)
from methylation_latent.latent_interpretation_protocol import (
    InterpretationParent,
    LatentInterpretationProtocol,
    load_latent_interpretation_protocol,
)
from methylation_latent.model import LatentMetric, normalize_vector_strict
from methylation_latent.storage import (
    EmbeddingMatrix,
    load_exact_safetensors,
    load_target_geometry,
)

_RESULT_SCHEMA = "methylation-latent.latent-interpretation-results.v1"
_MANIFEST_SCHEMA = "methylation-latent.latent-interpretation-manifest.v1"
_FINAL_SCHEMA = "methylation-latent.final-refit.v2"
_MODEL_KEYS = {"age_direction", "projection.weight"}
_PAIR_CHUNK_SIZE = 1024


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-config", type=Path, required=True)
    parser.add_argument("--protocol-config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--experiments", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _configure_runtime() -> None:
    t.set_num_threads(1)
    t.set_num_interop_threads(1)
    t.use_deterministic_algorithms(True)


def _load_json(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"expected JSON object: {path}")
    return cast(dict[str, object], raw)


def _object(record: dict[str, object], key: str) -> dict[str, object]:
    value = record.get(key)
    if not isinstance(value, dict) or any(not isinstance(name, str) for name in value):
        raise TypeError(f"{key} must be an object")
    return cast(dict[str, object], value)


def _integer(record: dict[str, object], key: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value


def _number(record: dict[str, object], key: str) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{key} must be finite")
    return result


def _string(record: dict[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise TypeError(f"{key} must be a non-empty string")
    return value


def _metrics(target: t.Tensor, prediction: t.Tensor) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], asdict(regression_metrics(target, prediction)))


def _summary(values: list[float]) -> dict[str, JsonValue]:
    if len(values) < 2 or any(not math.isfinite(value) for value in values):
        raise ValueError("summary requires at least two finite values")
    tensor = t.tensor(values, dtype=t.float64)
    return {
        "values": values,
        "mean": float(tensor.mean().item()),
        "standard_deviation": float(tensor.std(correction=0).item()),
        "minimum": float(tensor.min().item()),
        "maximum": float(tensor.max().item()),
    }


def _spectrum(eigenvalues: t.Tensor) -> dict[str, JsonValue]:
    if eigenvalues.dtype != t.float64 or eigenvalues.ndim != 1 or eigenvalues.numel() == 0:
        raise TypeError("spectrum requires a non-empty float64 vector")
    if bool(t.any(eigenvalues < -1.0e-10).item()):
        raise ValueError("spectrum contains a materially negative eigenvalue")
    values = t.sort(eigenvalues.clamp_min(0.0), descending=True).values
    total = values.sum()
    if float(total.item()) == 0.0:
        raise ValueError("zero spectrum is not interpretable")
    cumulative = values.cumsum(0) / total
    n90 = int(t.searchsorted(cumulative, t.tensor(0.9, dtype=t.float64)).item()) + 1
    n99 = int(t.searchsorted(cumulative, t.tensor(0.99, dtype=t.float64)).item()) + 1
    return {
        "eigenvalues": values.tolist(),
        "participation_rank": participation_rank(values),
        "stable_rank": float(total.div(values[0]).item()),
        "dimensions_for_90_percent_trace": n90,
        "dimensions_for_99_percent_trace": n99,
    }


def _load_parent_model(
    experiments: Path,
    parent: InterpretationParent,
    *,
    protocol: LatentInterpretationProtocol,
    primary_git_commit: str,
    data_sha256: str,
    target_sha256: str,
) -> tuple[LatentMetric, dict[str, object]]:
    directory = experiments / "final" / "full" / parent.split / f"window-{parent.window}"
    metadata_path = directory / "metadata.json"
    model_path = directory / "model.safetensors"
    if (
        sha256_file(metadata_path) != parent.metadata_sha256
        or sha256_file(model_path) != parent.model_sha256
    ):
        raise ValueError(f"parent final-model bytes differ: {directory}")
    metadata = _load_json(metadata_path)
    identity = _object(metadata, "identity")
    training = _object(metadata, "training")
    observed_identity = (
        metadata.get("schema"),
        _string(identity, "protocol_id"),
        _string(identity, "protocol_sha256"),
        _string(identity, "git_commit"),
        _string(identity, "data_sha256"),
        _string(identity, "target_sha256"),
        _string(identity, "split_name"),
        _integer(identity, "window_size"),
        _string(identity, "embedding_manifest_sha256"),
        metadata.get("model_file"),
        metadata.get("model_sha256"),
        _string(training, "mode"),
        _number(training, "weight_decay"),
    )
    expected_identity = (
        _FINAL_SCHEMA,
        protocol.protocol_id,
        protocol.protocol_sha256,
        primary_git_commit,
        data_sha256,
        target_sha256,
        parent.split,
        parent.window,
        parent.embedding_manifest_sha256,
        model_path.name,
        parent.model_sha256,
        "full",
        0.0,
    )
    if observed_identity != expected_identity:
        raise ValueError(f"parent final-model identity differs: {directory}")
    latent_dimension = parse_latent_dimension(_integer(training, "latent_dimension"))
    model = LatentMetric(embedding_dimension=256, latent_dimension=latent_dimension)
    model.load_state_dict(load_exact_safetensors(model_path, _MODEL_KEYS), strict=True)
    model.to("cpu")
    model.eval()
    return model, metadata


def _localize_pairs(left: t.Tensor, right: t.Tensor) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    count = left.numel()
    unique, inverse = t.unique(t.cat((left, right)), sorted=True, return_inverse=True)
    local_left = inverse[:count]
    local_right = inverse[count:]
    if not bool(
        t.equal(unique.index_select(0, local_left), left)
        and t.equal(unique.index_select(0, local_right), right)
    ):
        raise RuntimeError("localized pairs do not reconstruct global indices")
    return unique, local_left, local_right


def _component_metrics(
    decomposition: PairDecomposition,
    distance_class: t.Tensor,
) -> dict[str, JsonValue]:
    def report(mask: t.Tensor) -> dict[str, JsonValue]:
        target_total = decomposition.target_total[mask]
        target_age = decomposition.target_age[mask]
        target_residual = decomposition.target_residual[mask]
        prediction_total = decomposition.prediction_total[mask]
        prediction_age = decomposition.prediction_age[mask]
        prediction_residual = decomposition.prediction_residual[mask]
        return {
            "total": _metrics(target_total, prediction_total),
            "age_component": _metrics(target_age, prediction_age),
            "age_adjusted_residual": _metrics(target_residual, prediction_residual),
            "predicted_age_component_against_total": _metrics(target_total, prediction_age),
            "empirical_age_component_against_total": _metrics(target_total, target_age),
            "target_component_pearson": strict_pearson(target_age, target_residual),
            "target_total_mean_square": float(target_total.square().mean().item()),
            "target_age_mean_square": float(target_age.square().mean().item()),
            "target_residual_mean_square": float(target_residual.square().mean().item()),
        }

    overall_mask = t.ones(distance_class.shape, dtype=t.bool)
    by_distance: dict[str, JsonValue] = {}
    for distance in DistanceClass:
        mask = distance_class == int(distance)
        if bool(mask.any().item()):
            by_distance[DISTANCE_CLASS_LABELS[int(distance)]] = report(mask)
    return {"overall": report(overall_mask), "by_distance": by_distance}


def _representation_audit(pre_latent: t.Tensor, latent: t.Tensor) -> dict[str, JsonValue]:
    pre_centered = pre_latent.to(t.float64) - pre_latent.to(t.float64).mean(dim=0, keepdim=True)
    latent_centered = latent.to(t.float64) - latent.to(t.float64).mean(dim=0, keepdim=True)
    pre_covariance = pre_centered.mT @ pre_centered / pre_centered.shape[0]
    latent_covariance = latent_centered.mT @ latent_centered / latent_centered.shape[0]
    norms = t.linalg.vector_norm(pre_latent.to(t.float64), dim=1)
    return {
        "sample_count": pre_latent.shape[0],
        "pre_normalization_covariance": _spectrum(t.linalg.eigvalsh(pre_covariance)),
        "normalized_latent_covariance": _spectrum(t.linalg.eigvalsh(latent_covariance)),
        "pre_normalization_norm": {
            "mean": float(norms.mean().item()),
            "standard_deviation": float(norms.std(correction=0).item()),
            "minimum": float(norms.min().item()),
            "maximum": float(norms.max().item()),
        },
    }


def _weight_audit(
    model: LatentMetric, training: dict[str, object]
) -> tuple[dict[str, JsonValue], t.Tensor, t.Tensor]:
    weight = model.projection.weight.detach().to(t.float64)
    age = model.age_direction.detach().to(t.float64)
    seed = _integer(training, "seed")
    with t.random.fork_rng():
        t.manual_seed(seed)
        initial = LatentMetric(
            embedding_dimension=256,
            latent_dimension=parse_latent_dimension(weight.shape[0]),
        )
    initial_weight = initial.projection.weight.detach().to(t.float64)
    initial_age = initial.age_direction.detach().to(t.float64)
    metric = weight.mT @ weight
    initial_metric = initial_weight.mT @ initial_weight
    age_pullback = weight.mT @ normalize_vector_strict(age)
    initial_age_pullback = initial_weight.mT @ normalize_vector_strict(initial_age)
    return (
        {
            "latent_dimension": weight.shape[0],
            "metric_spectrum": _spectrum(t.linalg.eigvalsh(metric)),
            "final_initial_weight_cosine": normalized_frobenius_cosine(weight, initial_weight),
            "final_initial_metric_cosine": normalized_frobenius_cosine(metric, initial_metric),
            "final_initial_age_pullback_cosine": normalized_frobenius_cosine(
                age_pullback, initial_age_pullback
            ),
            "relative_weight_change": float(
                t.linalg.vector_norm(weight - initial_weight)
                .div(t.linalg.vector_norm(initial_weight))
                .item()
            ),
        },
        metric,
        age_pullback,
    )


def _annotation_record(
    global_index: int,
    probes: NonEmptyProbeSet,
    annotations: dict[str, ManifestInterpretationAnnotation],
) -> dict[str, JsonValue]:
    probe = probes.probes[global_index]
    annotation = annotations[str(probe.probe_id)]
    return {
        "global_index": global_index,
        "probe_id": str(probe.probe_id),
        "chromosome": int(probe.chromosome),
        "position": int(probe.position),
        "context": probe.context.value,
        "design": probe.design.value,
        "manifest_strand": probe.manifest_strand.value,
        "refgene_names": list(annotation.refgene_names),
        "refgene_groups": list(annotation.refgene_groups),
        "enhancer": annotation.enhancer,
        "regulatory_feature_group": annotation.regulatory_feature_group,
        "dhs": annotation.dhs,
    }


def _category_vectors(
    probes: NonEmptyProbeSet,
    annotations: dict[str, ManifestInterpretationAnnotation],
) -> tuple[dict[str, tuple[str, ...]], dict[str, t.Tensor]]:
    string_labels = {
        "context": tuple(probe.context.value for probe in probes.probes),
        "design": tuple(probe.design.value for probe in probes.probes),
        "manifest_strand": tuple(probe.manifest_strand.value for probe in probes.probes),
        "enhancer": tuple(
            str(annotations[str(probe.probe_id)].enhancer).lower() for probe in probes.probes
        ),
        "regulatory": tuple(
            str(annotations[str(probe.probe_id)].regulatory).lower() for probe in probes.probes
        ),
        "dhs": tuple(str(annotations[str(probe.probe_id)].dhs).lower() for probe in probes.probes),
    }
    contexts = tuple(GenomicContext)
    context_by_value = {value: index for index, value in enumerate(contexts)}
    context_index = t.tensor(
        tuple(context_by_value[probe.context] for probe in probes.probes), dtype=t.int64
    )
    numeric = {
        "context_index": context_index,
        "design_ii": t.tensor(
            tuple(probe.design == InfiniumDesign.TYPE_II for probe in probes.probes),
            dtype=t.float64,
        ),
        "strand_reverse": t.tensor(
            tuple(probe.manifest_strand == ManifestStrand.REVERSE for probe in probes.probes),
            dtype=t.float64,
        ),
        "enhancer": t.tensor(
            tuple(annotations[str(probe.probe_id)].enhancer for probe in probes.probes),
            dtype=t.float64,
        ),
        "regulatory": t.tensor(
            tuple(annotations[str(probe.probe_id)].regulatory for probe in probes.probes),
            dtype=t.float64,
        ),
        "dhs": t.tensor(
            tuple(annotations[str(probe.probe_id)].dhs for probe in probes.probes),
            dtype=t.float64,
        ),
    }
    return string_labels, numeric


def _design_matrix(
    sequence_features: t.Tensor,
    numeric_categories: dict[str, t.Tensor],
    indices: t.Tensor,
    *,
    expanded: bool,
) -> t.Tensor:
    selected_sequence = sequence_features.index_select(0, indices)
    columns = [t.ones((indices.numel(), 1), dtype=t.float64), selected_sequence]
    if expanded:
        context = t.nn.functional.one_hot(
            numeric_categories["context_index"].index_select(0, indices), num_classes=4
        ).to(t.float64)
        columns.extend(
            (
                context[:, 1:],
                *(
                    numeric_categories[name].index_select(0, indices)[:, None]
                    for name in (
                        "design_ii",
                        "strand_reverse",
                        "enhancer",
                        "regulatory",
                        "dhs",
                    )
                ),
            )
        )
    result = t.cat(columns, dim=1)
    if not bool(t.isfinite(result).all().item()):
        raise RuntimeError("interpretation design matrix is not finite")
    return result


def _surrogate_report(
    train_indices: t.Tensor,
    test_indices: t.Tensor,
    model_train: t.Tensor,
    model_test: t.Tensor,
    empirical_rho: t.Tensor,
    sequence_features: t.Tensor,
    numeric_categories: dict[str, t.Tensor],
) -> dict[str, JsonValue]:
    reports: dict[str, JsonValue] = {}
    for name, expanded in (("cpg_gc", False), ("cpg_gc_plus_annotations", True)):
        train_design = _design_matrix(
            sequence_features, numeric_categories, train_indices, expanded=expanded
        )
        test_design = _design_matrix(
            sequence_features, numeric_categories, test_indices, expanded=expanded
        )
        model_coefficients = fit_linear_surrogate(train_design, model_train.to(t.float64))
        empirical_coefficients = fit_linear_surrogate(
            train_design, empirical_rho.index_select(0, train_indices)
        )
        reports[name] = {
            "feature_count": train_design.shape[1],
            "model_output": _metrics(model_test.to(t.float64), test_design @ model_coefficients),
            "empirical_rho": _metrics(
                empirical_rho.index_select(0, test_indices),
                test_design @ empirical_coefficients,
            ),
        }
    return reports


def _group_report(
    test_indices: t.Tensor,
    empirical: t.Tensor,
    prediction: t.Tensor,
    category_labels: dict[str, tuple[str, ...]],
) -> dict[str, JsonValue]:
    output: dict[str, JsonValue] = {}
    global_indices = test_indices.tolist()
    for category, labels in category_labels.items():
        selected = tuple(labels[index] for index in global_indices)
        groups: dict[str, JsonValue] = {}
        for label in sorted(set(selected)):
            mask = t.tensor(tuple(value == label for value in selected), dtype=t.bool)
            target = empirical[mask]
            predicted = prediction[mask]
            groups[label] = {
                "count": int(mask.sum().item()),
                "empirical_mean": float(target.mean().item()),
                "empirical_standard_deviation": float(target.std(correction=0).item()),
                "prediction_mean": float(predicted.mean().item()),
                "prediction_standard_deviation": float(predicted.std(correction=0).item()),
                "mean_absolute_error": float((target - predicted).abs().mean().item()),
            }
        output[category] = groups
    return output


def _reliability_component(
    first: t.Tensor,
    second: t.Tensor,
) -> float:
    return strict_pearson(first.to(t.float64), second.to(t.float64))


def _reliability_report(
    beta: t.Tensor,
    raw_age: t.Tensor,
    split: SplitArtifact,
    pair_cache: EvaluationPairCache,
    protocol: LatentInterpretationProtocol,
    *,
    split_offset: int,
) -> dict[str, JsonValue]:
    cached = pair_cache.pair_sets[PairSetName.HELD_OUT_STRATIFIED]
    sample = stratified_sample_indices(
        cached.pairs.distance_class,
        maximum_per_class=protocol.reliability.maximum_pairs_per_distance_class,
        seed=protocol.reliability.pair_sampling_seed + split_offset,
    )
    left_global = cached.pairs.left.index_select(0, sample)
    right_global = cached.pairs.right.index_select(0, sample)
    distance_class = cached.pairs.distance_class.index_select(0, sample)
    global_to_local = t.full((split.universe_size,), -1, dtype=t.int64)
    global_to_local[split.test_indices] = t.arange(split.test_indices.numel(), dtype=t.int64)
    left = global_to_local.index_select(0, left_global)
    right = global_to_local.index_select(0, right_global)
    if bool(t.any((left < 0) | (right < 0)).item()):
        raise RuntimeError("reliability pair sample escapes the held-out partition")
    runs: list[dict[str, JsonValue]] = []
    for seed in protocol.reliability.subject_split_seeds:
        first_subjects, second_subjects = age_stratified_half_indices(raw_age, seed=seed)
        first_rows, first_rho = split_half_targets(
            beta, raw_age, split.test_indices, first_subjects
        )
        second_rows, second_rho = split_half_targets(
            beta, raw_age, split.test_indices, second_subjects
        )
        first_total = gather_pair_products(first_rows, left, right, chunk_size=_PAIR_CHUNK_SIZE)
        second_total = gather_pair_products(second_rows, left, right, chunk_size=_PAIR_CHUNK_SIZE)
        first_age = first_rho.index_select(0, left) * first_rho.index_select(0, right)
        second_age = second_rho.index_select(0, left) * second_rho.index_select(0, right)
        first_residual = first_total - first_age
        second_residual = second_total - second_age
        by_distance: dict[str, JsonValue] = {}
        for distance in DistanceClass:
            mask = distance_class == int(distance)
            if bool(mask.any().item()):
                by_distance[DISTANCE_CLASS_LABELS[int(distance)]] = {
                    "count": int(mask.sum().item()),
                    "total": _reliability_component(first_total[mask], second_total[mask]),
                    "age_component": _reliability_component(first_age[mask], second_age[mask]),
                    "age_adjusted_residual": _reliability_component(
                        first_residual[mask], second_residual[mask]
                    ),
                }
        runs.append(
            {
                "seed": seed,
                "first_subject_count": first_subjects.numel(),
                "second_subject_count": second_subjects.numel(),
                "age_mean_difference_years": abs(
                    float(raw_age.index_select(0, first_subjects).mean().item())
                    - float(raw_age.index_select(0, second_subjects).mean().item())
                ),
                "probe_age": _reliability_component(first_rho, second_rho),
                "pair_overall": {
                    "total": _reliability_component(first_total, second_total),
                    "age_component": _reliability_component(first_age, second_age),
                    "age_adjusted_residual": _reliability_component(
                        first_residual, second_residual
                    ),
                },
                "pair_by_distance": by_distance,
            }
        )

    def summarized(values: list[float]) -> dict[str, JsonValue]:
        summary = _summary(values)
        mean = cast(float, summary["mean"])
        full_reliability = spearman_brown(mean)
        summary["spearman_brown_from_mean"] = full_reliability
        summary["descriptive_noise_ceiling"] = (
            math.sqrt(full_reliability) if 0.0 <= full_reliability <= 1.0 else None
        )
        return summary

    age_values = [cast(float, run["probe_age"]) for run in runs]
    overall: dict[str, JsonValue] = {}
    for component in ("total", "age_component", "age_adjusted_residual"):
        overall[component] = summarized(
            [
                cast(float, cast(dict[str, JsonValue], run["pair_overall"])[component])
                for run in runs
            ]
        )
    by_distance_summary: dict[str, JsonValue] = {}
    observed_labels = tuple(cast(dict[str, JsonValue], runs[0]["pair_by_distance"]))
    for label in observed_labels:
        component_summary: dict[str, JsonValue] = {}
        for component in ("total", "age_component", "age_adjusted_residual"):
            component_summary[component] = summarized(
                [
                    cast(
                        float,
                        cast(
                            dict[str, JsonValue],
                            cast(dict[str, JsonValue], run["pair_by_distance"])[label],
                        )[component],
                    )
                    for run in runs
                ]
            )
        by_distance_summary[label] = component_summary
    return {
        "subject_split_count": len(runs),
        "subjects_per_half": raw_age.numel() // 2,
        "pair_sample_count": sample.numel(),
        "pair_sample_count_by_distance": t.bincount(distance_class, minlength=8).tolist(),
        "probe_age": summarized(age_values),
        "pair_overall": overall,
        "pair_by_distance": by_distance_summary,
        "age_mean_difference_years": _summary(
            [cast(float, run["age_mean_difference_years"]) for run in runs]
        ),
        "runs": runs,
    }


@dataclass(slots=True)
class CellRuntime:
    split_name: str
    window: int
    model: LatentMetric
    metric: t.Tensor
    age_pullback: t.Tensor
    test_indices: t.Tensor
    test_latent: t.Tensor
    test_age_prediction: t.Tensor
    pair_decompositions: dict[str, PairDecomposition]
    pair_cache: EvaluationPairCache
    result: dict[str, JsonValue]


def _run_cell(
    split_name: str,
    window: int,
    *,
    split_offset: int,
    protocol: LatentInterpretationProtocol,
    probes: NonEmptyProbeSet,
    split: SplitArtifact,
    pair_cache: EvaluationPairCache,
    embeddings: EmbeddingMatrix,
    sequence_features: SequenceFeatureArtifact,
    target_rows: t.Tensor,
    target_age: t.Tensor,
    target_rho: t.Tensor,
    experiments: Path,
    primary_git_commit: str,
    data_sha256: str,
    target_sha256: str,
    category_labels: dict[str, tuple[str, ...]],
    numeric_categories: dict[str, t.Tensor],
) -> CellRuntime:
    parent = protocol.parent(split_name, window)
    model, metadata = _load_parent_model(
        experiments,
        parent,
        protocol=protocol,
        primary_git_commit=primary_git_commit,
        data_sha256=data_sha256,
        target_sha256=target_sha256,
    )
    training = _object(metadata, "training")
    with t.inference_mode():
        test_embedding = embeddings.tensor.index_select(0, split.test_indices).to(t.float32)
        test_latent = model.latent(test_embedding)
        test_age_prediction = model.predict_age_from_latent(test_latent).to(t.float64)
    empirical_test_rho = target_rho.index_select(0, split.test_indices)
    age_metrics = _metrics(empirical_test_rho, test_age_prediction)
    train_local = deterministic_indices(
        split.primary_train_indices.numel(),
        protocol.geometry.covariance_sample_size,
        seed=protocol.geometry.covariance_sampling_seed + split_offset,
    )
    train_indices = split.primary_train_indices.index_select(0, train_local)
    with t.inference_mode():
        train_embedding = embeddings.tensor.index_select(0, train_indices).to(t.float32)
        pre_latent = model.projection(train_embedding)
        train_latent = model.latent(train_embedding)
        train_age_prediction = model.predict_age_from_latent(train_latent).to(t.float64)
    representation = _representation_audit(pre_latent, train_latent)
    feature_values = sequence_features.by_window[window]
    representation["pre_norm_cpg_density_pearson"] = strict_pearson(
        t.linalg.vector_norm(pre_latent.to(t.float64), dim=1),
        feature_values.index_select(0, train_indices)[:, 0],
    )
    representation["pre_norm_gc_content_pearson"] = strict_pearson(
        t.linalg.vector_norm(pre_latent.to(t.float64), dim=1),
        feature_values.index_select(0, train_indices)[:, 1],
    )
    weight_audit, metric, age_pullback = _weight_audit(model, training)
    pair_decompositions: dict[str, PairDecomposition] = {}
    pair_reports: dict[str, JsonValue] = {}
    for pair_name in (PairSetName.SEEN_STRATIFIED, PairSetName.HELD_OUT_STRATIFIED):
        cached = pair_cache.pair_sets[pair_name]
        unique, local_left, local_right = _localize_pairs(cached.pairs.left, cached.pairs.right)
        with t.inference_mode():
            local_latent = model.latent(embeddings.tensor.index_select(0, unique).to(t.float32))
        decomposition = exact_pair_decomposition(
            local_latent,
            model.age_direction.detach(),
            target_rows.index_select(0, unique),
            target_age,
            target_rho.index_select(0, unique),
            cached.targets,
            local_left,
            local_right,
            chunk_size=_PAIR_CHUNK_SIZE,
        )
        pair_decompositions[pair_name.value] = decomposition
        pair_reports[pair_name.value] = _component_metrics(
            decomposition, cached.pairs.distance_class
        )
    surrogate = _surrogate_report(
        train_indices,
        split.test_indices,
        train_age_prediction,
        test_age_prediction,
        target_rho,
        feature_values,
        numeric_categories,
    )
    groups = _group_report(
        split.test_indices,
        empirical_test_rho,
        test_age_prediction,
        category_labels,
    )
    result: dict[str, JsonValue] = {
        "split_name": split_name,
        "window_size": window,
        "parent": {
            "metadata_sha256": parent.metadata_sha256,
            "model_sha256": parent.model_sha256,
            "embedding_manifest_sha256": parent.embedding_manifest_sha256,
            "training_seed": _integer(training, "seed"),
            "latent_dimension": _integer(training, "latent_dimension"),
            "lambda_age": _number(training, "lambda_age"),
        },
        "held_out_probe_count": split.test_indices.numel(),
        "age": age_metrics,
        "weights": weight_audit,
        "representation": representation,
        "pair_decomposition": pair_reports,
        "surrogates": surrogate,
        "age_by_annotation": groups,
    }
    return CellRuntime(
        split_name=split_name,
        window=window,
        model=model,
        metric=metric,
        age_pullback=age_pullback,
        test_indices=split.test_indices,
        test_latent=test_latent,
        test_age_prediction=test_age_prediction,
        pair_decompositions=pair_decompositions,
        pair_cache=pair_cache,
        result=result,
    )


def _rank_curve(
    first: t.Tensor,
    second: t.Tensor,
    empirical: t.Tensor,
    *,
    positive: bool,
    ranks: tuple[int, ...],
) -> list[dict[str, JsonValue]]:
    maximum = ranks[-1]
    ordered = stable_extreme_indices(first, second, positive=positive, maximum=maximum)
    direction_mask = empirical > 0.0 if positive else empirical < 0.0
    output: list[dict[str, JsonValue]] = []
    for rank in ranks:
        selected = ordered[:rank]
        target = empirical.index_select(0, selected)
        prediction = (first.index_select(0, selected) + second.index_select(0, selected)) / 2.0
        output.append(
            {
                "rank": rank,
                "empirical_sign_agreement": float(
                    direction_mask.index_select(0, selected).to(t.float64).mean().item()
                ),
                "empirical_mean": float(target.mean().item()),
                "empirical_mean_absolute": float(target.abs().mean().item()),
                "prediction_mean": float(prediction.mean().item()),
                "prediction_empirical_pearson": strict_pearson(target, prediction),
                "population_empirical_sign_rate": float(direction_mask.to(t.float64).mean().item()),
            }
        )
    return output


def _age_candidate_table(
    first: CellRuntime,
    second: CellRuntime,
    target_rho: t.Tensor,
    probes: NonEmptyProbeSet,
    annotations: dict[str, ManifestInterpretationAnnotation],
    *,
    positive: bool,
    count: int,
) -> list[dict[str, JsonValue]]:
    selected = stable_extreme_indices(
        first.test_age_prediction,
        second.test_age_prediction,
        positive=positive,
        maximum=count,
    )
    output: list[dict[str, JsonValue]] = []
    for local_index in selected.tolist():
        global_index = int(first.test_indices[local_index].item())
        record = _annotation_record(global_index, probes, annotations)
        record.update(
            {
                "empirical_rho": float(target_rho[global_index].item()),
                "prediction_1kb": float(first.test_age_prediction[local_index].item()),
                "prediction_4kb": float(second.test_age_prediction[local_index].item()),
                "stable_magnitude": min(
                    abs(float(first.test_age_prediction[local_index].item())),
                    abs(float(second.test_age_prediction[local_index].item())),
                ),
            }
        )
        output.append(record)
    return output


def _pair_candidate_table(
    first: CellRuntime,
    second: CellRuntime,
    probes: NonEmptyProbeSet,
    annotations: dict[str, ManifestInterpretationAnnotation],
    *,
    distance_class: DistanceClass,
    positive: bool,
    count: int,
) -> list[dict[str, JsonValue]]:
    pair_name = PairSetName.HELD_OUT_STRATIFIED
    cached = first.pair_cache.pair_sets[pair_name]
    first_decomposition = first.pair_decompositions[pair_name.value]
    second_decomposition = second.pair_decompositions[pair_name.value]
    class_indices = t.nonzero(cached.pairs.distance_class == int(distance_class)).flatten()
    selected_local = stable_extreme_indices(
        first_decomposition.prediction_total.index_select(0, class_indices),
        second_decomposition.prediction_total.index_select(0, class_indices),
        positive=positive,
        maximum=count,
    )
    selected = class_indices.index_select(0, selected_local)
    output: list[dict[str, JsonValue]] = []
    for pair_index in selected.tolist():
        left = int(cached.pairs.left[pair_index].item())
        right = int(cached.pairs.right[pair_index].item())
        output.append(
            {
                "left": _annotation_record(left, probes, annotations),
                "right": _annotation_record(right, probes, annotations),
                "distance_class": DISTANCE_CLASS_LABELS[int(distance_class)],
                "empirical_correlation": float(first_decomposition.target_total[pair_index].item()),
                "empirical_age_component": float(first_decomposition.target_age[pair_index].item()),
                "empirical_age_adjusted_residual": float(
                    first_decomposition.target_residual[pair_index].item()
                ),
                "prediction_1kb": float(first_decomposition.prediction_total[pair_index].item()),
                "prediction_4kb": float(second_decomposition.prediction_total[pair_index].item()),
            }
        )
    return output


def _window_comparison(
    first: CellRuntime,
    second: CellRuntime,
    target_rho: t.Tensor,
    probes: NonEmptyProbeSet,
    annotations: dict[str, ManifestInterpretationAnnotation],
    protocol: LatentInterpretationProtocol,
    *,
    split_offset: int,
) -> dict[str, JsonValue]:
    if first.window != 1024 or second.window != 4096 or first.split_name != second.split_name:
        raise ValueError("window comparison requires aligned 1 kb and 4 kb cells")
    sample_local = deterministic_indices(
        first.test_indices.numel(),
        protocol.geometry.neighbour_sample_size,
        seed=protocol.geometry.neighbour_sampling_seed + split_offset,
    )
    first_sample = first.test_latent.index_select(0, sample_local)
    second_sample = second.test_latent.index_select(0, sample_local)
    empirical_age = target_rho.index_select(0, first.test_indices)
    pair_agreement: dict[str, JsonValue] = {}
    pair_displays: dict[str, JsonValue] = {}
    for pair_name in (PairSetName.SEEN_STRATIFIED, PairSetName.HELD_OUT_STRATIFIED):
        cached = first.pair_cache.pair_sets[pair_name]
        first_decomposition = first.pair_decompositions[pair_name.value]
        second_decomposition = second.pair_decompositions[pair_name.value]
        pair_agreement[pair_name.value] = {
            "total_prediction_pearson": strict_pearson(
                first_decomposition.prediction_total,
                second_decomposition.prediction_total,
            ),
            "age_component_prediction_pearson": strict_pearson(
                first_decomposition.prediction_age,
                second_decomposition.prediction_age,
            ),
            "residual_prediction_pearson": strict_pearson(
                first_decomposition.prediction_residual,
                second_decomposition.prediction_residual,
            ),
        }
        display_indices = stratified_sample_indices(
            cached.pairs.distance_class,
            maximum_per_class=protocol.display.pairs_per_distance_class,
            seed=protocol.display.sampling_seed
            + split_offset
            + int(pair_name == PairSetName.HELD_OUT_STRATIFIED),
        )
        pair_displays[pair_name.value] = {
            "source_count": cached.pairs.count,
            "display_count": display_indices.numel(),
            "distance_class": cached.pairs.distance_class.index_select(0, display_indices).tolist(),
            "target_total": first_decomposition.target_total.index_select(
                0, display_indices
            ).tolist(),
            "target_age": first_decomposition.target_age.index_select(0, display_indices).tolist(),
            "target_residual": first_decomposition.target_residual.index_select(
                0, display_indices
            ).tolist(),
            "prediction_1kb_total": first_decomposition.prediction_total.index_select(
                0, display_indices
            ).tolist(),
            "prediction_1kb_age": first_decomposition.prediction_age.index_select(
                0, display_indices
            ).tolist(),
            "prediction_1kb_residual": first_decomposition.prediction_residual.index_select(
                0, display_indices
            ).tolist(),
            "prediction_4kb_total": second_decomposition.prediction_total.index_select(
                0, display_indices
            ).tolist(),
            "prediction_4kb_age": second_decomposition.prediction_age.index_select(
                0, display_indices
            ).tolist(),
            "prediction_4kb_residual": second_decomposition.prediction_residual.index_select(
                0, display_indices
            ).tolist(),
        }
    age_display_local = deterministic_indices(
        first.test_indices.numel(),
        protocol.display.age_probe_count,
        seed=protocol.display.sampling_seed + 100 + split_offset,
    )
    age_display: list[dict[str, JsonValue]] = []
    for local_index in age_display_local.tolist():
        global_index = int(first.test_indices[local_index].item())
        record = _annotation_record(global_index, probes, annotations)
        record.update(
            {
                "empirical_rho": float(target_rho[global_index].item()),
                "prediction_1kb": float(first.test_age_prediction[local_index].item()),
                "prediction_4kb": float(second.test_age_prediction[local_index].item()),
            }
        )
        age_display.append(record)
    held_name = PairSetName.HELD_OUT_STRATIFIED
    held_cached = first.pair_cache.pair_sets[held_name]
    held_first = first.pair_decompositions[held_name.value]
    held_second = second.pair_decompositions[held_name.value]
    candidate_distance = (
        DistanceClass.TRANS
        if bool((held_cached.pairs.distance_class == int(DistanceClass.TRANS)).any().item())
        else DistanceClass.CIS_1MB_PLUS
    )
    candidate_mask = held_cached.pairs.distance_class == int(candidate_distance)
    candidate_first = held_first.prediction_total[candidate_mask]
    candidate_second = held_second.prediction_total[candidate_mask]
    candidate_target = held_first.target_total[candidate_mask]
    return {
        "split_name": first.split_name,
        "metric_cosine": normalized_frobenius_cosine(first.metric, second.metric),
        "age_pullback_cosine": normalized_frobenius_cosine(first.age_pullback, second.age_pullback),
        "age_prediction_pearson": strict_pearson(
            first.test_age_prediction, second.test_age_prediction
        ),
        "age_prediction_mean_absolute_difference": float(
            (first.test_age_prediction - second.test_age_prediction).abs().mean().item()
        ),
        "age_prediction_sign_agreement": float(
            ((first.test_age_prediction > 0.0) == (second.test_age_prediction > 0.0))
            .to(t.float64)
            .mean()
            .item()
        ),
        "linear_cka": linear_cka(first_sample, second_sample),
        "top_neighbour_overlap": mean_neighbour_overlap(
            first_sample,
            second_sample,
            neighbours=protocol.geometry.neighbour_count,
        ),
        "neighbour_sample_count": sample_local.numel(),
        "neighbour_count": protocol.geometry.neighbour_count,
        "pair_prediction_agreement": pair_agreement,
        "age_rank_curves": {
            "positive": _rank_curve(
                first.test_age_prediction,
                second.test_age_prediction,
                empirical_age,
                positive=True,
                ranks=protocol.display.candidate_ranks,
            ),
            "negative": _rank_curve(
                first.test_age_prediction,
                second.test_age_prediction,
                empirical_age,
                positive=False,
                ranks=protocol.display.candidate_ranks,
            ),
        },
        "pair_candidate_distance_class": DISTANCE_CLASS_LABELS[int(candidate_distance)],
        "pair_rank_curves": {
            "positive": _rank_curve(
                candidate_first,
                candidate_second,
                candidate_target,
                positive=True,
                ranks=protocol.display.candidate_ranks,
            ),
            "negative": _rank_curve(
                candidate_first,
                candidate_second,
                candidate_target,
                positive=False,
                ranks=protocol.display.candidate_ranks,
            ),
        },
        "age_candidates": {
            "positive": _age_candidate_table(
                first,
                second,
                target_rho,
                probes,
                annotations,
                positive=True,
                count=protocol.display.candidate_table_size,
            ),
            "negative": _age_candidate_table(
                first,
                second,
                target_rho,
                probes,
                annotations,
                positive=False,
                count=protocol.display.candidate_table_size,
            ),
        },
        "pair_candidates": {
            "positive": _pair_candidate_table(
                first,
                second,
                probes,
                annotations,
                distance_class=candidate_distance,
                positive=True,
                count=protocol.display.candidate_table_size,
            ),
            "negative": _pair_candidate_table(
                first,
                second,
                probes,
                annotations,
                distance_class=candidate_distance,
                positive=False,
                count=protocol.display.candidate_table_size,
            ),
        },
        "age_display": age_display,
        "pair_displays": pair_displays,
    }


def _publish(output: Path, results: dict[str, JsonValue], *, code_git_commit: str) -> None:
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{secrets.token_hex(16)}.tmp"
    temporary.mkdir()
    results_path = temporary / "results.json"
    write_canonical_json_exclusive(results_path, results)
    write_canonical_json_exclusive(
        temporary / "manifest.json",
        {
            "schema": _MANIFEST_SCHEMA,
            "status": "post_hoc_hypothesis_generating",
            "code_git_commit": code_git_commit,
            "results_file": results_path.name,
            "results_sha256": sha256_file(results_path),
        },
    )
    os.rename(temporary, output)


def main() -> None:
    arguments = _parser().parse_args()
    _configure_runtime()
    repository_root = Path(__file__).resolve().parents[1]
    code_git_commit = require_clean_git_commit(repository_root)
    print("stage=verify_protocol_and_bundle", flush=True)
    analysis_protocol = load_latent_interpretation_protocol(arguments.analysis_config)
    primary_config = load_protocol_config(arguments.protocol_config)
    if (
        primary_config.status != "frozen"
        or primary_config.protocol_id != analysis_protocol.protocol_id
        or sha256_file(arguments.protocol_config) != analysis_protocol.protocol_sha256
    ):
        raise ValueError("interpretation config and frozen primary protocol differ")
    bundle = verify_primary_data_bundle(
        arguments.data,
        protocol_id=primary_config.protocol_id,
        protocol_sha256=analysis_protocol.protocol_sha256,
    )
    probes = load_probe_table(arguments.data / "probes.tsv")
    print("stage=join_gpl13534_annotations", flush=True)
    annotations, annotation_audit = load_manifest_interpretation_annotations(
        arguments.manifest,
        probes,
        expected_sha256=analysis_protocol.gpl13534_sha256,
    )
    category_labels, numeric_categories = _category_vectors(probes, annotations)
    sequence_features = load_sequence_features(
        arguments.data / "sequence-features.safetensors",
        arguments.data / "sequence-features.json",
        probes=probes,
    )
    targets = load_target_geometry(arguments.data / "targets.safetensors")
    print("stage=load_cohort", flush=True)
    cohort_metadata = _load_json(arguments.data / "cohort.json")
    if cohort_metadata.get("probe_ids") != [str(probe.probe_id) for probe in probes.probes]:
        raise ValueError("cohort probe order differs from the retained probe table")
    cohort = load_exact_safetensors(
        arguments.data / "cohort.safetensors",
        {"beta", "age", "raw_sample_failure_fractions", "retained_raw_sample_indices"},
    )
    beta = cohort["beta"]
    raw_age = cohort["age"]
    if (
        beta.dtype != t.float64
        or beta.shape != targets.methylation.tensor.shape
        or raw_age.dtype != t.float64
        or raw_age.shape != targets.age.tensor.shape
        or not bool(t.all((beta >= 0.0) & (beta <= 1.0)).item())
    ):
        raise ValueError("cohort tensors differ from the sealed target axes")
    data_sha256 = sha256_file(arguments.data / "bundle.json")
    target_sha256 = sha256_file(arguments.data / "targets.safetensors")
    embeddings: dict[int, EmbeddingMatrix] = {}
    for window in analysis_protocol.windows:
        print(f"stage=load_embedding_cache window={window}", flush=True)
        parent_hashes = {
            analysis_protocol.parent(split, window).embedding_manifest_sha256
            for split in analysis_protocol.splits
        }
        if len(parent_hashes) != 1:
            raise ValueError("split parents disagree about one window embedding cache")
        directory = arguments.embeddings / f"window-{window}"
        if sha256_file(directory / "manifest.json") != next(iter(parent_hashes)):
            raise ValueError(f"window-{window} embedding manifest differs")
        embeddings[window] = load_embedding_cache(
            directory,
            probes,
            window_size=primary_config.splits.window_sizes[
                tuple(map(int, primary_config.splits.window_sizes)).index(window)
            ],
            expected_git_commit=bundle.git_commit,
        )
    runtimes: dict[tuple[str, int], CellRuntime] = {}
    reliability: dict[str, JsonValue] = {}
    window_comparisons: list[dict[str, JsonValue]] = []
    for split_offset, split_name in enumerate(analysis_protocol.splits):
        print(f"stage=load_split_and_pairs split={split_name}", flush=True)
        split_tensor = arguments.data / "splits" / f"{split_name}.safetensors"
        split_metadata = arguments.data / "splits" / f"{split_name}.json"
        split = load_split_artifact(split_tensor, split_metadata, probes=probes)
        split_sha256 = sha256_ordered_strings(
            (sha256_file(split_metadata), sha256_file(split_tensor))
        )
        pair_identity = EvaluationPairCacheIdentity(
            protocol_id=analysis_protocol.protocol_id,
            protocol_sha256=analysis_protocol.protocol_sha256,
            git_commit=bundle.git_commit,
            data_sha256=data_sha256,
            target_sha256=target_sha256,
            split_name=split_name,
            split_sha256=split_sha256,
            probe_order_sha256=sha256_ordered_strings(
                str(probe.probe_id) for probe in probes.probes
            ),
        )
        pair_cache = load_evaluation_pair_cache(
            arguments.experiments / "evaluation-pairs" / split_name,
            pair_identity,
        )
        reliability[split_name] = _reliability_report(
            beta,
            raw_age,
            split,
            pair_cache,
            analysis_protocol,
            split_offset=split_offset,
        )
        print(f"stage=reliability_complete split={split_name}", flush=True)
        for window in analysis_protocol.windows:
            print(f"stage=interpret_cell split={split_name} window={window}", flush=True)
            runtimes[(split_name, window)] = _run_cell(
                split_name,
                window,
                split_offset=split_offset,
                protocol=analysis_protocol,
                probes=probes,
                split=split,
                pair_cache=pair_cache,
                embeddings=embeddings[window],
                sequence_features=sequence_features,
                target_rows=targets.methylation.tensor,
                target_age=targets.age.tensor,
                target_rho=targets.rho.tensor,
                experiments=arguments.experiments,
                primary_git_commit=bundle.git_commit,
                data_sha256=data_sha256,
                target_sha256=target_sha256,
                category_labels=category_labels,
                numeric_categories=numeric_categories,
            )
            print(f"stage=cell_complete split={split_name} window={window}", flush=True)
        print(f"stage=compare_windows split={split_name}", flush=True)
        window_comparisons.append(
            _window_comparison(
                runtimes[(split_name, 1024)],
                runtimes[(split_name, 4096)],
                targets.rho.tensor,
                probes,
                annotations,
                analysis_protocol,
                split_offset=split_offset,
            )
        )
    cross_split: list[dict[str, JsonValue]] = []
    for window in analysis_protocol.windows:
        diverse = runtimes[("diverse-blocks", window)]
        chromosome = runtimes[("held-out-chromosome", window)]
        cross_split.append(
            {
                "window_size": window,
                "metric_cosine": normalized_frobenius_cosine(diverse.metric, chromosome.metric),
                "age_pullback_cosine": normalized_frobenius_cosine(
                    diverse.age_pullback, chromosome.age_pullback
                ),
            }
        )
    results: dict[str, JsonValue] = {
        "schema": _RESULT_SCHEMA,
        "status": "post_hoc_hypothesis_generating",
        "identity": {
            "code_git_commit": code_git_commit,
            "analysis_config_sha256": sha256_file(arguments.analysis_config),
            "protocol_id": analysis_protocol.protocol_id,
            "protocol_sha256": analysis_protocol.protocol_sha256,
            "data_bundle_sha256": data_sha256,
            "target_sha256": target_sha256,
            "probe_order_sha256": sha256_ordered_strings(
                str(probe.probe_id) for probe in probes.probes
            ),
            "manifest_sha256": annotation_audit.source_sha256,
            "parent_models": [
                {
                    "split_name": parent.split,
                    "window_size": parent.window,
                    "metadata_sha256": parent.metadata_sha256,
                    "model_sha256": parent.model_sha256,
                    "embedding_manifest_sha256": parent.embedding_manifest_sha256,
                }
                for parent in analysis_protocol.parents
            ],
        },
        "audit": {
            "gpu_used": False,
            "torch_threads": 1,
            "model_parameters_changed": False,
            "manifest_requested_probes": annotation_audit.requested_probes,
            "manifest_matched_probes": annotation_audit.matched_probes,
            "test_status": "already_viewed_post_hoc",
            "candidate_selection_reads_empirical_targets": False,
        },
        "reliability": reliability,
        "cells": [
            runtimes[(split, window)].result
            for split in analysis_protocol.splits
            for window in analysis_protocol.windows
        ],
        "window_comparisons": window_comparisons,
        "cross_split_weight_agreement": cross_split,
    }
    print("stage=publish", flush=True)
    _publish(arguments.output, results, code_git_commit=code_git_commit)
    print(f"stage=complete output={arguments.output}", flush=True)


if __name__ == "__main__":
    main()
