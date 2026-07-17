"""Audit the two age-association scatter lobes on every frozen test probe."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch as t
from safetensors import safe_open

from methylation_latent.age_clusters import (
    KMeansTwoConfig,
    adjusted_rand_index,
    categorical_association,
    continuous_contrast,
    fit_kmeans_two,
    permutation_invariant_label_agreement,
    subgroup_age_correlations,
)
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
    parse_latent_dimension,
)
from methylation_latent.embedding_cache import load_embedding_cache
from methylation_latent.evaluation import regression_metrics
from methylation_latent.experiment_data import load_sequence_features, load_split_artifact
from methylation_latent.metadata import Gender, parse_gse87571_series_metadata
from methylation_latent.model import LatentMetric
from methylation_latent.storage import load_exact_safetensors, load_target_geometry
from methylation_latent.targets import standardize_vector

_SCHEMA = "methylation-latent.age-scatter-cluster-audit.v1"
_EVALUATION_SCHEMA = "methylation-latent.held-out-evaluation.v2"
_FINAL_SCHEMA = "methylation-latent.final-refit.v2"
_DIRECT_SCHEMA = "methylation-latent.exploratory-direct-tanh-age.v1"
_MODEL_KEYS = {"age_direction", "projection.weight"}
_DIRECT_KEYS = {"prediction", "target", "test_indices"}
_COHORT_KEYS = {
    "age",
    "beta",
    "raw_sample_failure_fractions",
    "retained_raw_sample_indices",
}
_SPLITS = ("diverse-blocks", "held-out-chromosome")
_MAX_REPRODUCTION_ULPS = 8
_KMEANS = KMeansTwoConfig(restarts=64, maximum_iterations=200, seed=810_719)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--series-matrix", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--experiments", type=Path, required=True)
    parser.add_argument("--direct-age", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", type=int, nargs="+", required=True)
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


def _load_model(directory: Path) -> tuple[LatentMetric, dict[str, object]]:
    metadata = _load_json(directory / "metadata.json")
    training = _object(metadata, "training")
    model_path = directory / "model.safetensors"
    if metadata.get("schema") != _FINAL_SCHEMA or metadata.get("model_sha256") != sha256_file(
        model_path
    ):
        raise ValueError(f"final model identity differs: {directory}")
    model = LatentMetric(
        embedding_dimension=256,
        latent_dimension=parse_latent_dimension(_integer(training, "latent_dimension")),
    )
    model.load_state_dict(load_exact_safetensors(model_path, _MODEL_KEYS), strict=True)
    model.to("cpu")
    model.eval()
    return model, metadata


def _assert_metric_reproduction(
    expected: dict[str, object],
    target: t.Tensor,
    prediction: t.Tensor,
) -> None:
    observed = regression_metrics(target, prediction)
    fields = {
        "count": observed.count,
        "target_mean": observed.target_mean,
        "target_standard_deviation": observed.target_standard_deviation,
        "prediction_mean": observed.prediction_mean,
        "prediction_standard_deviation": observed.prediction_standard_deviation,
        "mse": observed.mse,
        "pearson": observed.pearson,
        "r_squared": observed.r_squared,
    }
    if _integer(expected, "count") != fields.pop("count"):
        raise ValueError("cluster-audit source count differs from the primary evaluation")
    for name, observed_value in fields.items():
        expected_value = _number(expected, name)
        tolerance = _MAX_REPRODUCTION_ULPS * max(math.ulp(observed_value), math.ulp(expected_value))
        if abs(observed_value - expected_value) > tolerance:
            raise ValueError(
                "cluster-audit prediction exceeds primary reproduction bound: "
                f"field={name}, difference={abs(observed_value - expected_value)}, "
                f"tolerance={tolerance}"
            )


def _pearson(first: t.Tensor, second: t.Tensor) -> float:
    return regression_metrics(first, second).pearson


def _categorical_json(
    labels: t.Tensor,
    values: tuple[str, ...],
    *,
    declared_order: tuple[str, ...] | None = None,
) -> dict[str, JsonValue]:
    levels = declared_order if declared_order is not None else tuple(sorted(set(values)))
    if set(levels) != set(values):
        raise ValueError("declared categorical levels differ from observed values")
    by_level = {value: index for index, value in enumerate(levels)}
    codes = t.tensor(tuple(by_level[value] for value in values), dtype=t.int64)
    association = categorical_association(labels, codes)
    if association.contingency.shape[1] != len(levels):
        raise RuntimeError("categorical contingency dropped a declared level")
    counts = {
        level: {
            "negative_cluster": int(association.contingency[0, index].item()),
            "positive_cluster": int(association.contingency[1, index].item()),
        }
        for index, level in enumerate(levels)
    }
    return {
        "levels": list(levels),
        "counts": cast(dict[str, JsonValue], counts),
        "cramer_v": association.cramer_v,
        "status": "defined" if len(levels) > 1 else "not_testable_single_level",
    }


def _continuous_json(labels: t.Tensor, values: t.Tensor) -> dict[str, JsonValue]:
    contrast = continuous_contrast(labels, values)
    return cast(dict[str, JsonValue], asdict(contrast))


def _age_distribution(values: t.Tensor) -> dict[str, JsonValue]:
    if values.dtype != t.float64 or values.ndim != 1 or values.numel() < 2:
        raise TypeError("age distribution requires at least two float64 values")
    return {
        "count": values.numel(),
        "minimum_years": float(values.min().item()),
        "maximum_years": float(values.max().item()),
        "mean_years": float(values.mean().item()),
        "standard_deviation_years": float(values.std(correction=1).item()),
    }


def _cluster_analysis(
    *,
    target: t.Tensor,
    prediction: t.Tensor,
    contexts: tuple[str, ...],
    designs: tuple[str, ...],
    strands: tuple[str, ...],
    chromosomes: tuple[str, ...],
    features: t.Tensor,
    female_rho: t.Tensor,
    male_rho: t.Tensor,
    female_count: int,
    male_count: int,
) -> tuple[dict[str, JsonValue], t.Tensor]:
    clustering = fit_kmeans_two(t.stack((target, prediction), dim=1), config=_KMEANS)
    labels = clustering.labels
    counts = t.bincount(labels, minlength=2)
    female_contrast = continuous_contrast(labels, female_rho)
    male_contrast = continuous_contrast(labels, male_rho)
    sex_delta = female_rho - male_rho
    record: dict[str, JsonValue] = {
        "kmeans": {
            "input": "column_standardized_empirical_rho_and_model_prediction",
            "k": 2,
            "config": cast(dict[str, JsonValue], asdict(_KMEANS)),
            "negative_cluster_count": int(counts[0].item()),
            "positive_cluster_count": int(counts[1].item()),
            "centers_empirical_then_predicted": clustering.centers.tolist(),
            "standardized_centers": clustering.standardized_centers.tolist(),
            "inertia": clustering.inertia,
            "converged_iterations_best_restart": clustering.iterations,
            "labels_in_frozen_test_order": labels.tolist(),
        },
        "categorical_associations": {
            "genomic_context": _categorical_json(
                labels,
                contexts,
                declared_order=tuple(context.value for context in GenomicContext),
            ),
            "probe_design": _categorical_json(
                labels,
                designs,
                declared_order=tuple(design.value for design in InfiniumDesign),
            ),
            "manifest_strand": _categorical_json(
                labels,
                strands,
                declared_order=tuple(strand.value for strand in ManifestStrand),
            ),
            "chromosome": _categorical_json(
                labels,
                chromosomes,
                declared_order=tuple(map(str, sorted(map(int, set(chromosomes))))),
            ),
        },
        "sequence_feature_contrasts": {
            "cpg_density": _continuous_json(labels, features[:, 0]),
            "gc_content": _continuous_json(labels, features[:, 1]),
        },
        "sex_stratified_targets": {
            "female": {
                "sample_count": female_count,
                "cluster_target_means": list(female_contrast.means),
                "cluster_target_standard_deviations": list(female_contrast.standard_deviations),
                "cluster_separation_cohen_d": female_contrast.standardized_mean_difference,
                "prediction_pearson": _pearson(female_rho, prediction),
                "all_subject_target_pearson": _pearson(female_rho, target),
            },
            "male": {
                "sample_count": male_count,
                "cluster_target_means": list(male_contrast.means),
                "cluster_target_standard_deviations": list(male_contrast.standard_deviations),
                "cluster_separation_cohen_d": male_contrast.standardized_mean_difference,
                "prediction_pearson": _pearson(male_rho, prediction),
                "all_subject_target_pearson": _pearson(male_rho, target),
            },
            "female_vs_male_target_pearson": _pearson(female_rho, male_rho),
            "female_minus_male_contrast": _continuous_json(labels, sex_delta),
        },
    }
    return record, labels


def main() -> None:
    arguments = _parser().parse_args()
    _configure_runtime()
    code_git_commit = require_clean_git_commit(Path(__file__).resolve().parents[1])
    config = load_protocol_config(arguments.config)
    if config.status != "frozen":
        raise ValueError("age-cluster audit requires the frozen primary protocol")
    protocol_sha256 = sha256_file(arguments.config)
    bundle = verify_primary_data_bundle(
        arguments.data,
        protocol_id=config.protocol_id,
        protocol_sha256=protocol_sha256,
    )
    probes = load_probe_table(arguments.data / "probes.tsv")
    targets = load_target_geometry(arguments.data / "targets.safetensors")
    features = load_sequence_features(
        arguments.data / "sequence-features.safetensors",
        arguments.data / "sequence-features.json",
        probes=probes,
    )
    cohort_metadata = _load_json(arguments.data / "cohort.json")
    cohort_tensor_path = arguments.data / "cohort.safetensors"
    with safe_open(cohort_tensor_path, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != _COHORT_KEYS:
            raise ValueError("cohort tensor keys differ during phenotype alignment")
        retained_raw_indices = handle.get_tensor("retained_raw_sample_indices")
        cohort_age = handle.get_tensor("age")
    raw_samples = parse_gse87571_series_metadata(arguments.series_matrix)
    retained_samples = tuple(raw_samples.samples[index] for index in retained_raw_indices.tolist())
    sample_gsm_ids = cohort_metadata.get("sample_gsm_ids")
    sample_sentrix_ids = cohort_metadata.get("sample_sentrix_ids")
    if (
        not isinstance(sample_gsm_ids, list)
        or not isinstance(sample_sentrix_ids, list)
        or sample_gsm_ids != [str(sample.gsm_accession) for sample in retained_samples]
        or sample_sentrix_ids != [str(sample.sentrix_identity) for sample in retained_samples]
    ):
        raise ValueError("retained GEO phenotype rows do not align to the sealed cohort")
    raw_retained_age = t.tensor(
        tuple(float(sample.age) for sample in retained_samples if sample.age is not None),
        dtype=t.float64,
    )
    if raw_retained_age.shape != cohort_age.shape or not t.equal(raw_retained_age, cohort_age):
        raise ValueError("retained GEO ages do not exactly reproduce sealed cohort ages")
    if not t.equal(standardize_vector(cohort_age).tensor, targets.age.tensor):
        raise ValueError("sealed standardized age does not exactly reproduce from cohort ages")
    genders = tuple(sample.gender for sample in retained_samples)
    if any(gender is None for gender in genders):
        raise ValueError("retained cohort unexpectedly contains missing gender")
    female_mask = t.tensor(tuple(gender == Gender.FEMALE for gender in genders), dtype=t.bool)
    male_mask = t.tensor(tuple(gender == Gender.MALE for gender in genders), dtype=t.bool)
    if not t.equal(female_mask | male_mask, t.ones_like(female_mask)) or bool(
        t.any(female_mask & male_mask).item()
    ):
        raise RuntimeError("female and male masks do not exactly partition retained samples")
    data_sha256 = sha256_file(arguments.data / "bundle.json")
    target_sha256 = sha256_file(arguments.data / "targets.safetensors")
    cohort_metadata_sha256 = sha256_file(arguments.data / "cohort.json")
    cohort_tensor_sha256 = sha256_file(cohort_tensor_path)
    series_matrix_sha256 = sha256_file(arguments.series_matrix)
    sequence_features_metadata_sha256 = sha256_file(arguments.data / "sequence-features.json")
    sequence_features_tensor_sha256 = sha256_file(arguments.data / "sequence-features.safetensors")
    configured_windows = {int(window): window for window in config.splits.window_sizes}
    windows = tuple(arguments.windows)
    if (
        not windows
        or tuple(sorted(set(windows))) != windows
        or any(window not in configured_windows for window in windows)
    ):
        raise ValueError("age-cluster windows must be increasing members of the frozen sweep")
    for split_name in _SPLITS:
        split_tensor = arguments.data / "splits" / f"{split_name}.safetensors"
        split_metadata = arguments.data / "splits" / f"{split_name}.json"
        split = load_split_artifact(split_tensor, split_metadata, probes=probes)
        split_sha256 = sha256_ordered_strings(
            (sha256_file(split_metadata), sha256_file(split_tensor))
        )
        test_indices = split.test_indices
        target = targets.rho.tensor.index_select(0, test_indices)
        test_methylation = targets.methylation.tensor.index_select(0, test_indices)
        female = subgroup_age_correlations(test_methylation, targets.age.tensor, female_mask)
        male = subgroup_age_correlations(test_methylation, targets.age.tensor, male_mask)
        test_probes = tuple(probes.probes[index] for index in test_indices.tolist())
        contexts = tuple(probe.context.value for probe in test_probes)
        designs = tuple(probe.design.value for probe in test_probes)
        strands = tuple(probe.manifest_strand.value for probe in test_probes)
        chromosomes = tuple(str(int(probe.chromosome)) for probe in test_probes)
        for window in windows:
            embedding_directory = arguments.embeddings / f"window-{window}"
            embedding_values = (
                load_embedding_cache(
                    embedding_directory,
                    probes,
                    window_size=configured_windows[window],
                )
                .tensor.index_select(0, test_indices)
                .to(dtype=t.float32)
            )
            evaluation_path = (
                arguments.experiments / "evaluation" / split_name / f"window-{window}.json"
            )
            evaluation = _load_json(evaluation_path)
            evaluation_identity = _object(evaluation, "identity")
            if (
                evaluation.get("schema") != _EVALUATION_SCHEMA
                or _string(evaluation_identity, "protocol_id") != config.protocol_id
                or _string(evaluation_identity, "protocol_sha256") != protocol_sha256
                or _string(evaluation_identity, "data_sha256") != data_sha256
                or _string(evaluation_identity, "split_name") != split_name
                or _string(evaluation_identity, "split_sha256") != split_sha256
                or _integer(evaluation_identity, "window_size") != window
            ):
                raise ValueError(f"age-cluster evaluation identity differs: {evaluation_path}")
            model_predictions: dict[str, t.Tensor] = {}
            model_identity: dict[str, JsonValue] = {}
            for model_name, stage, metric_name in (
                ("cosine_age_only", "age_only", "caduceus_age_only"),
                ("full_latent_metric", "full", "full_latent_metric"),
            ):
                directory = (
                    arguments.experiments / "final" / stage / split_name / f"window-{window}"
                )
                model, metadata = _load_model(directory)
                with t.inference_mode():
                    prediction = model.predict_age_from_latent(model.latent(embedding_values)).to(
                        t.float64
                    )
                _assert_metric_reproduction(
                    _object(_object(evaluation, "age_metrics"), metric_name),
                    target,
                    prediction,
                )
                model_predictions[model_name] = prediction
                model_identity[model_name] = {
                    "metadata_sha256": sha256_file(directory / "metadata.json"),
                    "model_sha256": _string(metadata, "model_sha256"),
                }
            direct_directory = arguments.direct_age / split_name / f"window-{window}"
            direct_metadata_path = direct_directory / "metadata.json"
            direct_metadata = _load_json(direct_metadata_path)
            direct_identity = _object(direct_metadata, "identity")
            direct_prediction_path = direct_directory / "predictions.safetensors"
            direct = load_exact_safetensors(direct_prediction_path, _DIRECT_KEYS)
            if (
                direct_metadata.get("schema") != _DIRECT_SCHEMA
                or _string(direct_identity, "protocol_id") != config.protocol_id
                or _string(direct_identity, "split_name") != split_name
                or _integer(direct_identity, "window_size") != window
                or _string(direct_identity, "data_sha256") != data_sha256
                or _string(direct_identity, "split_sha256") != split_sha256
                or not t.equal(direct["test_indices"], test_indices)
                or not t.equal(direct["target"], target)
            ):
                raise ValueError("direct-tanh age artifact differs during cluster audit")
            model_predictions["direct_tanh"] = direct["prediction"]
            model_identity["direct_tanh"] = {
                "metadata_sha256": sha256_file(direct_metadata_path),
                "predictions_sha256": sha256_file(direct_prediction_path),
            }
            analyses: dict[str, JsonValue] = {}
            labels_by_model: dict[str, t.Tensor] = {}
            test_features = features.by_window[window].index_select(0, test_indices)
            for model_name, prediction in model_predictions.items():
                analysis, labels = _cluster_analysis(
                    target=target,
                    prediction=prediction,
                    contexts=contexts,
                    designs=designs,
                    strands=strands,
                    chromosomes=chromosomes,
                    features=test_features,
                    female_rho=female.correlations,
                    male_rho=male.correlations,
                    female_count=female.sample_count,
                    male_count=male.sample_count,
                )
                analyses[model_name] = cast(JsonValue, analysis)
                labels_by_model[model_name] = labels
            agreements: dict[str, JsonValue] = {}
            names = tuple(model_predictions)
            for left_index, left in enumerate(names):
                for right in names[left_index + 1 :]:
                    agreements[f"{left}__{right}"] = {
                        "permutation_invariant_fraction": permutation_invariant_label_agreement(
                            labels_by_model[left], labels_by_model[right]
                        ),
                        "adjusted_rand_index": adjusted_rand_index(
                            labels_by_model[left], labels_by_model[right]
                        ),
                    }
            record: dict[str, JsonValue] = {
                "schema": _SCHEMA,
                "status": "post_hoc_explanatory_frozen_test_targets_used_for_diagnosis",
                "identity": {
                    "code_git_commit": code_git_commit,
                    "primary_git_commit": bundle.git_commit,
                    "protocol_id": config.protocol_id,
                    "protocol_sha256": protocol_sha256,
                    "data_sha256": data_sha256,
                    "target_sha256": target_sha256,
                    "cohort_metadata_sha256": cohort_metadata_sha256,
                    "cohort_tensor_sha256": cohort_tensor_sha256,
                    "series_matrix_sha256": series_matrix_sha256,
                    "sequence_features_metadata_sha256": sequence_features_metadata_sha256,
                    "sequence_features_tensor_sha256": sequence_features_tensor_sha256,
                    "split_name": split_name,
                    "split_sha256": split_sha256,
                    "test_indices_sha256": sha256_ordered_strings(map(str, test_indices.tolist())),
                    "window_size": window,
                    "embedding_manifest_sha256": sha256_file(embedding_directory / "manifest.json"),
                    "evaluation_sha256": sha256_file(evaluation_path),
                    "model_artifacts": model_identity,
                },
                "scope": {
                    "test_probe_count": test_indices.numel(),
                    "models": list(model_predictions),
                    "cluster_input_uses_empirical_target": True,
                    "confirmatory_status": "post_hoc_not_for_model_selection",
                },
                "phenotype_audit": {
                    "observed_series_fields": ["age", "gender", "tissue", "disease state"],
                    "tissue_levels": sorted({sample.tissue for sample in retained_samples}),
                    "disease_state_levels": sorted(
                        {sample.disease_state for sample in retained_samples}
                    ),
                    "female_age": _age_distribution(cohort_age[female_mask]),
                    "male_age": _age_distribution(cohort_age[male_mask]),
                    "cell_composition": {
                        "status": "not_testable_no_measured_or_precomputed_cell_proportions",
                        "reason": (
                            "The sealed GEO phenotype and preprocessing artifacts contain no "
                            "blood-cell proportion covariate. Inferring one from these methylation "
                            "targets inside this audit would be circular."
                        ),
                    },
                },
                "cross_model_cluster_agreement": agreements,
                "analyses": analyses,
            }
            output_path = arguments.output / split_name / f"window-{window}.json"
            if output_path.is_file():
                if _load_json(output_path) != record:
                    raise ValueError(f"existing age-cluster audit differs: {output_path}")
            else:
                write_canonical_json_exclusive(output_path, record)
            summary = cast(dict[str, object], analyses["full_latent_metric"])
            kmeans = _object(summary, "kmeans")
            categorical = _object(summary, "categorical_associations")
            print(
                f"age-clusters split={split_name} window={window} probes={test_indices.numel()} "
                f"counts=({_integer(kmeans, 'negative_cluster_count')},"
                f"{_integer(kmeans, 'positive_cluster_count')}) "
                f"context_v={_number(_object(categorical, 'genomic_context'), 'cramer_v'):.4f} "
                f"strand_v={_number(_object(categorical, 'manifest_strand'), 'cramer_v'):.4f} "
                f"output={output_path}"
            )


if __name__ == "__main__":
    main()
