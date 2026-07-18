"""Fit and evaluate post-hoc protein anchors in frozen probe geometries."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch as t

from methylation_latent.artifacts import (
    JsonValue,
    require_clean_git_commit,
    sha256_file,
    write_canonical_json_exclusive,
)
from methylation_latent.cohort import load_probe_table
from methylation_latent.domain import NonNegativeWeight, parse_latent_dimension, parse_window_size
from methylation_latent.embedding_cache import load_embedding_cache
from methylation_latent.evaluation import regression_metrics
from methylation_latent.experiment_data import load_split_artifact
from methylation_latent.model import LatentMetric, normalize_rows_strict, normalize_vector_strict
from methylation_latent.protein_extension import (
    FreeProteinVectors,
    LinearProteinMapper,
    ProteinSplit,
    protein_objective,
    unique_unordered_pair_mask,
)
from methylation_latent.storage import load_exact_safetensors, save_safetensors_exclusive

_TARGET_KEYS = {
    "probe_protein",
    "protein_gram",
    "direct_protein_age",
    "probe_age",
    "standardized_proteins",
    "common_cohort_indices",
    "protein_train_indices",
    "protein_validation_indices",
    "protein_test_indices",
}
_PARENT_MODEL_KEYS = {"age_direction", "projection.weight"}
_REPRESENTATIONS = (
    "free",
    "shared_tss",
    "tss_linear",
    "amino_acid_linear",
    "combined_linear",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--probe-embeddings", type=Path, required=True)
    parser.add_argument("--parent-experiments", type=Path, required=True)
    parser.add_argument("--protein-targets", type=Path, required=True)
    parser.add_argument("--protein-embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    return cast(dict[str, object], value)


def _string(table: dict[str, object], name: str) -> str:
    value = table.get(name)
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _integer(table: dict[str, object], name: str) -> int:
    value = table.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _number(table: dict[str, object], name: str) -> float:
    value = table.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not t.isfinite(t.tensor(result)).item():
        raise ValueError(f"{name} must be finite")
    return result


def _load_json(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return cast(dict[str, object], raw)


def _parent_model(directory: Path, *, device: str) -> tuple[LatentMetric, dict[str, object]]:
    metadata = _load_json(directory / "metadata.json")
    if (
        metadata.get("schema") != "methylation-latent.final-refit.v2"
        or metadata.get("model_file") != "model.safetensors"
        or metadata.get("model_sha256") != sha256_file(directory / "model.safetensors")
    ):
        raise ValueError(f"parent model metadata identity differs: {directory}")
    training = _object(metadata.get("training"), "parent training")
    latent_dimension = _integer(training, "latent_dimension")
    if _string(training, "mode") != "full" or _number(training, "lambda_age") != 0.1:
        raise ValueError("protein extension requires the selected full lambda=0.1 parent model")
    model = LatentMetric(
        embedding_dimension=256,
        latent_dimension=parse_latent_dimension(latent_dimension),
    )
    model.load_state_dict(
        load_exact_safetensors(directory / "model.safetensors", _PARENT_MODEL_KEYS),
        strict=True,
    )
    model.to(device)
    model.eval()
    return model, metadata


def _protein_features(
    root: Path,
    *,
    window_size: int,
    protocol_sha256: str,
    target_metadata_sha256: str,
) -> tuple[t.Tensor, t.Tensor]:
    tss_directory = root / f"tss-window-{window_size}"
    amino_directory = root / "amino-acid"
    tss_metadata = _load_json(tss_directory / "metadata.json")
    amino_metadata = _load_json(amino_directory / "metadata.json")
    if (
        tss_metadata.get("schema") != "methylation-latent.protein-tss-embeddings.v1"
        or tss_metadata.get("protocol_sha256") != protocol_sha256
        or tss_metadata.get("protein_target_metadata_sha256") != target_metadata_sha256
        or tss_metadata.get("window_size") != window_size
        or tss_metadata.get("tensor_sha256")
        != sha256_file(tss_directory / "embeddings.safetensors")
    ):
        raise ValueError("protein TSS embedding identity differs")
    if (
        amino_metadata.get("schema")
        != "methylation-latent.protein-amino-acid-embeddings.v1"
        or amino_metadata.get("protocol_sha256") != protocol_sha256
        or amino_metadata.get("protein_target_metadata_sha256") != target_metadata_sha256
        or amino_metadata.get("tensor_sha256")
        != sha256_file(amino_directory / "embeddings.safetensors")
    ):
        raise ValueError("protein amino-acid embedding identity differs")
    tss = load_exact_safetensors(tss_directory / "embeddings.safetensors", {"embeddings"})[
        "embeddings"
    ]
    amino = load_exact_safetensors(
        amino_directory / "embeddings.safetensors", {"embeddings"}
    )["embeddings"]
    if tss.shape != (52, 256) or tss.dtype != t.float16:
        raise ValueError("protein TSS embedding tensor differs")
    if amino.shape != (52, 320) or amino.dtype != t.float32:
        raise ValueError("protein amino-acid embedding tensor differs")
    return tss.to(t.float32), amino


def _run_seed(base_seed: int, split_name: str, window_size: int, representation: str) -> int:
    digest = hashlib.sha256(
        f"{base_seed}:{split_name}:{window_size}:{representation}".encode()
    ).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _optimizer(module: t.nn.Module, *, learning_rate: float) -> t.optim.Optimizer:
    optimizer = t.optim.Adam(module.parameters(), lr=learning_rate, weight_decay=0.0)
    if any(float(group["weight_decay"]) != 0.0 for group in optimizer.param_groups):
        raise RuntimeError("protein mapper optimizer must use zero weight decay")
    return optimizer


def _validation_score(
    *,
    probe_latent: t.Tensor,
    protein_latent: t.Tensor,
    validation_probes: t.Tensor,
    validation_proteins: t.Tensor,
    train_proteins: t.Tensor,
    cross_target: t.Tensor,
    gram_target: t.Tensor,
    lambda_pairs: float,
) -> tuple[float, float, float]:
    probe_values = probe_latent.index_select(0, validation_probes)
    validation_values = protein_latent.index_select(0, validation_proteins)
    train_values = protein_latent.index_select(0, train_proteins)
    cross = t.square(
        probe_values @ validation_values.mT
        - cross_target.index_select(0, validation_probes).index_select(1, validation_proteins)
    ).mean()
    protein_pairs = t.square(
        validation_values @ train_values.mT
        - gram_target.index_select(0, validation_proteins).index_select(1, train_proteins)
    ).mean()
    total = cross + lambda_pairs * protein_pairs
    return float(total.item()), float(cross.item()), float(protein_pairs.item())


def _train_mapper(
    *,
    features: t.Tensor,
    latent_dimension: int,
    probe_latent: t.Tensor,
    cross_target: t.Tensor,
    gram_target: t.Tensor,
    protein_split: ProteinSplit,
    optimization_probes: t.Tensor,
    validation_probes: t.Tensor,
    primary_train_probes: t.Tensor,
    steps: int,
    validation_interval: int,
    batch_size: int,
    learning_rate: float,
    lambda_pairs: float,
    seed: int,
    device: str,
) -> tuple[t.Tensor, int, list[dict[str, JsonValue]], dict[str, t.Tensor]]:
    train_proteins = protein_split.train_indices.to(device)
    validation_proteins = protein_split.validation_indices.to(device)
    optimization = optimization_probes.to(device)
    validation = validation_probes.to(device)
    generator = t.Generator(device="cpu").manual_seed(seed)
    t.manual_seed(seed)
    if t.device(device).type == "cuda":
        t.cuda.manual_seed_all(seed)
    mapper = LinearProteinMapper(
        feature_dimension=features.shape[1], latent_dimension=latent_dimension
    ).to(device)
    optimizer = _optimizer(mapper, learning_rate=learning_rate)
    curve: list[dict[str, JsonValue]] = []
    best_step = -1
    best_score = float("inf")
    for step in range(1, steps + 1):
        local = t.randint(
            optimization.numel(),
            (min(batch_size, optimization.numel()),),
            generator=generator,
        ).to(device)
        probe_indices = optimization.index_select(0, local)
        protein_latent = mapper.latent(features)
        terms = protein_objective(
            probe_latent.index_select(0, probe_indices),
            protein_latent.index_select(0, train_proteins),
            cross_target.index_select(0, probe_indices).index_select(1, train_proteins),
            gram_target.index_select(0, train_proteins).index_select(1, train_proteins),
            lambda_protein_pairs=NonNegativeWeight(lambda_pairs),
        )
        optimizer.zero_grad(set_to_none=True)
        terms.total.backward()
        optimizer.step()
        if step % validation_interval == 0:
            with t.inference_mode():
                score, cross, pair = _validation_score(
                    probe_latent=probe_latent,
                    protein_latent=mapper.latent(features),
                    validation_probes=validation,
                    validation_proteins=validation_proteins,
                    train_proteins=train_proteins,
                    cross_target=cross_target,
                    gram_target=gram_target,
                    lambda_pairs=lambda_pairs,
                )
            curve.append(
                {
                    "step": step,
                    "validation_total": score,
                    "validation_cross_mse": cross,
                    "validation_protein_pair_mse": pair,
                }
            )
            if score < best_score:
                best_score = score
                best_step = step
    if best_step <= 0:
        raise RuntimeError("protein mapper tuning selected no checkpoint")

    t.manual_seed(seed)
    if t.device(device).type == "cuda":
        t.cuda.manual_seed_all(seed)
    refit = LinearProteinMapper(
        feature_dimension=features.shape[1], latent_dimension=latent_dimension
    ).to(device)
    refit_optimizer = _optimizer(refit, learning_rate=learning_rate)
    refit_proteins = protein_split.refit_indices.to(device)
    primary_train = primary_train_probes.to(device)
    refit_generator = t.Generator(device="cpu").manual_seed(seed)
    for _ in range(best_step):
        local = t.randint(
            primary_train.numel(),
            (min(batch_size, primary_train.numel()),),
            generator=refit_generator,
        ).to(device)
        probe_indices = primary_train.index_select(0, local)
        protein_latent = refit.latent(features)
        terms = protein_objective(
            probe_latent.index_select(0, probe_indices),
            protein_latent.index_select(0, refit_proteins),
            cross_target.index_select(0, probe_indices).index_select(1, refit_proteins),
            gram_target.index_select(0, refit_proteins).index_select(1, refit_proteins),
            lambda_protein_pairs=NonNegativeWeight(lambda_pairs),
        )
        refit_optimizer.zero_grad(set_to_none=True)
        terms.total.backward()
        refit_optimizer.step()
    with t.inference_mode():
        latent = refit.latent(features).detach()
    return latent, best_step, curve, {
        name: value.detach().cpu().contiguous() for name, value in refit.state_dict().items()
    }


def _train_free(
    *,
    protein_count: int,
    latent_dimension: int,
    probe_latent: t.Tensor,
    cross_target: t.Tensor,
    gram_target: t.Tensor,
    optimization_probes: t.Tensor,
    validation_probes: t.Tensor,
    primary_train_probes: t.Tensor,
    steps: int,
    validation_interval: int,
    batch_size: int,
    learning_rate: float,
    lambda_pairs: float,
    seed: int,
    device: str,
) -> tuple[t.Tensor, int, list[dict[str, JsonValue]], dict[str, t.Tensor]]:
    all_proteins = t.arange(protein_count, dtype=t.int64, device=device)
    optimization = optimization_probes.to(device)
    validation = validation_probes.to(device)

    def initialized() -> FreeProteinVectors:
        t.manual_seed(seed)
        if t.device(device).type == "cuda":
            t.cuda.manual_seed_all(seed)
        return FreeProteinVectors(
            protein_count=protein_count, latent_dimension=latent_dimension
        ).to(device)

    module = initialized()
    optimizer = _optimizer(module, learning_rate=learning_rate)
    generator = t.Generator(device="cpu").manual_seed(seed)
    curve: list[dict[str, JsonValue]] = []
    best_step = -1
    best_score = float("inf")
    for step in range(1, steps + 1):
        local = t.randint(
            optimization.numel(),
            (min(batch_size, optimization.numel()),),
            generator=generator,
        ).to(device)
        probe_indices = optimization.index_select(0, local)
        terms = protein_objective(
            probe_latent.index_select(0, probe_indices),
            module.latent(),
            cross_target.index_select(0, probe_indices),
            gram_target,
            lambda_protein_pairs=NonNegativeWeight(lambda_pairs),
        )
        optimizer.zero_grad(set_to_none=True)
        terms.total.backward()
        optimizer.step()
        if step % validation_interval == 0:
            with t.inference_mode():
                protein_latent = module.latent()
                cross = t.square(
                    probe_latent.index_select(0, validation) @ protein_latent.mT
                    - cross_target.index_select(0, validation)
                ).mean()
                mask = ~t.eye(protein_count, dtype=t.bool, device=device)
                pair = t.square(protein_latent @ protein_latent.mT - gram_target)[mask].mean()
                score = cross + lambda_pairs * pair
            curve.append(
                {
                    "step": step,
                    "validation_total": float(score.item()),
                    "validation_cross_mse": float(cross.item()),
                    "validation_protein_pair_mse": float(pair.item()),
                }
            )
            if float(score.item()) < best_score:
                best_score = float(score.item())
                best_step = step
    if best_step <= 0:
        raise RuntimeError("free protein-vector tuning selected no checkpoint")
    refit = initialized()
    refit_optimizer = _optimizer(refit, learning_rate=learning_rate)
    primary_train = primary_train_probes.to(device)
    refit_generator = t.Generator(device="cpu").manual_seed(seed)
    for _ in range(best_step):
        local = t.randint(
            primary_train.numel(),
            (min(batch_size, primary_train.numel()),),
            generator=refit_generator,
        ).to(device)
        probe_indices = primary_train.index_select(0, local)
        terms = protein_objective(
            probe_latent.index_select(0, probe_indices),
            refit.latent().index_select(0, all_proteins),
            cross_target.index_select(0, probe_indices),
            gram_target,
            lambda_protein_pairs=NonNegativeWeight(lambda_pairs),
        )
        refit_optimizer.zero_grad(set_to_none=True)
        terms.total.backward()
        refit_optimizer.step()
    with t.inference_mode():
        latent = refit.latent().detach()
    return latent, best_step, curve, {
        "vectors": refit.vectors.detach().cpu().contiguous()
    }


def _metric_record(target: t.Tensor, prediction: t.Tensor) -> dict[str, JsonValue]:
    report = regression_metrics(
        target.flatten().detach().cpu().to(t.float64),
        prediction.flatten().detach().cpu().to(t.float64),
    )
    return cast(dict[str, JsonValue], asdict(report))


def _stratified_cross_metrics(
    *,
    probe_indices: t.Tensor,
    protein_indices: t.Tensor,
    prediction: t.Tensor,
    target: t.Tensor,
    probe_chromosomes: t.Tensor,
    probe_positions: t.Tensor,
    protein_chromosomes: t.Tensor,
    protein_tss: t.Tensor,
    window_size: int,
) -> dict[str, JsonValue]:
    probe_chr = probe_chromosomes.index_select(0, probe_indices.cpu())
    protein_chr = protein_chromosomes.index_select(0, protein_indices.cpu())
    same_chromosome = probe_chr[:, None] == protein_chr[None, :]
    distance = (
        probe_positions.index_select(0, probe_indices.cpu())[:, None]
        - protein_tss.index_select(0, protein_indices.cpu())[None, :]
    ).abs()
    overlap = same_chromosome & (distance < window_size)
    masks = {
        "all_overlap_safe": ~overlap,
        "cis_nonoverlap": same_chromosome & ~overlap,
        "trans": ~same_chromosome,
        "overlap_excluded": overlap,
    }
    result: dict[str, JsonValue] = {}
    prediction_cpu = prediction.detach().cpu()
    target_cpu = target.detach().cpu()
    for name, mask in masks.items():
        count = int(mask.sum().item())
        if count < 2:
            result[name] = {"count": count, "defined": False}
        else:
            result[name] = {"defined": True, **_metric_record(target_cpu[mask], prediction_cpu[mask])}
    return result


def _scatter_sample(
    target: t.Tensor,
    prediction: t.Tensor,
    *,
    maximum_points: int,
    seed: int,
) -> dict[str, JsonValue]:
    flat_target = target.flatten().detach().cpu().to(t.float32)
    flat_prediction = prediction.flatten().detach().cpu().to(t.float32)
    count = flat_target.numel()
    if count > maximum_points:
        generator = t.Generator().manual_seed(seed)
        indices = t.randperm(count, generator=generator)[:maximum_points]
        flat_target = flat_target.index_select(0, indices)
        flat_prediction = flat_prediction.index_select(0, indices)
    return {
        "population_count": count,
        "sample_count": flat_target.numel(),
        "empirical": flat_target.tolist(),
        "predicted": flat_prediction.tolist(),
    }


def _pearson(left: t.Tensor, right: t.Tensor) -> float:
    report = regression_metrics(
        left.detach().cpu().to(t.float64), right.detach().cpu().to(t.float64)
    )
    return report.pearson


def _evaluate(
    *,
    representation: str,
    protein_latent: t.Tensor,
    probe_latent: t.Tensor,
    parent_model: LatentMetric,
    cross_target: t.Tensor,
    gram_target: t.Tensor,
    direct_age: t.Tensor,
    probe_age: t.Tensor,
    split: ProteinSplit,
    primary_train_probes: t.Tensor,
    test_probes: t.Tensor,
    probe_chromosomes: t.Tensor,
    probe_positions: t.Tensor,
    protein_chromosomes: t.Tensor,
    protein_tss: t.Tensor,
    window_size: int,
    seed: int,
) -> dict[str, JsonValue]:
    seen_proteins = split.refit_indices.to(protein_latent.device)
    test_proteins = split.test_indices.to(protein_latent.device)
    primary_train = primary_train_probes.to(protein_latent.device)
    test_cpg = test_probes.to(protein_latent.device)

    def cross_population(probes: t.Tensor, proteins: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
        prediction = probe_latent.index_select(0, probes) @ protein_latent.index_select(
            0, proteins
        ).mT
        target = cross_target.index_select(0, probes).index_select(1, proteins)
        return prediction, target

    test_seen_prediction, test_seen_target = cross_population(test_cpg, seen_proteins)
    seen_test_prediction, seen_test_target = cross_population(primary_train, test_proteins)
    test_test_prediction, test_test_target = cross_population(test_cpg, test_proteins)
    cross_populations = {
        "heldout_cpg_seen_protein": _stratified_cross_metrics(
            probe_indices=test_probes,
            protein_indices=split.refit_indices,
            prediction=test_seen_prediction,
            target=test_seen_target,
            probe_chromosomes=probe_chromosomes,
            probe_positions=probe_positions,
            protein_chromosomes=protein_chromosomes,
            protein_tss=protein_tss,
            window_size=window_size,
        ),
        "seen_cpg_heldout_protein": _stratified_cross_metrics(
            probe_indices=primary_train_probes,
            protein_indices=split.test_indices,
            prediction=seen_test_prediction,
            target=seen_test_target,
            probe_chromosomes=probe_chromosomes,
            probe_positions=probe_positions,
            protein_chromosomes=protein_chromosomes,
            protein_tss=protein_tss,
            window_size=window_size,
        ),
        "heldout_cpg_heldout_protein": _stratified_cross_metrics(
            probe_indices=test_probes,
            protein_indices=split.test_indices,
            prediction=test_test_prediction,
            target=test_test_target,
            probe_chromosomes=probe_chromosomes,
            probe_positions=probe_positions,
            protein_chromosomes=protein_chromosomes,
            protein_tss=protein_tss,
            window_size=window_size,
        ),
    }
    seen_latent = protein_latent.index_select(0, seen_proteins)
    test_latent = protein_latent.index_select(0, test_proteins)
    pp_seen_test_prediction = seen_latent @ test_latent.mT
    pp_seen_test_target = gram_target.index_select(0, seen_proteins).index_select(
        1, test_proteins
    )
    pp_test_test_prediction = test_latent @ test_latent.mT
    pp_test_test_target = gram_target.index_select(0, test_proteins).index_select(
        1, test_proteins
    )
    unique_pairs = unique_unordered_pair_mask(
        test_proteins.numel(), device=protein_latent.device
    )
    protein_pair_metrics = {
        "seen_heldout": _metric_record(pp_seen_test_target, pp_seen_test_prediction),
        "heldout_heldout_off_diagonal": _metric_record(
            pp_test_test_target[unique_pairs], pp_test_test_prediction[unique_pairs]
        ),
    }
    age_direction = normalize_vector_strict(parent_model.age_direction.detach())
    model_age_cosine = protein_latent @ age_direction
    profile = t.tensor(
        tuple(
            _pearson(
                cross_target.index_select(0, test_cpg)[:, protein_index],
                probe_age.index_select(0, test_cpg),
            )
            for protein_index in range(protein_latent.shape[0])
        ),
        dtype=t.float64,
    )
    per_test_protein: list[dict[str, JsonValue]] = []
    for local_index, protein_index in enumerate(test_proteins.tolist()):
        per_test_protein.append(
            {
                "protein_index": protein_index,
                "cross_metrics": _metric_record(
                    test_test_target[:, local_index], test_test_prediction[:, local_index]
                ),
                "model_age_cosine": float(model_age_cosine[protein_index].item()),
                "direct_residualized_protein_age_correlation": float(
                    direct_age[protein_index].item()
                ),
                "empirical_methylation_profile_age_concordance": float(
                    profile[protein_index].item()
                ),
            }
        )
    return {
        "representation": representation,
        "protein_generalization_valid": representation
        not in {"free"},
        "cross_populations": cross_populations,
        "protein_pair_metrics": protein_pair_metrics,
        "age_edge_summary": {
            "heldout_protein_model_cosine_vs_direct_age_pearson": _pearson(
                model_age_cosine.index_select(0, test_proteins),
                direct_age.index_select(0, test_proteins),
            ),
            "heldout_protein_model_cosine_vs_profile_concordance_pearson": _pearson(
                model_age_cosine.index_select(0, test_proteins).to(t.float64),
                profile.index_select(0, test_proteins.cpu()),
            ),
            "maximum_absolute_direct_residualized_protein_age_correlation": float(
                direct_age.abs().max().item()
            ),
            "interpretation": "Model protein-age cosine is an imputed edge. The public protein residuals were adjusted for age, so direct residualized protein-age correlation is not an unadjusted biological age target.",
        },
        "per_heldout_protein": per_test_protein,
        "scatter_heldout_cpg_heldout_protein": _scatter_sample(
            test_test_target,
            test_test_prediction,
            maximum_points=5_000,
            seed=seed,
        ),
    }


def main() -> None:
    arguments = _parser().parse_args()
    repository = Path(__file__).resolve().parents[1]
    git_commit = require_clean_git_commit(repository)
    config = tomllib.loads(arguments.config.read_text(encoding="utf-8"))
    if (
        config.get("schema") != "methylation-latent.protein-extension-protocol.v1"
        or config.get("status") != "frozen"
    ):
        raise ValueError("protein extension protocol identity differs")
    protocol_sha256 = sha256_file(arguments.config)
    training = _object(config.get("training"), "training")
    representations = training.get("representations")
    if representations != list(_REPRESENTATIONS):
        raise ValueError("protein representation sweep differs from frozen order")
    steps = _integer(training, "steps")
    validation_interval = _integer(training, "validation_interval")
    batch_size = _integer(training, "probe_batch_size")
    learning_rate = _number(training, "learning_rate")
    lambda_pairs = _number(training, "lambda_protein_pairs")
    base_seed = _integer(training, "seed")
    if _number(training, "weight_decay") != 0.0:
        raise ValueError("protein extension weight decay must be zero")

    target_directory = arguments.protein_targets
    target_metadata_path = target_directory / "metadata.json"
    target_metadata = _load_json(target_metadata_path)
    target_tensor_path = target_directory / "targets.safetensors"
    if (
        target_metadata.get("schema") != "methylation-latent.protein-targets.v1"
        or target_metadata.get("protocol_sha256") != protocol_sha256
        or target_metadata.get("target_sha256") != sha256_file(target_tensor_path)
    ):
        raise ValueError("protein target artifact identity differs")
    target_tensors = load_exact_safetensors(target_tensor_path, _TARGET_KEYS)
    cross_target_cpu = target_tensors["probe_protein"]
    gram_target_cpu = target_tensors["protein_gram"]
    direct_age_cpu = target_tensors["direct_protein_age"]
    probe_age_cpu = target_tensors["probe_age"]
    if (
        cross_target_cpu.shape != (346_628, 52)
        or cross_target_cpu.dtype != t.float64
        or gram_target_cpu.shape != (52, 52)
        or direct_age_cpu.shape != (52,)
        or probe_age_cpu.shape != (346_628,)
    ):
        raise ValueError("protein target tensor axes differ")
    protein_split = ProteinSplit(
        protein_count=52,
        train_indices=target_tensors["protein_train_indices"],
        validation_indices=target_tensors["protein_validation_indices"],
        test_indices=target_tensors["protein_test_indices"],
    )
    genes_raw = target_metadata.get("gene_symbols")
    loci_raw = target_metadata.get("gene_loci")
    protein_names_raw = target_metadata.get("protein_names")
    if not all(isinstance(axis, list) and len(axis) == 52 for axis in (genes_raw, loci_raw, protein_names_raw)):
        raise ValueError("protein metadata axes differ")
    genes = tuple(str(value) for value in cast(list[object], genes_raw))
    protein_names = tuple(str(value) for value in cast(list[object], protein_names_raw))
    loci = tuple(_object(value, "gene locus") for value in cast(list[object], loci_raw))
    chromosome_codes = {str(value): index for index, value in enumerate((*map(str, range(1, 23)), "X", "Y"), start=1)}
    protein_chromosomes = t.tensor(
        tuple(chromosome_codes[_string(locus, "chromosome")] for locus in loci),
        dtype=t.int64,
    )
    protein_tss = t.tensor(tuple(_integer(locus, "tss") for locus in loci), dtype=t.int64)

    probes = load_probe_table(arguments.data / "probes.tsv")
    if len(probes) != cross_target_cpu.shape[0]:
        raise ValueError("probe table and protein target axes differ")
    probe_chromosomes = t.tensor(
        tuple(chromosome_codes[str(int(probe.chromosome))] for probe in probes.probes),
        dtype=t.int64,
    )
    probe_positions = t.tensor(
        tuple(int(probe.position) for probe in probes.probes), dtype=t.int64
    )
    split_paths = {
        "diverse-blocks": "diverse-blocks",
        "held-out-chromosome": "held-out-chromosome",
    }
    splits = {
        name: load_split_artifact(
            arguments.data / "splits" / f"{filename}.safetensors",
            arguments.data / "splits" / f"{filename}.json",
            probes=probes,
        )
        for name, filename in split_paths.items()
    }
    target_metadata_sha256 = sha256_file(target_metadata_path)
    cross_target = cross_target_cpu.to(device=arguments.device, dtype=t.float32)
    gram_target = gram_target_cpu.to(device=arguments.device, dtype=t.float32)
    direct_age = direct_age_cpu.to(device=arguments.device, dtype=t.float32)
    probe_age = probe_age_cpu.to(device=arguments.device, dtype=t.float32)
    arguments.output.mkdir(parents=True, exist_ok=False)
    model_root = arguments.output / "models"
    records: list[dict[str, JsonValue]] = []

    for split_name, split in splits.items():
        for window_size in (1024, 4096):
            parent_directory = (
                arguments.parent_experiments
                / "final"
                / "full"
                / split_name
                / f"window-{window_size}"
            )
            parent_model, parent_metadata = _parent_model(
                parent_directory, device=arguments.device
            )
            embeddings = load_embedding_cache(
                arguments.probe_embeddings / f"window-{window_size}",
                probes,
                window_size=parse_window_size(window_size),
            )
            with t.inference_mode():
                probe_latent = parent_model.latent(
                    embeddings.training_tensor(device=arguments.device)
                ).detach()
            tss, amino = _protein_features(
                arguments.protein_embeddings,
                window_size=window_size,
                protocol_sha256=protocol_sha256,
                target_metadata_sha256=target_metadata_sha256,
            )
            tss = tss.to(arguments.device)
            amino = amino.to(arguments.device)
            combined = t.cat(
                (normalize_rows_strict(tss), normalize_rows_strict(amino)), dim=1
            )
            feature_by_representation = {
                "tss_linear": tss,
                "amino_acid_linear": amino,
                "combined_linear": combined,
            }
            latent_dimension = parent_model.latent_dimension
            for representation in _REPRESENTATIONS:
                run_seed = _run_seed(base_seed, split_name, window_size, representation)
                curve: list[dict[str, JsonValue]] = []
                model_state: dict[str, t.Tensor] | None = None
                if representation == "shared_tss":
                    with t.inference_mode():
                        protein_latent = parent_model.latent(tss).detach()
                    selected_step = 0
                elif representation == "free":
                    protein_latent, selected_step, curve, model_state = _train_free(
                        protein_count=52,
                        latent_dimension=latent_dimension,
                        probe_latent=probe_latent,
                        cross_target=cross_target,
                        gram_target=gram_target,
                        optimization_probes=split.optimization_indices,
                        validation_probes=split.validation_indices,
                        primary_train_probes=split.primary_train_indices,
                        steps=steps,
                        validation_interval=validation_interval,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        lambda_pairs=lambda_pairs,
                        seed=run_seed,
                        device=arguments.device,
                    )
                else:
                    protein_latent, selected_step, curve, model_state = _train_mapper(
                        features=feature_by_representation[representation],
                        latent_dimension=latent_dimension,
                        probe_latent=probe_latent,
                        cross_target=cross_target,
                        gram_target=gram_target,
                        protein_split=protein_split,
                        optimization_probes=split.optimization_indices,
                        validation_probes=split.validation_indices,
                        primary_train_probes=split.primary_train_indices,
                        steps=steps,
                        validation_interval=validation_interval,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        lambda_pairs=lambda_pairs,
                        seed=run_seed,
                        device=arguments.device,
                    )
                model_sha256: str | None = None
                if model_state is not None:
                    model_directory = (
                        model_root / split_name / f"window-{window_size}" / representation
                    )
                    model_directory.mkdir(parents=True, exist_ok=False)
                    model_path = model_directory / "model.safetensors"
                    save_safetensors_exclusive(model_path, model_state)
                    model_sha256 = sha256_file(model_path)
                evaluation = _evaluate(
                    representation=representation,
                    protein_latent=protein_latent,
                    probe_latent=probe_latent,
                    parent_model=parent_model,
                    cross_target=cross_target,
                    gram_target=gram_target,
                    direct_age=direct_age,
                    probe_age=probe_age,
                    split=protein_split,
                    primary_train_probes=split.primary_train_indices,
                    test_probes=split.test_indices,
                    probe_chromosomes=probe_chromosomes,
                    probe_positions=probe_positions,
                    protein_chromosomes=protein_chromosomes,
                    protein_tss=protein_tss,
                    window_size=window_size,
                    seed=run_seed,
                )
                records.append(
                    {
                        "split": split_name,
                        "window_size": window_size,
                        "latent_dimension": latent_dimension,
                        "representation": representation,
                        "run_seed": run_seed,
                        "selected_step": selected_step,
                        "validation_curve": curve,
                        "model_sha256": model_sha256,
                        "parent_model_sha256": _string(parent_metadata, "model_sha256"),
                        "evaluation": evaluation,
                    }
                )
                strongest = cast(
                    dict[str, object],
                    cast(dict[str, object], evaluation["cross_populations"])[
                        "heldout_cpg_heldout_protein"
                    ],
                )
                overlap_safe = cast(dict[str, object], strongest["all_overlap_safe"])
                print(
                    f"completed split={split_name} window={window_size} "
                    f"representation={representation} selected_step={selected_step} "
                    f"heldout_mse={overlap_safe.get('mse')} heldout_pearson={overlap_safe.get('pearson')}",
                    flush=True,
                )
            del probe_latent, embeddings, tss, amino, combined, parent_model
            if t.device(arguments.device).type == "cuda":
                t.cuda.empty_cache()

    write_canonical_json_exclusive(
        arguments.output / "results.json",
        {
            "schema": "methylation-latent.protein-extension-results.v1",
            "protocol_id": _string(config, "protocol_id"),
            "protocol_sha256": protocol_sha256,
            "git_commit": git_commit,
            "scientific_status": _string(config, "scientific_status"),
            "device": arguments.device,
            "protein_target_metadata_sha256": target_metadata_sha256,
            "protein_target_tensor_sha256": sha256_file(target_tensor_path),
            "protein_names": list(protein_names),
            "gene_symbols": list(genes),
            "protein_split": {
                "train": protein_split.train_indices.tolist(),
                "validation": protein_split.validation_indices.tolist(),
                "test": protein_split.test_indices.tolist(),
            },
            "selection_warning": _string(
                _object(config.get("panel"), "panel"), "selection_warning"
            ),
            "source_adjustment": _string(_object(config.get("panel"), "panel"), "values"),
            "loss_formula": "mean((probe_latent @ protein_latent.T - R)^2) + lambda_protein_pairs * mean_offdiag((protein_latent @ protein_latent.T - G)^2)",
            "lambda_protein_pairs": lambda_pairs,
            "records": records,
        },
    )
    print(f"finalized results={arguments.output / 'results.json'} records={len(records)}")


if __name__ == "__main__":
    main()
