from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import torch as t

from methylation_latent.config import ProtocolConfig
from methylation_latent.domain import (
    GenomicContext,
    NonEmptyProbeSet,
    ProbeLocus,
    parse_latent_dimension,
    parse_non_negative_weight,
    parse_positive_int,
    parse_window_size,
)
from methylation_latent.embedding_cache import (
    finalize_embedding_cache_exclusive,
    save_embedding_shard_exclusive,
)
from methylation_latent.experiment_data import SequenceFeatureArtifact, SplitArtifact
from methylation_latent.storage import EmbeddingMatrix
from methylation_latent.targets import build_target_geometry
from methylation_latent.training import TrainingMode
from scripts.run_experiments import (
    ExperimentContext,
    run_evaluation,
    run_model_stage,
    run_pair_caches,
    run_sequence_baselines,
)


def _synthetic_probes(
    make_probe: Callable[..., ProbeLocus],
) -> NonEmptyProbeSet:
    positions = (
        (1, 10_000),
        (1, 10_500),
        (1, 12_000),
        (1, 20_000),
        (1, 40_000),
        (1, 100_000),
        (1, 400_000),
        (1, 1_500_000),
        (2, 10_000),
        (2, 10_500),
        (2, 12_000),
        (2, 20_000),
        (2, 40_000),
        (2, 100_000),
        (2, 400_000),
        (2, 1_500_000),
        (3, 10_000),
        (3, 10_500),
        (3, 12_000),
        (3, 20_000),
        (3, 40_000),
        (3, 100_000),
        (3, 400_000),
        (3, 1_500_000),
        (4, 10_000),
        (4, 10_500),
        (4, 12_000),
        (4, 20_000),
        (4, 40_000),
        (4, 100_000),
        (4, 400_000),
        (4, 1_500_000),
    )
    contexts = tuple(GenomicContext)
    return NonEmptyProbeSet(
        tuple(
            make_probe(
                index + 1,
                chromosome=chromosome,
                position=position,
                context=contexts[index % len(contexts)],
            )
            for index, (chromosome, position) in enumerate(positions)
        )
    )


def _synthetic_config() -> ProtocolConfig:
    window = parse_window_size(1_024)
    return cast(
        ProtocolConfig,
        SimpleNamespace(
            protocol_id="synthetic-orchestration-v1",
            splits=SimpleNamespace(window_sizes=(window,)),
            training=SimpleNamespace(
                batch_size=parse_positive_int(4),
                neighbourhood_width=parse_positive_int(2_000_000),
                tuning_steps=parse_positive_int(2),
                validation_interval=parse_positive_int(1),
                validation_pair_chunk_size=parse_positive_int(2),
                learning_rate=1.0e-3,
                training_seed=71,
            ),
            sweep=SimpleNamespace(
                latent_dimensions=(parse_latent_dimension(2),),
                lambda_age=(parse_non_negative_weight(1.0),),
                maximum_uniform_evaluation_pairs=parse_positive_int(4),
                maximum_pairs_per_distance_class=parse_positive_int(2),
                uniform_evaluation_seed=81,
                distance_evaluation_seed=91,
                projection_window_size=window,
            ),
        ),
    )


def _synthetic_split() -> SplitArtifact:
    return SplitArtifact(
        name="synthetic",
        universe_size=32,
        optimization_indices=t.arange(0, 12, dtype=t.int64),
        validation_indices=t.arange(12, 16, dtype=t.int64),
        test_indices=t.arange(16, 32, dtype=t.int64),
        primary_buffer_indices=t.empty(0, dtype=t.int64),
        validation_buffer_indices=t.empty(0, dtype=t.int64),
    )


def _synthetic_context(
    tmp_path: Path,
    make_probe: Callable[..., ProbeLocus],
) -> ExperimentContext:
    probes = _synthetic_probes(make_probe)
    generator = t.Generator().manual_seed(61)
    targets = build_target_geometry(
        t.rand((len(probes), 8), dtype=t.float64, generator=generator),
        t.linspace(20.0, 80.0, 8, dtype=t.float64),
    )
    feature_axis = t.linspace(0.05, 0.95, len(probes), dtype=t.float64)
    features = SequenceFeatureArtifact(
        probe_count=len(probes),
        by_window={1_024: t.stack((feature_axis, t.square(feature_axis)), dim=1)},
    )
    embeddings = EmbeddingMatrix(t.randn((len(probes), 256), dtype=t.float16, generator=generator))
    embedding_directory = tmp_path / "embeddings"
    window_directory = embedding_directory / "window-1024"
    save_embedding_shard_exclusive(
        window_directory,
        embeddings,
        probes,
        git_commit="a" * 40,
        window_size=parse_window_size(1_024),
        start=0,
        stop=len(probes),
    )
    finalize_embedding_cache_exclusive(
        window_directory,
        probes,
        git_commit="a" * 40,
        window_size=parse_window_size(1_024),
        shard_size=len(probes),
    )
    split = _synthetic_split()
    return ExperimentContext(
        git_commit="a" * 40,
        config=_synthetic_config(),
        protocol_sha256="1" * 64,
        data_directory=tmp_path / "data",
        embedding_directory=embedding_directory,
        output_directory=tmp_path / "experiments",
        probes=probes,
        targets=targets,
        features=features,
        splits={
            "diverse-blocks": split,
            "held-out-chromosome": split,
        },
        data_sha256="2" * 64,
        target_sha256="3" * 64,
        split_sha256={
            "diverse-blocks": "4" * 64,
            "held-out-chromosome": "5" * 64,
        },
        retained_sample_count=8,
    )


def test_complete_experiment_graph_is_restart_safe(
    tmp_path: Path,
    make_probe: Callable[..., ProbeLocus],
) -> None:
    context = _synthetic_context(tmp_path, make_probe)
    run_sequence_baselines(context)
    run_pair_caches(context)
    run_model_stage(context, mode=TrainingMode.AGE_ONLY, device="cpu")
    run_model_stage(context, mode=TrainingMode.FULL, device="cpu")
    run_evaluation(context, device="cpu")

    evaluation_paths = tuple(
        context.output_directory / "evaluation" / split / "window-1024.json"
        for split in ("diverse-blocks", "held-out-chromosome")
    )
    first_bytes = tuple(path.read_bytes() for path in evaluation_paths)
    for path in evaluation_paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["schema"] == "methylation-latent.held-out-evaluation.v2"
        assert record["identity"]["git_commit"] == "a" * 40
        assert set(record["pair_metrics"]) == {
            "seen_by_held_out",
            "held_out_by_held_out",
        }
        assert set(record["age_metrics"]) == {
            "sequence_features",
            "caduceus_age_only",
            "full_latent_metric",
        }
        assert len(record["projection"]["points"]) == 16

    run_sequence_baselines(context)
    run_pair_caches(context)
    run_model_stage(context, mode=TrainingMode.AGE_ONLY, device="cpu")
    run_model_stage(context, mode=TrainingMode.FULL, device="cpu")
    run_evaluation(context, device="cpu")
    assert tuple(path.read_bytes() for path in evaluation_paths) == first_bytes
