"""Restart-safe centre-CpG embedding generation in the pinned Caduceus runtime."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch as t

from methylation_latent.cohort import load_probe_table
from methylation_latent.domain import parse_window_size
from methylation_latent.embedding_cache import (
    embedding_shard_ranges,
    finalize_embedding_cache_exclusive,
    load_embedding_cache,
    load_embedding_shard,
    save_embedding_shard_exclusive,
)
from methylation_latent.embeddings import (
    EmbeddingInferenceConfig,
    InferencePrecision,
    embed_center_tokens,
    load_pinned_caduceus,
)
from methylation_latent.genome import IndexedFasta, extract_cpg_window
from methylation_latent.storage import EmbeddingMatrix


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--shard-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    probes = load_probe_table(arguments.probes)
    window = parse_window_size(arguments.window_size)
    ranges = embedding_shard_ranges(len(probes), arguments.shard_size)
    manifest = arguments.output / "manifest.json"
    if manifest.is_file():
        loaded = load_embedding_cache(
            arguments.output,
            probes,
            window_size=window,
        )
        print(
            f"verified_complete window={int(window)} probes={loaded.tensor.shape[0]} "
            f"shards={len(ranges)}"
        )
        return

    missing_ranges: list[tuple[int, int]] = []
    for start, stop in ranges:
        directory = arguments.output / f"shard-{start:09d}-{stop:09d}"
        if directory.is_dir():
            load_embedding_shard(
                directory,
                probes,
                window_size=window,
                start=start,
                stop=stop,
            )
            print(f"verified_existing window={int(window)} start={start} stop={stop}")
        else:
            missing_ranges.append((start, stop))
    if missing_ranges:
        config = EmbeddingInferenceConfig(
            device=arguments.device,
            precision=InferencePrecision.FLOAT16,
            batch_size=arguments.batch_size,
        )
        loaded = load_pinned_caduceus(
            cache_directory=arguments.model_cache,
            config=config,
        )
        with IndexedFasta(arguments.fasta) as reference:
            for start, stop in missing_ranges:
                batch_shards: list[t.Tensor] = []
                for batch_start in range(start, stop, config.batch_size):
                    batch_stop = min(batch_start + config.batch_size, stop)
                    sequences = tuple(
                        extract_cpg_window(reference, probe, window)
                        for probe in probes.probes[batch_start:batch_stop]
                    )
                    batch_shards.append(
                        embed_center_tokens(
                            sequences,
                            tokenizer=loaded.tokenizer,
                            model=loaded.model,
                            config=config,
                        )
                    )
                embeddings = EmbeddingMatrix(t.cat(batch_shards, dim=0))
                save_embedding_shard_exclusive(
                    arguments.output,
                    embeddings,
                    probes,
                    window_size=window,
                    start=start,
                    stop=stop,
                )
                print(f"completed window={int(window)} start={start} stop={stop}")
    finalize_embedding_cache_exclusive(
        arguments.output,
        probes,
        window_size=window,
        shard_size=arguments.shard_size,
    )
    verified = load_embedding_cache(
        arguments.output,
        probes,
        window_size=window,
    )
    print(f"finalized window={int(window)} probes={verified.tensor.shape[0]} shards={len(ranges)}")


if __name__ == "__main__":
    main()
