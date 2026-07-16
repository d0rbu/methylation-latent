"""Real-checkpoint smoke for the shared Caduceus embedding implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch as t

from methylation_latent.embeddings import (
    EmbeddingInferenceConfig,
    InferencePrecision,
    embed_center_tokens,
    load_pinned_caduceus,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-directory", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        choices=tuple(precision.value for precision in InferencePrecision),
        default=InferencePrecision.FLOAT16.value,
    )
    parser.add_argument("--window-size", type=int, default=1024)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.window_size <= 2 or arguments.window_size % 2 != 0:
        raise ValueError("smoke window size must be even and greater than two")
    half = arguments.window_size // 2
    sequence = "A" * half + "CG" + "T" * (half - 2)
    config = EmbeddingInferenceConfig(
        device=arguments.device,
        precision=InferencePrecision(arguments.precision),
        batch_size=1,
    )
    loaded = load_pinned_caduceus(
        cache_directory=arguments.cache_directory,
        config=config,
    )
    embedding = embed_center_tokens(
        (sequence,),
        tokenizer=loaded.tokenizer,
        model=loaded.model,
        config=config,
    )
    print(
        json.dumps(
            {
                "dtype": str(embedding.dtype),
                "finite": bool(t.isfinite(embedding).all().item()),
                "shape": tuple(embedding.shape),
                "snapshot": loaded.snapshot_path.name,
                "window_size": arguments.window_size,
            },
            allow_nan=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
