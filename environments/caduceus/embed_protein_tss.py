"""Embed GRCh37 gene TSS centres with the pinned Caduceus model."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import cast

import torch as t

from methylation_latent.artifacts import (
    JsonValue,
    require_clean_git_commit,
    sha256_file,
    write_canonical_json_exclusive,
)
from methylation_latent.domain import parse_window_size
from methylation_latent.embeddings import (
    CADUCEUS_CHECKPOINT_SHA256,
    CADUCEUS_REPOSITORY,
    CADUCEUS_REVISION,
    EmbeddingInferenceConfig,
    InferencePrecision,
    embed_unconstrained_center_tokens,
    load_pinned_caduceus,
)
from methylation_latent.genome import UCSC_HG19_FASTA_PAYLOAD_SHA256, IndexedFasta
from methylation_latent.protein_extension import GeneLocus, tss_window_interval
from methylation_latent.storage import save_safetensors_exclusive


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
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


def main() -> None:
    arguments = _parser().parse_args()
    repository = Path(__file__).resolve().parents[2]
    git_commit = require_clean_git_commit(repository)
    config = tomllib.loads(arguments.config.read_text(encoding="utf-8"))
    sequence = _object(config.get("sequence"), "sequence")
    if (
        _string(sequence, "caduceus_repository") != CADUCEUS_REPOSITORY
        or _string(sequence, "caduceus_revision") != CADUCEUS_REVISION
        or _string(sequence, "caduceus_checkpoint_sha256") != CADUCEUS_CHECKPOINT_SHA256
        or _string(sequence, "genome_build") != "GRCh37/hg19"
        or _string(sequence, "reference_strand") != "plus"
    ):
        raise ValueError("protein extension Caduceus or reference identity differs")
    window = parse_window_size(arguments.window_size)
    windows = sequence.get("window_sizes")
    if not isinstance(windows, list) or int(window) not in windows:
        raise ValueError("requested TSS window is outside the frozen sweep")

    metadata = json.loads(arguments.targets.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("schema") != "methylation-latent.protein-targets.v1":
        raise ValueError("protein target metadata schema differs")
    if metadata.get("protocol_sha256") != sha256_file(arguments.config):
        raise ValueError("protein target and embedding protocol fingerprints differ")
    genes_raw = metadata.get("gene_symbols")
    loci_raw = metadata.get("gene_loci")
    if not isinstance(genes_raw, list) or not isinstance(loci_raw, list) or len(genes_raw) != 52:
        raise ValueError("protein target gene axes differ")
    if len(loci_raw) != len(genes_raw):
        raise ValueError("protein target loci do not align to genes")
    genes = tuple(str(value) for value in genes_raw)
    loci = tuple(
        GeneLocus(
            gene_symbol=_string(_object(raw, "gene locus"), "gene"),
            chromosome=_string(_object(raw, "gene locus"), "chromosome"),
            start=_integer(_object(raw, "gene locus"), "start"),
            end=_integer(_object(raw, "gene locus"), "end"),
            strand=_integer(_object(raw, "gene locus"), "strand"),
        )
        for raw in loci_raw
    )
    if tuple(locus.gene_symbol for locus in loci) != genes:
        raise ValueError("protein target gene/locus order differs")
    observed_fasta_hash = sha256_file(arguments.fasta)
    if observed_fasta_hash != UCSC_HG19_FASTA_PAYLOAD_SHA256:
        raise ValueError(
            f"hg19 FASTA fingerprint differs: expected={UCSC_HG19_FASTA_PAYLOAD_SHA256}, "
            f"observed={observed_fasta_hash}"
        )

    sequences: list[str] = []
    intervals: list[dict[str, JsonValue]] = []
    with IndexedFasta(arguments.fasta) as reference:
        for locus in loci:
            chromosome = f"chr{locus.chromosome}"
            interval = tss_window_interval(
                locus,
                window,
                chromosome_length=reference.contig_length(chromosome),
            )
            sequence_value = reference.read_interval(interval)
            if len(sequence_value) != int(window) or set(sequence_value) - set("ACGT"):
                raise ValueError(f"gene TSS window is incomplete or non-ACGT: {locus.gene_symbol}")
            sequences.append(sequence_value)
            intervals.append(
                {
                    "gene": locus.gene_symbol,
                    "chromosome": interval.chromosome,
                    "start": interval.start,
                    "end": interval.end,
                    "center_base": sequence_value[int(window) // 2],
                    "reference_strand": "plus",
                }
            )

    inference = EmbeddingInferenceConfig(
        device=arguments.device,
        precision=InferencePrecision.FLOAT16,
        batch_size=arguments.batch_size,
    )
    loaded = load_pinned_caduceus(cache_directory=arguments.model_cache, config=inference)
    shards = tuple(
        embed_unconstrained_center_tokens(
            sequences[start : start + arguments.batch_size],
            tokenizer=loaded.tokenizer,
            model=loaded.model,
            config=inference,
        )
        for start in range(0, len(sequences), arguments.batch_size)
    )
    embeddings = t.cat(shards, dim=0)
    if embeddings.shape != (len(genes), 256) or embeddings.dtype != t.float16:
        raise ValueError("protein TSS embedding matrix shape or dtype differs")
    arguments.output.mkdir(parents=True, exist_ok=False)
    tensor_path = arguments.output / "embeddings.safetensors"
    save_safetensors_exclusive(tensor_path, {"embeddings": embeddings})
    write_canonical_json_exclusive(
        arguments.output / "metadata.json",
        {
            "schema": "methylation-latent.protein-tss-embeddings.v1",
            "protocol_sha256": sha256_file(arguments.config),
            "git_commit": git_commit,
            "protein_target_metadata_sha256": sha256_file(arguments.targets),
            "tensor_file": tensor_path.name,
            "tensor_sha256": sha256_file(tensor_path),
            "gene_symbols": list(genes),
            "window_size": int(window),
            "embedding_width": 256,
            "dtype": "float16",
            "model_repository": CADUCEUS_REPOSITORY,
            "model_revision": CADUCEUS_REVISION,
            "model_checkpoint_sha256": CADUCEUS_CHECKPOINT_SHA256,
            "fasta_sha256": observed_fasta_hash,
            "coordinate_convention": "GRCh37 one-based TSS to zero-based half-open hg19 plus-strand window",
            "pooling": "single center token",
            "intervals": intervals,
        },
    )
    print(
        f"embedded_tss proteins={len(genes)} window={int(window)} "
        f"sha256={sha256_file(tensor_path)}"
    )


if __name__ == "__main__":
    main()
