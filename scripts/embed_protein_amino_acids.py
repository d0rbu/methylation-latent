"""Embed canonical reviewed protein sequences with pinned ESM-2."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import tomllib
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
from methylation_latent.protein_extension import protein_chunk_ranges
from methylation_latent.storage import save_safetensors_exclusive

_HEADER = re.compile(r"^>sp\|([^|]+)\|")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
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


def _sequences(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    accession: str | None = None
    pieces: list[str] = []

    def finish() -> None:
        if accession is None:
            return
        sequence = "".join(pieces)
        if not sequence or accession in records:
            raise ValueError("UniProt FASTA contains an empty or duplicate selected sequence")
        records[accession] = sequence

    with gzip.open(path, mode="rt", encoding="ascii", newline="") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.startswith(">"):
                finish()
                match = _HEADER.match(line)
                if match is None:
                    raise ValueError(f"reviewed UniProt header differs: {line}")
                accession = match.group(1)
                pieces = []
            else:
                if accession is None:
                    raise ValueError("UniProt sequence appears before header")
                pieces.append(line)
    finish()
    return records


def main() -> None:
    arguments = _parser().parse_args()
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    git_commit = require_clean_git_commit(Path(__file__).resolve().parents[1])
    config = tomllib.loads(arguments.config.read_text(encoding="utf-8"))
    sources = _object(config.get("sources"), "sources")
    sequence = _object(config.get("sequence"), "sequence")
    observed_fasta_hash = sha256_file(arguments.fasta)
    if observed_fasta_hash != _string(sources, "uniprot_fasta_sha256"):
        raise ValueError("UniProt FASTA fingerprint differs")
    repository = _string(sequence, "esm_repository")
    revision = _string(sequence, "esm_revision")
    checkpoint_sha256 = _string(sequence, "esm_checkpoint_sha256")
    embedding_width = _integer(sequence, "esm_embedding_width")
    maximum_residues = _integer(sequence, "esm_maximum_residues")

    target_metadata = json.loads(arguments.targets.read_text(encoding="utf-8"))
    if (
        not isinstance(target_metadata, dict)
        or target_metadata.get("schema") != "methylation-latent.protein-targets.v1"
        or target_metadata.get("protocol_sha256") != sha256_file(arguments.config)
    ):
        raise ValueError("protein target metadata identity differs")
    accessions_raw = target_metadata.get("uniprot_accessions")
    expected_lengths = target_metadata.get("protein_sequence_lengths")
    expected_hashes = target_metadata.get("protein_sequence_sha256")
    if not all(isinstance(axis, list) and len(axis) == 52 for axis in (accessions_raw, expected_lengths, expected_hashes)):
        raise ValueError("protein target sequence axes differ")
    accessions = tuple(str(value) for value in cast(list[object], accessions_raw))
    records = _sequences(arguments.fasta)
    protein_sequences = tuple(records[accession] for accession in accessions)
    if [len(value) for value in protein_sequences] != expected_lengths:
        raise ValueError("UniProt sequence lengths differ from prepared targets")
    if [sha256_ordered_strings((value,)) for value in protein_sequences] != expected_hashes:
        raise ValueError("UniProt sequence hashes differ from prepared targets")

    snapshot = Path(
        snapshot_download(repo_id=repository, revision=revision, cache_dir=arguments.model_cache)
    )
    checkpoint = snapshot / "model.safetensors"
    if sha256_file(checkpoint) != checkpoint_sha256:
        raise ValueError("ESM-2 checkpoint fingerprint differs")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModel.from_pretrained(snapshot, local_files_only=True)
    if int(model.config.hidden_size) != embedding_width:
        raise ValueError("ESM-2 hidden width differs")
    model.to(arguments.device)
    model.eval()

    embeddings: list[t.Tensor] = []
    chunk_counts: list[int] = []
    with t.inference_mode():
        for sequence_value in protein_sequences:
            ranges = protein_chunk_ranges(
                len(sequence_value), maximum_residues=maximum_residues
            )
            residue_sum = t.zeros(embedding_width, dtype=t.float64)
            observed_residues = 0
            for chunk in ranges:
                subsequence = sequence_value[chunk.start : chunk.stop]
                encoded = tokenizer(subsequence, return_tensors="pt", add_special_tokens=True)
                input_ids = encoded.get("input_ids")
                if input_ids is None or input_ids.shape != (1, len(subsequence) + 2):
                    raise ValueError("ESM-2 token/residue identity failed")
                output = model(**{name: value.to(arguments.device) for name, value in encoded.items()})
                hidden = output.last_hidden_state
                if hidden.shape != (1, len(subsequence) + 2, embedding_width):
                    raise ValueError("ESM-2 hidden-state shape differs")
                residue_hidden = hidden[0, 1 : 1 + len(subsequence)]
                if not bool(t.isfinite(residue_hidden).all().item()):
                    raise ValueError("ESM-2 residue embeddings contain non-finite values")
                residue_sum += residue_hidden.sum(dim=0).cpu().to(t.float64)
                observed_residues += len(subsequence)
            if observed_residues != len(sequence_value):
                raise RuntimeError("ESM-2 chunk pooling omitted or duplicated residues")
            embeddings.append((residue_sum / observed_residues).to(t.float32))
            chunk_counts.append(len(ranges))
    matrix = t.stack(embeddings)
    if matrix.shape != (52, embedding_width) or not bool(t.isfinite(matrix).all().item()):
        raise ValueError("ESM-2 protein embedding matrix differs")

    arguments.output.mkdir(parents=True, exist_ok=False)
    tensor_path = arguments.output / "embeddings.safetensors"
    save_safetensors_exclusive(tensor_path, {"embeddings": matrix})
    write_canonical_json_exclusive(
        arguments.output / "metadata.json",
        {
            "schema": "methylation-latent.protein-amino-acid-embeddings.v1",
            "protocol_sha256": sha256_file(arguments.config),
            "git_commit": git_commit,
            "protein_target_metadata_sha256": sha256_file(arguments.targets),
            "tensor_file": tensor_path.name,
            "tensor_sha256": sha256_file(tensor_path),
            "uniprot_accessions": list(accessions),
            "sequence_lengths": [len(value) for value in protein_sequences],
            "chunk_counts": chunk_counts,
            "maximum_residues_per_chunk": maximum_residues,
            "pooling": _string(sequence, "esm_pooling"),
            "model_repository": repository,
            "model_revision": revision,
            "model_checkpoint_sha256": checkpoint_sha256,
            "embedding_width": embedding_width,
            "dtype": "float32",
            "source_fasta_sha256": observed_fasta_hash,
            "checks": cast(
                dict[str, JsonValue],
                {
                    "all_residues_embedded_exactly_once": True,
                    "long_sequences_truncated": False,
                },
            ),
        },
    )
    print(f"embedded_amino_acids proteins=52 sha256={sha256_file(tensor_path)}")


if __name__ == "__main__":
    main()
