"""Pinned Caduceus loading and centre-CpG embedding extraction."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

import torch as t
from beartype import beartype

from methylation_latent.hashing import sha256_file
from methylation_latent.storage import EmbeddingMatrix

CADUCEUS_REPOSITORY = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
CADUCEUS_REVISION = "d89eeb853136ea64da7feb3d0c8e909771b17ae6"
CADUCEUS_CHECKPOINT_SHA256 = "a3e6976fe90460ff5d90d371b457d0263dc65e156ea5651b4a452e98228daef2"
CADUCEUS_EMBEDDING_WIDTH = 256
CADUCEUS_RCPS_RAW_WIDTH = 2 * CADUCEUS_EMBEDDING_WIDTH


class InferencePrecision(StrEnum):
    FLOAT16 = "float16"
    FLOAT32 = "float32"

    @property
    def torch_dtype(self) -> t.dtype:
        return t.float16 if self == InferencePrecision.FLOAT16 else t.float32


@beartype
@dataclass(frozen=True, slots=True)
class EmbeddingInferenceConfig:
    """All model/runtime choices required to reproduce one embedding shard."""

    device: str
    precision: InferencePrecision
    batch_size: int
    repository: str = CADUCEUS_REPOSITORY
    revision: str = CADUCEUS_REVISION
    checkpoint_sha256: str = CADUCEUS_CHECKPOINT_SHA256
    embedding_width: int = CADUCEUS_EMBEDDING_WIDTH

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("embedding batch size must be positive")
        if self.repository != CADUCEUS_REPOSITORY or self.revision != CADUCEUS_REVISION:
            raise ValueError("embedding model identity must match the frozen protocol")
        if self.checkpoint_sha256 != CADUCEUS_CHECKPOINT_SHA256:
            raise ValueError("embedding checkpoint hash must match the frozen protocol")
        if self.embedding_width != CADUCEUS_EMBEDDING_WIDTH:
            raise ValueError("Caduceus embedding width must be exactly 256")


@runtime_checkable
class TokenizerProtocol(Protocol):
    def __call__(
        self,
        sequences: Sequence[str],
        *,
        add_special_tokens: bool,
        padding: bool,
        return_tensors: str,
    ) -> Mapping[str, t.Tensor]: ...


@runtime_checkable
class BackboneOutputProtocol(Protocol):
    last_hidden_state: t.Tensor


@runtime_checkable
class BackboneProtocol(Protocol):
    def __call__(self, **inputs: t.Tensor) -> BackboneOutputProtocol: ...


@dataclass(frozen=True, slots=True)
class LoadedCaduceus:
    tokenizer: TokenizerProtocol
    model: BackboneProtocol
    snapshot_path: Path


def _validate_sequences(sequences: Sequence[str]) -> int:
    if not sequences:
        raise ValueError("embedding batch must contain at least one sequence")
    lengths = {len(sequence) for sequence in sequences}
    if len(lengths) != 1:
        raise ValueError("all sequences in an embedding batch must have the same width")
    width = next(iter(lengths))
    if width <= 0 or width % 2 != 0:
        raise ValueError("embedding sequences must have a positive even width")
    invalid = sorted(set("".join(sequences).upper()) - set("ACGT"))
    if invalid:
        raise ValueError(f"embedding sequences contain non-ACGT characters: {invalid}")
    centre = width // 2
    failures = tuple(
        index
        for index, sequence in enumerate(sequences)
        if sequence[centre : centre + 2].upper() != "CG"
    )
    if failures:
        raise ValueError(f"embedding sequences are not centred on plus-strand CG: {failures[:10]}")
    return width


@beartype
def load_pinned_caduceus(
    *,
    cache_directory: Path,
    config: EmbeddingInferenceConfig,
) -> LoadedCaduceus:
    """Download a pinned snapshot, verify bytes, then load entirely from that snapshot."""

    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(
        snapshot_download(
            repo_id=config.repository,
            revision=config.revision,
            cache_dir=cache_directory,
        )
    )
    checkpoint = snapshot / "model.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    observed_hash = sha256_file(checkpoint)
    if observed_hash != config.checkpoint_sha256:
        raise ValueError(
            f"Caduceus checkpoint hash differs: expected={config.checkpoint_sha256}, "
            f"observed={observed_hash}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        snapshot,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=config.precision.torch_dtype,
    )
    if model.config.d_model != config.embedding_width or model.config.rcps is not True:
        raise ValueError(
            "Caduceus architecture differs from the frozen RCPS d_model=256 protocol: "
            f"d_model={model.config.d_model}, rcps={model.config.rcps}"
        )
    model.to(config.device)
    model.eval()
    return LoadedCaduceus(
        tokenizer=cast(TokenizerProtocol, tokenizer),
        model=cast(BackboneProtocol, model),
        snapshot_path=snapshot,
    )


@beartype
def embed_center_tokens(
    sequences: Sequence[str],
    *,
    tokenizer: TokenizerProtocol,
    model: BackboneProtocol,
    config: EmbeddingInferenceConfig,
) -> t.Tensor:
    """Embed only the plus-strand cytosine token, never a pooled representation."""

    width = _validate_sequences(sequences)
    encoded = tokenizer(
        sequences,
        add_special_tokens=False,
        padding=False,
        return_tensors="pt",
    )
    if "input_ids" not in encoded:
        raise ValueError("Caduceus tokenizer output lacks input_ids")
    input_ids = encoded["input_ids"]
    if input_ids.shape != (len(sequences), width):
        raise ValueError(
            f"token/base identity failed: expected {(len(sequences), width)}, "
            f"observed {tuple(input_ids.shape)}"
        )
    device_inputs = {name: tensor.to(config.device) for name, tensor in encoded.items()}
    device_type = t.device(config.device).type
    autocast_enabled = config.precision == InferencePrecision.FLOAT16
    if autocast_enabled and device_type != "cuda":
        raise ValueError("float16 Caduceus inference is allowed only on CUDA")
    with (
        t.inference_mode(),
        t.autocast(
            device_type=device_type,
            dtype=config.precision.torch_dtype,
            enabled=autocast_enabled,
        ),
    ):
        output = model(**device_inputs)
    hidden = output.last_hidden_state
    expected_shape = (len(sequences), width, CADUCEUS_RCPS_RAW_WIDTH)
    if hidden.shape != expected_shape:
        raise ValueError(
            f"Caduceus hidden state shape differs: expected={expected_shape}, "
            f"observed={tuple(hidden.shape)}"
        )
    if not bool(t.isfinite(hidden).all().item()):
        raise ValueError("Caduceus hidden state contains non-finite values")
    plus_strand = hidden[..., : config.embedding_width]
    return plus_strand[:, width // 2, :].to(device="cpu", dtype=t.float16).contiguous()


@beartype
def embed_all_sequences(
    sequences: Sequence[str],
    *,
    loaded: LoadedCaduceus,
    config: EmbeddingInferenceConfig,
) -> EmbeddingMatrix:
    """Embed one ordered probe universe in fixed, non-overlapping inference batches."""

    if not sequences:
        raise ValueError("embedding sequence collection must not be empty")
    shards = tuple(
        embed_center_tokens(
            sequences[start : start + config.batch_size],
            tokenizer=loaded.tokenizer,
            model=loaded.model,
            config=config,
        )
        for start in range(0, len(sequences), config.batch_size)
    )
    embeddings = t.cat(shards, dim=0)
    if embeddings.shape[0] != len(sequences):
        raise RuntimeError("embedding concatenation changed the probe count")
    return EmbeddingMatrix(embeddings)


@beartype
def embed_sequence_batches(
    batches: Iterable[Sequence[str]],
    *,
    expected_probe_count: int,
    loaded: LoadedCaduceus,
    config: EmbeddingInferenceConfig,
) -> EmbeddingMatrix:
    """Stream extracted sequence batches so 64-kb windows are never all resident in RAM."""

    if expected_probe_count <= 0:
        raise ValueError("expected probe count must be positive")
    shards: list[t.Tensor] = []
    observed = 0
    for batch in batches:
        if not 0 < len(batch) <= config.batch_size:
            raise ValueError("streamed sequence batch has an invalid size")
        shard = embed_center_tokens(
            batch,
            tokenizer=loaded.tokenizer,
            model=loaded.model,
            config=config,
        )
        shards.append(shard)
        observed += shard.shape[0]
        if observed > expected_probe_count:
            raise ValueError("streamed sequence batches exceed the expected probe count")
    if observed != expected_probe_count:
        raise ValueError(
            f"streamed sequence coverage differs: expected={expected_probe_count}, "
            f"observed={observed}"
        )
    return EmbeddingMatrix(t.cat(shards, dim=0))
