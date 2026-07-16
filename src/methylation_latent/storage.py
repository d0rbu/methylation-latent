"""Strict safetensors persistence for targets and precomputed embeddings."""

from __future__ import annotations

import os
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch as t
from safetensors.torch import load_file, save_file

from methylation_latent.targets import (
    CorrelationVector,
    TargetGeometry,
    UnitNormRows,
    UnitNormVector,
)


def save_safetensors_exclusive(path: Path, tensors: Mapping[str, t.Tensor]) -> None:
    """Atomically publish a new tensor file without overwriting an artifact."""

    if not tensors or any(not name for name in tensors):
        raise ValueError("safetensors payload requires named tensors")
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prepared: dict[str, t.Tensor] = {}
    for name, tensor in tensors.items():
        if tensor.layout != t.strided:
            raise ValueError(f"tensor {name!r} must use strided layout")
        if not bool(t.isfinite(tensor).all().item()):
            raise ValueError(f"tensor {name!r} must be finite")
        prepared[name] = tensor.detach().to(device="cpu").contiguous()
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(16)}.tmp")
    save_file(prepared, temporary)
    os.link(temporary, path)
    temporary.unlink()


def load_exact_safetensors(path: Path, expected_keys: set[str]) -> dict[str, t.Tensor]:
    tensors = load_file(path, device="cpu")
    observed = set(tensors)
    if observed != expected_keys:
        raise ValueError(
            f"safetensors keys differ: missing={sorted(expected_keys - observed)}, "
            f"unknown={sorted(observed - expected_keys)}"
        )
    if any(not bool(t.isfinite(tensor).all().item()) for tensor in tensors.values()):
        raise ValueError("safetensors payload contains non-finite values")
    return tensors


def save_target_geometry(path: Path, geometry: TargetGeometry) -> None:
    save_safetensors_exclusive(
        path,
        {
            "X": geometry.methylation.tensor,
            "age": geometry.age.tensor,
            "rho": geometry.rho.tensor,
        },
    )


def load_target_geometry(path: Path) -> TargetGeometry:
    tensors = load_exact_safetensors(path, {"X", "age", "rho"})
    if any(tensor.dtype != t.float64 for tensor in tensors.values()):
        raise TypeError("target geometry tensors must all be float64")
    return TargetGeometry(
        methylation=UnitNormRows(tensors["X"]),
        age=UnitNormVector(tensors["age"]),
        rho=CorrelationVector(tensors["rho"]),
    )


@dataclass(frozen=True, slots=True)
class EmbeddingMatrix:
    """Finite fp16 Caduceus centre-token embeddings with fixed width 256."""

    tensor: t.Tensor

    def __post_init__(self) -> None:
        if self.tensor.ndim != 2 or self.tensor.shape[0] == 0 or self.tensor.shape[1] != 256:
            raise ValueError("embedding matrix must have shape [non-empty probes, 256]")
        if self.tensor.dtype != t.float16:
            raise TypeError(f"cached embedding matrix must be float16, got {self.tensor.dtype}")
        if not bool(t.isfinite(self.tensor).all().item()):
            raise ValueError("embedding matrix must contain only finite values")

    def training_tensor(self, *, device: t.device | str) -> t.Tensor:
        values = self.tensor.to(device=device, dtype=t.float32)
        if values.dtype != t.float32:
            raise RuntimeError("embedding training conversion did not produce float32")
        return values


def save_embedding_matrix(path: Path, embeddings: EmbeddingMatrix) -> None:
    save_safetensors_exclusive(path, {"embeddings": embeddings.tensor})


def load_embedding_matrix(path: Path) -> EmbeddingMatrix:
    tensors = load_exact_safetensors(path, {"embeddings"})
    return EmbeddingMatrix(tensors["embeddings"])
