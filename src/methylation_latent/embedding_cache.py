"""Restart-safe immutable Caduceus embedding shards."""

from __future__ import annotations

import json
import os
import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch as t
from beartype import beartype

from methylation_latent.artifacts import (
    JsonValue,
    sha256_file,
    sha256_ordered_strings,
    write_canonical_json_exclusive,
)
from methylation_latent.domain import NonEmptyProbeSet, WindowSize
from methylation_latent.embeddings import (
    CADUCEUS_CHECKPOINT_SHA256,
    CADUCEUS_REPOSITORY,
    CADUCEUS_REVISION,
)
from methylation_latent.storage import (
    EmbeddingMatrix,
    load_embedding_matrix,
    save_embedding_matrix,
)

_SHARD_SCHEMA = "methylation-latent.caduceus-embedding-shard.v2"
_MANIFEST_SCHEMA = "methylation-latent.caduceus-embedding-cache.v2"
_SHARD_PAYLOAD = "embeddings.safetensors"
_SHARD_METADATA = "metadata.json"
_CACHE_MANIFEST = "manifest.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$", flags=re.ASCII)
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$", flags=re.ASCII)


def _probe_order_sha256(probes: NonEmptyProbeSet) -> str:
    return sha256_ordered_strings(str(probe.probe_id) for probe in probes.probes)


@dataclass(frozen=True, slots=True)
class EmbeddingShardIdentity:
    git_commit: str
    window_size: int
    start: int
    stop: int
    probe_order_sha256: str
    payload_sha256: str

    def __post_init__(self) -> None:
        if _GIT_COMMIT.fullmatch(self.git_commit) is None:
            raise ValueError("embedding shard Git commit must be 40 lowercase hex characters")
        if self.window_size <= 0 or self.window_size % 2 != 0:
            raise ValueError("embedding shard window must be positive and even")
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("embedding shard range must be non-empty and increasing")
        if (
            _SHA256.fullmatch(self.probe_order_sha256) is None
            or _SHA256.fullmatch(self.payload_sha256) is None
        ):
            raise ValueError("embedding shard fingerprints must be SHA-256 values")

    @property
    def row_count(self) -> int:
        return self.stop - self.start

    @property
    def directory_name(self) -> str:
        return f"shard-{self.start:09d}-{self.stop:09d}"


def embedding_shard_ranges(probe_count: int, shard_size: int) -> tuple[tuple[int, int], ...]:
    if probe_count <= 0 or shard_size <= 0:
        raise ValueError("embedding probe and shard counts must be positive")
    return tuple(
        (start, min(start + shard_size, probe_count)) for start in range(0, probe_count, shard_size)
    )


def _metadata_json(identity: EmbeddingShardIdentity) -> dict[str, JsonValue]:
    return {
        "schema": _SHARD_SCHEMA,
        "repository": CADUCEUS_REPOSITORY,
        "revision": CADUCEUS_REVISION,
        "checkpoint_sha256": CADUCEUS_CHECKPOINT_SHA256,
        "git_commit": identity.git_commit,
        "window_size": identity.window_size,
        "start": identity.start,
        "stop": identity.stop,
        "probe_order_sha256": identity.probe_order_sha256,
        "payload_file": _SHARD_PAYLOAD,
        "payload_sha256": identity.payload_sha256,
        "dtype": "float16",
        "embedding_width": 256,
    }


@beartype
def save_embedding_shard_exclusive(
    cache_directory: Path,
    embeddings: EmbeddingMatrix,
    probes: NonEmptyProbeSet,
    *,
    git_commit: str,
    window_size: WindowSize,
    start: int,
    stop: int,
) -> EmbeddingShardIdentity:
    if stop - start != embeddings.tensor.shape[0]:
        raise ValueError("embedding shard row count differs from its recorded range")
    if stop > len(probes):
        raise IndexError("embedding shard range exceeds the global probe universe")
    probe_hash = sha256_ordered_strings(str(probe.probe_id) for probe in probes.probes[start:stop])
    cache_directory.mkdir(parents=True, exist_ok=True)
    final_directory = cache_directory / f"shard-{start:09d}-{stop:09d}"
    if final_directory.exists():
        raise FileExistsError(final_directory)
    temporary = cache_directory / (f".{final_directory.name}.{secrets.token_hex(16)}.tmp")
    temporary.mkdir()
    payload = temporary / _SHARD_PAYLOAD
    save_embedding_matrix(payload, embeddings)
    identity = EmbeddingShardIdentity(
        git_commit=git_commit,
        window_size=int(window_size),
        start=start,
        stop=stop,
        probe_order_sha256=probe_hash,
        payload_sha256=sha256_file(payload),
    )
    write_canonical_json_exclusive(
        temporary / _SHARD_METADATA,
        _metadata_json(identity),
    )
    os.rename(temporary, final_directory)
    return identity


def _load_shard_metadata(path: Path) -> EmbeddingShardIdentity:
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema",
        "repository",
        "revision",
        "checkpoint_sha256",
        "git_commit",
        "window_size",
        "start",
        "stop",
        "probe_order_sha256",
        "payload_file",
        "payload_sha256",
        "dtype",
        "embedding_width",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("embedding shard metadata envelope differs")
    fixed_identity = (
        raw["schema"],
        raw["repository"],
        raw["revision"],
        raw["checkpoint_sha256"],
        raw["payload_file"],
        raw["dtype"],
        raw["embedding_width"],
    )
    expected_identity = (
        _SHARD_SCHEMA,
        CADUCEUS_REPOSITORY,
        CADUCEUS_REVISION,
        CADUCEUS_CHECKPOINT_SHA256,
        _SHARD_PAYLOAD,
        "float16",
        256,
    )
    if fixed_identity != expected_identity:
        raise ValueError("embedding shard model, schema, or tensor identity differs")
    return EmbeddingShardIdentity(
        git_commit=str(raw["git_commit"]),
        window_size=int(raw["window_size"]),
        start=int(raw["start"]),
        stop=int(raw["stop"]),
        probe_order_sha256=str(raw["probe_order_sha256"]),
        payload_sha256=str(raw["payload_sha256"]),
    )


@beartype
def load_embedding_shard(
    shard_directory: Path,
    probes: NonEmptyProbeSet,
    *,
    window_size: WindowSize,
    start: int,
    stop: int,
    expected_git_commit: str | None = None,
) -> tuple[EmbeddingShardIdentity, EmbeddingMatrix]:
    observed_names = {path.name for path in shard_directory.iterdir()}
    if observed_names != {_SHARD_PAYLOAD, _SHARD_METADATA}:
        raise ValueError("embedding shard file inventory differs")
    identity = _load_shard_metadata(shard_directory / _SHARD_METADATA)
    if expected_git_commit is not None and identity.git_commit != expected_git_commit:
        raise ValueError("embedding shard Git commit differs")
    if (
        identity.window_size != int(window_size)
        or identity.start != start
        or identity.stop != stop
        or shard_directory.name != identity.directory_name
    ):
        raise ValueError("embedding shard range, window, or directory name differs")
    expected_probe_hash = sha256_ordered_strings(
        str(probe.probe_id) for probe in probes.probes[start:stop]
    )
    if identity.probe_order_sha256 != expected_probe_hash:
        raise ValueError("embedding shard probe-order fingerprint differs")
    payload = shard_directory / _SHARD_PAYLOAD
    if sha256_file(payload) != identity.payload_sha256:
        raise ValueError("embedding shard payload fingerprint differs")
    embeddings = load_embedding_matrix(payload)
    if embeddings.tensor.shape[0] != identity.row_count:
        raise ValueError("embedding shard payload row count differs")
    return identity, embeddings


@beartype
def finalize_embedding_cache_exclusive(
    cache_directory: Path,
    probes: NonEmptyProbeSet,
    *,
    git_commit: str,
    window_size: WindowSize,
    shard_size: int,
) -> None:
    if _GIT_COMMIT.fullmatch(git_commit) is None:
        raise ValueError("embedding cache Git commit must be 40 lowercase hex characters")
    ranges = embedding_shard_ranges(len(probes), shard_size)
    identities = tuple(
        load_embedding_shard(
            cache_directory / f"shard-{start:09d}-{stop:09d}",
            probes,
            window_size=window_size,
            start=start,
            stop=stop,
            expected_git_commit=git_commit,
        )[0]
        for start, stop in ranges
    )
    expected_entries = {identity.directory_name for identity in identities}
    observed_entries = {
        path.name for path in cache_directory.iterdir() if path.name != _CACHE_MANIFEST
    }
    if observed_entries != expected_entries:
        raise ValueError(
            "embedding cache contains missing or unknown shard entries: "
            f"missing={sorted(expected_entries - observed_entries)[:10]}, "
            f"unknown={sorted(observed_entries - expected_entries)[:10]}"
        )
    write_canonical_json_exclusive(
        cache_directory / _CACHE_MANIFEST,
        {
            "schema": _MANIFEST_SCHEMA,
            "repository": CADUCEUS_REPOSITORY,
            "revision": CADUCEUS_REVISION,
            "checkpoint_sha256": CADUCEUS_CHECKPOINT_SHA256,
            "git_commit": git_commit,
            "window_size": int(window_size),
            "probe_count": len(probes),
            "probe_order_sha256": _probe_order_sha256(probes),
            "shard_size": shard_size,
            "shards": [
                {
                    "directory": identity.directory_name,
                    "git_commit": identity.git_commit,
                    "start": identity.start,
                    "stop": identity.stop,
                    "probe_order_sha256": identity.probe_order_sha256,
                    "payload_sha256": identity.payload_sha256,
                }
                for identity in identities
            ],
        },
    )


@beartype
def load_embedding_cache(
    cache_directory: Path,
    probes: NonEmptyProbeSet,
    *,
    window_size: WindowSize,
    expected_git_commit: str | None = None,
) -> EmbeddingMatrix:
    manifest_path = cache_directory / _CACHE_MANIFEST
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema",
        "repository",
        "revision",
        "checkpoint_sha256",
        "git_commit",
        "window_size",
        "probe_count",
        "probe_order_sha256",
        "shard_size",
        "shards",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("embedding-cache manifest envelope differs")
    git_commit = raw["git_commit"]
    if not isinstance(git_commit, str) or _GIT_COMMIT.fullmatch(git_commit) is None:
        raise ValueError("embedding-cache manifest Git commit is invalid")
    if expected_git_commit is not None and git_commit != expected_git_commit:
        raise ValueError("embedding-cache manifest Git commit differs")
    identity = (
        raw["schema"],
        raw["repository"],
        raw["revision"],
        raw["checkpoint_sha256"],
        raw["window_size"],
        raw["probe_count"],
        raw["probe_order_sha256"],
    )
    expected_identity = (
        _MANIFEST_SCHEMA,
        CADUCEUS_REPOSITORY,
        CADUCEUS_REVISION,
        CADUCEUS_CHECKPOINT_SHA256,
        int(window_size),
        len(probes),
        _probe_order_sha256(probes),
    )
    if identity != expected_identity:
        raise ValueError("embedding-cache manifest identity differs")
    shard_size = int(raw["shard_size"])
    ranges = embedding_shard_ranges(len(probes), shard_size)
    shards_raw = raw["shards"]
    if not isinstance(shards_raw, list) or len(shards_raw) != len(ranges):
        raise ValueError("embedding-cache shard manifest length differs")
    tensors: list[t.Tensor] = []
    for raw_shard, (start, stop) in zip(shards_raw, ranges, strict=True):
        if not isinstance(raw_shard, dict):
            raise ValueError("embedding-cache shard record must be an object")
        shard_record = cast(dict[str, object], raw_shard)
        expected_record = {
            "directory",
            "git_commit",
            "start",
            "stop",
            "probe_order_sha256",
            "payload_sha256",
        }
        if set(shard_record) != expected_record:
            raise ValueError("embedding-cache shard record fields differ")
        directory = cache_directory / str(shard_record["directory"])
        shard_identity, embeddings = load_embedding_shard(
            directory,
            probes,
            window_size=window_size,
            start=start,
            stop=stop,
        )
        manifest_record = (
            shard_record["git_commit"],
            shard_record["start"],
            shard_record["stop"],
            shard_record["probe_order_sha256"],
            shard_record["payload_sha256"],
        )
        observed_record = (
            shard_identity.git_commit,
            shard_identity.start,
            shard_identity.stop,
            shard_identity.probe_order_sha256,
            shard_identity.payload_sha256,
        )
        if manifest_record != observed_record:
            raise ValueError("embedding-cache shard record differs from its verified directory")
        if shard_identity.git_commit != git_commit:
            raise ValueError("embedding-cache shard Git commit differs from its manifest")
        tensors.append(embeddings.tensor)
    concatenated = t.cat(tensors, dim=0)
    if concatenated.shape != (len(probes), 256):
        raise RuntimeError("embedding-cache concatenation changed the expected global shape")
    return EmbeddingMatrix(concatenated)
