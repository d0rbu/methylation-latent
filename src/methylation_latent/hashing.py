"""Small deterministic hashing primitives shared with isolated runtimes."""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    if chunk_size <= 0:
        raise ValueError("hash chunk size must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Return MD5 only for matching externally published reference bytes."""

    if chunk_size <= 0:
        raise ValueError("hash chunk size must be positive")
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_gzip_payload_matches_file(
    archive_path: Path,
    payload_path: Path,
    *,
    chunk_size: int = 1 << 20,
) -> str:
    """Read through the gzip CRC and assert exact decompressed-file byte identity."""

    if chunk_size <= 0:
        raise ValueError("gzip comparison chunk size must be positive")
    digest = hashlib.sha256()
    observed_bytes = 0
    with gzip.open(archive_path, mode="rb") as archive, payload_path.open("rb") as payload:
        for archive_chunk in iter(lambda: archive.read(chunk_size), b""):
            payload_chunk = payload.read(len(archive_chunk))
            if payload_chunk != archive_chunk:
                raise ValueError("decompressed FASTA bytes differ from the pinned gzip payload")
            digest.update(archive_chunk)
            observed_bytes += len(archive_chunk)
        if payload.read(1) != b"":
            raise ValueError("decompressed FASTA has trailing bytes absent from the gzip payload")
    if observed_bytes == 0:
        raise ValueError("reference gzip payload is empty")
    return digest.hexdigest()
