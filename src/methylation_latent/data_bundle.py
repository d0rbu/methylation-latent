"""Exclusive sealing and exact verification of a complete primary-data directory."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

from beartype import beartype

from methylation_latent.artifacts import (
    JsonValue,
    sha256_file,
    write_canonical_json_exclusive,
)

_SCHEMA = "methylation-latent.primary-data-bundle.v2"
_BUNDLE_FILE = "bundle.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$", flags=re.ASCII)
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$", flags=re.ASCII)


@dataclass(frozen=True, slots=True)
class BundledFile:
    path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        parsed = PurePosixPath(self.path)
        if (
            not self.path
            or parsed.is_absolute()
            or ".." in parsed.parts
            or self.path == _BUNDLE_FILE
        ):
            raise ValueError("bundled paths must be safe relative POSIX paths")
        if self.size < 0:
            raise ValueError("bundled file size cannot be negative")
        if _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("bundled files require lowercase SHA-256 values")

    def as_json(self) -> dict[str, JsonValue]:
        return {"path": self.path, "size": self.size, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class PrimaryDataBundle:
    protocol_id: str
    protocol_sha256: str
    git_commit: str
    retained_probes: int
    retained_samples: int
    files: tuple[BundledFile, ...]

    def __post_init__(self) -> None:
        if not self.protocol_id or _SHA256.fullmatch(self.protocol_sha256) is None:
            raise ValueError("data bundle requires a protocol ID and SHA-256")
        if _GIT_COMMIT.fullmatch(self.git_commit) is None:
            raise ValueError("data bundle Git commit must be 40 lowercase hex characters")
        if self.retained_probes <= 0 or self.retained_samples <= 1:
            raise ValueError("data bundle retained dimensions are invalid")
        paths = tuple(file.path for file in self.files)
        if not paths or paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
            raise ValueError("data bundle paths must be non-empty, unique, and sorted")

    def as_json(self) -> dict[str, JsonValue]:
        return {
            "schema": _SCHEMA,
            "protocol_id": self.protocol_id,
            "protocol_sha256": self.protocol_sha256,
            "git_commit": self.git_commit,
            "retained_probes": self.retained_probes,
            "retained_samples": self.retained_samples,
            "files": [file.as_json() for file in self.files],
        }


def _inventory(directory: Path) -> tuple[BundledFile, ...]:
    files = tuple(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.relative_to(directory).as_posix() != _BUNDLE_FILE
    )
    return tuple(
        BundledFile(
            path=path.relative_to(directory).as_posix(),
            size=path.stat().st_size,
            sha256=sha256_file(path),
        )
        for path in sorted(files, key=lambda value: value.relative_to(directory).as_posix())
    )


@beartype
def seal_primary_data_bundle_exclusive(
    directory: Path,
    *,
    protocol_id: str,
    protocol_sha256: str,
    git_commit: str,
    retained_probes: int,
    retained_samples: int,
) -> PrimaryDataBundle:
    """Hash every primary-data file and exclusively publish one immutable inventory."""

    if not directory.is_dir():
        raise NotADirectoryError(directory)
    bundle = PrimaryDataBundle(
        protocol_id=protocol_id,
        protocol_sha256=protocol_sha256,
        git_commit=git_commit,
        retained_probes=retained_probes,
        retained_samples=retained_samples,
        files=_inventory(directory),
    )
    write_canonical_json_exclusive(directory / _BUNDLE_FILE, bundle.as_json())
    return bundle


def _parse_file(raw: object) -> BundledFile:
    if not isinstance(raw, dict) or set(raw) != {"path", "size", "sha256"}:
        raise ValueError("data-bundle file record fields differ")
    record = cast(dict[str, object], raw)
    path = record["path"]
    size = record["size"]
    sha256 = record["sha256"]
    if (
        not isinstance(path, str)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or not isinstance(sha256, str)
    ):
        raise TypeError("data-bundle file record types differ")
    return BundledFile(path=path, size=size, sha256=sha256)


@beartype
def verify_primary_data_bundle(
    directory: Path,
    *,
    protocol_id: str,
    protocol_sha256: str,
) -> PrimaryDataBundle:
    """Reject any missing, added, or byte-changed file in a sealed data directory."""

    bundle_path = directory / _BUNDLE_FILE
    raw = json.loads(bundle_path.read_text(encoding="utf-8"))
    expected = {
        "schema",
        "protocol_id",
        "protocol_sha256",
        "git_commit",
        "retained_probes",
        "retained_samples",
        "files",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("primary-data bundle envelope differs")
    files_raw = raw["files"]
    git_commit = raw["git_commit"]
    retained_probes = raw["retained_probes"]
    retained_samples = raw["retained_samples"]
    if (
        not isinstance(files_raw, list)
        or not isinstance(git_commit, str)
        or isinstance(retained_probes, bool)
        or not isinstance(retained_probes, int)
        or isinstance(retained_samples, bool)
        or not isinstance(retained_samples, int)
    ):
        raise TypeError("primary-data bundle field types differ")
    if (
        raw["schema"] != _SCHEMA
        or raw["protocol_id"] != protocol_id
        or raw["protocol_sha256"] != protocol_sha256
    ):
        raise ValueError("primary-data bundle protocol identity differs")
    bundle = PrimaryDataBundle(
        protocol_id=protocol_id,
        protocol_sha256=protocol_sha256,
        git_commit=git_commit,
        retained_probes=retained_probes,
        retained_samples=retained_samples,
        files=tuple(_parse_file(item) for item in files_raw),
    )
    observed = _inventory(directory)
    if observed != bundle.files:
        expected_by_path = {file.path: file for file in bundle.files}
        observed_by_path = {file.path: file for file in observed}
        missing = sorted(expected_by_path.keys() - observed_by_path.keys())
        added = sorted(observed_by_path.keys() - expected_by_path.keys())
        changed = sorted(
            path
            for path in expected_by_path.keys() & observed_by_path.keys()
            if expected_by_path[path] != observed_by_path[path]
        )
        raise ValueError(
            "primary-data bundle inventory differs: "
            f"missing={missing[:10]}, added={added[:10]}, changed={changed[:10]}"
        )
    return bundle
