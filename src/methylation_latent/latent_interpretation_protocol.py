"""Frozen configuration for post-hoc latent interpretation."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from beartype import beartype

_SCHEMA_V1 = "methylation-latent.latent-interpretation-config.v1"
_SCHEMA_V2 = "methylation-latent.latent-interpretation-config.v2"
_STATUS = "post_hoc_hypothesis_generating"
_SHA256 = re.compile(r"^[0-9a-f]{64}$", flags=re.ASCII)
_SPLITS = ("diverse-blocks", "held-out-chromosome")
_WINDOWS_BY_SCHEMA = {
    _SCHEMA_V1: (1024, 4096),
    _SCHEMA_V2: (1024, 4096, 16384),
}
_VERSION_BY_SCHEMA = {_SCHEMA_V1: 1, _SCHEMA_V2: 2}
_SUPPORTED_WINDOWS = tuple(
    dict.fromkeys(window for windows in _WINDOWS_BY_SCHEMA.values() for window in windows)
)


def _positive(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TypeError(f"{name} must be a positive integer")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise TypeError(f"{name} must be a lowercase SHA-256 value")
    return value


def _object(raw: dict[str, object], name: str, fields: set[str]) -> dict[str, object]:
    value = raw.get(name)
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} fields differ from the frozen interpretation config")
    return cast(dict[str, object], value)


def _positive_tuple(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise TypeError(f"{name} must be a non-empty integer array")
    parsed = tuple(_positive(item, name) for item in value)
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{name} must not contain duplicates")
    return parsed


@dataclass(frozen=True, slots=True)
class ReliabilityProtocol:
    subject_split_seeds: tuple[int, ...]
    maximum_pairs_per_distance_class: int
    pair_sampling_seed: int

    def __post_init__(self) -> None:
        if len(self.subject_split_seeds) < 2:
            raise ValueError("reliability requires at least two registered subject splits")
        if len(set(self.subject_split_seeds)) != len(self.subject_split_seeds):
            raise ValueError("reliability subject seeds must be unique")
        if (
            min(
                self.maximum_pairs_per_distance_class,
                self.pair_sampling_seed,
                *self.subject_split_seeds,
            )
            <= 0
        ):
            raise ValueError("reliability counts and seeds must be positive")


@dataclass(frozen=True, slots=True)
class GeometryProtocol:
    covariance_sample_size: int
    covariance_sampling_seed: int
    neighbour_sample_size: int
    neighbour_count: int
    neighbour_sampling_seed: int

    def __post_init__(self) -> None:
        if (
            min(
                self.covariance_sample_size,
                self.covariance_sampling_seed,
                self.neighbour_sample_size,
                self.neighbour_count,
                self.neighbour_sampling_seed,
            )
            <= 0
        ):
            raise ValueError("geometry counts and seeds must be positive")
        if self.neighbour_count >= self.neighbour_sample_size:
            raise ValueError("neighbour count must be smaller than its sample")


@dataclass(frozen=True, slots=True)
class DisplayProtocol:
    age_probe_count: int
    pairs_per_distance_class: int
    sampling_seed: int
    candidate_ranks: tuple[int, ...]
    candidate_table_size: int

    def __post_init__(self) -> None:
        if (
            min(
                self.age_probe_count,
                self.pairs_per_distance_class,
                self.sampling_seed,
                self.candidate_table_size,
                *self.candidate_ranks,
            )
            <= 0
        ):
            raise ValueError("display counts, ranks, and seeds must be positive")
        if tuple(sorted(self.candidate_ranks)) != self.candidate_ranks:
            raise ValueError("candidate ranks must be strictly increasing")
        if self.candidate_table_size > self.candidate_ranks[-1]:
            raise ValueError("candidate table cannot exceed the largest registered rank")


@dataclass(frozen=True, slots=True)
class InterpretationParent:
    split: str
    window: int
    metadata_sha256: str
    model_sha256: str
    embedding_manifest_sha256: str

    def __post_init__(self) -> None:
        if self.split not in _SPLITS or self.window not in _SUPPORTED_WINDOWS:
            raise ValueError("interpretation parent split or window differs")
        if any(
            _SHA256.fullmatch(value) is None
            for value in (
                self.metadata_sha256,
                self.model_sha256,
                self.embedding_manifest_sha256,
            )
        ):
            raise ValueError("interpretation parent hashes must be SHA-256 values")


@dataclass(frozen=True, slots=True)
class LatentInterpretationProtocol:
    protocol_id: str
    protocol_sha256: str
    gpl13534_sha256: str
    reliability: ReliabilityProtocol
    geometry: GeometryProtocol
    display: DisplayProtocol
    parents: tuple[InterpretationParent, ...]
    schema: str = _SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema not in _WINDOWS_BY_SCHEMA:
            raise ValueError("latent interpretation config schema differs")
        if not self.protocol_id or _SHA256.fullmatch(self.protocol_sha256) is None:
            raise ValueError("primary protocol identity is incomplete")
        if _SHA256.fullmatch(self.gpl13534_sha256) is None:
            raise ValueError("GPL13534 source hash is invalid")
        keys = tuple((parent.split, parent.window) for parent in self.parents)
        expected = tuple((split, window) for split in _SPLITS for window in self.windows)
        if keys != expected:
            raise ValueError("interpretation parents must cover the frozen split/window grid")

    @property
    def splits(self) -> tuple[str, ...]:
        return _SPLITS

    @property
    def windows(self) -> tuple[int, ...]:
        return _WINDOWS_BY_SCHEMA[self.schema]

    @property
    def artifact_version(self) -> int:
        return _VERSION_BY_SCHEMA[self.schema]

    def parent(self, split: str, window: int) -> InterpretationParent:
        matches = tuple(
            parent for parent in self.parents if parent.split == split and parent.window == window
        )
        if len(matches) != 1:
            raise KeyError(f"no unique interpretation parent for {split}/window-{window}")
        return matches[0]


@beartype
def load_latent_interpretation_protocol(path: Path) -> LatentInterpretationProtocol:
    raw_value = tomllib.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_value, dict):
        raise TypeError("latent interpretation config must be a TOML table")
    raw = cast(dict[str, object], raw_value)
    expected_top = {
        "schema",
        "protocol_id",
        "protocol_sha256",
        "status",
        "windows",
        "splits",
        "sources",
        "reliability",
        "geometry",
        "display",
        "parents",
    }
    if set(raw) != expected_top:
        raise ValueError("latent interpretation top-level config fields differ")
    schema = raw["schema"]
    if not isinstance(schema, str) or schema not in _WINDOWS_BY_SCHEMA or raw["status"] != _STATUS:
        raise ValueError("latent interpretation schema or status differs")
    expected_windows = _WINDOWS_BY_SCHEMA[schema]
    windows_raw = raw["windows"]
    splits_raw = raw["splits"]
    if (
        not isinstance(windows_raw, list)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in windows_raw)
        or tuple(windows_raw) != expected_windows
        or not isinstance(splits_raw, list)
        or any(not isinstance(value, str) for value in splits_raw)
        or tuple(splits_raw) != _SPLITS
    ):
        raise ValueError("latent interpretation split/window grid differs")
    sources = _object(raw, "sources", {"gpl13534_sha256"})
    reliability = _object(
        raw,
        "reliability",
        {
            "subject_split_seeds",
            "maximum_pairs_per_distance_class",
            "pair_sampling_seed",
        },
    )
    geometry = _object(
        raw,
        "geometry",
        {
            "covariance_sample_size",
            "covariance_sampling_seed",
            "neighbour_sample_size",
            "neighbour_count",
            "neighbour_sampling_seed",
        },
    )
    display = _object(
        raw,
        "display",
        {
            "age_probe_count",
            "pairs_per_distance_class",
            "sampling_seed",
            "candidate_ranks",
            "candidate_table_size",
        },
    )
    raw_parents = raw["parents"]
    if not isinstance(raw_parents, list):
        raise TypeError("latent interpretation parents must be an array of tables")
    parent_fields = {
        "split",
        "window",
        "metadata_sha256",
        "model_sha256",
        "embedding_manifest_sha256",
    }
    parents: list[InterpretationParent] = []
    for raw_parent in raw_parents:
        if not isinstance(raw_parent, dict) or set(raw_parent) != parent_fields:
            raise ValueError("latent interpretation parent fields differ")
        parent = cast(dict[str, object], raw_parent)
        split = parent["split"]
        if not isinstance(split, str):
            raise TypeError("parent split must be a string")
        parents.append(
            InterpretationParent(
                split=split,
                window=_positive(parent["window"], "parent.window"),
                metadata_sha256=_sha256(parent["metadata_sha256"], "parent.metadata_sha256"),
                model_sha256=_sha256(parent["model_sha256"], "parent.model_sha256"),
                embedding_manifest_sha256=_sha256(
                    parent["embedding_manifest_sha256"],
                    "parent.embedding_manifest_sha256",
                ),
            )
        )
    protocol_id = raw["protocol_id"]
    if not isinstance(protocol_id, str) or not protocol_id:
        raise TypeError("protocol_id must be a non-empty string")
    return LatentInterpretationProtocol(
        protocol_id=protocol_id,
        protocol_sha256=_sha256(raw["protocol_sha256"], "protocol_sha256"),
        gpl13534_sha256=_sha256(sources["gpl13534_sha256"], "sources.gpl13534_sha256"),
        reliability=ReliabilityProtocol(
            subject_split_seeds=_positive_tuple(
                reliability["subject_split_seeds"],
                "reliability.subject_split_seeds",
            ),
            maximum_pairs_per_distance_class=_positive(
                reliability["maximum_pairs_per_distance_class"],
                "reliability.maximum_pairs_per_distance_class",
            ),
            pair_sampling_seed=_positive(
                reliability["pair_sampling_seed"],
                "reliability.pair_sampling_seed",
            ),
        ),
        geometry=GeometryProtocol(
            covariance_sample_size=_positive(
                geometry["covariance_sample_size"], "geometry.covariance_sample_size"
            ),
            covariance_sampling_seed=_positive(
                geometry["covariance_sampling_seed"], "geometry.covariance_sampling_seed"
            ),
            neighbour_sample_size=_positive(
                geometry["neighbour_sample_size"], "geometry.neighbour_sample_size"
            ),
            neighbour_count=_positive(geometry["neighbour_count"], "geometry.neighbour_count"),
            neighbour_sampling_seed=_positive(
                geometry["neighbour_sampling_seed"], "geometry.neighbour_sampling_seed"
            ),
        ),
        display=DisplayProtocol(
            age_probe_count=_positive(display["age_probe_count"], "display.age_probe_count"),
            pairs_per_distance_class=_positive(
                display["pairs_per_distance_class"], "display.pairs_per_distance_class"
            ),
            sampling_seed=_positive(display["sampling_seed"], "display.sampling_seed"),
            candidate_ranks=_positive_tuple(display["candidate_ranks"], "display.candidate_ranks"),
            candidate_table_size=_positive(
                display["candidate_table_size"], "display.candidate_table_size"
            ),
        ),
        parents=tuple(parents),
        schema=schema,
    )
