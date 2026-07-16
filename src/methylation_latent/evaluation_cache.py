"""Immutable pair samples and exact target values shared by every model evaluation."""

from __future__ import annotations

import json
import os
import re
import secrets
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

import torch as t
from beartype import beartype

from methylation_latent.artifacts import (
    JsonValue,
    sha256_file,
    write_canonical_json_exclusive,
)
from methylation_latent.domain import NonEmptyProbeSet
from methylation_latent.evaluation import (
    DistanceBaseline,
    PairIndices,
    PairPopulation,
    build_cross_partition_pairs,
    build_stratified_cross_partition_pairs,
    build_stratified_within_partition_pairs,
    build_within_partition_pairs,
    fit_distance_baseline,
    gather_pair_targets_chunked,
)
from methylation_latent.storage import (
    load_exact_safetensors,
    save_safetensors_exclusive,
)
from methylation_latent.targets import UnitNormRows

_SCHEMA = "methylation-latent.evaluation-pair-cache.v1"
_PAYLOAD = "pairs.safetensors"
_METADATA = "metadata.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$", flags=re.ASCII)


class PairSetName(StrEnum):
    """Frozen pair samples with pooled and distance-stratified roles kept distinct."""

    TRAINING_STRATIFIED = "training_stratified"
    SEEN_UNIFORM = "seen_uniform"
    HELD_OUT_UNIFORM = "held_out_uniform"
    SEEN_STRATIFIED = "seen_stratified"
    HELD_OUT_STRATIFIED = "held_out_stratified"


_EXPECTED_POPULATIONS = {
    PairSetName.TRAINING_STRATIFIED: PairPopulation.TRAINING_BY_TRAINING,
    PairSetName.SEEN_UNIFORM: PairPopulation.SEEN_BY_HELD_OUT,
    PairSetName.HELD_OUT_UNIFORM: PairPopulation.HELD_OUT_BY_HELD_OUT,
    PairSetName.SEEN_STRATIFIED: PairPopulation.SEEN_BY_HELD_OUT,
    PairSetName.HELD_OUT_STRATIFIED: PairPopulation.HELD_OUT_BY_HELD_OUT,
}


@dataclass(frozen=True, slots=True)
class EvaluationPairCacheIdentity:
    """Complete scientific identity required before a pair cache may be reused."""

    protocol_id: str
    protocol_sha256: str
    data_sha256: str
    target_sha256: str
    split_name: str
    split_sha256: str
    probe_order_sha256: str

    def __post_init__(self) -> None:
        if not self.protocol_id or not self.split_name:
            raise ValueError("evaluation-cache protocol and split names must be non-empty")
        hashes = (
            self.protocol_sha256,
            self.data_sha256,
            self.target_sha256,
            self.split_sha256,
            self.probe_order_sha256,
        )
        if any(_SHA256.fullmatch(value) is None for value in hashes):
            raise ValueError("evaluation-cache identities require lowercase SHA-256 values")

    def as_json(self) -> dict[str, JsonValue]:
        return cast(dict[str, JsonValue], asdict(self))


@dataclass(frozen=True, slots=True)
class CachedPairSet:
    """One immutable pair-index population and its precomputed empirical correlations."""

    pairs: PairIndices
    targets: t.Tensor

    def __post_init__(self) -> None:
        if self.targets.dtype != t.float64 or self.targets.ndim != 1:
            raise TypeError("cached pair targets must be a float64 vector")
        if self.targets.shape != self.pairs.left.shape:
            raise ValueError("cached pair targets and indices have different lengths")
        if not bool(t.isfinite(self.targets).all().item()):
            raise ValueError("cached pair targets must be finite")
        if bool(t.any(self.targets.abs() > 1.0 + 1.0e-12).item()):
            raise ValueError("cached pair targets fall outside correlation bounds")


@dataclass(frozen=True, slots=True)
class EvaluationPairCache:
    """All pair samples needed for one split, including its training-only baseline."""

    pair_sets: dict[PairSetName, CachedPairSet]
    distance_baseline: DistanceBaseline

    def __post_init__(self) -> None:
        if set(self.pair_sets) != set(PairSetName):
            raise ValueError("evaluation pair cache does not contain the five frozen pair sets")
        for name, expected_population in _EXPECTED_POPULATIONS.items():
            if self.pair_sets[name].pairs.population != expected_population:
                raise ValueError(f"{name} has the wrong pair population")
        training = self.pair_sets[PairSetName.TRAINING_STRATIFIED]
        recomputed = fit_distance_baseline(training.pairs, training.targets)
        if not t.equal(recomputed.counts, self.distance_baseline.counts) or not t.equal(
            recomputed.means,
            self.distance_baseline.means,
        ):
            raise ValueError("cached distance baseline differs from its training-only pairs")


@beartype
def build_evaluation_pair_cache(
    probes: NonEmptyProbeSet,
    rows: UnitNormRows,
    train_indices: t.Tensor,
    test_indices: t.Tensor,
    *,
    maximum_uniform_pairs: int,
    maximum_pairs_per_distance_class: int,
    uniform_seed: int,
    distance_seed: int,
    target_chunk_size: int,
) -> EvaluationPairCache:
    """Build deterministic samples once and gather each empirical target once."""

    if rows.n_rows != len(probes):
        raise ValueError("target rows and global probe universe differ")
    if maximum_uniform_pairs <= 0 or maximum_pairs_per_distance_class <= 0:
        raise ValueError("evaluation pair caps must be positive")
    if target_chunk_size <= 0:
        raise ValueError("pair-target chunk size must be positive")
    pairs = {
        PairSetName.TRAINING_STRATIFIED: build_stratified_within_partition_pairs(
            probes,
            train_indices,
            population=PairPopulation.TRAINING_BY_TRAINING,
            maximum_pairs_per_distance_class=maximum_pairs_per_distance_class,
            seed=distance_seed,
        ),
        PairSetName.SEEN_UNIFORM: build_cross_partition_pairs(
            probes,
            train_indices,
            test_indices,
            maximum_pairs=maximum_uniform_pairs,
            seed=uniform_seed,
        ),
        PairSetName.HELD_OUT_UNIFORM: build_within_partition_pairs(
            probes,
            test_indices,
            population=PairPopulation.HELD_OUT_BY_HELD_OUT,
            maximum_pairs=maximum_uniform_pairs,
            seed=uniform_seed + 1,
        ),
        PairSetName.SEEN_STRATIFIED: build_stratified_cross_partition_pairs(
            probes,
            train_indices,
            test_indices,
            maximum_pairs_per_distance_class=maximum_pairs_per_distance_class,
            seed=distance_seed + 1,
        ),
        PairSetName.HELD_OUT_STRATIFIED: build_stratified_within_partition_pairs(
            probes,
            test_indices,
            population=PairPopulation.HELD_OUT_BY_HELD_OUT,
            maximum_pairs_per_distance_class=maximum_pairs_per_distance_class,
            seed=distance_seed + 2,
        ),
    }
    pair_sets = {
        name: CachedPairSet(
            pair_set,
            gather_pair_targets_chunked(
                rows,
                pair_set,
                chunk_size=target_chunk_size,
            ),
        )
        for name, pair_set in pairs.items()
    }
    training = pair_sets[PairSetName.TRAINING_STRATIFIED]
    return EvaluationPairCache(
        pair_sets=pair_sets,
        distance_baseline=fit_distance_baseline(training.pairs, training.targets),
    )


def _tensor_key(name: PairSetName, field: str) -> str:
    return f"{name.value}_{field}"


def _payload(cache: EvaluationPairCache) -> dict[str, t.Tensor]:
    tensors: dict[str, t.Tensor] = {
        "distance_baseline_means": cache.distance_baseline.means,
        "distance_baseline_counts": cache.distance_baseline.counts,
    }
    for name, cached in cache.pair_sets.items():
        tensors.update(
            {
                _tensor_key(name, "left"): cached.pairs.left,
                _tensor_key(name, "right"): cached.pairs.right,
                _tensor_key(name, "distance_class"): cached.pairs.distance_class,
                _tensor_key(name, "targets"): cached.targets,
            }
        )
    return tensors


def _pair_record(name: PairSetName, cached: CachedPairSet) -> dict[str, JsonValue]:
    counts = t.bincount(cached.pairs.distance_class, minlength=8)
    return {
        "name": name.value,
        "population": cached.pairs.population.value,
        "count": cached.pairs.count,
        "total_possible_pairs": cached.pairs.total_possible_pairs,
        "seed": cached.pairs.seed,
        "distance_class_counts": counts.tolist(),
    }


@beartype
def save_evaluation_pair_cache_exclusive(
    directory: Path,
    cache: EvaluationPairCache,
    identity: EvaluationPairCacheIdentity,
) -> None:
    """Publish a complete pair cache atomically without replacing prior evidence."""

    if directory.exists():
        raise FileExistsError(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = directory.parent / f".{directory.name}.{secrets.token_hex(16)}.tmp"
    temporary.mkdir()
    payload_path = temporary / _PAYLOAD
    save_safetensors_exclusive(payload_path, _payload(cache))
    write_canonical_json_exclusive(
        temporary / _METADATA,
        {
            "schema": _SCHEMA,
            "identity": identity.as_json(),
            "payload_file": _PAYLOAD,
            "payload_sha256": sha256_file(payload_path),
            "pair_sets": [_pair_record(name, cache.pair_sets[name]) for name in PairSetName],
        },
    )
    os.rename(temporary, directory)


def _required_int(record: dict[str, object], name: str) -> int:
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"evaluation-cache {name} must be an integer")
    return value


def _load_pair_records(raw: object) -> dict[PairSetName, dict[str, object]]:
    if not isinstance(raw, list) or len(raw) != len(PairSetName):
        raise ValueError("evaluation-cache pair-set manifest length differs")
    records: dict[PairSetName, dict[str, object]] = {}
    expected = {
        "name",
        "population",
        "count",
        "total_possible_pairs",
        "seed",
        "distance_class_counts",
    }
    for item in raw:
        if not isinstance(item, dict) or set(item) != expected:
            raise ValueError("evaluation-cache pair-set record fields differ")
        record = cast(dict[str, object], item)
        name = PairSetName(record["name"])
        if name in records:
            raise ValueError("evaluation-cache pair-set manifest contains duplicates")
        records[name] = record
    if set(records) != set(PairSetName):
        raise ValueError("evaluation-cache pair-set names differ")
    return records


@beartype
def load_evaluation_pair_cache(
    directory: Path,
    identity: EvaluationPairCacheIdentity,
) -> EvaluationPairCache:
    """Load only a byte-verified cache with an exact requested scientific identity."""

    if {path.name for path in directory.iterdir()} != {_PAYLOAD, _METADATA}:
        raise ValueError("evaluation-cache file inventory differs")
    raw = json.loads((directory / _METADATA).read_text(encoding="utf-8"))
    expected = {"schema", "identity", "payload_file", "payload_sha256", "pair_sets"}
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("evaluation-cache metadata envelope differs")
    if (
        raw["schema"] != _SCHEMA
        or raw["identity"] != identity.as_json()
        or raw["payload_file"] != _PAYLOAD
    ):
        raise ValueError("evaluation-cache scientific identity differs")
    payload_path = directory / _PAYLOAD
    if raw["payload_sha256"] != sha256_file(payload_path):
        raise ValueError("evaluation-cache payload fingerprint differs")
    records = _load_pair_records(raw["pair_sets"])
    expected_keys = {"distance_baseline_means", "distance_baseline_counts"} | {
        _tensor_key(name, field)
        for name in PairSetName
        for field in ("left", "right", "distance_class", "targets")
    }
    tensors = load_exact_safetensors(payload_path, expected_keys)
    pair_sets: dict[PairSetName, CachedPairSet] = {}
    for name in PairSetName:
        record = records[name]
        population = PairPopulation(record["population"])
        pairs = PairIndices(
            population=population,
            left=tensors[_tensor_key(name, "left")],
            right=tensors[_tensor_key(name, "right")],
            distance_class=tensors[_tensor_key(name, "distance_class")],
            total_possible_pairs=_required_int(record, "total_possible_pairs"),
            seed=_required_int(record, "seed"),
        )
        cached = CachedPairSet(
            pairs=pairs,
            targets=tensors[_tensor_key(name, "targets")],
        )
        counts = record["distance_class_counts"]
        if (
            _required_int(record, "count") != pairs.count
            or population != _EXPECTED_POPULATIONS[name]
            or not isinstance(counts, list)
            or counts != t.bincount(pairs.distance_class, minlength=8).tolist()
        ):
            raise ValueError("evaluation-cache pair-set manifest differs from its tensors")
        pair_sets[name] = cached
    baseline = DistanceBaseline(
        means=tensors["distance_baseline_means"],
        counts=tensors["distance_baseline_counts"],
    )
    return EvaluationPairCache(pair_sets=pair_sets, distance_baseline=baseline)
