from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch as t

from methylation_latent.domain import NonEmptyProbeSet, ProbeLocus, parse_window_size
from methylation_latent.embedding_cache import (
    EmbeddingShardIdentity,
    embedding_shard_ranges,
    finalize_embedding_cache_exclusive,
    load_embedding_cache,
    load_embedding_shard,
    save_embedding_shard_exclusive,
)
from methylation_latent.storage import EmbeddingMatrix


def test_embedding_cache_is_restart_safe_ordered_and_immutable(
    tmp_path: Path,
    make_probe: Callable[..., ProbeLocus],
) -> None:
    probes = NonEmptyProbeSet(tuple(make_probe(index + 1) for index in range(5)))
    window = parse_window_size(1024)
    generator = t.Generator().manual_seed(4)
    expected = t.randn((5, 256), generator=generator, dtype=t.float16)
    for start, stop in embedding_shard_ranges(len(probes), 2):
        identity = save_embedding_shard_exclusive(
            tmp_path,
            EmbeddingMatrix(expected[start:stop]),
            probes,
            window_size=window,
            start=start,
            stop=stop,
        )
        assert identity.directory_name == f"shard-{start:09d}-{stop:09d}"
        _, loaded = load_embedding_shard(
            tmp_path / identity.directory_name,
            probes,
            window_size=window,
            start=start,
            stop=stop,
        )
        assert t.equal(loaded.tensor, expected[start:stop])
        with pytest.raises(FileExistsError):
            save_embedding_shard_exclusive(
                tmp_path,
                EmbeddingMatrix(expected[start:stop]),
                probes,
                window_size=window,
                start=start,
                stop=stop,
            )
    finalize_embedding_cache_exclusive(
        tmp_path,
        probes,
        window_size=window,
        shard_size=2,
    )
    assert t.equal(
        load_embedding_cache(tmp_path, probes, window_size=window).tensor,
        expected,
    )


def test_embedding_cache_rejects_probe_order_drift(
    tmp_path: Path,
    make_probe: Callable[..., ProbeLocus],
) -> None:
    probes = NonEmptyProbeSet(tuple(make_probe(index + 1) for index in range(2)))
    window = parse_window_size(1024)
    identity = save_embedding_shard_exclusive(
        tmp_path,
        EmbeddingMatrix(t.ones((2, 256), dtype=t.float16)),
        probes,
        window_size=window,
        start=0,
        stop=2,
    )
    reversed_probes = NonEmptyProbeSet(tuple(reversed(probes.probes)))
    with pytest.raises(ValueError, match="probe-order"):
        load_embedding_shard(
            tmp_path / identity.directory_name,
            reversed_probes,
            window_size=window,
            start=0,
            stop=2,
        )


def test_embedding_cache_rejects_corrupt_shards_and_manifests(
    tmp_path: Path,
    make_probe: Callable[..., ProbeLocus],
) -> None:
    probes = NonEmptyProbeSet(tuple(make_probe(index + 1) for index in range(2)))
    window = parse_window_size(1024)
    embeddings = EmbeddingMatrix(t.ones((2, 256), dtype=t.float16))
    cache = tmp_path / "base"
    identity = save_embedding_shard_exclusive(
        cache,
        embeddings,
        probes,
        window_size=window,
        start=0,
        stop=2,
    )
    finalize_embedding_cache_exclusive(
        cache,
        probes,
        window_size=window,
        shard_size=2,
    )
    with pytest.raises(ValueError, match="must be positive"):
        embedding_shard_ranges(0, 2)
    with pytest.raises(ValueError, match="positive and even"):
        EmbeddingShardIdentity(3, 0, 1, "a" * 64, "b" * 64)
    with pytest.raises(ValueError, match="non-empty and increasing"):
        EmbeddingShardIdentity(2, 1, 1, "a" * 64, "b" * 64)
    with pytest.raises(ValueError, match="fingerprints"):
        EmbeddingShardIdentity(2, 0, 1, "short", "b" * 64)
    with pytest.raises(ValueError, match="row count differs"):
        save_embedding_shard_exclusive(
            tmp_path / "wrong-rows",
            embeddings,
            probes,
            window_size=window,
            start=0,
            stop=1,
        )
    with pytest.raises(IndexError, match="exceeds"):
        save_embedding_shard_exclusive(
            tmp_path / "out-of-range",
            EmbeddingMatrix(t.ones((3, 256), dtype=t.float16)),
            probes,
            window_size=window,
            start=0,
            stop=3,
        )

    def copied(name: str) -> Path:
        destination = tmp_path / name
        shutil.copytree(cache, destination)
        return destination

    extra_shard = copied("extra-shard-file")
    (extra_shard / identity.directory_name / "unknown").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="file inventory"):
        load_embedding_shard(
            extra_shard / identity.directory_name,
            probes,
            window_size=window,
            start=0,
            stop=2,
        )

    def shard_metadata_copy(name: str) -> tuple[Path, dict[str, Any]]:
        destination = copied(name)
        metadata = destination / identity.directory_name / "metadata.json"
        return destination, json.loads(metadata.read_text(encoding="utf-8"))

    malformed, raw = shard_metadata_copy("malformed-shard")
    raw["unknown"] = 1
    metadata = malformed / identity.directory_name / "metadata.json"
    metadata.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata envelope"):
        load_embedding_shard(
            malformed / identity.directory_name,
            probes,
            window_size=window,
            start=0,
            stop=2,
        )

    wrong_model, raw = shard_metadata_copy("wrong-model")
    raw["repository"] = "other"
    metadata = wrong_model / identity.directory_name / "metadata.json"
    metadata.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="model, schema"):
        load_embedding_shard(
            wrong_model / identity.directory_name,
            probes,
            window_size=window,
            start=0,
            stop=2,
        )

    wrong_range, raw = shard_metadata_copy("wrong-range")
    raw["window_size"] = 2048
    metadata = wrong_range / identity.directory_name / "metadata.json"
    metadata.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="range, window"):
        load_embedding_shard(
            wrong_range / identity.directory_name,
            probes,
            window_size=window,
            start=0,
            stop=2,
        )

    corrupt_payload = copied("corrupt-payload")
    payload = corrupt_payload / identity.directory_name / "embeddings.safetensors"
    payload.write_bytes(payload.read_bytes() + b"x")
    with pytest.raises(ValueError, match="payload fingerprint"):
        load_embedding_shard(
            corrupt_payload / identity.directory_name,
            probes,
            window_size=window,
            start=0,
            stop=2,
        )

    unknown_entry = copied("unknown-entry")
    (unknown_entry / "unexpected").mkdir()
    with pytest.raises(ValueError, match="missing or unknown shard"):
        finalize_embedding_cache_exclusive(
            unknown_entry,
            probes,
            window_size=window,
            shard_size=2,
        )

    def manifest_copy(name: str) -> tuple[Path, dict[str, Any]]:
        destination = copied(name)
        manifest = destination / "manifest.json"
        return destination, json.loads(manifest.read_text(encoding="utf-8"))

    malformed_manifest, raw = manifest_copy("malformed-manifest")
    raw["unknown"] = 1
    (malformed_manifest / "manifest.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest envelope"):
        load_embedding_cache(malformed_manifest, probes, window_size=window)

    wrong_identity, raw = manifest_copy("wrong-identity")
    raw["window_size"] = 2048
    (wrong_identity / "manifest.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest identity"):
        load_embedding_cache(wrong_identity, probes, window_size=window)

    wrong_length, raw = manifest_copy("wrong-length")
    raw["shards"] = []
    (wrong_length / "manifest.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest length"):
        load_embedding_cache(wrong_length, probes, window_size=window)

    non_object, raw = manifest_copy("non-object")
    raw["shards"][0] = "bad"
    (non_object / "manifest.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="record must be an object"):
        load_embedding_cache(non_object, probes, window_size=window)

    wrong_fields, raw = manifest_copy("wrong-fields")
    del raw["shards"][0]["start"]
    (wrong_fields / "manifest.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="record fields"):
        load_embedding_cache(wrong_fields, probes, window_size=window)

    wrong_record, raw = manifest_copy("wrong-record")
    raw["shards"][0]["start"] = 1
    (wrong_record / "manifest.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="record differs"):
        load_embedding_cache(wrong_record, probes, window_size=window)
