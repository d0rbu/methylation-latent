from __future__ import annotations

import gzip
import json
import subprocess
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import torch as t
from safetensors.torch import save_file

from methylation_latent.artifacts import (
    ArtifactMetadata,
    ArtifactReference,
    Eligibility,
    FullRunPrerequisites,
    PayloadRecord,
    ResultsRegistry,
    RunRecord,
    RunStage,
    ValidationCheck,
    canonical_json_bytes,
    finalize_metadata,
    least_eligible,
    load_metadata,
    require_clean_git_commit,
    sha256_file,
    sha256_ordered_strings,
    verify_payload,
    write_canonical_json_exclusive,
)
from methylation_latent.hashing import assert_gzip_payload_matches_file, md5_file
from methylation_latent.storage import (
    EmbeddingMatrix,
    load_embedding_matrix,
    load_exact_safetensors,
    load_target_geometry,
    save_embedding_matrix,
    save_safetensors_exclusive,
    save_target_geometry,
)
from methylation_latent.targets import build_target_geometry

HASH_A = "a" * 64
HASH_B = "b" * 64
COMMIT = "c" * 40


def test_hashing_is_framed_and_file_hash_matches_bytes(tmp_path: Path) -> None:
    assert sha256_ordered_strings(("ab", "c")) != sha256_ordered_strings(("a", "bc"))
    assert sha256_ordered_strings(()) != sha256_ordered_strings(("",))
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"research")
    assert (
        sha256_file(payload) == "66f62d1807d3821a3865f2573b69c74be033f1341240ac861fefc6d430bff5e0"
    )
    with pytest.raises(ValueError, match="positive"):
        sha256_file(payload, chunk_size=0)


def test_artifact_producer_requires_exact_clean_git_commit(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    for command in (
        ("git", "init", "--quiet"),
        ("git", "config", "user.email", "test@example.invalid"),
        ("git", "config", "user.name", "Test User"),
    ):
        subprocess.run(command, cwd=repository, check=True, capture_output=True)
    tracked = repository / "tracked.txt"
    tracked.write_text("committed\n", encoding="utf-8")
    subprocess.run(("git", "add", "tracked.txt"), cwd=repository, check=True)
    subprocess.run(
        ("git", "commit", "--quiet", "-m", "initial"),
        cwd=repository,
        check=True,
    )
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert require_clean_git_commit(repository) == commit

    nested = repository / "nested"
    nested.mkdir()
    with pytest.raises(ValueError, match="differs from Git toplevel"):
        require_clean_git_commit(nested)
    (nested / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean Git worktree"):
        require_clean_git_commit(repository)


def test_reference_archive_hash_and_decompressed_identity(tmp_path: Path) -> None:
    archive = tmp_path / "reference.fa.gz"
    payload = tmp_path / "reference.fa"
    payload.write_bytes(b">chr1\nACGT\n")
    with gzip.open(archive, mode="wb") as handle:
        handle.write(payload.read_bytes())
    assert len(md5_file(archive)) == 32
    assert assert_gzip_payload_matches_file(archive, payload) == sha256_file(payload)
    payload.write_bytes(b">chr1\nACGA\n")
    with pytest.raises(ValueError, match="differ"):
        assert_gzip_payload_matches_file(archive, payload)
    with pytest.raises(ValueError, match="positive"):
        md5_file(archive, chunk_size=0)
    with pytest.raises(ValueError, match="positive"):
        assert_gzip_payload_matches_file(archive, payload, chunk_size=0)


def test_canonical_json_is_stable_and_exclusive(tmp_path: Path) -> None:
    assert canonical_json_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}\n'
    with pytest.raises(ValueError):
        canonical_json_bytes({"bad": float("nan")})
    path = tmp_path / "nested" / "record.json"
    write_canonical_json_exclusive(path, {"ok": True})
    assert path.read_bytes() == b'{"ok":true}\n'
    with pytest.raises(FileExistsError):
        write_canonical_json_exclusive(path, {"ok": False})


def _finalized_metadata(tmp_path: Path) -> tuple[Path, ArtifactMetadata]:
    payload = tmp_path / "payload.txt"
    payload.write_text("validated\n", encoding="utf-8")
    metadata_path = tmp_path / "metadata.json"
    metadata = finalize_metadata(
        metadata_path=metadata_path,
        payload_path=payload,
        schema="test.v1",
        schema_version=1,
        artifact_id="test-artifact",
        protocol_id="protocol-v1",
        git_commit=COMMIT,
        command=("tool", "run"),
        environment_sha256=HASH_A,
        eligibility=Eligibility.AUDIT_ONLY,
        upstream=(ArtifactReference("upstream-one", HASH_B),),
        checks=(ValidationCheck("finite", True, True, None),),
        created_utc=datetime(2026, 7, 15, tzinfo=UTC),
    )
    return metadata_path, metadata


def test_metadata_roundtrip_and_payload_tamper_detection(tmp_path: Path) -> None:
    metadata_path, metadata = _finalized_metadata(tmp_path)
    loaded = load_metadata(metadata_path)
    assert loaded == metadata
    verify_payload(metadata_path, loaded)
    (tmp_path / "payload.txt").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash|size"):
        verify_payload(metadata_path, loaded)


def test_metadata_rejects_unknown_fields_and_invalid_semantics(tmp_path: Path) -> None:
    metadata_path, metadata = _finalized_metadata(tmp_path)
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    raw["unknown"] = 1
    invalid = tmp_path / "unknown.json"
    invalid.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_metadata(invalid)
    with pytest.raises(ValueError, match="UTC"):
        replace(metadata, created_utc="2026-07-15T12:00:00")
    with pytest.raises(ValueError, match="failed"):
        ValidationCheck("bad", False, 0, 0.1)
    with pytest.raises(ValueError, match="safe relative"):
        PayloadRecord("../payload", 1, HASH_A)
    with pytest.raises(ValueError, match="SHA-256"):
        ArtifactReference("valid-id", "bad")


@pytest.mark.parametrize(
    ("constructor", "message"),
    [
        (lambda: ArtifactReference("x", HASH_A), "artifact ID"),
        (lambda: ValidationCheck("", True, 1, None), "name"),
        (lambda: ValidationCheck("check", True, 1, -0.1), "negative"),
        (lambda: ValidationCheck("check", True, float("nan"), None), "finite"),
        (lambda: ValidationCheck("check", True, 1, float("inf")), "finite"),
        (lambda: PayloadRecord("payload", 0, HASH_A), "must not be empty"),
        (lambda: PayloadRecord("payload", 1, "bad"), "SHA-256"),
    ],
)
def test_artifact_value_objects_reject_every_invalid_field(
    constructor: Callable[[], object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        constructor()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda value: replace(value, schema=""), "schema"),
        (lambda value: replace(value, artifact_id="x"), "artifact ID"),
        (lambda value: replace(value, protocol_id=""), "protocol ID"),
        (lambda value: replace(value, git_commit="bad"), "Git commit"),
        (lambda value: replace(value, command=()), "command"),
        (lambda value: replace(value, environment_sha256="bad"), "environment"),
        (
            lambda value: replace(
                value,
                upstream=(
                    ArtifactReference("same-one", HASH_A),
                    ArtifactReference("same-one", HASH_B),
                ),
            ),
            "upstream",
        ),
        (lambda value: replace(value, checks=()), "validation checks"),
        (
            lambda value: replace(
                value,
                checks=(
                    ValidationCheck("same", True, 1, None),
                    ValidationCheck("same", True, 2, None),
                ),
            ),
            "check names",
        ),
    ],
)
def test_artifact_metadata_rejects_invalid_envelope_fields(
    tmp_path: Path,
    change: Callable[[ArtifactMetadata], ArtifactMetadata],
    message: str,
) -> None:
    _, metadata = _finalized_metadata(tmp_path)
    with pytest.raises(ValueError, match=message):
        change(metadata)


def test_finalize_metadata_requires_existing_colocated_payload(tmp_path: Path) -> None:
    def finalize(metadata_path: Path, payload_path: Path) -> ArtifactMetadata:
        return finalize_metadata(
            metadata_path=metadata_path,
            payload_path=payload_path,
            schema="test.v1",
            schema_version=1,
            artifact_id="artifact-one",
            protocol_id="protocol-v1",
            git_commit=COMMIT,
            command=("run",),
            environment_sha256=HASH_A,
            eligibility=Eligibility.AUDIT_ONLY,
            upstream=(),
            checks=(ValidationCheck("ok", True, True, None),),
        )

    with pytest.raises(FileNotFoundError):
        finalize(tmp_path / "metadata.json", tmp_path / "missing")
    other = tmp_path / "other"
    other.mkdir()
    payload = other / "payload"
    payload.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="share"):
        finalize(tmp_path / "metadata.json", payload)


@pytest.mark.parametrize(
    ("mutator", "exception", "message"),
    [
        (lambda raw: [], ValueError, "JSON object"),
        (lambda raw: {**raw, "payload": []}, ValueError, "nested"),
        (lambda raw: {**raw, "upstream": [1]}, ValueError, "reference"),
        (lambda raw: {**raw, "checks": [1]}, ValueError, "check"),
        (lambda raw: {**raw, "command": "run"}, ValueError, "JSON array"),
        (lambda raw: {**raw, "command": [1]}, TypeError, "arguments"),
        (lambda raw: {**raw, "schema_version": True}, TypeError, "integer"),
        (
            lambda raw: {**raw, "checks": [{**raw["checks"][0], "passed": "yes"}]},
            TypeError,
            "boolean",
        ),
        (
            lambda raw: {**raw, "checks": [{**raw["checks"][0], "measured": []}]},
            TypeError,
            "scalar",
        ),
        (
            lambda raw: {**raw, "checks": [{**raw["checks"][0], "tolerance": "low"}]},
            TypeError,
            "numeric",
        ),
    ],
)
def test_metadata_loader_rejects_wrong_json_types(
    tmp_path: Path,
    mutator: Callable[[dict[str, Any]], object],
    exception: type[Exception],
    message: str,
) -> None:
    metadata_path, _ = _finalized_metadata(tmp_path)
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    changed = mutator(raw)
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(exception, match=message):
        load_metadata(path)


def test_verify_payload_distinguishes_missing_size_and_hash(tmp_path: Path) -> None:
    metadata_path, metadata = _finalized_metadata(tmp_path)
    payload = tmp_path / "payload.txt"
    payload.unlink()
    with pytest.raises(FileNotFoundError):
        verify_payload(metadata_path, metadata)
    payload.write_text("validateD\n", encoding="utf-8")
    assert payload.stat().st_size == metadata.payload.byte_size
    with pytest.raises(ValueError, match="hash changed"):
        verify_payload(metadata_path, metadata)


def _run(
    artifact_id: str,
    stage: RunStage,
    *,
    eligibility: Eligibility = Eligibility.PRIMARY_CANDIDATE,
    prerequisites: FullRunPrerequisites | None = None,
) -> RunRecord:
    return RunRecord(
        artifact_id,
        stage,
        "protocol-v1",
        HASH_A,
        HASH_B,
        1024,
        eligibility,
        prerequisites,
    )


def test_results_registry_enforces_baseline_order_and_identity() -> None:
    registry = ResultsRegistry()
    with pytest.raises(ValueError, match="requires"):
        registry.append(_run("age-only", RunStage.CADUCEUS_AGE_BASELINE))
    registry = registry.append(_run("sequence-base", RunStage.SEQUENCE_AGE_BASELINE))
    registry = registry.append(_run("age-only", RunStage.CADUCEUS_AGE_BASELINE))
    prerequisites = FullRunPrerequisites("sequence-base", "age-only")
    registry = registry.append(
        _run("full-run", RunStage.FULL_LATENT_METRIC, prerequisites=prerequisites)
    )
    assert len(registry.records) == 3
    with pytest.raises(ValueError, match="duplicate"):
        registry.append(_run("full-run", RunStage.SEQUENCE_AGE_BASELINE))
    with pytest.raises(ValueError, match="absent"):
        registry.append(
            _run(
                "wrong-full",
                RunStage.FULL_LATENT_METRIC,
                prerequisites=FullRunPrerequisites("missing-base", "age-only"),
            )
        )


def test_registry_eligibility_is_monotonic() -> None:
    registry = ResultsRegistry().append(
        _run("sequence-base", RunStage.SEQUENCE_AGE_BASELINE, eligibility=Eligibility.AUDIT_ONLY)
    )
    registry = registry.append(
        _run("age-only", RunStage.CADUCEUS_AGE_BASELINE, eligibility=Eligibility.AUDIT_ONLY)
    )
    with pytest.raises(ValueError, match="more eligible"):
        registry.append(
            _run(
                "full-run",
                RunStage.FULL_LATENT_METRIC,
                eligibility=Eligibility.PRIMARY_VALIDATED,
                prerequisites=FullRunPrerequisites("sequence-base", "age-only"),
            )
        )
    assert (
        least_eligible((Eligibility.PRIMARY_VALIDATED, Eligibility.AUDIT_ONLY))
        == Eligibility.AUDIT_ONLY
    )
    with pytest.raises(ValueError, match="at least one"):
        least_eligible(())


def test_run_record_semantics_and_primary_filter() -> None:
    with pytest.raises(ValueError, match="prerequisite artifact"):
        FullRunPrerequisites("x", "age-only")
    with pytest.raises(ValueError, match="run artifact"):
        _run("x", RunStage.SEQUENCE_AGE_BASELINE)
    with pytest.raises(ValueError, match="protocol"):
        replace(_run("valid-run", RunStage.SEQUENCE_AGE_BASELINE), protocol_id="")
    with pytest.raises(ValueError, match="fingerprints"):
        replace(_run("valid-run", RunStage.SEQUENCE_AGE_BASELINE), data_sha256="bad")
    with pytest.raises(ValueError, match="requires"):
        _run("full-run", RunStage.FULL_LATENT_METRIC)
    with pytest.raises(ValueError, match="must not carry"):
        _run(
            "base-run",
            RunStage.SEQUENCE_AGE_BASELINE,
            prerequisites=FullRunPrerequisites("sequence-base", "age-only"),
        )
    validated = _run(
        "validated-run",
        RunStage.SEQUENCE_AGE_BASELINE,
        eligibility=Eligibility.PRIMARY_VALIDATED,
    )
    assert ResultsRegistry((validated,)).primary_validated() == (validated,)


def test_target_geometry_safetensors_roundtrip_is_strict(tmp_path: Path) -> None:
    beta = t.tensor([[0.1, 0.2, 0.7], [0.8, 0.4, 0.2]], dtype=t.float64)
    age = t.tensor([20.0, 40.0, 80.0], dtype=t.float64)
    geometry = build_target_geometry(beta, age)
    path = tmp_path / "targets.safetensors"
    save_target_geometry(path, geometry)
    loaded = load_target_geometry(path)
    assert t.equal(loaded.methylation.tensor, geometry.methylation.tensor)
    assert t.equal(loaded.rho.tensor, geometry.rho.tensor)
    with pytest.raises(FileExistsError):
        save_target_geometry(path, geometry)
    with pytest.raises(ValueError, match="keys differ"):
        load_exact_safetensors(path, {"wrong"})


def test_embedding_safetensors_and_semantic_type(tmp_path: Path) -> None:
    values = t.randn((3, 256), dtype=t.float16)
    embeddings = EmbeddingMatrix(values)
    path = tmp_path / "embeddings.safetensors"
    save_embedding_matrix(path, embeddings)
    loaded = load_embedding_matrix(path)
    assert t.equal(loaded.tensor, values)
    assert loaded.training_tensor(device="cpu").dtype == t.float32
    with pytest.raises(TypeError, match="float16"):
        EmbeddingMatrix(values.float())
    with pytest.raises(ValueError, match="shape"):
        EmbeddingMatrix(t.ones((3, 255), dtype=t.float16))
    with pytest.raises(ValueError, match="finite"):
        EmbeddingMatrix(t.full((1, 256), float("nan"), dtype=t.float16))


def test_safetensors_writer_rejects_empty_nonfinite_and_existing(tmp_path: Path) -> None:
    path = tmp_path / "values.safetensors"
    with pytest.raises(ValueError, match="named"):
        save_safetensors_exclusive(path, {})
    with pytest.raises(ValueError, match="finite"):
        save_safetensors_exclusive(path, {"x": t.tensor([float("inf")])})
    sparse = t.sparse_coo_tensor(t.tensor([[0]]), t.tensor([1.0]), size=(1,), check_invariants=True)
    with pytest.raises(ValueError, match="strided"):
        save_safetensors_exclusive(path, {"x": sparse})
    save_safetensors_exclusive(path, {"x": t.tensor([1.0])})
    with pytest.raises(FileExistsError):
        save_safetensors_exclusive(path, {"x": t.tensor([2.0])})


def test_safetensors_loaders_reject_nonfinite_and_wrong_target_dtype(tmp_path: Path) -> None:
    nonfinite = tmp_path / "nonfinite.safetensors"
    save_file({"x": t.tensor([float("nan")])}, nonfinite)
    with pytest.raises(ValueError, match="non-finite"):
        load_exact_safetensors(nonfinite, {"x"})
    wrong_dtype = tmp_path / "targets.safetensors"
    save_file(
        {
            "X": t.tensor([[-0.70710677, 0.70710677]], dtype=t.float32),
            "age": t.tensor([-0.70710677, 0.70710677], dtype=t.float32),
            "rho": t.tensor([1.0], dtype=t.float32),
        },
        wrong_dtype,
    )
    with pytest.raises(TypeError, match="float64"):
        load_target_geometry(wrong_dtype)
