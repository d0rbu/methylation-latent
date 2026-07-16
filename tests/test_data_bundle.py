from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from methylation_latent.data_bundle import (
    BundledFile,
    PrimaryDataBundle,
    seal_primary_data_bundle_exclusive,
    verify_primary_data_bundle,
)


def test_primary_data_bundle_rejects_any_post_seal_change(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "nested" / "b.txt").write_text("beta", encoding="utf-8")
    bundle = seal_primary_data_bundle_exclusive(
        tmp_path,
        protocol_id="protocol",
        protocol_sha256="a" * 64,
        git_commit="f" * 40,
        retained_probes=3,
        retained_samples=4,
    )
    assert tuple(file.path for file in bundle.files) == ("a.txt", "nested/b.txt")
    observed = verify_primary_data_bundle(
        tmp_path,
        protocol_id="protocol",
        protocol_sha256="a" * 64,
    )
    assert observed == bundle
    with pytest.raises(FileExistsError):
        seal_primary_data_bundle_exclusive(
            tmp_path,
            protocol_id="protocol",
            protocol_sha256="a" * 64,
            git_commit="f" * 40,
            retained_probes=3,
            retained_samples=4,
        )
    (tmp_path / "a.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match=r"changed=\['a.txt'\]"):
        verify_primary_data_bundle(
            tmp_path,
            protocol_id="protocol",
            protocol_sha256="a" * 64,
        )


def test_data_bundle_domain_types_reject_unsafe_or_incomplete_records() -> None:
    with pytest.raises(ValueError, match="safe relative"):
        BundledFile("../escape", 1, "a" * 64)
    with pytest.raises(ValueError, match="sorted"):
        PrimaryDataBundle(
            protocol_id="protocol",
            protocol_sha256="a" * 64,
            git_commit="f" * 40,
            retained_probes=2,
            retained_samples=3,
            files=(
                BundledFile("z", 1, "b" * 64),
                BundledFile("a", 1, "c" * 64),
            ),
        )
    with pytest.raises(ValueError, match="size cannot be negative"):
        BundledFile("x", -1, "a" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        BundledFile("x", 1, "not-a-hash")
    with pytest.raises(ValueError, match="protocol ID"):
        PrimaryDataBundle("", "a" * 64, "f" * 40, 2, 3, (BundledFile("x", 1, "b" * 64),))
    with pytest.raises(ValueError, match="Git commit"):
        PrimaryDataBundle("p", "a" * 64, "bad", 2, 3, (BundledFile("x", 1, "b" * 64),))
    with pytest.raises(ValueError, match="dimensions"):
        PrimaryDataBundle(
            "p",
            "a" * 64,
            "f" * 40,
            0,
            1,
            (BundledFile("x", 1, "b" * 64),),
        )
    with pytest.raises(ValueError, match="sorted"):
        PrimaryDataBundle("p", "a" * 64, "f" * 40, 2, 3, ())


def _sealed_directory(path: Path) -> Path:
    path.mkdir()
    (path / "source.txt").write_text("source", encoding="utf-8")
    seal_primary_data_bundle_exclusive(
        path,
        protocol_id="protocol",
        protocol_sha256="a" * 64,
        git_commit="f" * 40,
        retained_probes=2,
        retained_samples=3,
    )
    return path


def test_data_bundle_rejects_missing_added_and_unsealable_directories(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        seal_primary_data_bundle_exclusive(
            tmp_path / "missing",
            protocol_id="protocol",
            protocol_sha256="a" * 64,
            git_commit="f" * 40,
            retained_probes=2,
            retained_samples=3,
        )
    missing = _sealed_directory(tmp_path / "missing-file")
    (missing / "source.txt").unlink()
    with pytest.raises(ValueError, match=r"missing=\['source.txt'\]"):
        verify_primary_data_bundle(
            missing,
            protocol_id="protocol",
            protocol_sha256="a" * 64,
        )
    added = _sealed_directory(tmp_path / "added-file")
    (added / "extra.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(ValueError, match=r"added=\['extra.txt'\]"):
        verify_primary_data_bundle(
            added,
            protocol_id="protocol",
            protocol_sha256="a" * 64,
        )


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda raw: raw.update({"unknown": 1}), "envelope differs"),
        (lambda raw: raw.update({"retained_probes": True}), "field types differ"),
        (lambda raw: raw.update({"protocol_id": "other"}), "protocol identity differs"),
        (lambda raw: raw["files"].append("bad"), "file record fields differ"),
        (
            lambda raw: raw["files"][0].update({"size": "bad"}),
            "file record types differ",
        ),
    ],
)
def test_data_bundle_rejects_malformed_manifest_records(
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], object],
    error: str,
) -> None:
    directory = _sealed_directory(tmp_path / error.replace(" ", "-"))
    bundle_path = directory / "bundle.json"
    raw = json.loads(bundle_path.read_text(encoding="utf-8"))
    mutation(raw)
    bundle_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises((TypeError, ValueError), match=error):
        verify_primary_data_bundle(
            directory,
            protocol_id="protocol",
            protocol_sha256="a" * 64,
        )
