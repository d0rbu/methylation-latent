"""Exact compressed-IDAT inventory audits and deterministic materialization."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import secrets
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from beartype import beartype

from methylation_latent.artifacts import (
    sha256_file,
    sha256_ordered_strings,
    write_canonical_json_exclusive,
)
from methylation_latent.domain import SentrixIdentity, parse_sentrix_identity
from methylation_latent.metadata import Gse87571RawSampleSet


@dataclass(frozen=True, slots=True)
class CompressedIdat:
    """One gzip member whose full payload has valid CRC and IDAT magic."""

    sentrix_identity: SentrixIdentity
    channel: str
    compressed_path: Path
    compressed_bytes: int
    compressed_sha256: str
    decompressed_bytes: int
    decompressed_sha256: str

    def __post_init__(self) -> None:
        if self.channel not in {"Grn", "Red"}:
            raise ValueError(f"unexpected IDAT channel: {self.channel!r}")
        if min(self.compressed_bytes, self.decompressed_bytes) <= 4:
            raise ValueError("compressed and decompressed IDAT payloads must be non-empty")
        if len(self.compressed_sha256) != 64 or len(self.decompressed_sha256) != 64:
            raise ValueError("IDAT fingerprints must be SHA-256 values")

    @property
    def materialized_name(self) -> str:
        return f"{self.sentrix_identity}_{self.channel}.idat"


@dataclass(frozen=True, slots=True)
class CompressedIdatInventory:
    """Ordered green/red records aligned to the raw GEO sample order."""

    records: tuple[CompressedIdat, ...]

    def __post_init__(self) -> None:
        if not self.records or len(self.records) % 2 != 0:
            raise ValueError("compressed IDAT inventory must contain complete non-empty pairs")
        identities = tuple((record.sentrix_identity, record.channel) for record in self.records)
        if len(set(identities)) != len(identities):
            raise ValueError("compressed IDAT inventory contains duplicate identity/channel rows")
        by_sentrix: dict[SentrixIdentity, set[str]] = {}
        for record in self.records:
            by_sentrix.setdefault(record.sentrix_identity, set()).add(record.channel)
        if any(channels != {"Grn", "Red"} for channels in by_sentrix.values()):
            raise ValueError("compressed IDAT inventory contains an incomplete channel pair")

    @property
    def fingerprint(self) -> str:
        return sha256_ordered_strings(
            "\t".join(
                (
                    str(record.sentrix_identity),
                    record.channel,
                    str(record.compressed_bytes),
                    record.compressed_sha256,
                    str(record.decompressed_bytes),
                    record.decompressed_sha256,
                )
            )
            for record in self.records
        )

    def records_for(self, samples: Gse87571RawSampleSet) -> tuple[CompressedIdat, ...]:
        wanted = {sample.sentrix_identity for sample in samples.samples}
        records = tuple(record for record in self.records if record.sentrix_identity in wanted)
        if len(records) != 2 * len(samples):
            raise ValueError("requested sample set is not fully represented in IDAT inventory")
        return records


def save_compressed_idat_inventory_exclusive(
    path: Path,
    inventory: CompressedIdatInventory,
) -> None:
    write_canonical_json_exclusive(
        path,
        {
            "schema": "methylation-latent.compressed-idat-inventory.v1",
            "fingerprint": inventory.fingerprint,
            "records": [
                {
                    "sentrix_identity": str(record.sentrix_identity),
                    "channel": record.channel,
                    "compressed_name": record.compressed_path.name,
                    "compressed_bytes": record.compressed_bytes,
                    "compressed_sha256": record.compressed_sha256,
                    "decompressed_bytes": record.decompressed_bytes,
                    "decompressed_sha256": record.decompressed_sha256,
                }
                for record in inventory.records
            ],
        },
    )


@beartype
def load_compressed_idat_inventory(
    path: Path,
    *,
    compressed_directory: Path,
) -> CompressedIdatInventory:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"schema", "fingerprint", "records"}:
        raise ValueError("compressed IDAT inventory JSON has an invalid envelope")
    if raw["schema"] != "methylation-latent.compressed-idat-inventory.v1":
        raise ValueError("compressed IDAT inventory schema differs")
    raw_records = raw["records"]
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("compressed IDAT inventory records must be a non-empty array")
    records: list[CompressedIdat] = []
    expected_fields = {
        "sentrix_identity",
        "channel",
        "compressed_name",
        "compressed_bytes",
        "compressed_sha256",
        "decompressed_bytes",
        "decompressed_sha256",
    }
    for raw_record in raw_records:
        if not isinstance(raw_record, dict) or set(raw_record) != expected_fields:
            raise ValueError("compressed IDAT inventory record fields differ")
        compressed_name = raw_record["compressed_name"]
        if not isinstance(compressed_name, str) or Path(compressed_name).name != compressed_name:
            raise ValueError("compressed IDAT inventory contains an unsafe member name")
        records.append(
            CompressedIdat(
                sentrix_identity=parse_sentrix_identity(str(raw_record["sentrix_identity"])),
                channel=str(raw_record["channel"]),
                compressed_path=compressed_directory / compressed_name,
                compressed_bytes=int(raw_record["compressed_bytes"]),
                compressed_sha256=str(raw_record["compressed_sha256"]),
                decompressed_bytes=int(raw_record["decompressed_bytes"]),
                decompressed_sha256=str(raw_record["decompressed_sha256"]),
            )
        )
    inventory = CompressedIdatInventory(tuple(records))
    if raw["fingerprint"] != inventory.fingerprint:
        raise ValueError("compressed IDAT inventory fingerprint differs")
    return inventory


def _audit_gzip_idat(path: Path, *, chunk_size: int = 1 << 20) -> tuple[int, str]:
    if chunk_size <= 0:
        raise ValueError("IDAT gzip audit chunk size must be positive")
    digest = hashlib.sha256()
    decompressed_bytes = 0
    magic = b""
    with gzip.open(path, mode="rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            if not magic:
                magic = chunk[:4]
            digest.update(chunk)
            decompressed_bytes += len(chunk)
    if magic != b"IDAT":
        raise ValueError(f"decompressed file does not have IDAT magic bytes: {path}")
    if decompressed_bytes <= 4:
        raise ValueError(f"decompressed IDAT is empty: {path}")
    return decompressed_bytes, digest.hexdigest()


def _source_name(url: str) -> str:
    name = url.rsplit("/", maxsplit=1)[-1]
    if not name.endswith(".idat.gz"):
        raise ValueError(f"GEO source URL does not end in .idat.gz: {url!r}")
    return name


@beartype
def audit_gse87571_compressed_idats(
    directory: Path,
    samples: Gse87571RawSampleSet,
    *,
    expected_sizes: Mapping[str, int],
) -> CompressedIdatInventory:
    """Read every gzip through its CRC trailer and require exact GEO member sizes."""

    expected_names = {
        _source_name(url)
        for sample in samples.samples
        for url in (sample.green_url, sample.red_url)
    }
    if set(expected_sizes) != expected_names:
        raise ValueError(
            "GEO expected-size inventory differs from sample source URLs: "
            f"missing={sorted(expected_names - set(expected_sizes))[:10]}, "
            f"unknown={sorted(set(expected_sizes) - expected_names)[:10]}"
        )
    observed_names = {path.name for path in directory.glob("*.idat.gz")}
    if observed_names != expected_names:
        raise ValueError(
            "compressed IDAT directory differs from GEO sample inventory: "
            f"missing={sorted(expected_names - observed_names)[:10]}, "
            f"unknown={sorted(observed_names - expected_names)[:10]}"
        )

    records: list[CompressedIdat] = []
    for sample in samples.samples:
        for channel, url in (("Grn", sample.green_url), ("Red", sample.red_url)):
            name = _source_name(url)
            path = directory / name
            observed_size = path.stat().st_size
            expected_size = expected_sizes[name]
            if observed_size != expected_size:
                raise ValueError(
                    f"compressed IDAT size differs for {name}: "
                    f"expected={expected_size}, observed={observed_size}"
                )
            decompressed_bytes, decompressed_sha256 = _audit_gzip_idat(path)
            records.append(
                CompressedIdat(
                    sentrix_identity=sample.sentrix_identity,
                    channel=channel,
                    compressed_path=path,
                    compressed_bytes=observed_size,
                    compressed_sha256=sha256_file(path),
                    decompressed_bytes=decompressed_bytes,
                    decompressed_sha256=decompressed_sha256,
                )
            )
    return CompressedIdatInventory(tuple(records))


def _materialize_one(record: CompressedIdat, output_path: Path) -> None:
    if output_path.exists():
        if (
            output_path.stat().st_size != record.decompressed_bytes
            or sha256_file(output_path) != record.decompressed_sha256
        ):
            raise ValueError(f"existing materialized IDAT differs: {output_path}")
        return
    temporary = output_path.with_name(f".{output_path.name}.{secrets.token_hex(16)}.tmp")
    with (
        gzip.open(record.compressed_path, mode="rb") as source,
        temporary.open(mode="xb") as destination,
    ):
        shutil.copyfileobj(source, destination, length=1 << 20)
        destination.flush()
        os.fsync(destination.fileno())
    if (
        temporary.stat().st_size != record.decompressed_bytes
        or sha256_file(temporary) != record.decompressed_sha256
    ):
        temporary.unlink()
        raise RuntimeError(f"materialized IDAT verification failed: {record.compressed_path}")
    os.link(temporary, output_path)
    temporary.unlink()


@beartype
def materialize_idats(
    inventory: CompressedIdatInventory,
    samples: Gse87571RawSampleSet,
    *,
    output_directory: Path,
) -> tuple[Path, ...]:
    """Decompress an exact sample subset to Sentrix-named processor inputs."""

    output_directory.mkdir(parents=True, exist_ok=True)
    records = inventory.records_for(samples)
    expected_paths = tuple(output_directory / record.materialized_name for record in records)
    unknown = set(output_directory.glob("*.idat")) - set(expected_paths)
    if unknown:
        raise ValueError(
            f"materialized IDAT directory contains unknown files: {sorted(map(str, unknown))[:10]}"
        )
    for record, output_path in zip(records, expected_paths, strict=True):
        _materialize_one(record, output_path)
    return expected_paths
