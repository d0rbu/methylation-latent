"""Immutable artifact metadata and an order-enforcing results registry."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import TypeAlias, cast

from beartype import beartype

from methylation_latent.hashing import sha256_file

_SHA256 = re.compile(r"^[0-9a-f]{64}$", flags=re.ASCII)
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$", flags=re.ASCII)
_ARTIFACT_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$", flags=re.ASCII)

# The isolated pinned Caduceus runtime is Python 3.11 and imports artifact helpers.
JsonScalar: TypeAlias = None | bool | int | float | str  # noqa: UP040
JsonValue: TypeAlias = (  # noqa: UP040
    JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
)


class Eligibility(StrEnum):
    """Scientific eligibility carried monotonically through the artifact graph."""

    AUDIT_ONLY = "audit_only"
    PRIMARY_CANDIDATE = "primary_candidate"
    PRIMARY_VALIDATED = "primary_validated"


_ELIGIBILITY_ORDER = {
    Eligibility.AUDIT_ONLY: 0,
    Eligibility.PRIMARY_CANDIDATE: 1,
    Eligibility.PRIMARY_VALIDATED: 2,
}


@beartype
def least_eligible(eligibilities: Iterable[Eligibility]) -> Eligibility:
    values = tuple(eligibilities)
    if not values:
        raise ValueError("eligibility reduction requires at least one input")
    return min(values, key=_ELIGIBILITY_ORDER.__getitem__)


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    artifact_id: str
    sha256: str

    def __post_init__(self) -> None:
        if _ARTIFACT_ID.fullmatch(self.artifact_id) is None:
            raise ValueError(f"invalid artifact ID: {self.artifact_id!r}")
        if _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("artifact reference SHA-256 must be 64 lowercase hex characters")


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    """One named measured invariant, including its frozen tolerance when applicable."""

    name: str
    passed: bool
    measured: str | int | float | bool
    tolerance: float | None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("validation check name must not be empty")
        if not self.passed:
            raise ValueError(f"failed validation checks cannot finalize an artifact: {self.name}")
        if self.tolerance is not None and self.tolerance < 0.0:
            raise ValueError("validation tolerance cannot be negative")
        if isinstance(self.measured, float) and not math.isfinite(self.measured):
            raise ValueError("validation measurements must be finite")
        if self.tolerance is not None and not math.isfinite(self.tolerance):
            raise ValueError("validation tolerance must be finite")


@dataclass(frozen=True, slots=True)
class PayloadRecord:
    relative_path: str
    byte_size: int
    sha256: str

    def __post_init__(self) -> None:
        path = Path(self.relative_path)
        if path.is_absolute() or ".." in path.parts or self.relative_path == "":
            raise ValueError("payload path must be a non-empty safe relative path")
        if self.byte_size <= 0:
            raise ValueError("artifact payload must not be empty")
        if _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("payload SHA-256 must be 64 lowercase hex characters")


@dataclass(frozen=True, slots=True)
class ArtifactMetadata:
    """Strict common envelope for every immutable payload."""

    schema: str
    schema_version: int
    artifact_id: str
    created_utc: str
    protocol_id: str
    git_commit: str
    command: tuple[str, ...]
    environment_sha256: str
    eligibility: Eligibility
    payload: PayloadRecord
    upstream: tuple[ArtifactReference, ...]
    checks: tuple[ValidationCheck, ...]

    def __post_init__(self) -> None:
        if not self.schema or self.schema_version <= 0:
            raise ValueError("artifact schema and positive version are required")
        if _ARTIFACT_ID.fullmatch(self.artifact_id) is None:
            raise ValueError(f"invalid artifact ID: {self.artifact_id!r}")
        timestamp = datetime.fromisoformat(self.created_utc)
        if timestamp.tzinfo != UTC:
            raise ValueError("artifact creation time must use an explicit UTC offset")
        if not self.protocol_id:
            raise ValueError("artifact protocol ID must not be empty")
        if _GIT_COMMIT.fullmatch(self.git_commit) is None:
            raise ValueError("artifact Git commit must be 40 lowercase hex characters")
        if not self.command or any(argument == "" for argument in self.command):
            raise ValueError("artifact command must contain non-empty arguments")
        if _SHA256.fullmatch(self.environment_sha256) is None:
            raise ValueError("environment fingerprint must be a SHA-256")
        if len({reference.artifact_id for reference in self.upstream}) != len(self.upstream):
            raise ValueError("upstream artifact IDs must be unique")
        if not self.checks:
            raise ValueError("finalized artifact must contain validation checks")
        if len({check.name for check in self.checks}) != len(self.checks):
            raise ValueError("validation check names must be unique")


@beartype
def sha256_ordered_strings(values: Iterable[str]) -> str:
    """Hash a sequence with length framing, so concatenation cannot collide."""

    digest = hashlib.sha256()
    count = 0
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
        count += 1
    digest.update(count.to_bytes(8, byteorder="big", signed=False))
    return digest.hexdigest()


def canonical_json_bytes(value: JsonValue) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def write_canonical_json_exclusive(path: Path, value: JsonValue) -> None:
    """Write canonical JSON once; existing artifacts are immutable."""

    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()


def _metadata_as_json(metadata: ArtifactMetadata) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], asdict(metadata))


@beartype
def finalize_metadata(
    *,
    metadata_path: Path,
    payload_path: Path,
    schema: str,
    schema_version: int,
    artifact_id: str,
    protocol_id: str,
    git_commit: str,
    command: tuple[str, ...],
    environment_sha256: str,
    eligibility: Eligibility,
    upstream: tuple[ArtifactReference, ...],
    checks: tuple[ValidationCheck, ...],
    created_utc: datetime | None = None,
) -> ArtifactMetadata:
    """Hash an existing payload and exclusively finalize its strict metadata."""

    if not payload_path.is_file():
        raise FileNotFoundError(payload_path)
    if payload_path.parent.resolve() != metadata_path.parent.resolve():
        raise ValueError("payload and metadata must share an artifact directory")
    timestamp = created_utc or datetime.now(UTC)
    metadata = ArtifactMetadata(
        schema=schema,
        schema_version=schema_version,
        artifact_id=artifact_id,
        created_utc=timestamp.isoformat(),
        protocol_id=protocol_id,
        git_commit=git_commit,
        command=command,
        environment_sha256=environment_sha256,
        eligibility=eligibility,
        payload=PayloadRecord(
            relative_path=payload_path.name,
            byte_size=payload_path.stat().st_size,
            sha256=sha256_file(payload_path),
        ),
        upstream=upstream,
        checks=checks,
    )
    write_canonical_json_exclusive(metadata_path, _metadata_as_json(metadata))
    return metadata


_METADATA_FIELDS = {
    "schema",
    "schema_version",
    "artifact_id",
    "created_utc",
    "protocol_id",
    "git_commit",
    "command",
    "environment_sha256",
    "eligibility",
    "payload",
    "upstream",
    "checks",
}


def _require_exact_fields(raw: Mapping[str, object], expected: set[str], context: str) -> None:
    observed = set(raw)
    if observed != expected:
        raise ValueError(
            f"{context} fields differ: missing={sorted(expected - observed)}, "
            f"unknown={sorted(observed - expected)}"
        )


def _json_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"metadata field {name!r} must be a string")
    return value


def _json_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"metadata field {name!r} must be an integer")
    return value


def _json_boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"metadata field {name!r} must be a boolean")
    return value


def _json_measurement(value: object) -> str | int | float | bool:
    if not isinstance(value, str | int | float | bool):
        raise TypeError("validation measurement must be a JSON scalar")
    return value


def _json_tolerance(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("validation tolerance must be numeric or null")
    return float(value)


def load_metadata(path: Path) -> ArtifactMetadata:
    """Load only the current schema envelope; unknown fields are errors."""

    raw_object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_object, dict):
        raise ValueError("artifact metadata must be a JSON object")
    _require_exact_fields(raw_object, _METADATA_FIELDS, "metadata")
    payload_raw = raw_object["payload"]
    upstream_raw = raw_object["upstream"]
    checks_raw = raw_object["checks"]
    if (
        not isinstance(payload_raw, dict)
        or not isinstance(upstream_raw, list)
        or not isinstance(checks_raw, list)
    ):
        raise ValueError("artifact metadata contains invalid nested structures")
    _require_exact_fields(payload_raw, {"relative_path", "byte_size", "sha256"}, "payload")
    for item in upstream_raw:
        if not isinstance(item, dict):
            raise ValueError("upstream artifact reference must be an object")
        _require_exact_fields(item, {"artifact_id", "sha256"}, "upstream reference")
    for item in checks_raw:
        if not isinstance(item, dict):
            raise ValueError("validation check must be an object")
        _require_exact_fields(item, {"name", "passed", "measured", "tolerance"}, "check")
    command_raw = raw_object["command"]
    if not isinstance(command_raw, list):
        raise ValueError("artifact command must be a JSON array")
    if any(not isinstance(argument, str) for argument in command_raw):
        raise TypeError("artifact command arguments must be strings")
    return ArtifactMetadata(
        schema=_json_string(raw_object["schema"], "schema"),
        schema_version=_json_integer(raw_object["schema_version"], "schema_version"),
        artifact_id=_json_string(raw_object["artifact_id"], "artifact_id"),
        created_utc=_json_string(raw_object["created_utc"], "created_utc"),
        protocol_id=_json_string(raw_object["protocol_id"], "protocol_id"),
        git_commit=_json_string(raw_object["git_commit"], "git_commit"),
        command=tuple(command_raw),
        environment_sha256=_json_string(raw_object["environment_sha256"], "environment_sha256"),
        eligibility=Eligibility(_json_string(raw_object["eligibility"], "eligibility")),
        payload=PayloadRecord(
            relative_path=_json_string(payload_raw["relative_path"], "payload.relative_path"),
            byte_size=_json_integer(payload_raw["byte_size"], "payload.byte_size"),
            sha256=_json_string(payload_raw["sha256"], "payload.sha256"),
        ),
        upstream=tuple(
            ArtifactReference(
                artifact_id=_json_string(item["artifact_id"], "upstream.artifact_id"),
                sha256=_json_string(item["sha256"], "upstream.sha256"),
            )
            for item in upstream_raw
        ),
        checks=tuple(
            ValidationCheck(
                name=_json_string(item["name"], "check.name"),
                passed=_json_boolean(item["passed"], "check.passed"),
                measured=_json_measurement(item["measured"]),
                tolerance=_json_tolerance(item["tolerance"]),
            )
            for item in checks_raw
        ),
    )


@beartype
def verify_payload(metadata_path: Path, metadata: ArtifactMetadata) -> None:
    payload_path = metadata_path.parent / metadata.payload.relative_path
    if not payload_path.is_file():
        raise FileNotFoundError(payload_path)
    observed_size = payload_path.stat().st_size
    if observed_size != metadata.payload.byte_size:
        raise ValueError(
            f"payload byte size changed: expected={metadata.payload.byte_size}, "
            f"observed={observed_size}"
        )
    observed_hash = sha256_file(payload_path)
    if observed_hash != metadata.payload.sha256:
        raise ValueError(
            f"payload hash changed: expected={metadata.payload.sha256}, observed={observed_hash}"
        )


class RunStage(IntEnum):
    SEQUENCE_AGE_BASELINE = 1
    CADUCEUS_AGE_BASELINE = 2
    FULL_LATENT_METRIC = 3


@dataclass(frozen=True, slots=True)
class FullRunPrerequisites:
    sequence_age_baseline_id: str
    caduceus_age_baseline_id: str

    def __post_init__(self) -> None:
        for artifact_id in (self.sequence_age_baseline_id, self.caduceus_age_baseline_id):
            if _ARTIFACT_ID.fullmatch(artifact_id) is None:
                raise ValueError(f"invalid prerequisite artifact ID: {artifact_id!r}")


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One durable run pointer keyed to exact data, split, and input window."""

    artifact_id: str
    stage: RunStage
    protocol_id: str
    data_sha256: str
    split_sha256: str
    window_size: int
    eligibility: Eligibility
    prerequisites: FullRunPrerequisites | None

    def __post_init__(self) -> None:
        if _ARTIFACT_ID.fullmatch(self.artifact_id) is None:
            raise ValueError(f"invalid run artifact ID: {self.artifact_id!r}")
        if not self.protocol_id or self.window_size <= 0:
            raise ValueError("run protocol and positive window size are required")
        if (
            _SHA256.fullmatch(self.data_sha256) is None
            or _SHA256.fullmatch(self.split_sha256) is None
        ):
            raise ValueError("run data and split fingerprints must be SHA-256 values")
        if self.stage == RunStage.FULL_LATENT_METRIC and self.prerequisites is None:
            raise ValueError("full latent run requires both baseline artifact IDs")
        if self.stage != RunStage.FULL_LATENT_METRIC and self.prerequisites is not None:
            raise ValueError("baseline runs must not carry full-run prerequisites")

    @property
    def comparison_key(self) -> tuple[str, str, str, int]:
        return self.protocol_id, self.data_sha256, self.split_sha256, self.window_size


@dataclass(frozen=True, slots=True)
class ResultsRegistry:
    """Append-only logical registry that enforces stages 1 -> 2 -> 3."""

    records: tuple[RunRecord, ...] = ()

    def append(self, record: RunRecord) -> ResultsRegistry:
        if any(existing.artifact_id == record.artifact_id for existing in self.records):
            raise ValueError(f"duplicate run artifact ID: {record.artifact_id}")
        comparable = tuple(
            existing
            for existing in self.records
            if existing.comparison_key == record.comparison_key
        )
        if record.stage == RunStage.CADUCEUS_AGE_BASELINE and not any(
            existing.stage == RunStage.SEQUENCE_AGE_BASELINE for existing in comparable
        ):
            raise ValueError("Caduceus age baseline requires a completed sequence-feature baseline")
        if record.stage == RunStage.FULL_LATENT_METRIC:
            if record.prerequisites is None:
                raise RuntimeError("full-run prerequisite invariant was not enforced")
            by_id = {existing.artifact_id: existing for existing in comparable}
            required = (
                (record.prerequisites.sequence_age_baseline_id, RunStage.SEQUENCE_AGE_BASELINE),
                (record.prerequisites.caduceus_age_baseline_id, RunStage.CADUCEUS_AGE_BASELINE),
            )
            for artifact_id, stage in required:
                if artifact_id not in by_id or by_id[artifact_id].stage != stage:
                    raise ValueError(
                        f"full run prerequisite {artifact_id!r} is absent or has the wrong stage"
                    )
                if (
                    _ELIGIBILITY_ORDER[record.eligibility]
                    > _ELIGIBILITY_ORDER[by_id[artifact_id].eligibility]
                ):
                    raise ValueError("full run cannot be more eligible than either baseline")
        return replace(self, records=(*self.records, record))

    def primary_validated(self) -> tuple[RunRecord, ...]:
        return tuple(
            record for record in self.records if record.eligibility == Eligibility.PRIMARY_VALIDATED
        )
