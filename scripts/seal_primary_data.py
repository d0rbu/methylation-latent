"""Re-audit and immutably seal a completed primary-data directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import torch as t

from methylation_latent.artifacts import (
    require_clean_git_commit,
    sha256_file,
    sha256_ordered_strings,
)
from methylation_latent.cohort import (
    load_prepared_cohort,
    load_probe_table,
)
from methylation_latent.config import ProtocolConfig, load_protocol_config
from methylation_latent.data_bundle import seal_primary_data_bundle_exclusive
from methylation_latent.domain import GenomicContext, NonEmptyProbeSet, ProbeLocus
from methylation_latent.experiment_data import (
    SplitArtifact,
    load_sequence_features,
    load_split_artifact,
)
from methylation_latent.splits import (
    OverlapAudit,
    apply_maximum_window_buffer,
    assert_no_window_overlap,
)
from methylation_latent.storage import load_target_geometry
from methylation_latent.targets import (
    assert_correlation_identity,
    build_target_geometry,
)

_SPLITS = ("diverse-blocks", "held-out-chromosome")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    return parser


def _load_json(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"expected JSON object: {path}")
    return cast(dict[str, object], raw)


def _required_object(record: dict[str, object], key: str) -> dict[str, object]:
    value = record.get(key)
    if not isinstance(value, dict) or any(not isinstance(name, str) for name in value):
        raise TypeError(f"{key} must be an object with string keys")
    return cast(dict[str, object], value)


def _required_list(record: dict[str, object], key: str) -> list[object]:
    value = record.get(key)
    if not isinstance(value, list):
        raise TypeError(f"{key} must be an array")
    return cast(list[object], value)


def _required_int(record: dict[str, object], key: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value


def _required_str(record: dict[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def _subset(probes: NonEmptyProbeSet, indices: t.Tensor) -> NonEmptyProbeSet:
    if indices.dtype != t.int64 or indices.ndim != 1 or indices.numel() == 0:
        raise ValueError("semantic split audit requires non-empty int64 indices")
    return NonEmptyProbeSet(tuple(probes.probes[index] for index in indices.tolist()))


def _probe_ids(probes: NonEmptyProbeSet) -> set[str]:
    return {str(probe.probe_id) for probe in probes.probes}


def _assert_context_coverage(probes: NonEmptyProbeSet, name: str) -> None:
    observed = {probe.context for probe in probes.probes}
    if observed != set(GenomicContext):
        raise ValueError(f"{name} does not contain every genomic context: {observed}")


def _block_selected_indices(
    probes: NonEmptyProbeSet,
    base_indices: t.Tensor,
    section: dict[str, object],
) -> t.Tensor:
    blocks = _required_list(section, "blocks")
    parsed: list[tuple[str, int, int, str, str]] = []
    for raw in blocks:
        if not isinstance(raw, dict):
            raise TypeError("split block must be an object")
        block = cast(dict[str, object], raw)
        parsed.append(
            (
                _required_str(block, "chromosome"),
                _required_int(block, "start"),
                _required_int(block, "end"),
                _required_str(block, "anchor_probe_id"),
                _required_str(block, "anchor_context"),
            )
        )
    selected: list[int] = []
    for index in base_indices.tolist():
        probe = probes.probes[index]
        cytosine_zero = int(probe.position) - 1
        chromosome = f"chr{int(probe.chromosome)}"
        if any(
            block_chromosome == chromosome and start <= cytosine_zero < end
            for block_chromosome, start, end, _, _ in parsed
        ):
            selected.append(index)
    selected_ids = {str(probes.probes[index].probe_id) for index in selected}
    by_id = {str(probe.probe_id): probe for probe in probes.probes}
    for chromosome, start, end, anchor_id, anchor_context in parsed:
        if end - start <= 0:
            raise ValueError("split block interval is empty")
        if anchor_id not in selected_ids:
            raise ValueError(f"split block anchor is not selected: {anchor_id}")
        anchor = by_id[anchor_id]
        if f"chr{int(anchor.chromosome)}" != chromosome or anchor.context.value != anchor_context:
            raise ValueError("split block anchor chromosome or context differs")
    return t.tensor(selected, dtype=t.int64)


def _expected_overlap_json(audits: tuple[OverlapAudit, ...]) -> list[dict[str, int | None]]:
    return [
        {
            "window_size": int(audit.window_size),
            "shared_chromosomes": audit.shared_chromosomes,
            "minimum_cytosine_distance": audit.minimum_cytosine_distance,
        }
        for audit in audits
    ]


def _assert_section_summary(
    section: dict[str, object],
    *,
    train: NonEmptyProbeSet,
    test: NonEmptyProbeSet,
    buffer: tuple[ProbeLocus, ...],
    audits: tuple[OverlapAudit, ...],
    expected_kind: str,
    expected_seed: int,
    config: ProtocolConfig,
) -> None:
    if (
        _required_str(section, "kind") != expected_kind
        or _required_int(section, "seed") != expected_seed
        or _required_int(section, "train_probes") != len(train)
        or _required_int(section, "test_probes") != len(test)
        or _required_int(section, "buffer_excluded_probes") != len(buffer)
        or _required_list(section, "window_sizes")
        != [int(window) for window in config.splits.window_sizes]
        or _required_list(section, "overlap_audits") != _expected_overlap_json(audits)
    ):
        raise ValueError("split metadata summary differs from recomputed partitions")


def _assert_split_semantics(
    probes: NonEmptyProbeSet,
    split: SplitArtifact,
    metadata: dict[str, object],
    *,
    split_file_name: str,
    config: ProtocolConfig,
) -> None:
    primary_train = _subset(probes, split.primary_train_indices)
    test = _subset(probes, split.test_indices)
    optimization = _subset(probes, split.optimization_indices)
    validation = _subset(probes, split.validation_indices)
    primary_section = _required_object(metadata, "primary")
    validation_section = _required_object(metadata, "validation_within_primary_train")
    maximum_window = max(config.splits.window_sizes, key=int)

    if split_file_name == "diverse-blocks":
        expected_test = _block_selected_indices(
            probes,
            t.arange(len(probes), dtype=t.int64),
            primary_section,
        )
        expected_kind = "diverse_blocks"
        if len({int(probe.chromosome) for probe in test.probes}) < 2:
            raise ValueError("diverse primary test is not spread across chromosomes")
    else:
        expected_test = t.tensor(
            tuple(
                index
                for index, probe in enumerate(probes.probes)
                if probe.chromosome == config.splits.held_out_chromosome
            ),
            dtype=t.int64,
        )
        expected_kind = "held_out_chromosome"
        if _required_list(primary_section, "blocks"):
            raise ValueError("held-out-chromosome split must not contain block records")
    if not t.equal(t.sort(expected_test).values, t.sort(split.test_indices).values):
        raise ValueError("primary test tensor does not match its target-blind split definition")

    primary_test_index_set = set(split.test_indices.tolist())
    primary_candidates = NonEmptyProbeSet(
        tuple(
            probe
            for index, probe in enumerate(probes.probes)
            if index not in primary_test_index_set
        )
    )
    expected_primary_train, expected_primary_buffer = apply_maximum_window_buffer(
        primary_candidates,
        test,
        maximum_window,
    )
    if _probe_ids(expected_primary_train) != _probe_ids(primary_train) or {
        str(probe.probe_id) for probe in expected_primary_buffer
    } != {str(probes.probes[index].probe_id) for index in split.primary_buffer_indices.tolist()}:
        raise ValueError("primary train/buffer tensors differ from exact maximum-window buffering")
    primary_audits = assert_no_window_overlap(
        primary_train,
        test,
        config.splits.window_sizes,
    )
    _assert_section_summary(
        primary_section,
        train=primary_train,
        test=test,
        buffer=expected_primary_buffer,
        audits=primary_audits,
        expected_kind=expected_kind,
        expected_seed=config.splits.primary_seed,
        config=config,
    )

    expected_validation = _block_selected_indices(
        probes,
        split.primary_train_indices,
        validation_section,
    )
    if not t.equal(t.sort(expected_validation).values, t.sort(split.validation_indices).values):
        raise ValueError("validation tensor does not match its target-blind block definition")
    validation_index_set = set(split.validation_indices.tolist())
    validation_candidates = NonEmptyProbeSet(
        tuple(
            probe
            for index, probe in zip(
                split.primary_train_indices.tolist(),
                primary_train.probes,
                strict=True,
            )
            if index not in validation_index_set
        )
    )
    expected_optimization, expected_validation_buffer = apply_maximum_window_buffer(
        validation_candidates,
        validation,
        maximum_window,
    )
    if _probe_ids(expected_optimization) != _probe_ids(optimization) or {
        str(probe.probe_id) for probe in expected_validation_buffer
    } != {str(probes.probes[index].probe_id) for index in split.validation_buffer_indices.tolist()}:
        raise ValueError("optimization/validation-buffer tensors differ from exact buffering")
    validation_audits = assert_no_window_overlap(
        optimization,
        validation,
        config.splits.window_sizes,
    )
    _assert_section_summary(
        validation_section,
        train=optimization,
        test=validation,
        buffer=expected_validation_buffer,
        audits=validation_audits,
        expected_kind="diverse_blocks",
        expected_seed=config.splits.validation_seed,
        config=config,
    )
    _assert_context_coverage(primary_train, f"{split_file_name} primary train")
    _assert_context_coverage(test, f"{split_file_name} primary test")
    _assert_context_coverage(optimization, f"{split_file_name} optimization")
    _assert_context_coverage(validation, f"{split_file_name} validation")


def main() -> None:
    arguments = _parser().parse_args()
    git_commit = require_clean_git_commit(Path(__file__).resolve().parents[1])
    config = load_protocol_config(arguments.config)
    protocol_sha256 = sha256_file(arguments.config)
    probes = load_probe_table(arguments.data / "probes.tsv")
    cohort = load_prepared_cohort(
        arguments.data / "cohort.safetensors",
        arguments.data / "cohort.json",
        probe_universe=probes,
    )
    targets = load_target_geometry(arguments.data / "targets.safetensors")
    recomputed = build_target_geometry(cohort.beta, cohort.age)
    if (
        not t.equal(targets.methylation.tensor, recomputed.methylation.tensor)
        or not t.equal(targets.age.tensor, recomputed.age.tensor)
        or not t.equal(targets.rho.tensor, recomputed.rho.tensor)
    ):
        raise ValueError("stored target geometry differs from exact cohort recomputation")
    target_metadata = _load_json(arguments.data / "targets.json")
    if (
        _required_str(target_metadata, "tensor_sha256")
        != sha256_file(arguments.data / "targets.safetensors")
        or _required_int(target_metadata, "probe_count") != len(probes)
        or _required_int(target_metadata, "sample_count") != cohort.audit.retained_samples
        or _required_str(target_metadata, "probe_order_sha256")
        != sha256_ordered_strings(str(probe.probe_id) for probe in probes.probes)
        or _required_str(target_metadata, "sample_order_sha256") != cohort.sample_order_sha256
    ):
        raise ValueError("target metadata differs from its recomputed source identity")
    audit_probe_ids = _required_list(target_metadata, "correlation_audit_probe_ids")
    if any(not isinstance(probe_id, str) for probe_id in audit_probe_ids):
        raise TypeError("correlation audit probe IDs must be strings")
    by_probe_id = {str(probe.probe_id): index for index, probe in enumerate(probes.probes)}
    audit_indices = t.tensor(
        tuple(by_probe_id[cast(str, probe_id)] for probe_id in audit_probe_ids),
        dtype=t.int64,
    )
    observed_error = assert_correlation_identity(
        cohort.beta,
        targets.methylation,
        audit_indices,
    )
    recorded_error = target_metadata.get("correlation_identity_maximum_error")
    if (
        isinstance(recorded_error, bool)
        or not isinstance(recorded_error, int | float)
        or observed_error != float(recorded_error)
    ):
        raise ValueError("correlation-identity audit differs on bundle sealing")

    features = load_sequence_features(
        arguments.data / "sequence-features.safetensors",
        arguments.data / "sequence-features.json",
        probes=probes,
    )
    if set(features.by_window) != set(map(int, config.splits.window_sizes)):
        raise ValueError("sequence-feature windows differ from the frozen sweep")
    for split_name in _SPLITS:
        metadata_path = arguments.data / "splits" / f"{split_name}.json"
        split = load_split_artifact(
            arguments.data / "splits" / f"{split_name}.safetensors",
            metadata_path,
            probes=probes,
        )
        _assert_split_semantics(
            probes,
            split,
            _load_json(metadata_path),
            split_file_name=split_name,
            config=config,
        )

    bundle = seal_primary_data_bundle_exclusive(
        arguments.data,
        protocol_id=config.protocol_id,
        protocol_sha256=protocol_sha256,
        git_commit=git_commit,
        retained_probes=len(probes),
        retained_samples=cohort.audit.retained_samples,
    )
    print(
        f"sealed protocol={bundle.protocol_id} probes={bundle.retained_probes} "
        f"samples={bundle.retained_samples} files={len(bundle.files)}"
    )


if __name__ == "__main__":
    main()
