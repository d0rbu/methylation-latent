"""Add the immutable dual-probe latent report to an existing interim site."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path, PurePosixPath
from typing import Never, cast

from methylation_latent.artifacts import (
    JsonValue,
    require_clean_git_commit,
    sha256_file,
    write_canonical_json_exclusive,
)
from methylation_latent.cohort import load_probe_table
from methylation_latent.config import load_protocol_config
from methylation_latent.data_bundle import verify_primary_data_bundle
from methylation_latent.domain import NonEmptyProbeSet
from methylation_latent.dual_probe_protocol import load_dual_probe_protocol

_MANIFEST_SCHEMA = "methylation-latent.dual-probe-manifest.v1"
_RESULTS_SCHEMA = "methylation-latent.dual-probe-results.v1"
_SELECTION_SCHEMA = "methylation-latent.dual-probe-selection.v1"
_TUNING_SCHEMA = "methylation-latent.dual-probe-tuning.v1"
_REFIT_SCHEMA = "methylation-latent.dual-probe-refit.v1"
_SITE_SCHEMA = "methylation-latent.dual-probe-site-data.v1"
_SPLITS = ("diverse-blocks", "held-out-chromosome")
_WINDOWS = (1_024, 4_096)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent-config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dual-results", type=Path, required=True)
    parser.add_argument("--base-site", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _reject_nonfinite_json(value: str) -> Never:
    raise ValueError(f"dual-probe artifact contains non-finite JSON number {value}")


def _load_json(path: Path) -> dict[str, object]:
    raw = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_nonfinite_json,
    )
    if not isinstance(raw, dict):
        raise TypeError(f"expected JSON object: {path}")
    return cast(dict[str, object], raw)


def _object(record: dict[str, object], key: str) -> dict[str, object]:
    value = record.get(key)
    if not isinstance(value, dict) or any(not isinstance(name, str) for name in value):
        raise TypeError(f"{key} must be an object")
    return cast(dict[str, object], value)


def _list(record: dict[str, object], key: str) -> list[object]:
    value = record.get(key)
    if not isinstance(value, list):
        raise TypeError(f"{key} must be an array")
    return cast(list[object], value)


def _string(record: dict[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise TypeError(f"{key} must be a non-empty string")
    return value


def _integer(record: dict[str, object], key: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value


def _safe_relative(root: Path, value: str) -> Path:
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or ".." in parsed.parts or not parsed.parts:
        raise ValueError("dual-probe manifest contains an unsafe relative path")
    path = root.joinpath(*parsed.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("dual-probe manifest path escapes its artifact root")
    return path


def _validate_display(results: dict[str, object]) -> None:
    display = _object(results, "display")
    if (
        display.get("sampling")
        != "target_blind_uniform_offset_sample_before_target_or_prediction_access"
        or _integer(display, "maximum_points") != 5_000
    ):
        raise ValueError("dual-probe display sampling contract differs")
    age = _object(display, "age")
    age_count = len(_list(age, "target"))
    if (
        not 0 < age_count <= 5_000
        or len(_list(age, "global_indices")) != age_count
        or len(_list(age, "mean_prediction")) != age_count
        or len(_list(age, "by_seed")) != 3
    ):
        raise ValueError("dual-probe age display axes differ")
    pairs = _object(display, "pairs")
    if set(pairs) != {"seen_by_held_out", "held_out_by_held_out"}:
        raise ValueError("dual-probe display pair populations differ")
    for population, raw in pairs.items():
        if not isinstance(raw, dict):
            raise TypeError("dual-probe pair display must be an object")
        record = cast(dict[str, object], raw)
        count = len(_list(record, "target"))
        if not 0 < count <= 5_000:
            raise ValueError("dual-probe pair display count differs")
        for key in (
            "pair_offsets",
            "left_global_indices",
            "right_global_indices",
            "distance_class",
        ):
            if len(_list(record, key)) != count:
                raise ValueError("dual-probe pair display axes differ")
        predictions = _object(record, "predictions")
        expected_modes = (
            {"sequence_inductive", "hybrid_learned_seen"}
            if population == "seen_by_held_out"
            else {"sequence_inductive"}
        )
        if set(predictions) != expected_modes:
            raise ValueError("dual-probe display prediction modes differ")
        for raw_mode in predictions.values():
            if not isinstance(raw_mode, dict):
                raise TypeError("dual-probe display prediction mode must be an object")
            mode = cast(dict[str, object], raw_mode)
            if len(_list(mode, "mean_prediction")) != count or len(_list(mode, "by_seed")) != 3:
                raise ValueError("dual-probe display prediction axes differ")


def _validate_results(
    results: dict[str, object],
    *,
    protocol_id: str,
    protocol_sha256: str,
    split_name: str,
    window: int,
) -> None:
    identity = _object(results, "identity")
    if (
        results.get("schema") != _RESULTS_SCHEMA
        or results.get("status") != "post_hoc_hypothesis_generating_selected_only_test_evaluation"
        or results.get("test_evaluated_strategy_count") != 1
        or _string(identity, "protocol_id") != protocol_id
        or _string(identity, "protocol_config_sha256") != protocol_sha256
        or _string(identity, "split_name") != split_name
        or _integer(identity, "window_size") != window
        or len(_list(results, "seeds")) != 3
        or len(_list(results, "seed_results")) != 3
    ):
        raise ValueError("dual-probe result identity or selected-only policy differs")
    selected = _string(results, "selected_alpha_strategy")
    if any(
        isinstance(raw_seed, dict)
        and any(key in raw_seed for key in ("alpha_strategy", "candidate"))
        for raw_seed in _list(results, "seed_results")
    ):
        raise ValueError("dual-probe test result exposes an unselected strategy")
    if selected == "":  # pragma: no cover - _string rejects this
        raise RuntimeError("selected alpha strategy unexpectedly empty")
    _validate_display(results)


def _validate_selection_and_tuning(
    *,
    root: Path,
    selection_path: Path,
    results: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, JsonValue]]]:
    selection = _load_json(selection_path)
    if (
        selection.get("schema") != _SELECTION_SCHEMA
        or selection.get("test_metrics_read") is not False
        or _string(selection, "selected_alpha_strategy")
        != _string(results, "selected_alpha_strategy")
        or len(_list(selection, "candidates")) != 7
    ):
        raise ValueError("dual-probe validation-only selection differs")
    if _string(_object(results, "identity"), "dual_selection_sha256") != sha256_file(
        selection_path
    ):
        raise ValueError("dual-probe result does not pin its selection record")
    expected_names = {
        "fixed_0",
        "fixed_0.25",
        "fixed_0.5",
        "fixed_0.75",
        "fixed_1",
        "linear_0_to_1",
        "linear_1_to_0",
    }
    observed_names: set[str] = set()
    histories: list[dict[str, JsonValue]] = []
    for raw_candidate in _list(selection, "candidates"):
        if not isinstance(raw_candidate, dict):
            raise TypeError("dual-probe selection candidate must be an object")
        candidate = cast(dict[str, object], raw_candidate)
        name = _string(candidate, "alpha_strategy")
        observed_names.add(name)
        seeds = _list(candidate, "seeds")
        if len(seeds) != 3:
            raise ValueError("dual-probe selection candidate seed count differs")
        seed_histories: list[dict[str, JsonValue]] = []
        for raw_seed in seeds:
            if not isinstance(raw_seed, dict):
                raise TypeError("dual-probe selection seed must be an object")
            seed = cast(dict[str, object], raw_seed)
            tuning_path = _safe_relative(root, _string(seed, "tuning_record"))
            if sha256_file(tuning_path) != _string(seed, "tuning_record_sha256"):
                raise ValueError("dual-probe tuning record hash differs")
            tuning = _load_json(tuning_path)
            if (
                tuning.get("schema") != _TUNING_SCHEMA
                or _string(_object(tuning, "identity"), "alpha_strategy") != name
                or _integer(_object(tuning, "partition"), "test_target_access_count") != 0
                or tuning.get("selected_step") != seed.get("selected_step")
            ):
                raise ValueError("dual-probe tuning record identity differs")
            seed_histories.append(
                {
                    "seed": _integer(seed, "seed"),
                    "initialization": cast(dict[str, JsonValue], _object(tuning, "initialization")),
                    "validation_history": cast(
                        list[JsonValue], _list(tuning, "validation_history")
                    ),
                }
            )
        histories.append(
            {
                "alpha_strategy": name,
                "alpha_mode": _string(candidate, "alpha_mode"),
                "alpha_start": cast(float, candidate["alpha_start"]),
                "alpha_end": cast(float, candidate["alpha_end"]),
                "mean_seed_minimum_validation_score": cast(
                    float, candidate["mean_seed_minimum_validation_score"]
                ),
                "standard_deviation_seed_minimum_validation_score": cast(
                    float,
                    candidate["standard_deviation_seed_minimum_validation_score"],
                ),
                "seeds": seed_histories,
            }
        )
    if observed_names != expected_names:
        raise ValueError("dual-probe alpha strategy grid differs")
    return selection, histories


def _validate_refits(root: Path, results: dict[str, object]) -> None:
    identity = _object(results, "identity")
    expected_hashes = _list(identity, "refit_metadata_sha256")
    seeds = tuple(
        _integer(cast(dict[str, object], row), "seed") for row in _list(results, "seed_results")
    )
    if len(expected_hashes) != len(seeds):
        raise ValueError("dual-probe refit hash and seed axes differ")
    split_name = _string(identity, "split_name")
    window = _integer(identity, "window_size")
    for seed, expected_hash in zip(seeds, expected_hashes, strict=True):
        directory = root / split_name / f"window-{window}" / "refit" / f"seed-{seed}"
        metadata_path = directory / "metadata.json"
        metadata = _load_json(metadata_path)
        model_path = directory / "model.safetensors"
        if (
            not isinstance(expected_hash, str)
            or sha256_file(metadata_path) != expected_hash
            or metadata.get("schema") != _REFIT_SCHEMA
            or metadata.get("model_file") != model_path.name
            or metadata.get("model_sha256") != sha256_file(model_path)
            or _integer(_object(metadata, "partition"), "held_out_learned_row_count") != 0
            or _integer(_object(metadata, "partition"), "held_out_ols_row_count") != 0
        ):
            raise ValueError("dual-probe refit artifact differs")


def _annotation_rows(
    cells: list[dict[str, JsonValue]],
    probes: NonEmptyProbeSet,
) -> dict[str, JsonValue]:
    rows = probes.probes
    indices: set[int] = set()
    for cell in cells:
        results = cast(dict[str, object], cell["results"])
        display = _object(results, "display")
        indices.update(cast(list[int], _list(_object(display, "age"), "global_indices")))
        for raw_pair in _object(display, "pairs").values():
            pair = cast(dict[str, object], raw_pair)
            indices.update(cast(list[int], _list(pair, "left_global_indices")))
            indices.update(cast(list[int], _list(pair, "right_global_indices")))
    annotations: dict[str, JsonValue] = {}
    for index in sorted(indices):
        if not 0 <= index < len(rows):
            raise IndexError("dual-probe display annotation index is outside probe universe")
        probe = rows[index]
        annotations[str(index)] = {
            "probe_id": str(probe.probe_id),
            "chromosome": int(probe.chromosome),
            "position": int(probe.position),
            "context": probe.context.value,
            "design": probe.design.value,
            "manifest_strand": probe.manifest_strand.value,
        }
    return annotations


def main() -> None:
    arguments = _parser().parse_args()
    repository = Path(__file__).resolve().parents[1]
    compiler_git_commit = require_clean_git_commit(repository)
    protocol = load_dual_probe_protocol(arguments.config)
    protocol_sha256 = sha256_file(arguments.config)
    if sha256_file(arguments.parent_config) != protocol.parent_protocol_sha256:
        raise ValueError("dual report parent protocol hash differs")
    parent = load_protocol_config(arguments.parent_config)
    if parent.protocol_id != protocol.parent_protocol_id:
        raise ValueError("dual report parent protocol ID differs")
    if sha256_file(arguments.data / "bundle.json") != protocol.data_bundle_sha256:
        raise ValueError("dual report data-bundle hash differs")
    verify_primary_data_bundle(
        arguments.data,
        protocol_id=protocol.parent_protocol_id,
        protocol_sha256=protocol.parent_protocol_sha256,
    )
    probes = load_probe_table(arguments.data / "probes.tsv")
    manifest_path = arguments.dual_results / "manifest.json"
    manifest = _load_json(manifest_path)
    if (
        manifest.get("schema") != _MANIFEST_SCHEMA
        or manifest.get("protocol_id") != protocol.protocol_id
        or manifest.get("protocol_config_sha256") != protocol_sha256
        or len(_list(manifest, "cells")) != 4
    ):
        raise ValueError("dual-probe manifest identity or completeness differs")
    cells: list[dict[str, JsonValue]] = []
    observed: set[tuple[str, int]] = set()
    artifact_git_commits: set[str] = set()
    for raw_cell in _list(manifest, "cells"):
        if not isinstance(raw_cell, dict):
            raise TypeError("dual-probe manifest cell must be an object")
        cell = cast(dict[str, object], raw_cell)
        split_name = _string(cell, "split_name")
        window = _integer(cell, "window_size")
        observed.add((split_name, window))
        results_path = _safe_relative(
            arguments.dual_results,
            _string(cell, "results_file"),
        )
        if sha256_file(results_path) != _string(cell, "results_sha256"):
            raise ValueError("dual-probe result hash differs from manifest")
        results = _load_json(results_path)
        _validate_results(
            results,
            protocol_id=protocol.protocol_id,
            protocol_sha256=protocol_sha256,
            split_name=split_name,
            window=window,
        )
        artifact_git_commits.add(_string(_object(results, "identity"), "code_git_commit"))
        selection_path = results_path.parent / "selection.json"
        selection, histories = _validate_selection_and_tuning(
            root=arguments.dual_results,
            selection_path=selection_path,
            results=results,
        )
        _validate_refits(arguments.dual_results, results)
        cells.append(
            {
                "split_name": split_name,
                "window_size": window,
                "selection": cast(dict[str, JsonValue], selection),
                "validation_histories": histories,
                "results": cast(dict[str, JsonValue], results),
                "results_sha256": sha256_file(results_path),
            }
        )
    expected = {(split, window) for split in _SPLITS for window in _WINDOWS}
    if observed != expected or len(observed) != 4 or len(artifact_git_commits) != 1:
        raise ValueError("dual-probe report cells or producer Git identity differ")
    cells.sort(key=lambda cell: (_SPLITS.index(str(cell["split_name"])), int(cell["window_size"])))
    site_data: dict[str, JsonValue] = {
        "schema": _SITE_SCHEMA,
        "scientific_status": "post_hoc_hypothesis_generating",
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol_sha256,
        "artifact_git_commit": next(iter(artifact_git_commits)),
        "compiler_git_commit": compiler_git_commit,
        "manifest_sha256": sha256_file(manifest_path),
        "cells": cells,
        "probe_annotations": _annotation_rows(cells, probes),
    }
    if arguments.output.exists():
        raise FileExistsError(arguments.output)
    required_base = {
        "index.html",
        "style.css",
        "diverse-blocks",
        "held-out-chromosome",
    }
    if not required_base.issubset(path.name for path in arguments.base_site.iterdir()):
        raise ValueError("dual report base-site inventory differs")
    template_files = ("dual-probe.html", "dual-probe.css", "dual-probe.js")
    if any(not (arguments.template / filename).is_file() for filename in template_files):
        raise FileNotFoundError("dual-probe site template is incomplete")
    shutil.copytree(arguments.base_site, arguments.output)
    for filename in template_files:
        shutil.copyfile(arguments.template / filename, arguments.output / filename)
    write_canonical_json_exclusive(arguments.output / "dual-probe-data.json", site_data)
    index_path = arguments.output / "index.html"
    index = index_path.read_text(encoding="utf-8")
    anchor = (
        '        <a href="protein-extension.html">\n'
        "          <strong>Post-hoc protein extension</strong>\n"
        "          <span>Protein anchors from gene-locus DNA and amino-acid sequence</span>\n"
        "        </a>\n"
    )
    if index.count(anchor) != 1:
        raise ValueError("dual report landing-page anchor differs")
    index_path.write_text(
        index.replace(
            anchor,
            anchor
            + '        <a href="dual-probe.html">\n'
            + "          <strong>Dual probe latent experiment</strong>\n"
            + "          <span>Learned probe geometry with gradient-routed sequence catching</span>\n"
            + "        </a>\n",
        ),
        encoding="utf-8",
        newline="",
    )
    print(f"compiled output={arguments.output} manifest_sha256={sha256_file(manifest_path)}")


if __name__ == "__main__":
    main()
