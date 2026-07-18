"""Add the immutable post-hoc latent-interpretation report to an interim site."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Never, cast

from methylation_latent.artifacts import (
    JsonValue,
    require_clean_git_commit,
    sha256_file,
    write_canonical_json_exclusive,
)
from methylation_latent.latent_interpretation_protocol import (
    LatentInterpretationProtocol,
    load_latent_interpretation_protocol,
)

_MANIFEST_SCHEMA = "methylation-latent.latent-interpretation-manifest.v1"
_RESULTS_SCHEMA = "methylation-latent.latent-interpretation-results.v1"
_STATUS = "post_hoc_hypothesis_generating"
_SPLITS = ("diverse-blocks", "held-out-chromosome")
_WINDOWS = (1024, 4096)
_POPULATIONS = ("seen_stratified", "held_out_stratified")
_COMPONENTS = ("total", "age_component", "age_adjusted_residual")
_DISTANCE_CLASSES = (
    "cis_0_1kb",
    "cis_1_4kb",
    "cis_4_16kb",
    "cis_16_64kb",
    "cis_64_256kb",
    "cis_256kb_1mb",
    "cis_1mb_plus",
    "trans",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--base-site", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _reject_nonfinite_json(value: str) -> Never:
    raise ValueError(f"latent-interpretation artifact contains non-finite number {value}")


def _load_object(path: Path) -> dict[str, object]:
    raw = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_nonfinite_json,
    )
    if not isinstance(raw, dict):
        raise TypeError(f"expected a JSON object: {path}")
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


def _metric(record: dict[str, object], key: str) -> dict[str, object]:
    metric = _object(record, key)
    required = {
        "count",
        "mse",
        "pearson",
        "prediction_mean",
        "prediction_standard_deviation",
        "r_squared",
        "target_mean",
        "target_standard_deviation",
    }
    if set(metric) != required or _integer(metric, "count") <= 1:
        raise ValueError(f"{key} metric fields or count differ")
    return metric


def _validate_reliability(
    results: dict[str, object], protocol: LatentInterpretationProtocol
) -> None:
    reliability = _object(results, "reliability")
    if set(reliability) != set(_SPLITS):
        raise ValueError("latent reliability split axis differs")
    expected_replicates = len(protocol.reliability.subject_split_seeds)
    for split in _SPLITS:
        raw_split = reliability[split]
        if not isinstance(raw_split, dict):
            raise TypeError("latent reliability split must be an object")
        split_record = cast(dict[str, object], raw_split)
        summaries = [_object(split_record, "probe_age")]
        overall = _object(split_record, "pair_overall")
        if set(overall) != set(_COMPONENTS):
            raise ValueError("latent reliability component axis differs")
        summaries.extend(cast(dict[str, object], overall[name]) for name in _COMPONENTS)
        by_distance = _object(split_record, "pair_by_distance")
        expected_distances = (
            set(_DISTANCE_CLASSES)
            if split == "diverse-blocks"
            else set(_DISTANCE_CLASSES) - {"trans"}
        )
        if set(by_distance) != expected_distances:
            raise ValueError("latent reliability distance axis differs")
        for raw_distance in by_distance.values():
            if not isinstance(raw_distance, dict) or set(raw_distance) != set(_COMPONENTS):
                raise ValueError("latent reliability distance components differ")
            distance = cast(dict[str, object], raw_distance)
            summaries.extend(cast(dict[str, object], distance[name]) for name in _COMPONENTS)
        for summary in summaries:
            values = _list(summary, "values")
            if len(values) != expected_replicates:
                raise ValueError("latent reliability replicate count differs")


def _validate_cell(
    cell: dict[str, object],
    *,
    protocol: LatentInterpretationProtocol,
) -> tuple[str, int]:
    split = _string(cell, "split_name")
    window = _integer(cell, "window_size")
    parent = protocol.parent(split, window)
    parent_record = _object(cell, "parent")
    if (
        parent_record.get("metadata_sha256") != parent.metadata_sha256
        or parent_record.get("model_sha256") != parent.model_sha256
        or parent_record.get("embedding_manifest_sha256") != parent.embedding_manifest_sha256
    ):
        raise ValueError("latent interpretation parent hashes differ")
    age = _metric(cell, "age")
    if _integer(age, "count") != _integer(cell, "held_out_probe_count"):
        raise ValueError("latent interpretation age and held-out counts differ")
    pair_decomposition = _object(cell, "pair_decomposition")
    if set(pair_decomposition) != set(_POPULATIONS):
        raise ValueError("latent pair-population axis differs")
    for raw_population in pair_decomposition.values():
        if not isinstance(raw_population, dict):
            raise TypeError("latent pair population must be an object")
        population = cast(dict[str, object], raw_population)
        overall = _object(population, "overall")
        for component in _COMPONENTS:
            _metric(overall, component)
        by_distance = _object(population, "by_distance")
        if not set(by_distance).issubset(_DISTANCE_CLASSES) or not by_distance:
            raise ValueError("latent pair distance axis differs")
        for raw_distance in by_distance.values():
            if not isinstance(raw_distance, dict):
                raise TypeError("latent pair distance record must be an object")
            distance = cast(dict[str, object], raw_distance)
            for component in _COMPONENTS:
                _metric(distance, component)
    surrogates = _object(cell, "surrogates")
    if set(surrogates) != {"cpg_gc", "cpg_gc_plus_annotations"}:
        raise ValueError("latent surrogate axis differs")
    for raw_surrogate in surrogates.values():
        if not isinstance(raw_surrogate, dict):
            raise TypeError("latent surrogate must be an object")
        surrogate = cast(dict[str, object], raw_surrogate)
        _metric(surrogate, "model_output")
        _metric(surrogate, "empirical_rho")
    weights = _object(cell, "weights")
    representation = _object(cell, "representation")
    if _integer(weights, "latent_dimension") != _integer(parent_record, "latent_dimension"):
        raise ValueError("latent dimension differs between model and weight audit")
    for key in ("metric_spectrum",):
        if "participation_rank" not in _object(weights, key):
            raise ValueError("latent metric spectrum is incomplete")
    for key in ("normalized_latent_covariance", "pre_normalization_covariance"):
        if "participation_rank" not in _object(representation, key):
            raise ValueError("latent representation spectrum is incomplete")
    return split, window


def _validate_window_comparisons(
    results: dict[str, object], protocol: LatentInterpretationProtocol
) -> None:
    comparisons = _list(results, "window_comparisons")
    if len(comparisons) != len(_SPLITS):
        raise ValueError("latent window-comparison count differs")
    observed: set[str] = set()
    for raw_comparison in comparisons:
        if not isinstance(raw_comparison, dict):
            raise TypeError("latent window comparison must be an object")
        comparison = cast(dict[str, object], raw_comparison)
        observed.add(_string(comparison, "split_name"))
        age_display = _list(comparison, "age_display")
        if len(age_display) != protocol.display.age_probe_count:
            raise ValueError("latent age display count differs")
        pair_displays = _object(comparison, "pair_displays")
        if set(pair_displays) != set(_POPULATIONS):
            raise ValueError("latent pair display population axis differs")
        for raw_display in pair_displays.values():
            if not isinstance(raw_display, dict):
                raise TypeError("latent pair display must be an object")
            display = cast(dict[str, object], raw_display)
            count = _integer(display, "display_count")
            axes = (
                "distance_class",
                "target_total",
                "target_age",
                "target_residual",
                "prediction_1kb_total",
                "prediction_1kb_age",
                "prediction_1kb_residual",
                "prediction_4kb_total",
                "prediction_4kb_age",
                "prediction_4kb_residual",
            )
            if count <= 0 or any(len(_list(display, axis)) != count for axis in axes):
                raise ValueError("latent pair display axes differ")
        for key in ("age_candidates", "pair_candidates"):
            candidates = _object(comparison, key)
            if set(candidates) != {"positive", "negative"} or any(
                len(_list(candidates, direction)) != protocol.display.candidate_table_size
                for direction in ("positive", "negative")
            ):
                raise ValueError("latent candidate table axis differs")
        for key in ("age_rank_curves", "pair_rank_curves"):
            curves = _object(comparison, key)
            if set(curves) != {"positive", "negative"} or any(
                tuple(
                    _integer(cast(dict[str, object], row), "rank")
                    for row in _list(curves, direction)
                )
                != protocol.display.candidate_ranks
                for direction in ("positive", "negative")
            ):
                raise ValueError("latent candidate-rank axis differs")
    if observed != set(_SPLITS):
        raise ValueError("latent window-comparison split axis differs")


def _validate_results(
    results: dict[str, object],
    *,
    protocol: LatentInterpretationProtocol,
    config_sha256: str,
) -> None:
    identity = _object(results, "identity")
    audit = _object(results, "audit")
    if (
        results.get("schema") != _RESULTS_SCHEMA
        or results.get("status") != _STATUS
        or identity.get("analysis_config_sha256") != config_sha256
        or identity.get("protocol_id") != protocol.protocol_id
        or identity.get("protocol_sha256") != protocol.protocol_sha256
        or identity.get("manifest_sha256") != protocol.gpl13534_sha256
        or audit.get("gpu_used") is not False
        or audit.get("model_parameters_changed") is not False
        or audit.get("candidate_selection_reads_empirical_targets") is not False
        or audit.get("test_status") != "already_viewed_post_hoc"
        or audit.get("manifest_requested_probes") != audit.get("manifest_matched_probes")
    ):
        raise ValueError("latent interpretation identity or audit contract differs")
    cells = _list(results, "cells")
    observed = {
        _validate_cell(cast(dict[str, object], raw_cell), protocol=protocol)
        for raw_cell in cells
        if isinstance(raw_cell, dict)
    }
    expected = {(split, window) for split in _SPLITS for window in _WINDOWS}
    if len(cells) != len(expected) or observed != expected:
        raise ValueError("latent interpretation cell grid differs")
    _validate_reliability(results, protocol)
    _validate_window_comparisons(results, protocol)
    cross_split = _list(results, "cross_split_weight_agreement")
    if len(cross_split) != len(_WINDOWS) or {
        _integer(cast(dict[str, object], record), "window_size")
        for record in cross_split
        if isinstance(record, dict)
    } != set(_WINDOWS):
        raise ValueError("latent cross-split weight-agreement axis differs")


def main() -> None:
    arguments = _parser().parse_args()
    repository = Path(__file__).resolve().parents[1]
    compiler_git_commit = require_clean_git_commit(repository)
    protocol = load_latent_interpretation_protocol(arguments.config)
    config_sha256 = sha256_file(arguments.config)
    manifest_path = arguments.results / "manifest.json"
    results_path = arguments.results / "results.json"
    manifest = _load_object(manifest_path)
    if (
        manifest.get("schema") != _MANIFEST_SCHEMA
        or manifest.get("status") != _STATUS
        or manifest.get("results_file") != results_path.name
        or manifest.get("results_sha256") != sha256_file(results_path)
    ):
        raise ValueError("latent interpretation manifest identity differs")
    results = _load_object(results_path)
    _validate_results(results, protocol=protocol, config_sha256=config_sha256)
    if manifest.get("code_git_commit") != _object(results, "identity").get("code_git_commit"):
        raise ValueError("latent interpretation producer commits differ")
    if arguments.output.exists():
        raise FileExistsError(arguments.output)
    required_base = {
        "index.html",
        "style.css",
        "diverse-blocks",
        "held-out-chromosome",
        "dual-probe.html",
    }
    if not required_base.issubset(path.name for path in arguments.base_site.iterdir()):
        raise ValueError("latent interpretation base-site inventory differs")
    template_files = (
        "latent-interpretation.html",
        "latent-interpretation.css",
        "latent-interpretation.js",
    )
    if any(not (arguments.template / filename).is_file() for filename in template_files):
        raise FileNotFoundError("latent-interpretation site template is incomplete")
    shutil.copytree(arguments.base_site, arguments.output)
    for filename in template_files:
        shutil.copyfile(arguments.template / filename, arguments.output / filename)
    shutil.copyfile(results_path, arguments.output / "latent-interpretation-data.json")
    provenance: dict[str, JsonValue] = {
        "schema": "methylation-latent.latent-interpretation-site-provenance.v1",
        "compiler_git_commit": compiler_git_commit,
        "producer_git_commit": _string(manifest, "code_git_commit"),
        "config_sha256": config_sha256,
        "analysis_manifest_sha256": sha256_file(manifest_path),
        "analysis_results_sha256": sha256_file(results_path),
    }
    write_canonical_json_exclusive(
        arguments.output / "latent-interpretation-provenance.json",
        provenance,
    )
    index_path = arguments.output / "index.html"
    index = index_path.read_text(encoding="utf-8")
    anchor = (
        '        <a href="dual-probe.html">\n'
        "          <strong>Dual probe latent experiment</strong>\n"
        "          <span>Learned probe geometry with gradient-routed sequence catching</span>\n"
        "        </a>\n"
    )
    if index.count(anchor) != 1:
        raise ValueError("latent report landing-page anchor differs")
    index_path.write_text(
        index.replace(
            anchor,
            anchor
            + '        <a href="latent-interpretation.html">\n'
            + "          <strong>What the learned geometry contains</strong>\n"
            + "          <span>Reliability, age decomposition, coarse features, and candidates</span>\n"
            + "        </a>\n",
        ),
        encoding="utf-8",
        newline="",
    )
    expected_files = {
        "latent-interpretation.html",
        "latent-interpretation.css",
        "latent-interpretation.js",
        "latent-interpretation-data.json",
        "latent-interpretation-provenance.json",
    }
    if not expected_files.issubset(path.name for path in arguments.output.iterdir()):
        raise RuntimeError("compiled latent report lacks required files")
    print(f"compiled output={arguments.output} results_sha256={sha256_file(results_path)}")


if __name__ == "__main__":
    main()
