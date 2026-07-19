"""Add the immutable post-hoc latent-interpretation report to an interim site."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Never, cast

from methylation_latent.artifacts import (
    JsonValue,
    require_clean_git_commit,
    sha256_file,
    write_canonical_json_exclusive,
)
from methylation_latent.latent_interpretation import deterministic_window_pairs
from methylation_latent.latent_interpretation_protocol import (
    LatentInterpretationProtocol,
    load_latent_interpretation_protocol,
)

_MANIFEST_SCHEMAS = {
    1: "methylation-latent.latent-interpretation-manifest.v1",
    2: "methylation-latent.latent-interpretation-manifest.v2",
}
_RESULTS_SCHEMAS = {
    1: "methylation-latent.latent-interpretation-results.v1",
    2: "methylation-latent.latent-interpretation-results.v2",
}
_PROVENANCE_SCHEMAS = {
    1: "methylation-latent.latent-interpretation-site-provenance.v1",
    2: "methylation-latent.latent-interpretation-site-provenance.v2",
}
_STATUS = "post_hoc_hypothesis_generating"
_SPLITS = ("diverse-blocks", "held-out-chromosome")
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


def _number(record: dict[str, object], key: str) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{key} must be finite")
    return result


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


def _validate_v1_window_comparisons(
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


def _validate_window_prediction_map(
    record: dict[str, object],
    key: str,
    windows: tuple[int, ...],
) -> dict[str, object]:
    predictions = _object(record, key)
    expected = {str(window) for window in windows}
    if set(predictions) != expected:
        raise ValueError(f"{key} window axis differs")
    return predictions


def _validate_v2_pairwise_comparison(
    comparison: dict[str, object],
    *,
    expected_pair: tuple[int, int],
    protocol: LatentInterpretationProtocol,
) -> None:
    expected_fields = {
        "first_window_size",
        "second_window_size",
        "metric_cosine",
        "age_pullback_cosine",
        "age_prediction_pearson",
        "age_prediction_mean_absolute_difference",
        "age_prediction_sign_agreement",
        "linear_cka",
        "top_neighbour_overlap",
        "neighbour_sample_count",
        "neighbour_count",
        "pair_prediction_agreement",
    }
    if (
        set(comparison) != expected_fields
        or (
            _integer(comparison, "first_window_size"),
            _integer(comparison, "second_window_size"),
        )
        != expected_pair
    ):
        raise ValueError("latent pairwise window-comparison identity differs")
    bounded = {
        "metric_cosine": (-1.0, 1.0),
        "age_pullback_cosine": (-1.0, 1.0),
        "age_prediction_pearson": (-1.0, 1.0),
        "age_prediction_sign_agreement": (0.0, 1.0),
        "linear_cka": (0.0, 1.0),
        "top_neighbour_overlap": (0.0, 1.0),
    }
    for key, (minimum, maximum) in bounded.items():
        value = _number(comparison, key)
        if not minimum - 1.0e-8 <= value <= maximum + 1.0e-8:
            raise ValueError(f"latent pairwise {key} is outside its bounds")
    if _number(comparison, "age_prediction_mean_absolute_difference") < 0.0:
        raise ValueError("latent pairwise age difference must be non-negative")
    if (
        _integer(comparison, "neighbour_sample_count") <= protocol.geometry.neighbour_count
        or _integer(comparison, "neighbour_count") != protocol.geometry.neighbour_count
    ):
        raise ValueError("latent pairwise neighbour contract differs")
    agreement = _object(comparison, "pair_prediction_agreement")
    if set(agreement) != set(_POPULATIONS):
        raise ValueError("latent pairwise population axis differs")
    agreement_fields = {
        "total_prediction_pearson",
        "age_component_prediction_pearson",
        "residual_prediction_pearson",
    }
    for raw_population in agreement.values():
        if not isinstance(raw_population, dict) or set(raw_population) != agreement_fields:
            raise ValueError("latent pairwise prediction-agreement fields differ")
        population = cast(dict[str, object], raw_population)
        for key in agreement_fields:
            if not -1.0 - 1.0e-8 <= _number(population, key) <= 1.0 + 1.0e-8:
                raise ValueError("latent pairwise prediction agreement is outside [-1,1]")


def _validate_v2_prediction_rows(
    rows: list[object],
    *,
    windows: tuple[int, ...],
    expected_count: int,
    name: str,
) -> None:
    if len(rows) != expected_count:
        raise ValueError(f"latent {name} count differs")
    for raw_row in rows:
        if not isinstance(raw_row, dict):
            raise TypeError(f"latent {name} row must be an object")
        row = cast(dict[str, object], raw_row)
        predictions = _validate_window_prediction_map(row, "predictions", windows)
        for window in windows:
            _number(predictions, str(window))


def _validate_v2_window_comparisons(
    results: dict[str, object], protocol: LatentInterpretationProtocol
) -> None:
    comparisons = _list(results, "window_comparisons")
    if len(comparisons) != len(protocol.splits):
        raise ValueError("latent window-comparison count differs")
    expected_comparison_fields = {
        "split_name",
        "window_sizes",
        "pairwise_window_comparisons",
        "age_rank_curves",
        "pair_candidate_distance_class",
        "pair_rank_curves",
        "age_candidates",
        "pair_candidates",
        "age_display",
        "pair_displays",
    }
    observed_splits: list[str] = []
    expected_pairs = deterministic_window_pairs(protocol.windows)
    for raw_comparison in comparisons:
        if not isinstance(raw_comparison, dict):
            raise TypeError("latent window comparison must be an object")
        comparison = cast(dict[str, object], raw_comparison)
        if set(comparison) != expected_comparison_fields:
            raise ValueError("latent window-comparison fields differ")
        observed_splits.append(_string(comparison, "split_name"))
        windows = _list(comparison, "window_sizes")
        if windows != list(protocol.windows):
            raise ValueError("latent window-comparison window axis differs")
        raw_pairwise = _list(comparison, "pairwise_window_comparisons")
        if len(raw_pairwise) != len(expected_pairs):
            raise ValueError("latent pairwise window-comparison count differs")
        for raw_pair, expected_pair in zip(raw_pairwise, expected_pairs, strict=True):
            if not isinstance(raw_pair, dict):
                raise TypeError("latent pairwise window comparison must be an object")
            _validate_v2_pairwise_comparison(
                cast(dict[str, object], raw_pair),
                expected_pair=expected_pair,
                protocol=protocol,
            )
        _validate_v2_prediction_rows(
            _list(comparison, "age_display"),
            windows=protocol.windows,
            expected_count=protocol.display.age_probe_count,
            name="age display",
        )
        pair_displays = _object(comparison, "pair_displays")
        if set(pair_displays) != set(_POPULATIONS):
            raise ValueError("latent pair display population axis differs")
        for raw_display in pair_displays.values():
            if not isinstance(raw_display, dict):
                raise TypeError("latent pair display must be an object")
            display = cast(dict[str, object], raw_display)
            if set(display) != {
                "source_count",
                "display_count",
                "distance_class",
                "targets",
                "predictions",
            }:
                raise ValueError("latent pair display fields differ")
            count = _integer(display, "display_count")
            if count <= 0 or _integer(display, "source_count") < count:
                raise ValueError("latent pair display counts differ")
            distances = _list(display, "distance_class")
            if len(distances) != count or any(
                isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 7
                for value in distances
            ):
                raise ValueError("latent pair display distance axis differs")
            targets = _object(display, "targets")
            if set(targets) != set(_COMPONENTS) or any(
                len(_list(targets, component)) != count for component in _COMPONENTS
            ):
                raise ValueError("latent pair display target axes differ")
            predictions = _validate_window_prediction_map(display, "predictions", protocol.windows)
            for window in protocol.windows:
                raw_components = predictions[str(window)]
                if not isinstance(raw_components, dict):
                    raise TypeError("latent pair display prediction must be an object")
                components = cast(dict[str, object], raw_components)
                if set(components) != set(_COMPONENTS) or any(
                    len(_list(components, component)) != count for component in _COMPONENTS
                ):
                    raise ValueError("latent pair display prediction axes differ")
        for key in ("age_candidates", "pair_candidates"):
            candidates = _object(comparison, key)
            if set(candidates) != {"positive", "negative"}:
                raise ValueError("latent candidate direction axis differs")
            for direction in ("positive", "negative"):
                _validate_v2_prediction_rows(
                    _list(candidates, direction),
                    windows=protocol.windows,
                    expected_count=protocol.display.candidate_table_size,
                    name=f"{key} {direction}",
                )
        for key in ("age_rank_curves", "pair_rank_curves"):
            curves = _object(comparison, key)
            if set(curves) != {"positive", "negative"}:
                raise ValueError("latent candidate-rank direction axis differs")
            for direction in ("positive", "negative"):
                rows = _list(curves, direction)
                if len(rows) != len(protocol.display.candidate_ranks) or any(
                    not isinstance(row, dict) for row in rows
                ):
                    raise ValueError("latent candidate-rank records differ")
                if (
                    tuple(_integer(cast(dict[str, object], row), "rank") for row in rows)
                    != protocol.display.candidate_ranks
                ):
                    raise ValueError("latent candidate-rank axis differs")
        distance_class = _string(comparison, "pair_candidate_distance_class")
        if distance_class not in _DISTANCE_CLASSES:
            raise ValueError("latent candidate distance class differs")
    if tuple(observed_splits) != protocol.splits:
        raise ValueError("latent window-comparison split order differs")


def _validate_results(
    results: dict[str, object],
    *,
    protocol: LatentInterpretationProtocol,
    config_sha256: str,
) -> None:
    identity = _object(results, "identity")
    audit = _object(results, "audit")
    if (
        set(results)
        != {
            "schema",
            "status",
            "identity",
            "audit",
            "reliability",
            "cells",
            "window_comparisons",
            "cross_split_weight_agreement",
        }
        or results.get("schema") != _RESULTS_SCHEMAS[protocol.artifact_version]
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
    raw_parents = _list(identity, "parent_models")
    if len(raw_parents) != len(protocol.parents):
        raise ValueError("latent interpretation parent identity count differs")
    for raw_parent, parent in zip(raw_parents, protocol.parents, strict=True):
        if not isinstance(raw_parent, dict):
            raise TypeError("latent interpretation parent identity must be an object")
        parent_record = cast(dict[str, object], raw_parent)
        if set(parent_record) != {
            "split_name",
            "window_size",
            "metadata_sha256",
            "model_sha256",
            "embedding_manifest_sha256",
        } or (
            _string(parent_record, "split_name"),
            _integer(parent_record, "window_size"),
            _string(parent_record, "metadata_sha256"),
            _string(parent_record, "model_sha256"),
            _string(parent_record, "embedding_manifest_sha256"),
        ) != (
            parent.split,
            parent.window,
            parent.metadata_sha256,
            parent.model_sha256,
            parent.embedding_manifest_sha256,
        ):
            raise ValueError("latent interpretation parent identity differs")
    cells = _list(results, "cells")
    observed = {
        _validate_cell(cast(dict[str, object], raw_cell), protocol=protocol)
        for raw_cell in cells
        if isinstance(raw_cell, dict)
    }
    expected = {(split, window) for split in protocol.splits for window in protocol.windows}
    if len(cells) != len(expected) or observed != expected:
        raise ValueError("latent interpretation cell grid differs")
    _validate_reliability(results, protocol)
    if protocol.artifact_version == 1:
        _validate_v1_window_comparisons(results, protocol)
    else:
        _validate_v2_window_comparisons(results, protocol)
    cross_split = _list(results, "cross_split_weight_agreement")
    if (
        len(cross_split) != len(protocol.windows)
        or any(not isinstance(record, dict) for record in cross_split)
        or tuple(_integer(cast(dict[str, object], record), "window_size") for record in cross_split)
        != protocol.windows
    ):
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
        manifest.get("schema") != _MANIFEST_SCHEMAS[protocol.artifact_version]
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
        "schema": _PROVENANCE_SCHEMAS[protocol.artifact_version],
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
