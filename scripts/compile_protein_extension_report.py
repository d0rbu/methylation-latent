"""Add the immutable protein report to an existing compiled interim site."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Never, cast

from methylation_latent.artifacts import require_clean_git_commit, sha256_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-site", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _reject_nonfinite_json(value: str) -> Never:
    raise ValueError(f"protein results contain non-finite JSON number {value}")


def _validate_results(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise TypeError("protein results must be a JSON object")
    results = cast(dict[str, object], raw)
    if results.get("schema") != "methylation-latent.protein-extension-results.v1":
        raise ValueError("protein results use an unknown schema")
    if results.get("scientific_status") != "post_hoc_hypothesis_generating":
        raise ValueError("protein results lost their post-hoc scientific-status warning")
    proteins = results.get("protein_names")
    records = results.get("records")
    if not isinstance(proteins, list) or len(proteins) != 52:
        raise ValueError("protein results require the frozen 52-protein axis")
    if not isinstance(records, list) or len(records) != 20:
        raise ValueError("protein results are incomplete")
    expected = {
        (split, window, representation)
        for split in ("diverse-blocks", "held-out-chromosome")
        for window in (1024, 4096)
        for representation in (
            "free",
            "shared_tss",
            "tss_linear",
            "amino_acid_linear",
            "combined_linear",
        )
    }
    observed: set[tuple[str, int, str]] = set()
    for raw_record in records:
        if not isinstance(raw_record, dict):
            raise TypeError("protein experiment record must be an object")
        record = cast(dict[str, object], raw_record)
        split = record.get("split")
        window = record.get("window_size")
        representation = record.get("representation")
        if not isinstance(split, str) or not isinstance(window, int) or not isinstance(
            representation, str
        ):
            raise TypeError("protein experiment identity fields have invalid types")
        observed.add((split, window, representation))
        evaluation = record.get("evaluation")
        if not isinstance(evaluation, dict):
            raise TypeError("protein experiment evaluation must be an object")
        expected_generalization = representation != "free"
        if evaluation.get("protein_generalization_valid") is not expected_generalization:
            raise ValueError("protein generalization flag contradicts representation")
        pair_metrics = evaluation.get("protein_pair_metrics")
        if not isinstance(pair_metrics, dict):
            raise TypeError("protein-pair metrics must be an object")
        heldout_pairs = pair_metrics.get("heldout_heldout_off_diagonal")
        if not isinstance(heldout_pairs, dict) or heldout_pairs.get("count") != 28:
            raise ValueError("eight held-out proteins must yield 28 unique unordered pairs")
    if observed != expected:
        raise ValueError("protein experiment matrix is incomplete or duplicated")
    return results


def main() -> None:
    arguments = _parser().parse_args()
    repository = Path(__file__).resolve().parents[1]
    require_clean_git_commit(repository)
    raw = json.loads(
        arguments.results.read_text(encoding="utf-8"),
        parse_constant=_reject_nonfinite_json,
    )
    _validate_results(raw)
    if arguments.output.exists():
        raise FileExistsError(arguments.output)
    if not (arguments.base_site / "index.html").is_file():
        raise FileNotFoundError(arguments.base_site / "index.html")
    shutil.copytree(arguments.base_site, arguments.output)
    template = repository / "protein-site-template"
    for filename in (
        "protein-extension.html",
        "protein-extension.css",
        "protein-extension.js",
    ):
        shutil.copyfile(template / filename, arguments.output / filename)
    shutil.copyfile(arguments.results, arguments.output / "protein-extension-data.json")
    index_path = arguments.output / "index.html"
    index = index_path.read_text(encoding="utf-8")
    anchor = '        <a href="../held-out-chromosome/">Held-out chromosome 7</a>\n'
    if index.count(anchor) != 1:
        raise ValueError("compiled interim index navigation anchor differs")
    index_path.write_text(
        index.replace(
            anchor,
            anchor + '        <a href="protein-extension.html">Protein extension</a>\n',
        ),
        encoding="utf-8",
        newline="",
    )
    expected = {
        "index.html",
        "protein-extension.html",
        "protein-extension.css",
        "protein-extension.js",
        "protein-extension-data.json",
    }
    if not expected.issubset(path.name for path in arguments.output.iterdir()):
        raise RuntimeError("compiled protein report lacks required root files")
    print(
        f"compiled output={arguments.output} results_sha256={sha256_file(arguments.results)}"
    )


if __name__ == "__main__":
    main()
