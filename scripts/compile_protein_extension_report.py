"""Add the immutable protein report to an existing compiled interim site."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from methylation_latent.artifacts import require_clean_git_commit, sha256_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-site", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    repository = Path(__file__).resolve().parents[1]
    require_clean_git_commit(repository)
    raw = json.loads(arguments.results.read_text(encoding="utf-8"))
    if (
        not isinstance(raw, dict)
        or raw.get("schema") != "methylation-latent.protein-extension-results.v1"
        or not isinstance(raw.get("records"), list)
        or len(raw["records"]) != 20
    ):
        raise ValueError("protein results are incomplete or use an unknown schema")
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
