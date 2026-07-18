"""Prepare immutable common-subject protein correlation targets."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import tomllib
from pathlib import Path
from typing import cast
from xml.etree import ElementTree
from zipfile import ZipFile

import torch as t

from methylation_latent.artifacts import (
    JsonValue,
    require_clean_git_commit,
    sha256_file,
    sha256_ordered_strings,
    write_canonical_json_exclusive,
)
from methylation_latent.cohort import load_prepared_cohort, load_probe_table
from methylation_latent.domain import parse_window_size
from methylation_latent.protein_extension import (
    GeneLocus,
    ProteinPanel,
    build_protein_targets,
    target_blind_protein_split,
    tss_window_interval,
)
from methylation_latent.storage import save_safetensors_exclusive
from methylation_latent.targets import (
    assert_correlation_identity,
    standardize_rows,
    standardize_vector,
)

_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_CELL = re.compile(r"^([A-Z]+)[0-9]+$", flags=re.ASCII)
_FASTA_ACCESSION = re.compile(r"^>sp\|([^|]+)\|")
_FASTA_GENE = re.compile(r"(?:^| )GN=([^ ]+)(?: |$)")
_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a TOML table")
    return cast(dict[str, object], value)


def _string(table: dict[str, object], name: str) -> str:
    value = table.get(name)
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _integer(table: dict[str, object], name: str) -> int:
    value = table.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _column_index(reference: str) -> int:
    match = _CELL.fullmatch(reference)
    if match is None:
        raise ValueError(f"invalid XLSX cell reference: {reference!r}")
    value = 0
    for letter in match.group(1):
        value = value * 26 + ord(letter) - ord("A") + 1
    return value - 1


def _xlsx_rows(path: Path) -> dict[int, dict[int, str]]:
    with ZipFile(path) as archive:
        shared: tuple[str, ...] = ()
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = tuple(
                "".join(node.text or "" for node in item.iterfind(f".//{{{_NS}}}t"))
                for item in root.findall(f"{{{_NS}}}si")
            )
        sheet_path = "xl/worksheets/sheet1.xml"
        if sheet_path not in archive.namelist():
            raise ValueError(f"XLSX lacks {sheet_path}: {path}")
        root = ElementTree.fromstring(archive.read(sheet_path))
    rows: dict[int, dict[int, str]] = {}
    for row in root.findall(f".//{{{_NS}}}sheetData/{{{_NS}}}row"):
        number = int(row.attrib["r"])
        if number in rows:
            raise ValueError(f"XLSX contains duplicate row {number}: {path}")
        values: dict[int, str] = {}
        for cell in row.findall(f"{{{_NS}}}c"):
            column = _column_index(cell.attrib["r"])
            if column in values:
                raise ValueError(f"XLSX row {number} contains duplicate column {column}")
            cell_type = cell.attrib.get("t")
            raw_node = cell.find(f"{{{_NS}}}v")
            raw = "" if raw_node is None else raw_node.text or ""
            if cell_type == "s":
                raw = shared[int(raw)]
            elif cell_type == "inlineStr":
                raw = "".join(node.text or "" for node in cell.iterfind(f".//{{{_NS}}}t"))
            values[column] = raw
        rows[number] = values
    if not rows:
        raise ValueError(f"XLSX contains no rows: {path}")
    return rows


def _normalized_biomarker(value: str) -> str:
    return "".join(character for character in value.strip().lower() if character.isalnum())


def _published_selection_audit(
    *,
    workbook_headers: tuple[str, ...],
    gwas_path: Path,
    ewas_path: Path,
    panel_config: dict[str, object],
) -> dict[str, JsonValue]:
    gwas_rows = _xlsx_rows(gwas_path)
    ewas_rows = _xlsx_rows(ewas_path)
    if gwas_rows.get(15, {}).get(0) != "Biomarker" or ewas_rows.get(9, {}).get(0) != "Biomarker":
        raise ValueError("published supplement biomarker headers differ")
    gwas = {
        _normalized_biomarker(row.get(0, ""))
        for number, row in gwas_rows.items()
        if number >= 16 and row.get(0, "").strip()
    }
    ewas = {
        _normalized_biomarker(row.get(0, ""))
        for number, row in ewas_rows.items()
        if number >= 10 and row.get(0, "").strip()
    }
    public = {_normalized_biomarker(name) for name in workbook_headers}
    missing = _normalized_biomarker(_string(panel_config, "public_union_missing_biomarker"))
    if len(gwas) != _integer(panel_config, "published_gwas_biomarkers"):
        raise ValueError("published GWAS biomarker count differs")
    if len(ewas) != _integer(panel_config, "published_ewas_biomarkers"):
        raise ValueError("published EWAS biomarker count differs")
    if len(gwas | ewas) != _integer(panel_config, "published_union_biomarkers"):
        raise ValueError("published GWAS/EWAS biomarker union count differs")
    if public != (gwas | ewas) - {missing} or missing in public:
        raise ValueError("public protein columns do not equal the published hit union minus CXCL9")
    return {
        "gwas_biomarkers": len(gwas),
        "ewas_biomarkers": len(ewas),
        "union_biomarkers": len(gwas | ewas),
        "public_biomarkers": len(public),
        "public_union_missing_biomarker": _string(
            panel_config, "public_union_missing_biomarker"
        ),
        "public_is_outcome_selected": True,
    }


def _parse_uniprot(path: Path) -> dict[str, tuple[str | None, str]]:
    records: dict[str, tuple[str | None, str]] = {}
    header: str | None = None
    pieces: list[str] = []

    def finish() -> None:
        if header is None:
            return
        accession_match = _FASTA_ACCESSION.search(header)
        if accession_match is None:
            raise ValueError(f"reviewed UniProt FASTA header lacks an accession: {header}")
        accession = accession_match.group(1)
        gene_match = _FASTA_GENE.search(header)
        gene = None if gene_match is None else gene_match.group(1)
        sequence = "".join(pieces)
        if not sequence or not sequence.isalpha() or not sequence.isupper():
            raise ValueError(f"UniProt sequence is empty or non-alphabetic: {accession}")
        if accession in records:
            raise ValueError(f"duplicate UniProt accession: {accession}")
        records[accession] = (gene, sequence)

    with gzip.open(path, mode="rt", encoding="ascii", newline="") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.startswith(">"):
                finish()
                header = line
                pieces = []
            else:
                if header is None:
                    raise ValueError("UniProt sequence appears before its header")
                pieces.append(line)
    finish()
    if not records:
        raise ValueError("UniProt FASTA contains no records")
    return records


def _load_gene_loci(
    directory: Path,
    genes: tuple[str, ...],
    *,
    biotype_exceptions: dict[str, str],
) -> tuple[tuple[GeneLocus, ...], tuple[str, ...]]:
    if not set(biotype_exceptions) <= set(genes) or any(
        not gene or not biotype for gene, biotype in biotype_exceptions.items()
    ):
        raise ValueError("GRCh37 biotype exceptions must name configured genes and biotypes")
    loci: list[GeneLocus] = []
    ordered_hash_input: list[str] = []
    for gene in sorted(genes):
        path = directory / f"{gene}.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "assembly_name",
            "biotype",
            "canonical_transcript",
            "db_type",
            "description",
            "display_name",
            "end",
            "id",
            "logic_name",
            "object_type",
            "seq_region_name",
            "source",
            "species",
            "start",
            "strand",
            "version",
        }
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError(f"Ensembl lookup envelope differs: {path}")
        if (
            raw["assembly_name"] != "GRCh37"
            or raw["biotype"] != biotype_exceptions.get(gene, "protein_coding")
            or raw["object_type"] != "Gene"
            or raw["display_name"] != gene
            or not isinstance(raw["canonical_transcript"], str)
            or not raw["canonical_transcript"].startswith("ENST")
        ):
            raise ValueError(f"Ensembl lookup identity differs: {gene}")
        loci.append(
            GeneLocus(
                gene_symbol=gene,
                chromosome=str(raw["seq_region_name"]),
                start=int(raw["start"]),
                end=int(raw["end"]),
                strand=int(raw["strand"]),
            )
        )
        ordered_hash_input.extend((path.name, sha256_file(path)))
    by_gene = {locus.gene_symbol: locus for locus in loci}
    return tuple(by_gene[gene] for gene in genes), tuple(ordered_hash_input)


def _assert_s1_coordinates(path: Path, loci: tuple[GeneLocus, ...]) -> None:
    rows = _xlsx_rows(path)
    if rows.get(6, {}).get(3) != "HGNC Symbol" or rows.get(6, {}).get(4) != "Gene location (Build 37)":
        raise ValueError("biomarker supplement headers differ")
    by_gene: dict[str, tuple[str, str]] = {}
    for number, row in rows.items():
        if number < 7 or not row.get(3, "").strip():
            continue
        gene = row[3].strip()
        if gene in by_gene:
            raise ValueError(f"biomarker supplement contains duplicate gene symbol: {gene}")
        if row.get(5) not in {"Yes", "No"}:
            raise ValueError(f"biomarker QC flag differs for {gene}")
        by_gene[gene] = (row.get(4, ""), row[5])
    coordinate_pattern = re.compile(r"^chr([^:]+):([0-9]+)-([0-9]+)$")
    for locus in loci:
        if locus.gene_symbol not in by_gene:
            raise ValueError(f"selected gene is absent from biomarker supplement: {locus.gene_symbol}")
        coordinate, passed = by_gene[locus.gene_symbol]
        match = coordinate_pattern.fullmatch(coordinate)
        if match is None or passed != "Yes":
            raise ValueError(f"selected gene lacks passed-QC Build 37 coordinates: {locus.gene_symbol}")
        chromosome, start, end = match.groups()
        if chromosome != locus.chromosome or max(int(start), locus.start) > min(int(end), locus.end):
            raise ValueError(f"Ensembl and biomarker-table gene intervals do not overlap: {locus.gene_symbol}")


def main() -> None:
    arguments = _parser().parse_args()
    repository = Path(__file__).resolve().parents[1]
    git_commit = require_clean_git_commit(repository)
    config = tomllib.loads(arguments.config.read_text(encoding="utf-8"))
    if config.get("schema") != "methylation-latent.protein-extension-protocol.v1":
        raise ValueError("protein extension protocol schema differs")
    if config.get("status") != "frozen":
        raise ValueError("protein extension protocol must be frozen")
    sources_config = _object(config.get("sources"), "sources")
    panel_config = _object(config.get("panel"), "panel")
    split_config = _object(config.get("protein_split"), "protein_split")
    sequence_config = _object(config.get("sequence"), "sequence")
    proteins_raw = config.get("proteins")
    if not isinstance(proteins_raw, list):
        raise TypeError("proteins must be an array of tables")
    proteins = tuple(_object(value, "protein") for value in proteins_raw)
    names = tuple(_string(value, "name") for value in proteins)
    genes = tuple(_string(value, "gene") for value in proteins)
    accessions = tuple(_string(value, "uniprot") for value in proteins)
    if len(proteins) != _integer(panel_config, "selected_proteins"):
        raise ValueError("configured selected protein count differs")

    source_paths = {
        "additional_characteristics": arguments.sources
        / "GSE87571_additional_sample_characteristics.xlsx",
        "biomarker_table": arguments.sources / "Ahsan2017_S1_biomarkers.xlsx",
        "gwas_table": arguments.sources / "Ahsan2017_S3_primary_GWAS.xlsx",
        "ewas_table": arguments.sources / "Ahsan2017_S7_EWAS_hits.xlsx",
        "uniprot_fasta": arguments.sources / "uniprot-human-reviewed-UP000005640-20260717.fasta.gz",
    }
    for name, path in source_paths.items():
        expected = _string(sources_config, f"{name}_sha256")
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"source hash differs for {name}: expected={expected}, observed={observed}")

    workbook_rows = _xlsx_rows(source_paths["additional_characteristics"])
    header = workbook_rows.get(3)
    if header is None or header.get(0) != "geo accession":
        raise ValueError("additional-characteristics workbook header differs")
    first = _integer(panel_config, "workbook_first_column_zero_based")
    last = _integer(panel_config, "workbook_last_column_zero_based")
    columns = tuple(range(first, last + 1))
    workbook_headers = tuple(
        header.get(column, "").removeprefix("characteristics: ") for column in columns
    )
    if len(workbook_headers) != 64 or any(not value for value in workbook_headers):
        raise ValueError("public protein header axis differs from 64 complete names")
    selection_audit = _published_selection_audit(
        workbook_headers=workbook_headers,
        gwas_path=source_paths["gwas_table"],
        ewas_path=source_paths["ewas_table"],
        panel_config=panel_config,
    )
    data_rows = tuple(workbook_rows[number] for number in sorted(workbook_rows) if number > 3)
    by_gsm: dict[str, dict[int, str]] = {}
    for row in data_rows:
        gsm = row.get(0, "")
        if not gsm or gsm in by_gsm:
            raise ValueError("additional-characteristics GSM axis is empty or duplicate")
        by_gsm[gsm] = row

    probes = load_probe_table(arguments.data / "probes.tsv")
    cohort = load_prepared_cohort(
        arguments.data / "cohort.safetensors",
        arguments.data / "cohort.json",
        probe_universe=probes,
    )
    if not set(cohort.sample_gsm_ids) <= set(by_gsm):
        raise ValueError("sealed methylation subjects are absent from the protein workbook")
    retained_gsms = set(cohort.sample_gsm_ids)
    coverage = {
        workbook_headers[offset]: sum(
            row.get(column, "") not in {"", "NA"} and row[0] in retained_gsms
            for row in data_rows
        )
        for offset, column in enumerate(columns)
    }
    eligible_headers = tuple(
        name
        for name in workbook_headers
        if coverage[name] >= _integer(panel_config, "minimum_retained_cohort_coverage")
    )
    if set(eligible_headers) != set(names) or len(eligible_headers) != len(names):
        raise ValueError(
            "retained-cohort coverage-selected protein names differ from the frozen 52-protein panel"
        )
    selected_columns = tuple(columns[workbook_headers.index(name)] for name in names)
    common_gsms = tuple(
        gsm
        for gsm in cohort.sample_gsm_ids
        if all(by_gsm[gsm].get(column, "") not in {"", "NA"} for column in selected_columns)
    )
    if len(common_gsms) != _integer(panel_config, "common_subjects"):
        raise ValueError("complete protein/methylation common-subject count differs")
    cohort_by_gsm = {gsm: index for index, gsm in enumerate(cohort.sample_gsm_ids)}
    common_indices = t.tensor(tuple(cohort_by_gsm[gsm] for gsm in common_gsms), dtype=t.int64)
    protein_values = t.tensor(
        tuple(
            tuple(float(by_gsm[gsm][column]) for gsm in common_gsms)
            for column in selected_columns
        ),
        dtype=t.float64,
    )
    panel = ProteinPanel(names, genes, accessions, common_gsms, protein_values)

    uniprot = _parse_uniprot(source_paths["uniprot_fasta"])
    sequences: list[str] = []
    for gene, accession in zip(genes, accessions, strict=True):
        if accession not in uniprot:
            raise ValueError(f"selected UniProt accession is absent: {accession}")
        observed_gene, sequence = uniprot[accession]
        if observed_gene != gene:
            raise ValueError(
                f"UniProt gene mapping differs: accession={accession}, expected={gene}, observed={observed_gene}"
            )
        if set(sequence) - _AMINO_ACIDS:
            raise ValueError(f"selected UniProt sequence contains unsupported residues: {accession}")
        sequences.append(sequence)

    exceptions_raw = sequence_config.get("grch37_biotype_exceptions")
    if not isinstance(exceptions_raw, dict) or exceptions_raw != {
        "MMP12": "processed_transcript"
    }:
        raise ValueError("frozen GRCh37 biotype exception set differs")
    loci, ensembl_hash_input = _load_gene_loci(
        arguments.sources / "ensembl-grch37-protein-genes",
        genes,
        biotype_exceptions=cast(dict[str, str], exceptions_raw),
    )
    if sha256_ordered_strings(ensembl_hash_input) != _string(
        sources_config, "ensembl_lookup_set_sha256"
    ):
        raise ValueError("Ensembl GRCh37 lookup set fingerprint differs")
    _assert_s1_coordinates(source_paths["biomarker_table"], loci)

    split = target_blind_protein_split(
        genes,
        validation_count=_integer(split_config, "validation_count"),
        test_count=_integer(split_config, "test_count"),
        seed=_integer(split_config, "seed"),
    )
    frozen_partitions = {
        "train": tuple(cast(list[str], split_config["train_genes"])),
        "validation": tuple(cast(list[str], split_config["validation_genes"])),
        "test": tuple(cast(list[str], split_config["test_genes"])),
    }
    observed_partitions = {
        "train": tuple(genes[index] for index in split.train_indices.tolist()),
        "validation": tuple(genes[index] for index in split.validation_indices.tolist()),
        "test": tuple(genes[index] for index in split.test_indices.tolist()),
    }
    if observed_partitions != frozen_partitions:
        raise ValueError("target-blind protein split differs from frozen partition")

    beta = cohort.beta.index_select(1, common_indices)
    age = cohort.age.index_select(0, common_indices)
    probe_rows = standardize_rows(beta)
    standardized_age = standardize_vector(age)
    targets = build_protein_targets(probe_rows, panel.values, standardized_age)
    probe_age = probe_rows.tensor @ standardized_age.tensor
    audit_indices = t.linspace(0, len(probes) - 1, steps=64, dtype=t.int64)
    correlation_error = assert_correlation_identity(beta, probe_rows, audit_indices)
    joined = t.cat(
        (
            beta.index_select(0, audit_indices),
            panel.values,
            age[None, :],
        ),
        dim=0,
    )
    joined_corr = t.corrcoef(joined)
    expected_cross = joined_corr[:64, 64 : 64 + len(panel.display_names)]
    cross_error = float(
        (targets.probe_protein.tensor.index_select(0, audit_indices) - expected_cross)
        .abs()
        .max()
        .item()
    )
    if cross_error > 1.0e-10:
        raise ValueError(f"probe-protein dot-product identity failed: {cross_error}")

    windows_raw = sequence_config.get("window_sizes")
    if not isinstance(windows_raw, list) or windows_raw != [1024, 4096]:
        raise ValueError("protein TSS window sweep differs")
    windows = cast(list[int], windows_raw)
    protein_overlap_audit: dict[str, JsonValue] = {}
    for width in windows:
        intervals = tuple(
            tss_window_interval(
                locus,
                parse_window_size(width),
                chromosome_length=10**9,
            )
            for locus in loci
        )
        train_like = set(split.refit_indices.tolist())
        test = set(split.test_indices.tolist())
        overlaps = tuple(
            (left, right)
            for left in train_like
            for right in test
            if intervals[left].overlaps(intervals[right])
        )
        if overlaps:
            raise ValueError(f"protein refit/test TSS windows overlap at width {width}: {overlaps}")
        protein_overlap_audit[str(width)] = {
            "refit_test_overlaps": 0,
            "reference_strand": "plus",
            "coordinate_system": "GRCh37 one-based gene TSS to zero-based half-open hg19 FASTA",
        }

    arguments.output.mkdir(parents=True, exist_ok=False)
    tensor_path = arguments.output / "targets.safetensors"
    save_safetensors_exclusive(
        tensor_path,
        {
            "probe_protein": targets.probe_protein.tensor,
            "protein_gram": targets.protein_gram.tensor,
            "direct_protein_age": targets.direct_age.tensor,
            "probe_age": probe_age,
            "standardized_proteins": targets.standardized_proteins.tensor,
            "common_cohort_indices": common_indices,
            "protein_train_indices": split.train_indices,
            "protein_validation_indices": split.validation_indices,
            "protein_test_indices": split.test_indices,
        },
    )
    write_canonical_json_exclusive(
        arguments.output / "metadata.json",
        {
            "schema": "methylation-latent.protein-targets.v1",
            "protocol_id": _string(config, "protocol_id"),
            "protocol_sha256": sha256_file(arguments.config),
            "git_commit": git_commit,
            "parent_bundle_sha256": _string(config, "parent_bundle_sha256"),
            "target_file": tensor_path.name,
            "target_sha256": sha256_file(tensor_path),
            "protein_names": list(panel.display_names),
            "gene_symbols": list(panel.gene_symbols),
            "uniprot_accessions": list(panel.uniprot_accessions),
            "protein_sequence_lengths": [len(sequence) for sequence in sequences],
            "protein_sequence_sha256": [
                sha256_ordered_strings((sequence,)) for sequence in sequences
            ],
            "gene_loci": [
                {
                    "gene": locus.gene_symbol,
                    "chromosome": locus.chromosome,
                    "start": locus.start,
                    "end": locus.end,
                    "strand": locus.strand,
                    "tss": locus.transcription_start,
                }
                for locus in loci
            ],
            "common_subject_count": len(common_gsms),
            "common_subject_order_sha256": sha256_ordered_strings(common_gsms),
            "selection_audit": selection_audit,
            "selection_warning": _string(panel_config, "selection_warning"),
            "source_adjustment": _string(panel_config, "values"),
            "coverage_by_public_protein": coverage,
            "protein_split": {name: list(values) for name, values in observed_partitions.items()},
            "tss_overlap_audit": protein_overlap_audit,
            "checks": {
                "probe_correlation_identity_max_error": correlation_error,
                "probe_protein_correlation_identity_max_error": cross_error,
                "protein_gram_symmetric": True,
                "protein_gram_unit_diagonal": True,
                "public_panel_equals_published_hit_union_minus_cxcl9": True,
                "all_selected_uniprot_mappings_unique": True,
                "all_selected_genes_overlap_published_grch37_intervals": True,
            },
        },
    )
    print(
        f"prepared proteins={len(panel.display_names)} subjects={len(common_gsms)} "
        f"probes={len(probes)} target_sha256={sha256_file(tensor_path)}"
    )


if __name__ == "__main__":
    main()
