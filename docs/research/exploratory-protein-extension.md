# Exploratory protein extension

Status: **post-hoc, hypothesis-generating; not part of frozen protocol v2**

This experiment places circulating-protein measurements into the already fitted probe latent
geometries. It does not reopen or modify the primary methylation protocol. The exact new choices
are frozen separately in `configs/protein-extension-v1.toml` before protein-map fitting.

## Source-data limitations

The public GSE87571 workbook contains 64 protein columns. A byte-pinned audit against Ahsan et al.
Supplementary Tables S3 and S7 shows that these columns are exactly the union of proteins with a
reported significant GWAS or EWAS result, except CXCL9. This is an outcome-selected disease
biomarker set, not an unbiased proteome panel. Results can generate hypotheses within this panel
but cannot estimate proteome-wide generalization.

The released protein values are rank-normalized residuals adjusted by the source study for age,
sex, BMI, collection year, and assay plate. Consequently:

- direct protein-age correlation in this file is expected to be near zero;
- cosine proximity between a protein vector and the parent age vector is a model-imputed edge;
- that cosine is not identified as an observed, unadjusted protein-aging correlation;
- the report separately compares the cosine with direct residualized protein-age correlation and
  with concordance between the protein's CpG-correlation profile and the CpG-age profile.

Of the 64 public proteins, the protocol retains the 52 with at least 656 observations within the
sealed 706-subject methylation cohort. Their complete intersection contains exactly 651 subjects.
All probe-protein, protein-protein, probe-age, and direct protein-age targets are rebuilt on this
one common subject axis. They are never joined from separately standardized cohorts.

## Fixed representations

The frozen parent probe metric `W` and age direction remain unchanged for each of the two completed
windows and two genomic split families. Five protein representations are compared:

1. one free vector per protein, a transductive upper bound with no unseen-protein claim;
2. Caduceus at the GRCh37 gene TSS, passed through the frozen probe `W`;
3. Caduceus TSS embedding through a new shared bias-free linear map;
4. reviewed canonical UniProt amino-acid sequence embedded by pinned ESM-2 8M, then a shared
   bias-free linear map;
5. row-normalized TSS and amino-acid embeddings concatenated before a shared bias-free linear map.

TSS windows use Ensembl GRCh37 gene coordinates and the hg19 plus reference strand. The gene
strand chooses the TSS coordinate but never reverse-complements the extracted input. Caduceus uses
the single centre token. Protein sequences use unique reviewed UniProt accessions. ESM-2 payloads
are capped at 1,022 residues per chunk; longer sequences are partitioned into non-overlapping
chunks, every residue is embedded exactly once, and residue embeddings are averaged with exact
residue-count weighting. Silent truncation is forbidden.

Ensembl GRCh37 labels 51 selected genes `protein_coding`. Its MMP12 record is the single pinned
exception (`processed_transcript`); that record still has the exact MMP12 HGNC display name, a
canonical ENST transcript, a reviewed MMP12 UniProt sequence, and an interval overlapping the
source paper's Build-37 MMP12 interval. Any additional biotype exception is an error.

The GRCh37 Ensembl symbol endpoint places PECAM1 on alternate contig `HG183_PATCH`; it is the only
selected non-primary placement. The byte-pinned paper supplement gives the primary placement
`chr17:62396775-62491136`. The minus-strand TSS, `62491136`, agrees exactly with the patch record.
The protocol therefore pins that primary chr17 interval as the sole placement exception. All
other loci must resolve directly to chromosomes 1–22, X, or Y.

## Target-blind protein split and objective

SHA-256 ordering of `seed:HGNC-symbol` fixes 36 training, 8 validation, and 8 test proteins without
reading any methylation or protein target. Sequence maps train only on training proteins and
optimization CpGs. Checkpoint selection uses validation proteins and nested-validation CpGs. The
selected number of steps is refit from the same deterministic initialization on training plus
validation proteins and complete primary-train CpGs. Test proteins never enter map fitting.

For standardized common-subject protein rows `P`, the cached targets are

```text
R = X_common @ P.T
G = P @ P.T
q = P @ age_common
```

For unit protein latents `z`, training minimizes

```text
mean((probe_latent @ z.T - R)^2)
  + lambda_protein_pairs * mean_offdiag((z @ z.T - G)^2)
```

The two blocks are averaged separately and `lambda_protein_pairs = 1`. The Gram diagonal is
excluded because it is fixed to one by normalization.

## Evaluation populations

The report keeps these claims separate:

- held-out CpG × seen protein;
- seen CpG × held-out protein;
- held-out CpG × held-out protein, the strongest inductive claim;
- seen × held-out and held-out × held-out protein-protein edges.

CpG-protein metrics are additionally split into cis and trans. Any pair whose CpG input window
overlaps the protein TSS input window is counted and excluded from the main metric. Protein refit
and test TSS windows are asserted not to overlap at 1,024 or 4,096 bases.

The free-vector model sees all protein identities and targets during fitting. Its numbers are an
upper bound only and are marked invalid for protein-generalization claims.

## Reproduction order

Artifact producers require a clean committed worktree:

```bash
uv run python scripts/prepare_protein_extension.py ...
uv run --project environments/caduceus python \
  environments/caduceus/embed_protein_tss.py ... --window-size 1024
uv run --project environments/caduceus python \
  environments/caduceus/embed_protein_tss.py ... --window-size 4096
uv run python scripts/embed_protein_amino_acids.py ...
uv run python scripts/run_protein_extension.py ...
uv run python scripts/compile_protein_extension_report.py ...
```

Every output directory is exclusive. A failed partial directory is quarantined rather than
accepted or overwritten.
