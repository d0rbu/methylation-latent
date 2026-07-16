# Data preprocessing

## Cohort choice

The original GSE40279 plan could not support a faithful primary run because GEO does not expose
its paired IDATs. The user approved switching to
[GSE87571](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE87571), a whole-blood
HumanMethylation450 cohort with authentic raw arrays and age metadata.

The source audit found:

- 732 distinct Sentrix identities;
- exactly 1,464 gzip-valid IDAT files, one red and one green channel per identity;
- 729 samples with age and gender;
- three raw arrays with no usable phenotype record;
- raw-inventory fingerprint
  `20fccbecf4b7e7b36f3099daf6084d9042fc3d9074bfa4bf0ab154eeca694ae0`.

Every join is one-to-one across GSM accession, supplemental filename, Sentrix identity, IDAT
prefix, processor output, phenotype row, and final tensor column. Lexical filename order is never
used as sample order.

## Why seSAMe is primary

Python `methylprep` 1.7.1 was tested first because it would have simplified orchestration. A
target-blind 12-array subset selected across array identities was independently processed with
methylprep and pinned seSAMe. The preregistered parity thresholds were not met:

| Check | Required | Observed |
|---|---:|---:|
| overall detection agreement | at least 0.999 | 0.978631 |
| minimum per-array detection agreement | at least 0.995 | 0.973857 |
| beta median absolute error | at most 0.001 | 0.002801 |
| beta 99.9th-percentile absolute error | at most 0.01 | 0.074160 |
| minimum per-array beta correlation | at least 0.999 | 0.999329 |
| quality-mask union disagreement | 0 | 653 |

High beta correlation did not rescue failed interchangeability. The audit records
`decision = sesame_primary`; thresholds were not loosened.

The primary processor is therefore R/Bioconductor seSAMe:

- official Bioconductor image pinned at
  `sha256:b10002b39efa30c3779ad839549806ebdbb29b3266f0d2428478b04426e55929`;
- R 4.6.0, Bioconductor 3.23, seSAMe 1.30.1, sesameData 1.30.0;
- complete dependency graph in `environments/sesame/renv.lock`;
- per-array `QCD`, raw pOOBAH p-values, `pOOBAH(..., pval.threshold=0.05)`, and `noob`;
- exclusive float64 beta/detection outputs and quality-mask bytes.

This is less convenient than all-Python preprocessing but is the faithful path. The downstream
pipeline remains Torch/Python.

## Primary QC order

1. Verify exact source hashes, 732 paired raw identities, platform HM450, and phenotype joins.
2. Run seSAMe without reading age or methylation targets.
3. Start from the 403,573-locus common sequence universe defined below.
4. Exclude phenotype-missing arrays.
5. Mark a sample failed when more than 5% of candidate probes have pOOBAH
   `p >= 0.05`; 23 phenotype-eligible samples failed.
6. Across the 706 retained samples, require every retained probe to have
   `p < 0.05` in every sample; 37,946 probes failed.
7. Exclude any probe marked by the union of per-array seSAMe quality masks; 18,999 probes failed.
8. Reject missing, non-finite, out-of-range, or constant beta rows; no additional constant rows
   remained.

This yields 346,628 probes by 706 samples: 374 female and 332 male subjects spanning ages 14–94.
There is no imputation and no pair-specific sample axis. Gender is audited metadata, not a
regression covariate in the frozen target definition.

An auxiliary full-cohort preprocessing at pOOBAH threshold 0.01 produced beta values exactly
identical to the 0.05 run over 353,132,172 values because exported betas are unmasked and the
threshold affects QC decisions, not the noob beta computation. Protocol v2 nevertheless freezes
0.05 for both sample and complete-case probe decisions.

## Probe and reference exclusions

The target-blind static universe applies:

- canonical `cg` identifiers only, dropping `rs`, `ch`, controls, and malformed IDs;
- autosomes 1–22 only;
- GPL13534 hg19 coordinates only;
- seSAMe quality filtering downstream across retained samples;
- Chen et al. cross-reactive/non-specific probes;
- Zhou et al. hg19 `MASK_general`, including mapping and common-SNP risks;
- full 65,536-base plus-strand window availability and A/C/G/T-only reference sequence.

Pinned source hashes:

- GPL13534 v1.1:
  `df3d5009d9b5b878507ba330bbcd7d2012455f6d0aa92c2df897f53a28452c33`;
- Chen list:
  `4e962d36821f6f6fcd8b81cc0558090c028e54fbdb2c039a5712f9b471d9d89e`;
- Zhou 2018-08-08 hg19 table:
  `94aaa52274738bbf8d767e56512b8fc36893c2cabffa202870531950bf522f0e`.

GPL13534 contains 485,577 rows and 470,870 manifest-valid autosomal CpGs. The Chen/Zhou union
leaves 405,610. Every one is checked against UCSC hg19; 15 fail the maximum-window boundary and
2,022 contain a non-ACGT base, leaving 403,573. Overlapping exclusion reasons are preserved in the
ledger rather than silently deduplicated.

## Build and strand

The reference is UCSC hg19/GRCh37:

- compressed FASTA MD5 `806c02398f5ac5da8ffd6da2d1d5d1a9`;
- exact decompressed payload SHA-256
  `92b96d16b307d824f3b6b9dc63feb237a75226f82ca2166d51ecec039bee4449`.

`MAPINFO` is interpreted once as a one-based plus-strand cytosine coordinate. No liftOver is used.
All 405,610 post-mask coordinates were verified to point to `CG` on the reference plus strand.

## Frozen outputs

`scripts/prepare_gse87571.py` creates a new directory exclusively. It writes:

- complete cohort and static exclusion ledgers;
- ordered probe tables and order fingerprints;
- float64 beta, `X`, standardized age, and `rho` tensors;
- target, sequence-feature, split, sensitivity, and summary metadata.

`scripts/seal_primary_data.py` then independently reloads and recomputes the high-value
invariants, verifies all expected files, and creates the 17-file `bundle.json`. Any extra or
modified file invalidates the bundle. The bundle also records the exact clean Git commit of the
sealer; sealing is refused when the worktree contains tracked or untracked changes.

## Sources

- [GSE87571 accession](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE87571)
- [GPL13534 accession](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL13534)
- [seSAMe package and manual](https://bioconductor.org/packages/sesame)
- [Zhou HM450 annotation](https://zwdzwd.github.io/InfiniumAnnotation/manifest.html)
- [Chen et al. cross-reactive probe paper](https://doi.org/10.4161/epi.23470)
