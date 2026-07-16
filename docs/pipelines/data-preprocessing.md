# Data preprocessing

## Public-source audit

[GSE40279](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE40279) contains 656
HumanMethylation450 whole-blood samples. GEO currently lists:

- processed average-beta tables;
- a 5.0 GB compressed signal table with `AVG_Beta`, total intensity, `SignalA`, and `SignalB`;
- a sample key and series matrix;
- `GSE40279_RAW.tar`.

The last name is misleading for this purpose: inspection shows that the tar contains only four
GPL13534 manifest files. It contains no red/green IDAT pairs. The signal table also lacks the
negative/control-probe measurements required by detection-p, noob, and functional normalization.
The minfi guide explicitly notes that GenomeStudio exports without controls support only part of
the preprocessing stack.

Therefore:

- a `processed_geo_audit` artifact may exercise metadata, probe filtering, target math, splits,
  and models;
- it is permanently `audit_only` and cannot populate primary figures or baseline comparisons;
- the primary experiment is blocked until authentic IDATs are obtained from the data generators,
  an archival source, or another source whose identity can be matched exactly to all 656 GEO
  samples.

Do not estimate detection p-values from the low tail of ordinary probes or claim that the supplied
BeadStudio beta values are noob/funnorm output.

## Tool choice

The primary implementation uses
[`methylprep`](https://life-epigenetics-methylprep.readthedocs-hosted.com/) 1.7.1 in its default
seSAMe-compatible mode because it preserves a Python orchestration path while implementing:

- pOOBAH out-of-band detection p-values;
- HM450 quality masking;
- noob background correction;
- nonlinear dye-bias correction;
- Type-I channel-switch inference;
- export of control probes, uncorrected intensities, and per-probe p-values.

This choice is conditional on a parity gate. Before processing the full cohort, a preregistered
set of at least 12 arrays spanning plates and positions is processed independently with the pinned
Bioconductor seSAMe `QCDPB` pipeline. Protocol v1 pins R 4.6, Bioconductor 3.23, seSAMe 1.30.1,
and sesameData 1.30.0; an exact resolved R dependency lock remains a protocol-freeze requirement.
Probe inclusion, pOOBAH calls, and normalized beta values are compared. Protocol v1 fixes the
subset at exactly 12 arrays, chosen deterministically from plate and Sentrix-position metadata
without age, beta, or target values. It requires:

- identical sample-retention decisions;
- identical all-retained-array complete-probe sets;
- at least 99.9% overall and 99.5% per-array pOOBAH pass/fail agreement;
- normalized-beta median absolute difference at most 0.001 and 99.9th-percentile absolute
  difference at most 0.01;
- per-array beta correlation of at least 0.999.

The executable gate is `assert_processor_parity`. A failed audit switches the primary processor to
seSAMe; it does not loosen thresholds or average the two outputs.

The methylprep environment is isolated from the Torch analysis environment because its published
release targets an older Python scientific stack. The exact command, package resolution, standard
output, and checksums are captured in the preprocessing run record.

## Primary QC order

1. Verify two IDATs per sample, Sentrix ID/position, file hashes, readable control probes, platform
   size, and exact linkage through `GSE40279_sample_key.txt.gz` to all 656 GSM records.
2. Parse age, gender, ethnicity, source, and plate from the series matrix; reject duplicate,
   missing, or contradictory fields.
3. Compute pOOBAH at `p < 0.01` from uncorrected out-of-band/control signals.
4. Remove samples for which more than 1% of candidate CpG probes fail.
5. On retained samples, retain only probes that pass pOOBAH in **every** sample. This preserves one
   common sample axis and forbids imputation or pair-specific observation sets.
6. Apply noob and nonlinear dye-bias correction and export float64 beta values.
7. Apply the independent probe exclusions below.
8. Reject non-finite/out-of-range beta values and constant probe rows.

Threshold changes require a new protocol ID. Detection decisions are made before age or target
construction.

## Independent probe exclusions

The exclusion ledger is the union of:

- non-`cg` IDs (`rs`, `ch`, controls);
- chromosomes X and Y;
- methylprep/seSAMe HM450 recommended quality mask;
- Chen et al. (2013) 29,233-probe cross-reactive/non-specific list;
- Zhou et al. (2017)
  [`MASK.general`](https://zwdzwd.github.io/InfiniumAnnotation/manifest.html), which combines
  mapping, 30-base non-uniqueness, extension-base, Type-I color-switch, and common-SNP flags.

The downloaded list bytes, upstream URL, version, SHA-256, and counts by reason are mandatory
metadata. The union is by probe ID and preserves every source/reason; overlapping lists are not
silently deduplicated in the audit ledger.

The v1 bytes are pinned as follows:

- Chen `48639-non-specific-probes-Illumina450k.csv`, recovered from a commit-pinned mirror because
  the paper's original SickKids download endpoint is no longer reliable: SHA-256
  `4e962d36821f6f6fcd8b81cc0558090c028e54fbdb2c039a5712f9b471d9d89e`, exactly 29,233 CpGs.
- Zhou versioned hg19 HM450 table `InfiniumAnnotation/20180808`: SHA-256
  `94aaa52274738bbf8d767e56512b8fc36893c2cabffa202870531950bf522f0e`, exactly 485,577 rows,
  65,574 `MASK_general=TRUE` rows, and 64,987 masked `cg` IDs.

The strict union audit over the GPL13534 autosomal CpG universe retains 405,610 of 470,870 probes
before detection QC. It writes 91,628 source/reason ledger rows because Chen/Zhou overlaps are
preserved; it does not mistake ledger-row count for unique excluded-probe count.

The subsequent exhaustive hg19 audit confirms `CG` at all 405,610 one-based `MAPINFO` coordinates.
The common 64 kb sequence-universe gate retains 403,573 after 15 boundary and 2,022 non-ACGT
reference exclusions. These are deterministic reference exclusions, not detection-QC outcomes.

## Manifest and build

Use GPL13534 v1.1 CSV, SHA-256
`df3d5009d9b5b878507ba330bbcd7d2012455f6d0aa92c2df897f53a28452c33`.
Every retained row must say genome build `37`, have an autosomal chromosome and positive `MAPINFO`,
and contain one `[CG]` marker in `Forward_Sequence`.

The reference is UCSC hg19. No liftOver is part of the primary protocol. An alternate build is a
new experiment and must include explicit chain, source/target build, unique-mapping, strand, and
round-trip audit artifacts.

## Metadata alignment

Sample order is never inferred from lexical filename order. The series matrix maps GSM accession
to subject metadata; the sample key maps subject ID to Sentrix ID/position; the IDAT processor emits
Sentrix identity. Joining these must be one-to-one at every edge, and the final beta columns and age
rows share one stored sample-order fingerprint.

The fully downloaded series matrix must pass its gzip CRC trailer before parsing. Protocol v1 pins
the series-matrix, sample-key, manifest, and final ordered-sample hashes in TOML; a correct sample
count with different bytes or order is still an error.

## Relevant sources

- [GSE40279 accession](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE40279)
- [GPL13534 accession](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL13534)
- [methylprep CLI and processing semantics](https://life-epigenetics-methylprep.readthedocs-hosted.com/en/latest/docs/cli.html)
- [minfi user guide](https://bioconductor.org/packages/release/bioc/vignettes/minfi/inst/doc/minfi.html)
- [seSAMe manual](https://bioconductor.org/packages/release/bioc/manuals/sesame/man/sesame.pdf)
