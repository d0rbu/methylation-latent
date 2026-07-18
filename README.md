# methylation-latent

Can DNA sequence predict a CpG locus's cross-person co-methylation structure and
association with age?

This repository tests that question with a leak-resistant locus-level experiment. Frozen
Caduceus-PS embeddings of CpG-centred hg19 sequence are mapped through a bias-free linear metric.
Cosine similarity in the learned space is trained against empirical probe-probe correlations, and
an independent latent age vector is trained against each probe's age correlation.

## Status

Protocol `gse87571-hg19-caduceus-ps-v2` is frozen. The primary cohort is
[GSE87571](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE87571), selected after the
original GSE40279 plan was blocked by the absence of authentic paired IDATs. GSE87571 supplies 732
paired HumanMethylation450 arrays; raw-IDAT preprocessing retains 706 subjects and 346,628
autosomal CpGs.

The primary data bundle is sealed and its first baseline is complete. The Caduceus embedding and
model sweeps are run artifacts outside Git and are being generated in the fixed order below. This
README will report results only after the final evaluation artifacts pass their identity checks.

## Audited data facts

- preprocessing: seSAMe 1.30.1, `QCD-pOOBAH@0.05-noob`, pinned container digest;
- source universe: 403,573 loci with valid full 65,536-base plus-strand hg19 windows;
- retained cohort: 706 subjects and 346,628 probes after phenotype, sample, quality, and
  complete-case detection filtering;
- target construction: float64 centering and L2 normalization;
- maximum observed `corr(i,j) - X[i] @ X[j]` error: `2.7755575615628914e-15`;
- target rank ceiling: `706 - 1 = 705`; useful learned-metric ceiling: Caduceus width 256;
- primary diverse split: 2,250 test probes and 900 buffer exclusions;
- strict split: all 20,938 chromosome-7 probes held out;
- minimum same-chromosome train/test cytosine separation in the diverse split: 65,605 bases,
  exceeding the 65,536-base maximum window.

The exact source, data, target, split, and model fingerprints are pinned in
[`configs/protocol-v2.toml`](configs/protocol-v2.toml) and adjacent immutable artifact metadata.

## Validity contract

A result is eligible only when all of these pass:

- paired raw IDAT identity, sample metadata, and byte-pinned source audits;
- detection-p sample/probe QC, seSAMe quality masking and noob normalization, and pinned
  Chen/Zhou exclusions;
- one hg19/GRCh37 coordinate system and plus-strand sequence convention throughout;
- exact centred `CG` and A/C/G/T-only sequence windows;
- zero-mean, unit-L2 probe rows and age, with dot-product correlation parity;
- target-blind split construction and explicit non-overlap checks at every window size;
- validation nested inside the buffered primary training set;
- immutable target, pair, embedding, baseline, selection, and evaluation artifacts;
- separate seen-by-held-out, held-out-by-held-out, and age reporting;
- pair results split by genomic distance with a training-only distance reference.

There are no graceful scientific fallbacks. A missing or mismatched prerequisite is an error.

## Run order

1. CpG density plus GC content to held-out probe-age correlation.
2. Frozen Caduceus-PS embedding plus age head, with no Gram loss.
3. Full shared latent metric, sweeping 1,024, 4,096, 16,384, and 65,536 bases; latent dimension
   and age-loss weight are secondary validation-selected axes.
4. One sealed evaluation pass on both test splits.
5. Static result-site compilation from validated evaluation artifacts.

Baseline results are immutable. The full-model stage refuses to run unless the matching sequence
and age-only baselines already exist.

## Quickstart

```bash
uv sync
uv run pre-commit install
uv run pre-commit run --all-files
uv run methylation-latent check-config --config configs/protocol-v2.toml
```

The raw data and multi-gigabyte artifacts are intentionally outside Git. Reproduction commands and
their required inputs are documented in
[`docs/onboarding/getting-started.md`](docs/onboarding/getting-started.md).
Artifact-producing commands require a committed, clean worktree and record the exact producer Git
commit; they refuse to run when tracked or untracked files differ.

## Documentation

| Topic | Source of truth |
|---|---|
| Frozen research protocol | [`docs/research/protocol.md`](docs/research/protocol.md) |
| Dated prior-work search | [`docs/research/literature.md`](docs/research/literature.md) |
| Raw-IDAT preprocessing | [`docs/pipelines/data-preprocessing.md`](docs/pipelines/data-preprocessing.md) |
| Genomic splits and leakage | [`docs/pipelines/splits.md`](docs/pipelines/splits.md) |
| Caduceus embedding cache | [`docs/pipelines/embeddings.md`](docs/pipelines/embeddings.md) |
| Training lifecycle | [`docs/pipelines/experiment-lifecycle.md`](docs/pipelines/experiment-lifecycle.md) |
| Evaluation and baselines | [`docs/pipelines/evaluation.md`](docs/pipelines/evaluation.md) |
| Results site and tunnel | [`docs/operations/results-site.md`](docs/operations/results-site.md) |
| Post-hoc protein extension | [`docs/research/exploratory-protein-extension.md`](docs/research/exploratory-protein-extension.md) |

## License

MIT. See [`LICENSE`](LICENSE).
