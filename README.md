# methylation-latent

Can DNA sequence predict a CpG locus's cross-person co-methylation structure and
association with age?

This repository tests that question with a deliberately leak-resistant pipeline. It maps
CpG-centred hg19 sequence embeddings into a learned cosine geometry, then asks whether
that geometry reproduces empirical probe-probe correlations and probe-age correlations in
the GSE40279 whole-blood cohort.

## Status

The repository contains the correctness scaffold and experiment protocol. It does **not**
yet contain a scientific result. In particular, public GSE40279 provides processed beta and
signal tables but not the IDAT/control-probe data required to reproduce detection-p filtering
and noob/funnorm. Runs based only on those processed files are audit-only and cannot produce
headline metrics.

Protocol v1 also remains deliberately `draft`: the optimization schedule and validation-based
checkpoint-selection rule and the exact seSAMe R dependency lock must be added before stage 1
begins. The tested fixed-step trainer is a primitive, not permission to improvise those choices
during a run.

## Validity contract

A result is eligible for comparison only when all of these are recorded and pass:

- the methylation artifact came from raw IDATs with detection-p sample/probe QC,
  chemistry normalization, and pinned Chen/Zhou probe exclusions;
- GPL13534 coordinates and the reference are both hg19/GRCh37;
- every extracted plus-strand window has `CG` at the declared centre cytosine;
- standardized probe rows and age have zero mean and unit L2 norm;
- correlation targets are dot products of those standardized vectors;
- split construction never reads methylation targets;
- no train/test sequence windows overlap at any configured window size;
- seen-by-held-out, held-out-by-held-out, and age metrics remain separate;
- pair metrics are broken out by same-chromosome distance and a separate trans bin.

The executable contracts live in the package and tests. The detailed protocol is in
[`docs/research/protocol.md`](docs/research/protocol.md).

## Planned run order

1. CpG density plus GC content to held-out probe-age correlation.
2. Frozen Caduceus-PS embedding plus age head, without the Gram loss.
3. Full shared latent metric, sweeping sequence window first and latent dimension and age
   loss weight second.

Baseline results are immutable run artifacts. The full model cannot be registered unless
matching baseline artifacts already exist for the same data and split fingerprints.

## Quickstart

```bash
uv sync
uv run pre-commit install
uv run pre-commit run --all-files
```

## Documentation

| Topic | Source of truth |
|---|---|
| Research question and preregistration | [`docs/research/protocol.md`](docs/research/protocol.md) |
| Prior-work search | [`docs/research/literature.md`](docs/research/literature.md) |
| Data and normalization | [`docs/pipelines/data-preprocessing.md`](docs/pipelines/data-preprocessing.md) |
| Genomic splits and leakage | [`docs/pipelines/splits.md`](docs/pipelines/splits.md) |
| Sequence and Caduceus embeddings | [`docs/pipelines/embeddings.md`](docs/pipelines/embeddings.md) |
| Metrics and baselines | [`docs/pipelines/evaluation.md`](docs/pipelines/evaluation.md) |
| Artifact schemas | [`docs/reference/artifacts.md`](docs/reference/artifacts.md) |
| Results site and Cloudflare tunnel | [`docs/operations/results-site.md`](docs/operations/results-site.md) |

## License

MIT. See [`LICENSE`](LICENSE).
