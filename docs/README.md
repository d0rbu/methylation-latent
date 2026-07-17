# Documentation

These documents are the source of truth for `methylation-latent`. Protocol v2 uses raw
GSE87571 IDATs; references to the unavailable GSE40279 primary run are historical only.

## Research

- [`research/protocol.md`](research/protocol.md) — frozen question, cohort, model, grids, run
  order, validity conditions, and interpretation limits
- [`research/literature.md`](research/literature.md) — dated search for direct and neighboring
  prior work
- [`research/exploratory-distance-integration.md`](research/exploratory-distance-integration.md)
  — post-hoc PSD distance-plus-sequence kernel analysis and leakage boundary

## Pipelines

- [`pipelines/experiment-lifecycle.md`](pipelines/experiment-lifecycle.md) — immutable stage graph
- [`pipelines/data-preprocessing.md`](pipelines/data-preprocessing.md) — GSE87571 raw-IDAT
  provenance, seSAMe preprocessing, exclusions, cohort construction, and observed audit counts
- [`pipelines/splits.md`](pipelines/splits.md) — diverse blocks, chromosome 7, nested validation,
  buffers, and overlap proofs
- [`pipelines/embeddings.md`](pipelines/embeddings.md) — hg19 windows and frozen Caduceus-PS
  centre-token cache
- [`pipelines/evaluation.md`](pipelines/evaluation.md) — immutable pair populations, baselines,
  distance strata, and metrics

## Development

- [`development/correctness.md`](development/correctness.md) — scalar, tensor, coordinate,
  target, rank, and artifact contracts
- [`development/testing.md`](development/testing.md) — unit/property suite and real-data gates

## Operations

- [`operations/results-site.md`](operations/results-site.md) — static compilation, local serving,
  and Cloudflare tunnel

## Reference

- [`reference/architecture.md`](reference/architecture.md) — module boundaries and data flow
- [`reference/artifacts.md`](reference/artifacts.md) — immutable artifact schemas and identities
- [`reference/configuration.md`](reference/configuration.md) — tool and protocol configuration
- [`reference/file-reference.md`](reference/file-reference.md) — file-by-file map

## Onboarding

- [`onboarding/getting-started.md`](onboarding/getting-started.md) — environments and exact stage
  commands
- [`onboarding/workflows.md`](onboarding/workflows.md) — common safe workflows
- [`onboarding/glossary.md`](onboarding/glossary.md) — project vocabulary
