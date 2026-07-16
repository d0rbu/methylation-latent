# Documentation

These documents are the source of truth for `methylation-latent`.

## Research

- [`research/protocol.md`](research/protocol.md) - preregistered question, model, run order,
  validity conditions, and interpretation limits
- [`research/literature.md`](research/literature.md) - dated prior-work search and closest work

## Pipelines

- [`pipelines/experiment-lifecycle.md`](pipelines/experiment-lifecycle.md) - gated run lifecycle
- [`pipelines/data-preprocessing.md`](pipelines/data-preprocessing.md) - GSE40279 audit,
  raw-IDAT requirement, normalization, exclusions, and metadata
- [`pipelines/splits.md`](pipelines/splits.md) - diverse locus blocks, held-out chromosome, and
  sequence-overlap buffer
- [`pipelines/embeddings.md`](pipelines/embeddings.md) - hg19 windows and frozen Caduceus-PS
  centre-token embeddings
- [`pipelines/evaluation.md`](pipelines/evaluation.md) - baselines, pair classes, distance bins,
  and metrics

## Development

- [`development/correctness.md`](development/correctness.md) - domain and tensor contracts
- [`development/testing.md`](development/testing.md) - test strategy and data-dependent gates

## Operations

- [`operations/results-site.md`](operations/results-site.md) - static site and Cloudflare tunnel

## Reference

- [`reference/architecture.md`](reference/architecture.md) - module boundaries and data flow
- [`reference/artifacts.md`](reference/artifacts.md) - artifact schemas and eligibility
- [`reference/configuration.md`](reference/configuration.md) - tool and experiment configuration
- [`reference/file-reference.md`](reference/file-reference.md) - file-by-file map

## Onboarding

- [`onboarding/getting-started.md`](onboarding/getting-started.md) - local setup
- [`onboarding/workflows.md`](onboarding/workflows.md) - common workflows
- [`onboarding/glossary.md`](onboarding/glossary.md) - project vocabulary
