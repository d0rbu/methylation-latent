# AGENTS.md - methylation-latent

This is a correctness-first research repository. A plausible result produced by a leaked or
misregistered pipeline is worse than a crash.

## Read before changing code

- [`README.md`](README.md)
- [`docs/README.md`](docs/README.md)
- [`docs/research/protocol.md`](docs/research/protocol.md)
- [`docs/development/correctness.md`](docs/development/correctness.md)
- [`docs/pipelines/experiment-lifecycle.md`](docs/pipelines/experiment-lifecycle.md)
- [`docs/pipelines/data-preprocessing.md`](docs/pipelines/data-preprocessing.md)
- [`docs/pipelines/splits.md`](docs/pipelines/splits.md)
- [`docs/reference/architecture.md`](docs/reference/architecture.md)
- [`docs/reference/artifacts.md`](docs/reference/artifacts.md)

## Non-negotiable conventions

- Use `uv` and invoke project tools with `uv run`.
- Use PyTorch as `import torch as t`. Do not import NumPy in project code or tests.
- Use `jaxtyping` for tensor shapes/dtypes, `beartype` at untrusted boundaries, and
  `phantom-types` for scalar domain invariants.
- Represent semantic tensor invariants with validated wrappers. A raw tensor is not a
  `UnitNormRows` or a `CorrelationMatrix` merely because a comment says so.
- Do not catch an invariant failure and continue. There are no substitute data sources,
  guessed coordinate conventions, silent clipping, missing-value imputation, or graceful
  preprocessing fallbacks.
- Keep configuration groups atomic. If several fields are jointly optional, make a dataclass
  containing required fields and make that dataclass optional.
- Keep imports at the top of files. Prefer `functools` and `itertools` when they make control
  flow shorter without obscuring invariants.
- Generated data, embeddings, checkpoints, metrics, and sites stay outside Git.
- Update docs and [`docs/reference/file-reference.md`](docs/reference/file-reference.md) when
  codepaths or contracts change.

## Scientific guardrails

- `hg19`/GRCh37 is the only accepted build. GPL13534 `MAPINFO` is a one-based coordinate of
  the plus-strand CpG cytosine; extraction converts it once to a zero-based half-open interval.
- Probe strand annotations are retained for audits but never reverse-complement the reference.
- Raw-IDAT preprocessing and processed-GEO audit artifacts have distinct types and eligibility.
- Split generation may use only probe ID, chromosome, coordinate, and genomic context. It may
  not load beta values, `X`, `Y`, `rho`, or embeddings.
- The maximum configured sequence window defines the common train/test buffer for all window
  experiments. Every individual window size is also checked explicitly.
- Never mix seen-by-held-out and held-out-by-held-out pairs. Never pool cis distance bins with
  trans pairs.
- A full-model run must point to completed baseline artifacts for the identical data, sequence,
  and split fingerprints.
- Caduceus embeddings are frozen, centre-token representations and are computed once per
  window size. Training must never invoke the sequence model.
- Optimizers for the latent metric use zero weight decay.
- Only one profiling benchmark may run at a time, and benchmarks are end-to-end with a recorded
  baseline.

## Required checks

```bash
uv run pre-commit run --all-files
```

Data-dependent integration gates are documented in
[`docs/pipelines/experiment-lifecycle.md`](docs/pipelines/experiment-lifecycle.md). A unit-test
pass does not make an unrun data gate pass.
