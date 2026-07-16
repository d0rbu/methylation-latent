# Configuration

Tool configuration lives in `pyproject.toml`. Scientific choices and external identities live in
the strict, versioned `configs/protocol-v1.toml`; unknown fields are errors.

## Package Management

Use `uv`.

```bash
uv sync
uv add torch
uv add --dev pytest
```

NumPy is deliberately not a project dependency. Project code and tests use PyTorch as
`import torch as t`.

## Linting

`ruff` is configured for Python 3.13 with common correctness-oriented rule families:

- `E`, `F`, `W`
- `I`
- `UP`
- `B`
- `C4`
- `SIM`
- `RET`

Run:

```bash
uv run ruff check .
```

## Pre-Commit

`pre-commit` uses local hooks that invoke the locked `uv` environment.

Install:

```bash
uv run pre-commit install
```

Run:

```bash
uv run pre-commit run --all-files
```

Configured hooks:

- `uv lock --check`
- `uv run ruff check .`
- `uv run ty check`
- `uv run pytest`

The two isolated legacy/native runtimes are locked independently:

```bash
uv lock --project environments/methylprep --check
uv lock --project environments/caduceus --check
```

## Type Checking

`ty` is configured for Python 3.13.

Run:

```bash
uv run ty check
```

## Testing

`pytest` collects from `tests/`, runs with strict config and strict markers, and reports branch
coverage for `methylation_latent`. The suite fails below 95%.

Run:

```bash
uv run pytest
```

Markers:

- `property`: property-based tests powered by Hypothesis
- `slow`: useful but expensive tests excluded from ad hoc focused runs

## Frozen protocol

`configs/protocol-v1.toml` pins:

- GSE40279/GPL13534/sample-key bytes and the ordered-sample fingerprint;
- hg19 reference archive identity;
- exhaustive hg19 coordinate/window counts and eligible probe-order fingerprint;
- detection/sample QC and the conditional methylprep processor;
- exact Chen and Zhou list URLs and hashes;
- Caduceus repository, revision, checkpoint hash, and width;
- window, split, latent-dimension, lambda, pair-cap, and seed grids.

The committed status is `draft` because authentic IDAT provenance and the seSAMe parity gate are
not available, and because training/batching/checkpoint-selection fields remain to be reviewed and
added. The exact resolved R dependency lock for the pinned seSAMe reference also remains to be
generated on an R 4.6 host. Setting the protocol to `frozen` is a scientific action, not a way to
bypass any blocker.
