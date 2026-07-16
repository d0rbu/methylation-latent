# Testing

## Default Suite

```bash
uv run pytest
```

The default suite includes unit tests, property tests, and coverage.

## Pre-Commit

Install hooks once per clone:

```bash
uv run pre-commit install
```

Run the full configured gate manually:

```bash
uv run pre-commit run --all-files
```

The local hooks run:

- `uv lock --check`
- `uv lock --project environments/methylprep --check`
- `uv lock --project environments/caduceus --check`
- `uv run ruff check .`
- `uv run ty check`
- `uv run pytest`

## Focused Runs

```bash
uv run pytest tests/test_domain_targets.py
uv run pytest tests/test_genome_splits.py -k overlap
uv run pytest tests/test_metadata_manifest_preprocessing.py -k manifest
uv run pytest -m property
```

## Coverage

Coverage is configured in `pyproject.toml` and currently fails below 95%.

Use coverage as a guardrail, not a substitute for meaningful assertions. The high-value tests in
this repository check scientific invariants: target/dot-product parity, one-based hg19 coordinate
conversion, every primary window's train/test non-overlap, target-blind split construction, exact
sample joins, immutable run order, and rejection of malformed artifacts.

## Data-dependent gates

Unit tests use synthetic fixtures and do not imply that external inputs passed. Before any primary
run, separately record:

- full gzip CRC and byte hashes for the series matrix, sample key, and manifest;
- strict Chen/Zhou source-list schema, hash, and count audits;
- all 656 sample joins and exact paired-IDAT inventory;
- methylprep/seSAMe parity on the preregistered arrays;
- exhaustive hg19 plus-strand window extraction;
- finalized split overlap audits for every window;
- fp16/float32 Caduceus precision drift.

The real-checkpoint Caduceus smoke is a separate GPU integration gate documented in
[`../pipelines/embeddings.md`](../pipelines/embeddings.md); it is deliberately not a CPU CI test.

An absent data-dependent gate remains absent. Never encode it as an expected pass in a unit test.
