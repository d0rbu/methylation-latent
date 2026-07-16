# Testing

## Repository gate

```bash
uv run pre-commit run --all-files
```

The gate checks all three lockfiles, Ruff, ty, and pytest. The direct equivalents are:

```bash
uv lock --check
uv lock --project environments/methylprep --check
uv lock --project environments/caduceus --check
uv run ruff check .
uv run ty check src tests scripts
uv run pytest
```

The suite uses strict pytest configuration, Hypothesis, and branch coverage. Coverage fails below
95%; the current full suite has 296 tests and 95.11% coverage.

For a focused test, disable the repository-wide coverage threshold explicitly:

```bash
uv run pytest --no-cov tests/test_data_bundle.py -q
uv run pytest --no-cov tests/test_genome_splits.py -k overlap -q
```

## What the tests prove

Synthetic and property tests cover:

- bounded scalar refinements and semantic tensor wrappers;
- float64 standardization and dot-product/correlation parity;
- target rank and model-dimension ceilings;
- one-based hg19 conversion and exact plus-strand centre-CG extraction;
- target-blind diverse/chromosome splits and brute-force interval-overlap parity;
- nested validation containment and buffer semantics at each window;
- exact GSE87571 metadata/IDAT joins and complete-case detection filtering;
- immutable data, embedding, pair, model, selection, evaluation, and site artifacts;
- separate age and pair objectives, zero weight decay, and deterministic training seeds;
- malformed, partial, reordered, duplicate, non-finite, stale, and extra-file rejection.

## What unit tests do not prove

Real-data gates are adjacent artifacts, not mocked expected passes. Protocol v2 separately records:

- all 1,464 compressed IDAT files for 732 arrays and their exact inventory fingerprint;
- byte hashes for phenotype sources, GPL13534, Chen, Zhou, and hg19;
- a deterministic 12-array methylprep/seSAMe parity audit whose failure selected seSAMe;
- pinned seSAMe container/package identities and both full-cohort preprocessing runs;
- observed cohort/filter counts and complete source/reason exclusion ledgers;
- exhaustive reference and split audits;
- sealed target tensors and the observed correlation-identity error;
- a real-checkpoint Caduceus smoke/throughput audit and immutable embedding shards;
- frozen evaluation-pair indices and target values.

An absent external gate remains absent. Never turn it into a fixture that merely says it passed.

## Long-running checks

Embedding generation and training are restart-safe stages, not CI tests. Run one GPU workload at a
time and retain the PTY/log, process ID, manifest, shard count, and elapsed time. Do not run
profiling benchmarks concurrently: their timing and memory numbers would be meaningless.
