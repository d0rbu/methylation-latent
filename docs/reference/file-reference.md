# File Reference

## Top Level

| File | Purpose |
|---|---|
| `README.md` | Project summary, quickstart, and doc links |
| `AGENTS.md` | Agent entry point and repo conventions |
| `CLAUDE.md` | Claude-specific pointer to agent conventions |
| `pyproject.toml` | Package metadata and tool configuration |
| `uv.lock` | Locked dependency graph |
| `.pre-commit-config.yaml` | Local commit hooks for lockfile, lint, type, and test checks |
| `.python-version` | Python version for local tooling |
| `.gitignore` | Local artifacts excluded from git |
| `LICENSE` | MIT license |
| `configs/protocol-v1.toml` | Strict draft protocol and immutable source/model identities |
| `results/registry.json` | Truthful audit-only site registry with no demo metrics |
| `site-template/` | Static results-site HTML, JavaScript, and CSS |

## Source package

| Module | Purpose |
|---|---|
| `domain.py` | Phantom scalar types and immutable locus/sample-independent domain records |
| `metadata.py` | Exact GEO metadata/sample-key joins and paired-IDAT audits |
| `manifest.py` | GPL13534 and byte-pinned Chen/Zhou parsing and exclusion union |
| `preprocessing.py` | Sample-first detection-p and complete-probe filtering |
| `genome.py` | Indexed hg19 extraction, exhaustive window audit, and sequence features |
| `targets.py` | Standardization, semantic target wrappers, dot-product identity, rank ceiling |
| `splits.py` | Diverse/chromosome splits, maximum buffer, and overlap audits |
| `batching.py` | Target-blind local genomic batches |
| `embeddings.py` | Pinned Caduceus load and centre-token embedding cache construction |
| `hashing.py` | Python 3.11-compatible file hashing shared with isolated model loading |
| `model.py` | Bias-free latent metric, independent age vector, and separate loss terms |
| `training.py` | Fixed-step cached-embedding age-only/full training |
| `evaluation.py` | Immutable pair populations, distance reference, metrics, and projection |
| `storage.py` | Exclusive safetensors persistence |
| `artifacts.py` | Canonical metadata, hashes, eligibility, and baseline run order |
| `site.py` | Strict result schema and static-site generator |
| `cli.py` | Public input/list/reference audits, protocol check, and site build commands |

## Tests

| File | Purpose |
|---|---|
| `tests/test_domain_targets.py` | Scalar/semantic types and correlation/rank contracts |
| `tests/test_genome_splits.py` | FASTA coordinate and exhaustive split-overlap contracts |
| `tests/test_metadata_manifest_preprocessing.py` | GEO, IDAT, manifest, exclusion, and detection-QC contracts |
| `tests/test_model_training.py` | Batching, cosine model, objectives, optimizer, and training |
| `tests/test_evaluation.py` | Pair populations, distance baseline, metrics, and projection |
| `tests/test_artifacts_storage.py` | Immutable metadata, eligibility graph, and safetensors |
| `tests/test_config_site_embeddings_baselines.py` | Frozen config, Caduceus IO, baselines, CLI, and site schema |

## Isolated environments

| Path | Purpose |
|---|---|
| `environments/methylprep/` | Python 3.10 lock for methylprep 1.7.1 raw-IDAT preprocessing |
| `environments/caduceus/` | Python 3.11 native-stack lock and real-checkpoint shared-code smoke |

## Docs

| Path | Purpose |
|---|---|
| `docs/README.md` | Documentation index |
| `docs/onboarding/` | Setup and day-to-day workflows |
| `docs/development/` | Correctness and testing guidance |
| `docs/pipelines/` | Research workflow templates |
| `docs/reference/` | Architecture, configuration, and file map |

## CI

| File | Purpose |
|---|---|
| `.github/workflows/ci.yml` | Runs `ruff`, `ty`, and `pytest` on pushes and pull requests |
