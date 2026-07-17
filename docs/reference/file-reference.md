# File reference

## Top level

| Path | Purpose |
|---|---|
| `configs/protocol-v2.toml` | Frozen scientific choices and immutable external identities |
| `scripts/prepare_gse87571.py` | Build primary cohort, targets, features, and splits |
| `scripts/seal_primary_data.py` | Recompute invariants and seal the 17-file data bundle |
| `scripts/run_experiments.py` | Ordered sequence/pair/age/full/evaluate stages |
| `scripts/run_distance_integration.py` | Post-hoc validation-fitted PSD distance integration |
| `scripts/run_direct_tanh_age.py` | Post-hoc direct scalar tanh age-only baseline |
| `scripts/compile_interim_results_site.py` | Compile completed windows, validation sweeps, and explicitly post-hoc panels into a non-primary site |
| `scripts/compile_results_site.py` | Compile validated evaluation records into static sites |
| `results/` | Compact validated site-data artifacts after experiment completion |
| `site-template/` | Split-specific result-page HTML, JavaScript, and CSS |
| `interim-site-template/` | Mixed-evidence interim page with validation sweeps and post-hoc labeling |
| `interim-site-root-template/` | Interim split-family landing page |
| `site-root-template/` | Root landing page for the two split sites |

## Source package

| Module | Purpose |
|---|---|
| `domain.py` | Phantom scalar types and immutable locus/context records |
| `metadata.py` | GEO phenotype and Sentrix identity parsing |
| `manifest.py` | GPL13534 plus Chen/Zhou parsing |
| `processor_io.py` | Strict seSAMe/methylprep processor output adapters |
| `preprocessing.py` | Detection-p sample/probe filtering |
| `cohort.py` | GSE87571 universe, cohort, ledgers, and probe table |
| `genome.py` | Indexed hg19 extraction and deterministic sequence features |
| `targets.py` | Standardization, semantic targets, correlation/rank contracts |
| `splits.py` | Target-blind splits, buffers, and overlap checks |
| `experiment_data.py` | Nested split and sequence-feature storage |
| `data_bundle.py` | Closed-world primary-data verification |
| `embeddings.py` | Pinned Caduceus centre-token inference |
| `embedding_cache.py` | Restart-safe embedding shards and finalization |
| `batching.py` | Local target-blind genomic batches |
| `model.py` | Bias-free latent metric and age vector |
| `training.py` | Age-only/full tuning, validation, and refit |
| `evaluation.py` | Pair sampling, distance reference, metrics, and projection |
| `evaluation_cache.py` | Immutable pair-index/target artifacts |
| `distance_integration.py` | PSD distance kernels and exact simplex-constrained mixtures |
| `artifacts.py` | Canonical JSON and hashing |
| `storage.py` | Strict safetensors persistence |
| `site.py` | Site-data validation and static generation |
| `config.py` / `cli.py` | Strict protocol and public commands |

## Isolated environments

| Path | Purpose |
|---|---|
| `environments/sesame/` | Pinned R/Bioconductor primary preprocessor |
| `environments/methylprep/` | Python 3.10 parity-only environment |
| `environments/caduceus/` | Python 3.11 native model environment and embedding entry point |

## Tests

The tests mirror the boundaries above. High-value additions include
`test_cohort_preparation.py`, `test_data_bundle.py`, `test_embedding_cache.py`,
`test_experiment_data.py`, and the pair-cache/failure-contract cases in the existing evaluation
and ingestion suites.

## CI

`.github/workflows/ci.yml` runs the same locked Ruff, ty, and pytest gates used locally. GPU and
raw-IDAT integration stages are external artifacts and are not replaced by CI mocks.
