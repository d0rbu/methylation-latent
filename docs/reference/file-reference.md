# File reference

## Top level

| Path | Purpose |
|---|---|
| `configs/protocol-v2.toml` | Frozen scientific choices and immutable external identities |
| `configs/protein-extension-v1.toml` | Separate frozen post-hoc protein protocol, source hashes, protein identities, split, representations, and objective |
| `configs/dual-probe-latent-v1.toml` | Frozen post-hoc alpha/schedule sweep and train-only dual-latent contract |
| `scripts/prepare_gse87571.py` | Build primary cohort, targets, features, and splits |
| `scripts/seal_primary_data.py` | Recompute invariants and seal the 17-file data bundle |
| `scripts/run_experiments.py` | Ordered sequence/pair/age/full/evaluate stages |
| `scripts/run_distance_integration.py` | Post-hoc validation-fitted PSD distance integration |
| `scripts/run_direct_tanh_age.py` | Post-hoc direct scalar tanh age-only baseline |
| `scripts/run_prediction_scatter.py` | Display-only predicted-versus-empirical correlation samples |
| `scripts/analyze_age_scatter_clusters.py` | Target-aware post-hoc k=2 audit of age-scatter lobes and phenotype/sequence associations |
| `scripts/run_latent_tsne.py` | Deterministic exact-Torch t-SNE of context-balanced validation latent spaces |
| `scripts/run_latent_umap.py` | Deterministic exact-Torch UMAP from hyperspherical validation latents onto the unit two-sphere plus the learned age direction |
| `scripts/compile_interim_results_site.py` | Compile completed windows, validation sweeps, and explicitly post-hoc panels into a non-primary site |
| `scripts/prepare_protein_extension.py` | Audit the selected public protein panel and build common-subject probe/protein targets |
| `scripts/embed_protein_amino_acids.py` | Pinned reviewed-UniProt to ESM-2 protein embedding cache without truncation |
| `scripts/run_protein_extension.py` | Fit free, TSS, amino-acid, and combined protein anchors in frozen probe geometries |
| `scripts/compile_protein_extension_report.py` | Add the immutable protein experiment page to an existing compiled interim site |
| `scripts/run_dual_probe_latent.py` | Tune, select, refit, and evaluate the dual-probe latent experiment without test-driven alpha selection |
| `scripts/compile_dual_probe_report.py` | Add the selected-only dual-probe results page to an existing compiled interim site |
| `scripts/compile_results_site.py` | Compile validated evaluation records into static sites |
| `results/` | Compact validated site-data artifacts after experiment completion |
| `site-template/` | Split-specific result-page HTML, JavaScript, and CSS |
| `interim-site-template/` | Mixed-evidence interim page with validation sweeps and post-hoc labeling |
| `interim-site-root-template/` | Interim split-family landing page |
| `protein-site-template/` | Post-hoc protein report page, scatter plots, tables, and caveat panels |
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
| `direct_age.py` | Direct scalar tanh age baseline with no latent dimension or lambda |
| `scatter.py` | Target-blind display sampling and bounded correlation scatter records |
| `age_clusters.py` | Deterministic k=2, association effect sizes, label agreement, and sex-stratified age correlations |
| `tsne.py` | Deterministic exact Torch t-SNE and balanced metadata sampling |
| `umap.py` | Exact Torch fuzzy-neighborhood UMAP with Euclidean or intrinsic spherical-geodesic inputs, planar or unit-two-sphere outputs, spectral initialization, distortion diagnostics, and full cross-entropy objectives |
| `protein_extension.py` | Common-axis protein targets, target-blind protein split, hyperspherical maps, loss blocks, TSS intervals, and no-truncation chunking |
| `dual_probe.py` | Learned probe table, OLS initialization audit, alpha schedules, routed catch loss, and dual-latent training |
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

`environments/caduceus/embed_protein_tss.py` is the protein TSS centre-token producer; it uses the
same pinned Caduceus checkpoint but does not impose a centre-CpG motif on gene TSS windows.

## Tests

The tests mirror the boundaries above. High-value additions include
`test_cohort_preparation.py`, `test_data_bundle.py`, `test_embedding_cache.py`,
`test_experiment_data.py`, and the pair-cache/failure-contract cases in the existing evaluation
and ingestion suites.

## CI

`.github/workflows/ci.yml` runs the same locked Ruff, ty, and pytest gates used locally. GPU and
raw-IDAT integration stages are external artifacts and are not replaced by CI mocks.
