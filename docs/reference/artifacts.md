# Artifact schemas

Artifacts are immutable payloads with canonical JSON metadata and SHA-256 identities. Multi-file
directories use a manifest or atomic final-directory rename.

## Primary data

| Schema | Required payload/contract |
|---|---|
| `gse87571-static-universe.v1` | pinned manifest/masks/reference and exhaustive window counts |
| `gse87571-prepared-cohort.v1` | float64 beta matrix, ordered subjects/probes, QC audit |
| `target-geometry-audit.v1` | `X`, age, `rho`, order hashes, correlation error, rank ceilings |
| `sequence-features.v1` | two float64 features for every probe and window |
| `nested-genomic-split.v1` | primary plus nested validation indices and overlap audits |
| `primary-data-summary.v1` | compact observed counts and payload hashes |
| `primary-data-bundle.v2` | producer Git commit plus exhaustive path, size, and hash for all 17 files |

The bundle defines a closed world: missing, extra, modified, or reordered files are errors.

## Embeddings

| Schema | Contract |
|---|---|
| `caduceus-embedding-shard.v2` | producer Git commit, immutable probe slice, `[count,256]` fp16 tensor, window/order/hash |
| `caduceus-embedding-cache.v2` | one producer commit, exact gap-free shard coverage, finalized aggregate identity |

A finalized cache cannot accept another shard.

## Experiment artifacts

| Schema | Contract |
|---|---|
| `sequence-age-baseline.v2` | producer commit, complete-train coefficients, held-out age metrics |
| `evaluation-pair-cache.v2` | producer commit, five deterministic pair sets, exact targets, distance reference |
| `latent-tuning-run.v2` | producer commit, one configuration, validation history, selected step, checkpoint hash |
| `hyperparameter-selection.v2` | producer commit, candidate hashes, frozen tie-break, selected configuration/step |
| `final-refit.v2` | producer commit and selected-step refit on complete primary training |
| `held-out-evaluation.v2` | producer commit, separate age/pair metrics, optional 16-kb projection |
| `site-data.v1` | complete split-specific panels and all contributing artifact IDs |
| `interim-site-data.v5` | completed-window primary panels plus nested-validation sweeps, probe-annotation-colored age scatters, target-aware age-lobe diagnostics, direct tanh age, distance integration, and validation-only interactive two-sphere UMAP with age direction and distortion metrics; cannot substitute for `site-data.v1` |
| `exploratory-distance-integration.v2` | post-hoc validation-fitted PSD distance/sequence mixture with a deterministic single-thread CPU runtime; never primary-validated |
| `exploratory-direct-tanh-age.v1` | post-hoc direct `tanh(v·embedding+b)` age baseline with no latent dimension or lambda; never primary-validated on the viewed test partitions |
| `validation-latent-tsne.v1` | deterministic post-hoc exact-Torch t-SNE of context-balanced nested-validation loci; no test loci or targets |
| `validation-latent-umap.v3` | deterministic post-hoc exact-Torch UMAP using intrinsic geodesic input distances and unit-`S^2` geodesic output distances for context-balanced nested-validation loci plus age; includes Stress-1, distance correlation, and neighbor recall; no test loci or targets |
| `prediction-scatter.v1` | display-only target-blind samples from frozen evaluation pairs and held-out age predictions; carries source hashes and performs no fitting |
| `age-scatter-cluster-audit.v1` | target-aware post-hoc k=2 diagnosis on every frozen-test age-scatter point with context, sequence-feature, probe, chromosome, and sex-stratified associations; never a model-selection artifact |
| `protein-targets.v1` | 52 proteins and 651 complete common subjects; float64 probe-protein, protein-protein, probe-age, and direct residualized protein-age targets plus target-blind protein partitions |
| `protein-tss-embeddings.v1` | 52 GRCh37 TSS-centred hg19 plus-strand Caduceus embeddings for one frozen window; records every interval and centre base |
| `protein-amino-acid-embeddings.v1` | 52 reviewed canonical UniProt sequences embedded by pinned ESM-2; all residues covered exactly once and long proteins never truncated |
| `protein-extension-results.v1` | four frozen parent probe geometries by five protein representations, separate inductive populations, cis/trans and overlap audits, protein-pair metrics, age-edge diagnostics, and display-only scatters |

## Identity rules

Downstream metadata names:

- exact producer Git commit from a clean worktree;
- protocol ID and protocol-file SHA-256;
- sealed data-bundle SHA-256;
- target, probe-order, split, embedding, prerequisite, selection, model, and pair-cache hashes;
- split name and window size;
- exact training configuration and seed.

Existing outputs are accepted only after full revalidation. A same-sized but different payload is
not compatible. Unknown schemas/fields and missing prerequisites fail; there is no implicit
upgrade.

Every artifact-producing entry point resolves the repository top level, records `HEAD`, and
requires `git status --porcelain=v1 --untracked-files=all` to be empty before doing work. A dirty
worktree is an error because an uncommitted producer cannot be reconstructed from its claimed
commit.

## Publication eligibility

The site compiler labels a split `primary_validated` only after all four window evaluation files
match the sealed data and split identity and contain:

- both pair populations;
- uniform and distance-stratified model/reference metrics;
- all three age stages;
- the complete held-out projection at the registered window;
- final model, pair cache, and baseline hashes.

The compiler does not infer eligibility from filenames or process exit status.
