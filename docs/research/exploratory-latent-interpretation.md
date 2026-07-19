# Post-hoc latent interpretation

This analysis asks what can be learned from the already fitted 1 kb, 4 kb, and 16 kb full-model
weights and probe representations. It was designed after the primary test results were viewed.
Every result is therefore exploratory and hypothesis-generating; it cannot revise protocol v2 or
turn the single GSE87571 cohort into independent biological replication.

The original two-window artifact and `configs/latent-interpretation-v1.toml` remain immutable. The
three-window extension is separately frozen in `configs/latent-interpretation-v2.toml`, including
all six exact parent-model and embedding-cache identities. The analysis never invokes Caduceus and
never changes a model. It reconstructs probe latents deterministically from the immutable
embedding caches and final `W`/age checkpoints.

## Identifiable geometry

Individual rows of `W` and named latent coordinates are not identifiable. For every orthogonal
matrix `Q`, replacing `W` by `QW` and the age direction by `Qw_age` leaves every prediction
unchanged. The analysis therefore uses only rotation-invariant objects:

- `M = W.T @ W` and its eigenvalue spectrum;
- the pulled-back Caduceus-space age numerator `W.T @ normalize(w_age)`;
- normalized latent Gram values, linear CKA, and nearest-neighbour sets;
- covariance spectra of the actual normalized probe representations.

The deterministic training initialization is reconstructed from the registered seed. Final versus
initial agreement is reported so shared initialization cannot be mistaken for learned stability.

## Exact age decomposition

For normalized probe latent `z_i` and normalized age direction `a`, define

```text
s_i = z_i @ a
r_i = z_i - s_i * a
```

Then the predicted pair correlation decomposes exactly:

```text
Y_hat_ij = s_i * s_j + r_i @ r_j.
```

The empirical geometry has the matching decomposition. For standardized probe row `X_i`,
standardized age `A`, and `rho_i = X_i @ A`, define `q_i = X_i - rho_i * A`. Then

```text
Y_ij = rho_i * rho_j + q_i @ q_j.
```

The implementation asserts both identities. It reports total, age-component, and age-adjusted
residual metrics separately for the frozen seen-by-held-out and held-out-by-held-out stratified
pair populations, never pooling populations or distance classes.

## Target reliability

Weak prediction can reflect either a weak model or noisy empirical targets. Reliability is
estimated without changing the model:

1. sort the 706 subjects by age;
2. pair adjacent subjects and randomly assign one member of every pair to each half;
3. independently center and unit-normalize beta rows and age within each 353-subject half;
4. recompute held-out `rho`, pair correlations, the age rank-one component, and the age-adjusted
   residual;
5. compare halves across 16 registered seeds.

The adjacent-age pairing keeps the two age distributions closely matched. Pair reliability uses a
target-blind distance-stratified sample of at most 12,500 frozen held-out pairs per non-empty
distance class. Raw split-half correlations and Spearman-Brown estimates are descriptive noise
audits, not independent-cohort validation. Pair entries share probes and are not independent
replicates.

## Window and representation stability

At identical held-out loci, compare every ordered window pair `(1 kb, 4 kb)`, `(1 kb, 16 kb)`, and
`(4 kb, 16 kb)` for age predictions, pair predictions, linear CKA, and top-25 hyperspherical
neighbours. Weight spectra, actual latent covariance spectra, and pre-normalization norms are
reported. A target-blind sample is used when an all-probe operation would require a quadratic
matrix.

Linear surrogates fitted on the registered target-blind 50,000-probe primary-training sample
measure how much held-out model output and empirical age association are explained by:

1. CpG density and GC content;
2. those features plus CpG context, Infinium design, manifest strand, enhancer, regulatory, and
   DHS indicators.

These are confound diagnostics, not competing neural models. A high assay-annotation surrogate
score weakens a biological interpretation even though the annotations are fixed properties of a
locus.

## Candidate loci and pairs

GPL13534 v1.1 annotations are accepted only at the byte-pinned source hash. Every retained probe
must join exactly once with identical probe ID, build 37 chromosome, and one-based `MAPINFO`.
Candidate rankings use model predictions only:

- age candidates require a common sign across all three windows and rank by the smallest absolute
  prediction;
- pair candidates require a common sign across all three windows and rank by the smallest absolute
  pair prediction;
- empirical values are revealed only after ranking.

RefGene names/groups, enhancer, regulatory-feature, and DHS annotations are displayed as context.
No pathway enrichment is treated as confirmatory because probe coverage is nonuniform, the strict
split is chromosome 7 only, whole-blood targets may contain cell-composition effects, and there is
no independent cohort.

## Interpretation boundaries

- The primary test set has already been viewed; all tables are discovery material.
- Reliability within one cohort does not measure cross-cohort transport.
- GSE87571 targets were not residualized for sex, slide, cell composition, or other covariates.
- UMAP coordinates are never used to define neighbours, clusters, candidates, or statistics.
- The freely learned dual-probe table is excluded from biological candidate generation because it
  has no held-out rows and failed sequence transfer. It remains a diagnostic empirical
  factorization only.
