# Research protocol

Protocol status: **draft, no eligible data artifact and no result yet**.

## Question

Does reference DNA sequence around an autosomal CpG contain enough information to predict:

1. its empirical cross-person correlation with other CpGs in whole blood; and
2. its empirical cross-person Pearson correlation with chronological age?

The unit of prediction is a locus. The model never receives an individual's genome or age.
Accordingly, a positive result supports sequence predictability of cohort-level locus properties;
it does not by itself establish a causal mechanism or within-person epigenetic drift.

## Cohort and assay

- GSE40279: 656 whole-blood samples measured on Illumina HumanMethylation450.
- Sample metadata: age, gender, ethnicity, source, and plate from the series matrix.
- GPL13534 v1.1 manifest: GRCh37/hg19 coordinates.
- Reference: UCSC hg19 plus strand, archive MD5
  `806c02398f5ac5da8ffd6da2d1d5d1a9` for `hg19.fa.gz`.

The exhaustive reference audit verifies all 405,610 post-mask coordinates as plus-strand CpGs.
At the common 65,536-base window, 15 loci fail chromosome boundaries and 2,022 contain non-ACGT
reference bases, leaving 403,573 loci. Their ordered probe fingerprint is
`6a647c74a664bc693f67013c51046d163499839c843486f1c26ce6c5aa4ab4cf`; the exact decompressed
FASTA SHA-256 is `92b96d16b307d824f3b6b9dc63feb237a75226f82ca2166d51ecec039bee4449`.

The complete public-input audit pins the series matrix SHA-256
`1f6ef8b5ce08f9a5f1c09b047b36d1f3342757b6e8e19dce0398581bb76807eb`, sample-key SHA-256
`dbb7c510da6a90bd2ff203564288b3b3400298c9d50217377a347f2ef2a9e7f9`, and resulting ordered
sample fingerprint `f0bf579bb37e25dee7a7f9c3c29a5d498a506ec2226514ee1651c833b4707d61`.

The primary run requires raw paired red/green IDATs. GEO does not currently provide them; see
[`../pipelines/data-preprocessing.md`](../pipelines/data-preprocessing.md).

## Probe universe

Starting from GPL13534, remove:

- explicit `rs` and `ch` assay probes;
- any locus not on chromosomes 1 through 22;
- probes failing detection QC after sample QC;
- seSAMe HM450 quality-mask probes;
- the union of Chen et al. cross-reactive/polymorphic probes and Zhou et al. `MASK.general`;
- probes with missing, non-finite, or constant beta vectors;
- loci that cannot provide every configured full-length plus-strand hg19 window with an exact
  centred `CG` and only A/C/G/T bases.

Every removal is recorded as `(probe_id, reason, source)` in an exclusion ledger. Filtering never
uses age correlation or co-methylation.

## Targets

Let `B` be the retained probe-by-sample beta matrix. For each probe row:

```text
X_i = (B_i - mean(B_i)) / ||B_i - mean(B_i)||_2
```

Standardize chronological age identically to obtain `a`. Then:

```text
Y_ij = X_i @ X_j
rho_i = X_i @ a
```

`X` and `rho` are cached. `Y` is never materialized globally; exact blocks are gathered by matrix
multiplication from cached `X`. The implementation asserts numerical agreement with
`torch.corrcoef` on deterministic audit blocks.

Because every row of `X` is centred over `n_samples`, `rank(Y) <= n_samples - 1` (655 if all 656
samples survive). This is a construction invariant, not an empirical model claim.

## Sequence representation

Primary window widths are 1,024, 4,096, 16,384, and 65,536 bases. An even window is centred on the
plus-strand cytosine token; the following token must be guanine.

Use frozen
[`kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16`](https://huggingface.co/kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16)
at revision `d89eeb853136ea64da7feb3d0c8e909771b17ae6`. Tokenization disables added special tokens and
must return one token per base. The RCPS backbone returns two 256-channel orientation halves, so
its raw final state must be exactly 512-wide. Cache the first, plus-strand 256-channel half at the
cytosine index in fp16. The cached embedding width must be exactly 256.

No mean pooling, reverse-complement augmentation, fine-tuning, or per-epoch embedding is part of
the primary protocol.

## Model

For embedding `e_i in R^256`, latent dimension `d`, matrix `W in R^(d x 256)`, and an independent
age vector `w_age in R^d`:

```text
z_i       = normalize(W e_i)
Y_hat_ij  = z_i @ z_j
rho_hat_i = z_i @ normalize(w_age)

loss = mean((Y_hat - Y)^2) + lambda_age * mean((rho_hat - rho)^2)
```

There is no bias and no weight decay. `w_age` is not a row of the Gram batch and is not produced by
`W`. `d` is swept only through values at most `min(256, n_samples - 1)`. With `d >= 256`, the
positive-semidefinite metric `M = W^T W` is unrestricted in rank; larger `d` does not add metric
expressivity.

## Batching

Sample a training anchor, collect allowed training probes within a fixed genomic span around it,
and repeat until the requested unique batch size is filled. The batch sampler sees chromosome and
coordinate but not `Y`, `rho`, or embeddings. Loss uses all within-batch pairs, including the
zero-loss diagonal.

## Splits

Two split families are mandatory:

- diverse held-out locus blocks, with anchors stratified by autosomal chromosome and island,
  shore, shelf, or open-sea context;
- a held-out chromosome.

Split selection is target-blind. A common training set is built by excluding every training locus
within the maximum configured window width of any test locus. Half-open sequence intervals are
then checked pairwise for every individual window width. Model selection uses a separately buffered
validation subset drawn from the training universe; test targets remain sealed until evaluation.

## Run order and grids

1. For every split and window, ordinary least squares on CpG density and GC content predicts `rho`.
2. For every split and window, the frozen Caduceus embedding trains only the age objective.
3. The full model sweeps window width first, then `d in {16, 32, 64, 128, 256}` and
   `lambda_age in {0.1, 1, 10}`.

The secondary grids may be reduced using training-validation results, but no value is selected on
the test set. Split and evaluation seeds are already fixed. Before this draft can become frozen,
the strict configuration must also add batch construction, optimizer, step budget, validation
cadence, checkpoint-selection rule, and training seeds. The current fixed-step trainer exists to
test mathematical and data-flow contracts; it is not yet a frozen run orchestrator. Stage 1 is
forbidden until those remaining choices are reviewed, implemented, and versioned.

## Evaluation

Report independently:

- seen-by-held-out pair prediction;
- held-out-by-held-out pair prediction;
- Pearson correlation of `rho_hat` and `rho` across held-out probes.

For pair predictions report MSE, Pearson correlation, and R-squared. Same-chromosome pairs are
split into fixed absolute-distance bins; different-chromosome pairs form a separate `trans` bin.
Fit the nonparametric distance-only reference from training pairs and freeze it before evaluating
test pairs.

## Interpretation limits and diagnostics

- This is whole-blood age association, potentially including blood-cell composition effects; do
  not call it cell-intrinsic drift without a separate analysis.
- Caduceus was pretrained on a human reference genome. Locus holdout tests target-label
  generalization, not sequence novelty relative to foundation-model pretraining.
- Array probe density and design are nonuniform. Chemistry/design type and manifest context must be
  reported as diagnostic strata even after normalization.
- Nearby held-out pairs may have overlapping inputs. They remain valid for held-out-by-held-out
  placement but are isolated in distance bins; seen/test windows never overlap.
- Hyperparameter selection, failure analysis, or visualization of test results cannot revise this
  protocol without creating a new protocol ID.
