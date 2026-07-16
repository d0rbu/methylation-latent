# Research protocol

Protocol ID: `gse87571-hg19-caduceus-ps-v2`
Status: **frozen before neural embedding/training**

## Question

Does reference DNA sequence around an autosomal CpG contain enough information to predict:

1. its empirical cross-person correlation with other CpGs in whole blood; and
2. its empirical cross-person Pearson correlation with chronological age?

The unit of prediction is a locus. The model never receives an individual's genome, beta vector,
or age. A positive result would establish predictability of cohort-level locus properties from
reference sequence; it would not by itself establish a causal mechanism or within-person drift.

## Cohort and assay

- accession: GSE87571;
- assay: Illumina HumanMethylation450 whole blood;
- raw input: 732 exact red/green IDAT pairs;
- phenotype-eligible: 729;
- retained after sample detection QC: 706;
- retained metadata: 374 female and 332 male subjects, ages 14–94;
- manifest: GPL13534 v1.1, GRCh37/hg19;
- reference: UCSC hg19 plus strand.

The primary switch from GSE40279 was made because the latter's authentic IDATs were unavailable.
GSE87571 was selected before target or model results were available because it provides raw arrays
and age metadata. Its series matrix does not provide the ethnicity field available in the original
GSE40279 plan; ethnicity is therefore neither imputed nor used.

## Preprocessing

Pinned seSAMe 1.30.1 under R 4.6.0/Bioconductor 3.23 is the primary processor. The exact pipeline
per array is `QCD`, pOOBAH p-values and threshold 0.05, then noob. Python methylprep was rejected
after a preregistered 12-array parity audit failed detection, quality-mask, and beta-difference
thresholds.

Starting from canonical autosomal GPL13534 CpGs:

- union Chen cross-reactive and Zhou `MASK_general` exclusions;
- full 65,536-base plus-strand hg19 window, centred `CG`, A/C/G/T only;
- sample failure when more than 5% of candidate probes have detection `p >= 0.05`;
- probe retention only when detection `p < 0.05` in every retained sample;
- union of per-array seSAMe quality masks;
- finite beta values in `[0,1]` and nonconstant rows.

The static sequence universe contains 403,573 loci. Cohort QC excludes 23 samples, 18,999
quality-masked probes, and 37,946 detection-failing probes, leaving 346,628 probes by 706 samples.
No values are imputed.

## Targets

Let `B` be the retained probe-by-sample float64 beta matrix. Define:

```text
X_i = (B_i - mean(B_i)) / ||B_i - mean(B_i)||_2
a   = (age - mean(age)) / ||age - mean(age)||_2

Y_ij  = X_i @ X_j
rho_i = X_i @ a
```

Cache `X`, `a`, and `rho`. Do not materialize global `Y`; obtain training blocks and evaluation
targets by exact dot products from `X`. The independently sealed audit compared a deterministic
256-probe block with `torch.corrcoef`; maximum absolute error was
`2.7755575615628914e-15`.

Centering gives `rank(Y) <= 706 - 1 = 705`. Because sequence embeddings are 256-wide, the learned
metric has a stricter useful dimension ceiling of 256.

## Sequence representation

Primary windows are 1,024, 4,096, 16,384, and 65,536 bases. Each uses the hg19 plus strand with
the probe cytosine at `window / 2` and asserts that it is followed by G.

Use frozen
[`kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16`](https://huggingface.co/kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16)
at revision `d89eeb853136ea64da7feb3d0c8e909771b17ae6`, checkpoint SHA-256
`a3e6976fe90460ff5d90d371b457d0263dc65e156ea5651b4a452e98228daef2`.

Disable special tokens and require one token per base. From the `[batch,window,512]` RCPS final
backbone state, cache `hidden[:, window // 2, :256]` as fp16. Do not mean-pool, fine-tune,
reverse-complement average, or re-embed per epoch.

## Model

For cached embedding `e_i in R^256`, latent dimension `d`, matrix
`W in R^(d x 256)`, and independent `w_age in R^d`:

```text
z_i       = normalize(W e_i)
Y_hat_ij  = z_i @ z_j
rho_hat_i = z_i @ normalize(w_age)

loss = MSE(Y_hat, Y) + lambda_age * MSE(rho_hat, rho)
```

`W` has no bias and no weight decay. `w_age` is a free parameter, not a row of the pair Gram matrix
and not produced by `W`.

Since

```text
cos(W e_i, W e_j)
  = e_i.T M e_j / sqrt((e_i.T M e_i)(e_j.T M e_j)),
M = W.T W,
```

the pair head learns a positive-semidefinite Mahalanobis cosine metric. At `d = 256`, every rank
allowed by the embedding width is available; larger `d` adds no expressivity.

## Batching and optimization

Sample an anchor from the allowed optimization partition and collect unique allowed probes within
a 1,048,576-base genomic neighborhood, repeating until batch size 512. The sampler sees only
chromosome, position, and partition membership. It never clusters using targets.

Use Adam, learning rate 0.001, weight decay 0, 2,000 tuning steps, validation every 100 steps, and
training seed 851733. The full batch loss contains all within-batch pairs, including the exact
zero-loss diagonal.

## Splits

Two primary split families are mandatory:

- 16 context-stratified 262,144-base held-out blocks spread across autosomes;
- all chromosome-7 probes held out.

Primary counts:

| Split | Test | Buffered out of train | Complete train |
|---|---:|---:|---:|
| diverse blocks | 2,250 | 900 | 343,478 |
| chromosome 7 | 20,938 | 0 | 325,690 |

The diverse split's closest shared-chromosome train/test cytosines are 65,605 bases apart. The
chromosome split shares no chromosome. Exact half-open window non-overlap is asserted separately
for all four windows.

Model selection uses a second target-blind diverse-block split nested inside primary training:

| Primary split | Validation | Validation buffer | Optimization |
|---|---:|---:|---:|
| diverse blocks | 1,535 | 530 | 341,413 |
| chromosome 7 | 920 | 324 | 324,446 |

Primary test targets never select hyperparameters or checkpoints.

## Frozen run order

1. Fit float64 OLS `rho ~ 1 + CpG density + GC content` for each split/window on complete primary
   training.
2. For each split/window, sweep age-only latent dimension
   `d in {16,32,64,128,256}`.
3. For each split/window, sweep the full model over those dimensions and
   `lambda_age in {0.1,1,10}`.
4. Refit selected configurations from initialization on complete primary training.
5. Run one final evaluation and static-site compilation.

Age-only selection minimizes validation age MSE. Full-model selection minimizes unweighted
validation pair MSE plus age MSE so lambda candidates are judged on a common scale. Ties prefer
smaller `d`, then smaller lambda.

## Evaluation

Report separately for both split families:

- seen-by-held-out pair prediction;
- held-out-by-held-out pair prediction;
- held-out `corr(rho_hat,rho)` for sequence, age-only, and full stages.

For pair prediction, report uniform pooled metrics and the primary distance-stratified metrics in
seven cis bins plus trans. Compare on identical frozen pair indices with a class-mean
`f(|delta position|)` fit only from training-training targets.

Metrics are count, MSE, Pearson correlation, R-squared, mean, and standard deviation. Undefined
Pearson values for a constant distance reference are labeled undefined; model predictions are not
allowed to be silently constant.

## Interpretation limits

- Whole-blood associations can include blood-cell composition effects.
- Targets are not residualized for sex, slide, or other cohort covariates; autosomal associations
  with those variables can therefore contribute to the learned population geometry.
- GSE87571 is one public cohort; this protocol has no independent-cohort replication.
- Caduceus was pretrained on reference-genome sequence, so locus holdout is label generalization,
  not sequence novelty relative to pretraining.
- Probe placement, chemistry, and manifest context remain nonuniform despite normalization.
- Nearby held-out probes can have overlapping inputs; distance-stratified held-out×held-out
  results expose rather than hide that ease.
- A positive pair metric is predictive evidence, not proof that sequence causally determines
  methylation covariance.
- One frozen training seed measures the registered experiment, not seed-to-seed uncertainty.

Any change after test evaluation requires a new protocol ID.
