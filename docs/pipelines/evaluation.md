# Evaluation

## Run order is enforced

For an identical data/split/window fingerprint:

1. fit CpG density plus GC content to `rho` with training probes;
2. fit the Caduceus age-only head with no pair objective;
3. train and evaluate the full metric.

The registry rejects stage 3 without completed stage-1 and stage-2 records. Numbers are read from
artifacts, never copied from logs or memory.

## Hand-crafted baseline

For each exact input window:

- GC content is the fraction of bases equal to G or C;
- CpG density is the fraction of adjacent base pairs equal to CG, with denominator `window - 1`.

Fit an intercept plus these two features by float64 ordinary least squares on training probes only.
Constant or rank-deficient design columns are errors. Evaluate on held-out probes.

## Pair populations

Never combine:

- **seen-by-held-out:** one training probe and one test probe;
- **held-out-by-held-out:** two distinct test probes, counted once as an unordered pair.

Diagonal pairs are excluded from evaluation. Any sampling cap, seed, and stratum counts are stored;
all compared models use the same pair-index artifact.

## Distance strata

For loci on the same chromosome, bin absolute cytosine distance using frozen half-open edges:

```text
[0, 1 kb), [1, 4 kb), [4, 16 kb), [16, 64 kb),
[64, 256 kb), [256 kb, 1 Mb), [1 Mb, infinity)
```

Different-chromosome pairs form `trans`; they have no `|delta position|` and are never placed into
a cis bin. Report every pair metric per bin and the number of pairs. A pooled summary may be shown
only as a secondary aggregate adjacent to, never instead of, the bins.

## Distance-only reference

Fit `f(|delta position|)` from training-training target pairs only, using the same fixed bins and
the mean target correlation in each bin. The trans baseline is a separate training trans mean.
Freeze and hash that table before applying it to either test pair population. Empty bins are errors;
there is no interpolation fallback.

## Metrics

For every model, pair population, and distance class report:

- pair count;
- target and prediction means/standard deviations;
- MSE;
- Pearson correlation between predicted and target correlations;
- R-squared against the target mean for that exact evaluation stratum.

For age report held-out probe count, MSE, and Pearson `corr(rho_hat, rho)`, shown next to both
baseline stages. Confidence intervals, if added, bootstrap held-out genomic blocks rather than
individual highly correlated pairs.

## Latent projection

The site uses a deterministic 2D PCA/SVD projection of held-out latent vectors, with probes colored
by genomic context. Projection is descriptive and receives no metric status. Fit projection axes on
training latent vectors and apply them to held-out vectors to avoid using test geometry to choose
the view.
