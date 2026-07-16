# Evaluation

## Immutable run order

For each exact data/split/window identity:

1. fit CpG density plus GC content to `rho`;
2. fit Caduceus plus an age-only latent head;
3. tune and refit the full pair-plus-age metric;
4. evaluate all frozen models once on the test partition.

Stages 1 and 2 are durable prerequisites. Stage 3 cannot run without their matching hashes.

## Sequence baseline

For each exact window:

- GC content is `(G + C) / window`;
- CpG density is the number of adjacent `CG` dinucleotides divided by `window - 1`.

An intercept plus both features is fit by float64 ordinary least squares on complete primary
training probes. Rank-deficient designs are errors.

The frozen baseline is already complete. At 1,024 bases it reaches age-correlation Pearson 0.4626
on the diverse-block test set and 0.5126 on held-out chromosome 7. These are the bars the neural
age heads must beat; later results never replace them by hand.

## Pair populations

Two populations are reported independently:

- **seen-by-held-out:** one complete-primary-training probe and one test probe;
- **held-out-by-held-out:** two distinct test probes, stored once as an unordered pair.

Diagonals are excluded. The evaluation cache was constructed before neural training and stores
the pair indices, distance class, and exact `X_i @ X_j` target for:

- up to 1,000,000 uniformly sampled pairs per test population;
- up to 100,000 pairs per distance class per test population;
- 100,000 training-training pairs per distance class for the distance reference.

Seeds and realized counts are in the cache metadata. Compared models cannot resample.

## Distance classes

Same-chromosome absolute cytosine distance uses fixed half-open bins:

```text
[0,1 kb), [1,4 kb), [4,16 kb), [16,64 kb),
[64,256 kb), [256 kb,1 Mb), [1 Mb,infinity)
```

Different-chromosome pairs are `trans`; they never receive a fake genomic distance. Pooled uniform
metrics are secondary summaries. Distance-stratified results are the primary interpretation.

## Distance-only reference

For each split, `f(|delta position|)` is the mean target correlation in each class among the frozen
training-training pairs. The trans mean is fit separately. The table is hashed before application
to either test population.

The held-out-chromosome seen-by-held-out population is necessarily all trans. In that case a
constant distance-reference prediction has undefined Pearson correlation; the artifact records
the null status explicitly rather than manufacturing a number.

## Model metrics

For each pair population report:

- uniform pair count, MSE, Pearson correlation, R-squared, target/prediction mean, and standard
  deviation;
- the same fields separately for every realized distance class;
- the training-only distance-reference metrics on the identical pair indices.

For age report held-out probe count, MSE, Pearson correlation, and R-squared for:

- CpG density plus GC;
- Caduceus age-only;
- full latent metric.

No metric is mixed across split families, pair populations, or distance classes.

## Hyperparameter selection

Each split has a separately buffered validation partition nested inside primary training.
Candidates run for 2,000 steps and validate every 100 steps.

- age-only candidates select minimum validation age MSE;
- full candidates select minimum unweighted pair MSE plus age MSE;
- ties prefer smaller latent dimension, then smaller lambda;
- the selected step/configuration is refit from initialization on complete primary training;
- test targets are loaded only by the final evaluation stage.

The full training objective still uses configured `lambda_age`; the unweighted validation sum is
only the frozen cross-lambda selection rule.

## Latent projection

At the 16,384-base window, deterministic two-dimensional PCA axes are fit on complete primary
training latent vectors and applied to every test vector. Points are colored by island, shore,
shelf, or open-sea context. This panel is descriptive and never selects a model.
