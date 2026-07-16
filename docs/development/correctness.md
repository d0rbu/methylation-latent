# Correctness

The experiment is designed so that invalid scientific state is rejected at the boundary where it
enters. A crash is preferable to a plausible result from mismatched data.

## Scalar and semantic types

Use `phantom-types` for bounded or structured primitives: beta values in `[0,1]`, correlations in
`[-1,1]`, one-based genomic coordinates, even positive windows, positive dimensions and batch
sizes, and non-negative loss weights. Refine raw values once.

`jaxtyping` expresses shape and dtype; validated frozen wrappers own properties it cannot express:

- `UnitNormRows`: finite rank-2 tensor, non-empty axes, zero row means, unit row L2 norms;
- `UnitNormVector`: finite, non-empty, zero mean, unit L2 norm;
- `CorrelationVector`: finite values bounded by one;
- `CorrelationMatrix`: square, symmetric, bounded, unit diagonal;
- non-empty probe, embedding, split, and pair-set records whose order hashes are explicit.

Public tensor APIs use `beartype` and `jaxtyping`. Private helpers operate on already validated
values.

## Correlation identity

For a nonconstant beta row `b`,

```text
c = b - mean(b)
x = c / ||c||_2
```

and therefore `corr(b_i, b_j) = x_i @ x_j` in exact arithmetic. Age is transformed identically.
The primary artifact uses float64, caches `X` and `rho = X @ age`, and computes `Y` blocks only as
matrix products from `X`.

Protocol v2 independently reloaded the sealed tensors and compared a deterministic 256-probe
block against `torch.corrcoef`. The maximum absolute error was
`2.7755575615628914e-15`. Missing values, imputation, clipping, and pair-specific sample axes are
forbidden.

## Rank contract

With `n` retained samples, centering gives `rank(X) <= n - 1` and
`rank(XX^T) <= n - 1`. The primary cohort has `n = 706`, so the target rank ceiling is 705.

Caduceus embeddings are 256-wide. Because `M = W.T @ W`, dimensions above 256 do not enlarge the
learned positive-semidefinite metric even though the empirical target ceiling is 705. Protocol v2
therefore rejects `d > min(256, n - 1)` and sweeps only `{16,32,64,128,256}`.

The earlier value 655 applied only to the abandoned 656-sample GSE40279 plan; it is not the rank
ceiling of this experiment.

## Coordinates and strand

GPL13534 `MAPINFO` is the one-based coordinate of the plus-strand CpG cytosine. For even window
width `w`:

```text
c0 = MAPINFO - 1
start0 = c0 - w / 2
end0 = start0 + w
centre_index = w / 2
```

The FASTA interval is zero-based and half-open. Extraction always uses the UCSC hg19 plus strand,
regardless of the assay-probe strand column, and asserts
`sequence[centre_index:centre_index+2] == "CG"`.

The exhaustive audit verified 405,610 masked loci at their declared coordinates. Fifteen cannot
support a full 65,536-base window and 2,022 contain non-ACGT reference bases, leaving a common
403,573-locus sequence universe. Windows are never padded, lifted over, or strand-flipped.

## Split and evaluation boundaries

Split selection receives locus metadata and seeded Torch RNG state, never beta values, `X`, `Y`,
`rho`, or embeddings. The primary test split is constructed before nested validation. Both buffer
passes use the maximum 65,536-base window, then exact half-open interval overlap is checked again
for all four windows.

Evaluation pair indices and target correlations are cached before neural training. Model
comparisons must use those identical indices. Distance-only reference values are fit from
training-training targets; test targets never select a distance curve, checkpoint, dimension, or
lambda.

## Artifact boundary

Every multi-file artifact has a canonical JSON manifest with schema, upstream identities, tensor
shape/order facts, and payload SHA-256. The sealed primary bundle enumerates all 17 files and
rejects absent, extra, resized, or modified files. Writes are exclusive or atomic-directory
renames. Existing results are validated and reused; they are never overwritten. Every producing
entry point also requires a clean Git worktree and records the exact 40-character producer commit;
the commit claim is rejected before computation if tracked or untracked files differ.

## No fallback policy

The following are errors:

- wrong source bytes, raw-IDAT inventory, build, coordinate, centre dinucleotide, or alphabet;
- missing/duplicate samples or probes, non-finite beta values, or constant retained rows;
- sample/probe order drift between adjacent artifacts;
- train/test sequence overlap at any configured window;
- `d` above the useful ceiling or weight decay other than zero;
- absent baseline, pair-cache, selection, embedding, or model prerequisites;
- constant model predictions for a metric that requires Pearson correlation;
- mixed split families or pair populations in one reported metric.

There is no processed-data or cross-build fallback for the primary experiment.
