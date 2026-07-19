# Post-hoc dual probe latents

This experiment asks whether a train-only free probe geometry can act as an optimization scaffold
for the sequence-derived latent metric. It is post-hoc and hypothesis-generating: the 1 kb and
4 kb primary test results were already viewed before this protocol was created. It cannot replace
or revise protocol v2.

## Versioned window scope

Dual-probe protocol v1 and its 1 kb/4 kb artifact root remain immutable. A separately frozen
dual-probe protocol v2 may register a canonical supported window subset containing 16 kb, including
a 16-kb-only grid. Every registered split/window cell must pin its own selected parent dimension,
age-loss weight, selection record, evaluation record, embedding manifest, and pair-cache metadata.
The v2 run writes a separate artifact root and a v2 completeness manifest; it does not append to or
reinterpret the v1 root.

The report compiler may combine validated v1 and v2 roots only when they share the same parent
protocol and sealed data bundle. It records protocol, manifest, and producer provenance per source
and rejects any split/window cell claimed by more than one source. No v2 result is eligible until
the v2 configuration is frozen with the exact parent artifact identities.

## Model

For each fitting partition, retain one learned vector `u_i` per fitting probe and compute one
sequence vector `z_i = normalize(W e_i)` from its frozen Caduceus centre-token embedding. The
learned table has no rows that can be used for validation or test evaluation.

The learned geometry receives the empirical objective

```text
L_geometry = mean((u u.T - Y)^2) + lambda_age * mean((u a - rho)^2)
```

where every row and the shared age direction `a` are normalized before cosine products. The
sequence branch receives no direct `Y` or `rho` loss in this experiment. It learns through the
gradient-routed catch objective

```text
L_catch = alpha * mean(1 - cos(u, stopgrad(z)))
        + (1 - alpha) * mean(1 - cos(stopgrad(u), z))

L = L_geometry + 0.1 * L_catch
```

The stop-gradient operators make the endpoint semantics exact:

- `alpha = 0`: the learned table follows only empirical geometry; the sequence map chases it;
- `alpha = 1`: the sequence map receives no training gradient after initialization; the learned
  table is pulled toward the fixed sequence geometry;
- intermediate values split the catch gradient between the two representations.

Fixed alpha candidates are `{0, 0.25, 0.5, 0.75, 1}`. Linear `0 -> 1` and `1 -> 0` schedules
span the full 2,000-step run and include their endpoints exactly. Catch weight `0.1` is frozen, not
selected on test. The dense sequence/age parameters use Adam and the row-indexed learned table
uses SparseAdam; both use learning rate `0.001`, and neither applies weight decay. Sparse table
updates are mathematically restricted to the unique probe rows present in the minibatch.

## Initialization

Learned rows start as seeded iid standard-normal vectors and are normalized. Before optimization,
`W` is initialized by the bias-free least-squares map from the fitting partition's frozen
embeddings to those random unit vectors. A pre-run real-data audit found that the untruncated
normal matrix is positive definite but has condition number about `2.28e11` at 1 kb. Direct
inversion would amplify nearly-null embedding directions. The frozen implementation therefore
uses an eigendecomposed minimum-norm solve, discarding directions that would make the retained
condition number exceed `1e8`. It records raw and retained condition numbers, effective rank,
discarded spectral mass, discarded cross-moment fraction, and retained normal-equation residual;
the solve crashes if discarded cross-moment fraction exceeds `0.005` or retained residual exceeds
`1e-5`. Validation or test embeddings are forbidden from the solve.

Initialization, neighborhood batches, and learned rows are repeated for seeds `{851733, 851734,
851735}`. All alpha candidates for one split/window/seed begin from identical state and consume
identical target-blind batches.

## Selection and refit

The latent dimension and age weight are copied from the already validation-selected parent full
model for the same split and window. This isolates alpha rather than reopening `d` or lambda.

For each seed and alpha strategy, checkpoint selection minimizes exact sequence-only validation
pair MSE plus age MSE. Candidate selection then minimizes the mean of the three seed-specific best
validation scores. Only that alpha strategy is refit on complete primary training, once per seed,
for its seed-specific selected step count. Test targets never select alpha, schedule, checkpoint,
dimension, catch weight, or seed.

## Evaluation populations

All metrics use the frozen protocol-v2 pair indices and targets. Report these separately:

- **fully inductive seen x held-out:** sequence-derived vectors on both sides;
- **hybrid learned-seen x held-out:** complete-train learned vectors on the seen side and sequence
  vectors on the held-out side;
- **fully inductive held-out x held-out:** sequence-derived vectors on both sides;
- **age:** sequence-derived held-out vectors against the shared fitted age direction.

Uniform, distance-stratified, and training-only distance-reference metrics remain separate. Test
results are reported for all three refit seeds, with means and standard deviations; matrix entries
that share probes are not independent replicates.

## Leakage assertions

The implementation and real-data artifact must prove that:

1. tuning learned rows equal the optimization partition exactly;
2. tuning OLS sees only optimization embeddings;
3. validation selection uses sequence vectors only;
4. refit learned rows equal complete primary training exactly;
5. test indices are absent from both learned tables and both OLS solves;
6. hybrid evaluation maps every seen global index to exactly one learned row;
7. held-out predictions are functions only of frozen embeddings, `W`, and the shared age vector;
8. all unselected-alpha test metrics are absent from the artifact.

Any failure aborts publication. A positive hybrid result is not evidence that sequence predicts
new-new geometry; that claim belongs only to held-out x held-out sequence metrics.
