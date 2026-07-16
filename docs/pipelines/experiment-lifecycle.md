# Experiment lifecycle

Protocol v2 is frozen. Every stage consumes immutable upstream artifacts and either validates an
existing output or publishes a new one exclusively.

## 1. Source and processor audit

Verify byte hashes, paired IDAT inventory, sample identities, phenotype joins, platform, build,
reference, exclusion lists, processor environment, and the methylprep/seSAMe parity decision.

Gate: [`data-preprocessing.md`](data-preprocessing.md).

## 2. Primary preprocessing

Run pinned seSAMe, apply sample-first detection QC, complete-case probe detection, quality masks,
Chen/Zhou masks, autosome/canonical-CpG rules, and maximum-window sequence eligibility. Preserve
all exclusion reasons.

Observed output: 346,628 probes by 706 samples.

## 3. Target geometry

Standardize beta rows and age in float64. Cache `X`, standardized age, and `rho = X @ age`.
Never materialize global `Y`; gather exact blocks or pair targets as `X @ X.T` products. Record the
correlation audit and rank ceiling.

Observed output: maximum correlation-identity error `2.7755575615628914e-15`, rank ceiling 705.

## 4. Primary and validation splits

Construct diverse-block and chromosome-7 test partitions from locus metadata only. Remove the
maximum-window primary buffer. Inside each primary training partition, construct and buffer a
second diverse-block validation partition. Recheck exact interval non-overlap for every window.

Gate: [`splits.md`](splits.md).

## 5. Seal the data bundle

Reload all data products, recompute target and split invariants, reject unexpected files, and
write the canonical 17-file bundle manifest with the exact producer Git commit. Every later stage
verifies the full bundle before loading tensors. All artifact-producing commands refuse a dirty
worktree, so the recorded commit fully describes the producing implementation.

## 6. Freeze sequence baseline and evaluation pairs

Fit the CpG-density/GC baseline on complete primary training. Separately generate deterministic
uniform and distance-stratified pair indices and exact targets. Fit the distance curve from
training-training pairs only. These artifacts exist before neural training.

## 7. Cache Caduceus embeddings

For each window, extract the hg19 plus-strand sequence and store the pinned model's CpG-centre
representation in restart-safe fp16 shards. Every shard and the final manifest carry the same
clean producer commit. Finalize only after exact full coverage.

Gate: [`embeddings.md`](embeddings.md).

## 8. Tune and refit age-only models

For every split/window, sweep latent dimension `{16,32,64,128,256}` with no Gram objective.
Validate every 100 of 2,000 steps, select minimum validation age MSE, then refit the selected
configuration/step on complete primary training.

## 9. Tune and refit full models

For every split/window, sweep the same dimensions and `lambda_age in {0.1,1,10}`. Select the
minimum unweighted validation pair-plus-age MSE, then refit on complete primary training. Adam uses
learning rate 0.001, batch size 512, a 1,048,576-base target-blind neighborhood sampler, one fixed
training seed, and exactly zero weight decay.

## 10. Evaluate once

Load final sequence, age-only, and full artifacts plus the frozen pair cache. Report separate split,
pair-population, distance-class, and age metrics. Fit no choices on the test partition.

Gate: [`evaluation.md`](evaluation.md).

## 11. Compile and expose the static site

Compile complete panels from validated evaluation files, verify locally, then expose the
read-only directory through `cloudflared`.

Gate: [`../operations/results-site.md`](../operations/results-site.md).

## Restart semantics

Completed files/directories are immutable and revalidated. Embedding shards and candidate model
runs may be resumed only by rerunning the identical command. Temporary directories do not count
as completion. Manual renaming, overwriting, or partial-result registration is forbidden.
