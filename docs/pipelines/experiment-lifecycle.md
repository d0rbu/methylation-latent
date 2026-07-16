# Experiment lifecycle

Every run advances through immutable, fingerprinted artifacts. A later stage refuses an artifact
whose schema, provenance, eligibility, or upstream fingerprint is wrong.

## 0. Freeze the protocol

Record the hypothesis, split family, window grid, latent-dimension grid, age-loss grid, metrics,
distance bins, batch construction, optimizer and step budget, validation/checkpoint rule, random
seeds, software versions, and source checksums before looking at held-out results. Changes after
evaluation start create a new protocol ID. The committed v1 protocol is still a draft because its
optimization/checkpoint fields are not yet frozen; no stage below may produce a primary artifact
until that is resolved.

## 1. Audit source data

Verify counts, checksums, platform, build, sample identity mapping, required phenotypes, paired
IDAT filenames, and control-probe availability. A GSE40279 processed-only artifact is marked
`audit_only`; an IDAT artifact can be marked `primary_candidate`.

Gate: [`data-preprocessing.md`](data-preprocessing.md).

## 2. Preprocess methylation

Run pOOBAH/detection QC, drop failed samples, require every retained probe to pass in every retained
sample, apply noob and nonlinear dye correction, union the pinned Chen and Zhou exclusions, remove
`rs`, `ch`, X, and Y probes, and export beta values plus a full exclusion ledger.

Gate: no missing/non-finite beta values, values in `[0,1]`, non-empty sample/probe axes, exact
sample metadata alignment, and an independent seSAMe parity audit on a preregistered subset.

## 3. Build targets once

Standardize beta rows and age in float64. Cache `X` and `rho = X @ age`; compute Gram blocks as
`X[index] @ X[index].T` on demand. Store correlation-identity errors and the `n_samples - 1`
rank ceiling. Training batches never fit or recompute target statistics.

## 4. Build sequence universe and splits

Parse hg19 loci, extract the maximum window on the plus strand, require centred `CG` and A/C/G/T
only, then generate both split families using locus metadata alone. The diverse-block split is
stratified by chromosome and island/shore/shelf/open-sea anchor context. The chromosome split holds
out the configured chromosome.

Apply the maximum-window buffer once to define a common training universe. Explicitly recheck
non-overlap for every configured window size.

Gate: [`splits.md`](splits.md).

## 5. Precompute embeddings

For each window size, run the pinned Caduceus revision exactly once, assert its raw RCPS final state
is two 256-channel halves, take the plus-strand half at the cytosine token, and write fp16
safetensors plus a probe-order fingerprint. Training has no dependency on Transformers or FASTA.

Gate: [`embeddings.md`](embeddings.md).

## 6. Run baselines in order

1. Fit CpG density plus GC content to `rho` on training probes only.
2. Fit the frozen-Caduceus age head with the Gram loss disabled.

Register both artifacts under the exact data, split, window, and probe-order fingerprints. A full
run cannot start without them.

## 7. Train the full model

Use genomic-window batches drawn only from training probes. Optimize pair MSE plus the configured
age MSE weight with zero weight decay. Checkpoint selection uses a training-derived validation
partition with its own sequence buffer; the held-out test loci are never used for model selection.
The fixed-step training primitive is implemented, but the versioned optimization and checkpoint
orchestration named in stage 0 must be completed before this stage is eligible to run.

## 8. Evaluate once

Emit separate seen-by-held-out, held-out-by-held-out, and held-out age records. Break pair records
into fixed cis distance bins and a trans bin. Fit the distance-only reference curve using training
pairs only.

Gate: [`evaluation.md`](evaluation.md).

## 9. Publish artifacts and site

Generate the static site only from validated result records. It must display protocol and artifact
fingerprints and visibly label audit-only or incomplete runs. Serve the generated directory locally
and expose that server through a named Cloudflare tunnel as documented in
[`../operations/results-site.md`](../operations/results-site.md).

## 10. Profile only after correctness

Record an end-to-end baseline before optimizing. Run one benchmark process at a time and compare
identical inputs, outputs, devices, and artifact hashes. Microbenchmarks do not justify pipeline
performance claims.
