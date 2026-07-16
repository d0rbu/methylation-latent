# Genomic splits and leakage prevention

## Threat model

The primary leakage risk is not duplicate labels; it is duplicate input sequence. Two symmetric
windows of width `w` centred `delta` bases apart overlap whenever `abs(delta) < w`. A random probe
split therefore leaks heavily at local boundaries.

Split code is physically target-blind: it accepts only validated probe IDs, autosomal chromosome,
one-based cytosine coordinate, genomic context, chromosome lengths, and a Torch RNG seed.

## Diverse held-out locus blocks

The word *block* avoids confusing the split unit with UCSC CpG islands. Anchors are sampled from
every available chromosome-by-context stratum, where context is one of island, shore, shelf, and
open sea. Each anchor defines a fixed physical held-out interval, and every eligible probe in that
interval is held out. Candidate anchors must be far enough apart that held-out blocks do not
overlap.

The result is many small regions distributed across the genome, not one contiguous chromosome
chunk and not blocks discovered from methylation correlations. The split artifact records anchors,
intervals, per-chromosome/context counts, seed, and manifest fingerprint. Construction fails if
either train or test lacks a requested context.

## Held-out chromosome

The second split holds out all eligible probes on a configured autosome. No same-chromosome
train/test pair exists. The chromosome is chosen in the frozen protocol, not after comparing model
performance.

## Common maximum-window buffer

To keep the window sweep comparable, the training universe for every window uses the largest
configured width `W_max`. A candidate training cytosine on the same chromosome is removed when its
distance from any test cytosine is at most `W_max`. Using `<=` is one base more conservative than
the half-open non-overlap condition.

Validation anchors are selected from the remaining training universe and receive the same buffer
relative to optimization probes. Test probes are never used for early stopping.

## Required assertions

For every configured width and every train/test locus pair on the same chromosome:

```text
train_interval.end <= test_interval.start
or
test_interval.end <= train_interval.start
```

The implementation checks nearest neighbors in sorted coordinate arrays rather than materializing
all pairs. Tests include:

- equality at exact window width;
- adjacent and identical positions;
- different chromosomes;
- even-window coordinate conversion;
- every width in the configured 1,024/4,096/16,384/65,536 sweep;
- randomized property cases comparing the optimized assertion with a brute-force oracle.

The split manifest is written only after all widths pass. Embedding and training artifacts require
that manifest fingerprint.
