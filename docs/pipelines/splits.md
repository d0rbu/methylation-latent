# Genomic splits and leakage prevention

The split is a locus split, not a subject split. All 706 subjects define every retained locus
target; held-out labels are probe rows and probe-pair correlations.

## Target blindness

Split construction receives only probe ID, chromosome, one-based hg19 cytosine position, genomic
context, and seeded Torch RNG state. It cannot read beta values, `X`, `rho`, `Y`, embeddings, or
model outputs. Co-methylated blocks are never inferred from the targets.

## Diverse-block primary split

Protocol v2 samples four 262,144-base blocks per context across chromosomes for each of island,
shore, shelf, and open sea. The realized test partition contains 2,250 probes across 16 anchor
blocks. It is deliberately distributed rather than one contiguous genome segment.

Every otherwise training-eligible probe whose maximum-window interval could overlap a test
interval is removed. This excludes 900 probes and leaves 343,478 complete-primary-training probes.

At all four window widths, the closest train/test cytosines on a shared chromosome are 65,605
bases apart. For the 65,536-base even windows, exact half-open intervals are therefore disjoint;
the same explicit check also passes for 1,024, 4,096, and 16,384 bases.

## Held-out chromosome

Chromosome 7 is the strict second split:

- 20,938 test probes;
- 325,690 complete-primary-training probes;
- no chromosome shared between train and test;
- no possible train/test sequence overlap at any window.

The two split families are evaluated and displayed independently.

## Nested validation

Hyperparameters and checkpoint steps cannot use either primary test partition. A second,
target-blind diverse-block split is drawn inside each primary training universe using seed 991027,
then receives its own maximum-window buffer.

| Primary split | Validation probes | Validation-buffer exclusions | Optimization probes |
|---|---:|---:|---:|
| diverse blocks | 1,535 | 530 | 341,413 |
| held-out chromosome 7 | 920 | 324 | 324,446 |

The minimum optimization/validation cytosine separation is 65,612 bases for the diverse primary
split and 65,604 for the chromosome primary split. Exact non-overlap passes at every window.

Final selected configurations are refit on the complete primary training partition, not merely the
optimization subset.

## Overlap definition

For a probe cytosine at one-based `p` and even width `w`, use:

```text
start0 = (p - 1) - w / 2
end0 = start0 + w
```

Two windows overlap iff their chromosomes match and
`max(start_a,start_b) < min(end_a,end_b)`. Touching half-open boundaries are not overlap.

The maximum-window buffer is an efficient construction step; it is not accepted as proof.
`assert_no_window_overlap` independently checks the final index sets for each configured width,
and the sealer repeats those checks after deserialization.

## Evaluation pairs within the test set

Two held-out probes may have overlapping sequence windows. That does not leak a training input
into a test input, but it makes nearby held-out-by-held-out pairs easier and dependent. Those pairs
remain valid placement questions and are therefore retained, but every pair result is reported by
distance class and compared with the training-only distance baseline.
