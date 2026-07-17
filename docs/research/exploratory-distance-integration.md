# Exploratory distance integration

Status: **post-hoc exploratory; not part of frozen protocol v2**

This analysis was designed after inspecting the primary 1,024- and 4,096-base test metrics. It
must never be presented as a confirmatory result on those test partitions. A new sealed holdout or
independent cohort is required for confirmation.

## Question

The frozen sequence-only metric has positive within-distance and trans pair Pearson correlation,
but worse MSE than the training-fitted distance-class reference. Test whether a positive-semidefinite
distance component can retain sequence ranking while shrinking predictions toward genomic
distance structure.

## Kernel family

For probe pair `(i,j)`, construct:

- a global constant kernel;
- same-chromosome Laplacian kernels `exp(-|position_i-position_j| / scale)` at 1,024, 4,096,
  16,384, 65,536, 262,144, 1,048,576, and 4,194,304 bases;
- the selected full model's sequence-cosine kernel.

Fit non-negative weights constrained to sum to one. Every component is PSD with unit diagonal, so
the mixture is a PSD unit-diagonal Gram matrix and is equivalent to concatenating scaled latent
feature maps. The global plus distance-only mixture and the distance-plus-sequence mixture are fit
separately.

## Leakage boundary

Weights minimize exact MSE over every distinct unordered pair in the nested validation partition.
The selected tuning checkpoint supplies validation sequence latents; it was trained only on the
optimization partition. The weights are then frozen and applied to the complete-train refit on the
existing uniform and distance-stratified test pair caches. No test target enters weight fitting.

The producer forces deterministic Torch algorithms and one CPU intra-op and inter-op thread before
loading artifacts. This is intentionally slower: exact restart identity is part of the artifact
contract, and threaded least-squares reductions can otherwise differ in their final floating bits.

This preserves the mechanical train/validation/test boundary, but it does not undo the post-hoc
origin of the hypothesis. Results use a separate schema and output tree and cannot populate the
primary-validated site panels.
