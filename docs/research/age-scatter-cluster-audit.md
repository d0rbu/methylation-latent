# Post-hoc age-scatter lobe audit

## Status and question

This is a target-aware, post-hoc explanatory audit. It asks why predicted-versus-empirical age
association plots appear to contain two lobes. It does not train or select a model, and none of its
outputs are confirmatory metrics.

For each split, completed window, and age model, deterministic 64-restart k-means with `k = 2` is
fit to all frozen-test points in standardized two-dimensional `(empirical rho, predicted rho)`
space. Cluster labels are ordered by their empirical-rho center. Using empirical rho makes the
labels useful for diagnosis but invalid for model selection.

## Findings

The lobes are principally negative-age-association versus positive-age-association CpGs. The
strongest measured separator is genomic CpG context and local CpG density, not chromosome, strand,
or sex. For the full latent metric:

| Split | Window | context Cramer's V | design V | strand V | chromosome V | CpG-density Cohen's d | female/male rho correlation | female/male lobe d |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| diverse blocks | 1 kb | 0.577 | 0.346 | 0.036 | 0.175 | 1.747 | 0.933 | 1.642 / 1.931 |
| diverse blocks | 4 kb | 0.543 | 0.321 | 0.031 | 0.211 | 0.978 | 0.933 | 1.702 / 2.036 |
| chromosome 7 | 1 kb | 0.660 | 0.365 | 0.008 | not testable | 2.018 | 0.952 | 1.868 / 2.225 |
| chromosome 7 | 4 kb | 0.663 | 0.350 | 0.014 | not testable | 1.243 | 0.952 | 1.870 / 2.267 |

In the chromosome-7 holdout, both lobes remain although chromosome is constant. In the diverse
holdout, chromosome association is substantially weaker than context association. Strand
association is near zero in every panel, which also argues against a manifest-strand or extraction
convention artifact. Probe design is associated, but less strongly than context and is itself
confounded with CpG context; it must not be interpreted as an independent cause from this audit.

The female and male age-correlation vectors remain highly concordant, and the positive-versus-
negative separation remains large when correlations are recomputed within each sex. Sex can modify
effect magnitude, but it does not generate the two-lobe structure by itself. Female and male ages
also have similar distributions in the retained cohort: 374 female samples have mean 46.42 years
and standard deviation 20.98; 332 male samples have mean 47.96 and standard deviation 20.95.

Cell composition is not testable here. The sealed GEO phenotype has age, gender, tissue, and
disease state, but no measured or precomputed leukocyte proportions. All retained samples are
normal whole blood. Inferring cell proportions from the same methylation targets inside this audit
would be circular, so cell-type composition remains a plausible unresolved contributor rather than
something this analysis rules in or out.

The age-only cosine and full latent model assign nearly the same points to the two lobes: agreement
is 95.4-96.9%, with adjusted Rand index 0.823-0.879 across the four panels. That agreement and the
context composition indicate that the visible split is a property of the underlying age-association
target structure that both models learn, rather than a quirk unique to the full pair-loss head.

These are associations, not causal effects. CpG context/local density is the best measured
explanation among the audited variables; unmeasured blood-cell composition and other locus biology
remain possible contributors.

## Reproduction

```bash
uv run python scripts/analyze_age_scatter_clusters.py \
  --config configs/protocol-v2.toml \
  --data "$RUN_ROOT/data" \
  --series-matrix "$DATA_ROOT/sources/GSE87571_series_matrix.txt.gz" \
  --embeddings "$RUN_ROOT/embeddings" \
  --experiments "$RUN_ROOT/experiments" \
  --direct-age "$RUN_ROOT/exploratory/direct-tanh-age-v1" \
  --output "$RUN_ROOT/exploratory/age-scatter-cluster-audit-v1" \
  --windows 1024 4096
```

Existing outputs are accepted only if their canonical records exactly match a rerun. The script
reproduces the sealed age target, both registered model metrics, and test-index alignment before it
fits k-means.
