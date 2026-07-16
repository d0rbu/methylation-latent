# Glossary

| Term | Meaning |
|---|---|
| Probe locus | One HM450 `cg` assay mapped to a one-based hg19 plus-strand cytosine. |
| Static universe | Target-blind loci surviving manifest, autosome, published-mask, and maximum-window reference checks. |
| Target geometry | Cached unit-norm `X`, standardized age, and `rho = X @ age`; `Y` blocks are `X @ X.T`. |
| Diverse-block split | Context-stratified physical blocks spread across chromosomes without reading methylation targets. |
| Primary training | All probes outside a primary test partition and its maximum-window buffer. |
| Optimization partition | Primary training after removing nested validation and its buffer. |
| Buffer | A training probe excluded because its sequence interval could overlap a held-out interval. |
| Seen-by-held-out | Pair with one complete-primary-training locus and one primary test locus. |
| Held-out-by-held-out | Unordered pair of two distinct primary test loci. |
| Distance reference | Mean target correlation per cis/trans class fit only on training-training pairs. |
| Sequence baseline | Float64 OLS age-association predictor using CpG density and GC content. |
| Age-only model | Caduceus projection and free age vector trained without the pair loss. |
| Full metric | Shared Caduceus projection trained with pair and separately weighted age losses. |
| Phantom type | Runtime primitive narrowed by a predicate and carried as a richer static type. |
| Semantic tensor | Shape/dtype plus validated meaning such as zero-mean unit-norm rows. |
| Sealed bundle | Closed-world primary-data manifest that hashes every expected file. |
| Primary validated | Complete held-out result whose entire frozen artifact chain passed. |
| `MASK_general` | Zhou HM450 annotation union covering mapping, non-uniqueness, extension-base, color-switch, and common-SNP risks. |
