# Glossary

| Term | Meaning |
|---|---|
| Phantom type | A runtime primitive narrowed by a predicate and represented as a richer static type after validation. |
| Refinement function | A function such as `parse_probability` that validates raw input and returns a phantom type. |
| Runtime boundary | A place where untrusted values enter the system, such as CLI args, config files, data files, model outputs, or public APIs. |
| Property test | A Hypothesis test that checks an invariant across many generated examples. |
| Array contract | A dtype and shape expectation expressed with `jaxtyping` and enforced with `beartype`. |
| Audit-only | An artifact useful for exercising codepaths but ineligible for scientific figures or baseline comparisons. |
| Primary candidate | An artifact with eligible provenance whose downstream checks are not all complete yet. |
| Primary validated | An artifact whose complete upstream graph and required checks passed under the frozen protocol. |
| Probe locus | One HM450 CpG assay mapped to a one-based hg19 plus-strand cytosine coordinate. |
| Target geometry | Cached unit-norm probe rows `X`, standardized age, and `rho = X @ age`; Gram blocks are computed from `X` on demand. |
| Diverse block split | Context-stratified physical locus blocks spread across chromosomes and selected without target values. |
| Buffer | Training probes removed because their sequence windows could overlap a held-out window. |
| Seen-by-held-out | Pair population with one training locus and one test locus. |
| Held-out-by-held-out | Unordered pair population containing two distinct test loci. |
| `MASK_general` | Zhou HM450 annotation flag combining mapping, sequence non-uniqueness, extension-base, and common-SNP exclusions. |
