# Prior-work search

Search date: 2026-07-15.

## Question searched

Has a learned model already mapped reference DNA sequence at CpG loci directly to an empirical
cross-person CpG-by-CpG co-methylation/correlation geometry?

Searches combined terms for DNA sequence, CpG, co-methylation, covariance/correlation matrices,
graph prediction, methylation-correlated blocks, deep learning, and foundation models. The search
covered PubMed/PMC, arXiv, and direct paper/repository pages. This is a best-effort literature
search, not proof of absence.

## Conclusion

No exact prior implementation was found. Existing work falls into distinct neighboring categories:

- **Observed co-methylation discovery.** MCB, coMethDMR, and correlated methylation unit methods
  construct blocks or graphs from measured methylation correlations. For example, Gunasekara et
  al. build correlation matrices from observed 450K/EPIC cohorts and segment them into correlated
  methylation units; they do not predict the matrix from sequence.
- **Per-site methylation prediction.** DeepCpG and related methods use local DNA sequence, often
  together with neighboring observed methylation, to impute a cell/sample's methylation state.
  PretiMeth explicitly uses observed neighboring or co-methylated loci as features. These targets
  are per-sample methylation, not a locus-by-locus population covariance map.
- **Mechanistic spatial covariance.** Mean-field methylation models can predict covariance between
  nearby sites under a specified stochastic methylation process. They do not learn an empirical
  cross-person covariance map from genomic sequence representations.
- **Age prediction with observed graphs.** GraphAge and related work use observed methylation and
  a predefined co-methylation graph to predict an individual's age. They solve the inverse unit of
  prediction: individual age from methylation, not locus age association from sequence.

The closest work found is the May 2026 preprint
[*Bridging Sequence and Graph Structure for Epigenetic Age Prediction*](https://arxiv.org/abs/2605.10541).
It combines eight hand-crafted sequence features (and a CNN ablation) with a graph whose edges are
constructed from observed co-methylation, genomic proximity, and gene membership, then predicts
individual age from individual methylation values. It does **not** predict held-out locus-locus
correlations from sequence alone. Its finding that hand-crafted sequence features outperform its
CNN makes the proposed CpG-density/GC baseline especially important.

## Key neighboring references

- Gunasekara et al. (2023), [*A first-generation genome-wide map of correlated DNA
  methylation*](https://doi.org/10.1101/gr.276547.122).
- Angermueller et al. (2017), [*DeepCpG: accurate prediction of single-cell DNA methylation states
  using deep learning*](https://doi.org/10.1186/s13059-017-1189-z).
- Li et al. (2020), [*PretiMeth: precise prediction models for DNA methylation based on single
  methylation mark*](https://doi.org/10.1186/s12859-020-3500-6).
- Affinito et al. (2020), [*Nucleotide distance influences co-methylation between nearby CpG
  sites*](https://doi.org/10.1016/j.ygeno.2018.05.007).
- Li et al. (2026), [*Bridging Sequence and Graph Structure for Epigenetic Age
  Prediction*](https://arxiv.org/abs/2605.10541).
- Schiff et al. (2024), [*Caduceus: Bi-Directional Equivariant Long-Range DNA Sequence
  Modeling*](https://arxiv.org/abs/2403.03234).

## Novelty statement to use cautiously

The working delta is: **out-of-locus prediction of an empirical population co-methylation geometry
and age-association vector from frozen reference-sequence embeddings, evaluated on sequence-buffered
held-out loci.** Re-run the search immediately before a paper or public novelty claim.
