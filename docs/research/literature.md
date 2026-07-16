# Prior-work search

Search date: 2026-07-16.

## Question searched

Has a learned model already mapped reference DNA sequence at CpG loci directly to empirical
cross-person CpG-by-CpG co-methylation or correlation geometry?

Searches combined DNA sequence, CpG, co-methylation, covariance/correlation matrix, graph
prediction, methylation-correlated blocks, deep learning, foundation model, and epigenetic age.
Sources included PubMed/PMC, publisher pages, arXiv, and OpenReview. This is a dated best-effort
search, not proof of absence.

## Conclusion

No exact prior implementation was found. The nearest work uses one of the two ingredients while
targeting a different object:

- **Observed co-methylation discovery.** MCBs, correlated methylation units, and coMethDMR infer
  regions or graphs from methylation correlations already measured in a cohort. They do not place
  unseen loci from sequence.
- **Per-site methylation prediction.** DeepCpG and related sequence models predict an
  individual's or cell's methylation state, often with neighboring observed methylation. Their
  target is not a locus-by-locus population covariance map.
- **Mechanistic local covariance.** Stochastic and mean-field methylation models derive covariance
  among nearby CpGs under specified dynamics; they do not learn the empirical cross-person matrix
  from foundation-model sequence features.
- **Age models with observed graphs.** RelAge-GNN and related graph models use observed
  co-methylation, genomic proximity, gene membership, and individual methylation to predict an
  individual's age. The prediction unit and direction are different.
- **Multimodal methylation-profile prediction.** MethylProphet predicts methylation profiles from
  sequence and expression. It does not target cross-person locus covariance.

## Closest 2026 preprint

[*Bridging Sequence and Graph Structure for Epigenetic Age
Prediction*](https://arxiv.org/abs/2605.10541) combines eight sequence statistics with a graph
whose edges include observed co-methylation, genomic proximity, and gene membership, then predicts
individual age from individual methylation. It does not predict held-out locus-locus correlations
from sequence alone. Its report that hand-crafted sequence statistics can be competitive makes
the registered CpG-density/GC baseline especially important.

RelAge-GNN
([arXiv:2605.07175](https://arxiv.org/abs/2605.07175)) is another close age/graph neighbor, but its
graphs are inputs built from observed methylation and annotations rather than sequence-predicted
covariance.

## Key neighboring references

- Gunasekara et al. (2023), [a genome-wide map of correlated DNA
  methylation](https://doi.org/10.1101/gr.276547.122).
- Gomez et al. (2019), [coMethDMR](https://academic.oup.com/nar/article/47/17/e98/5530673).
- Angermueller et al. (2017), [DeepCpG](https://doi.org/10.1186/s13059-017-1189-z).
- Zhang et al. (2015), [sequence-based methylation-state
  prediction](https://pubmed.ncbi.nlm.nih.gov/25616342/).
- Affinito et al. (2020), [distance and nearby CpG
  co-methylation](https://doi.org/10.1016/j.ygeno.2018.05.007).
- Lövkvist et al. (2022), [mean-field neighboring-site
  covariance](https://pubmed.ncbi.nlm.nih.gov/35078341/).
- MethylProphet (2025/2026), [profile prediction from sequence and
  expression](https://pmc.ncbi.nlm.nih.gov/articles/PMC11839017/).
- Schiff et al. (2024), [Caduceus](https://arxiv.org/abs/2403.03234).

## Cautious novelty statement

The working delta is: **out-of-locus prediction of empirical population co-methylation geometry
and age-association from frozen reference-sequence embeddings, evaluated on sequence-buffered
held-out loci.**

Repeat the search immediately before any paper submission or unqualified novelty claim.
