# Post-hoc latent UMAP visualization

## Scope

This visualization describes how the full model's normalized latent geometry changes over
`d = 16, 32, 64, 128` at `lambda = 0.1` for the completed 1 kb and 4 kb windows. It uses nested
validation loci only and is not a model-selection metric.

Every panel within a split uses the same 300 target-blind validation probes, sampled without
replacement and balanced to 75 island, shore, shelf, and open-sea loci. The normalized learned age
direction is appended as a 301st point before graph construction and is displayed as a star. No
test probe, empirical age target, or pair target enters the projection.

## Deterministic Torch implementation

The implementation follows UMAP's fuzzy-neighborhood construction while staying entirely in
Torch:

- exact Euclidean k-nearest neighbors on unit latent rows, equivalent to cosine ordering;
- locally scaled directed memberships with `n_neighbors = 15`, then fuzzy union;
- deterministic normalized-Laplacian spectral initialization;
- the fitted low-dimensional UMAP distance curve with `min_dist = 0.1` and `spread = 1`;
- exact full fuzzy Bernoulli cross-entropy over all pairs, without negative-sampling
  approximation;
- 750 deterministic CPU optimization steps.

Axis orientation and global distances are not identified and must not be compared as if they were
biological coordinates. The panels are for neighborhood inspection. The age star shows where the
learned age direction lies in each independently optimized graph; it is not a fitted 2D regression
coefficient.

## Reproduction

```bash
uv run python scripts/run_latent_umap.py \
  --config configs/protocol-v2.toml \
  --data "$RUN_ROOT/data" \
  --embeddings "$RUN_ROOT/embeddings" \
  --experiments "$RUN_ROOT/experiments" \
  --output "$RUN_ROOT/exploratory/validation-latent-umap-v1" \
  --windows 1024 4096 \
  --dimensions 16 32 64 128
```

The output records the exact sample, tuning checkpoint, embedding manifest, graph diagnostics,
optimization diagnostics, and coordinates. An existing file must match the canonical rerun exactly.

## Method references

- McInnes, Healy, and Melville, *UMAP: Uniform Manifold Approximation and Projection for Dimension
  Reduction*, 2018: <https://arxiv.org/abs/1802.03426>
- UMAP documentation, *How UMAP Works*: <https://umap-learn.readthedocs.io/en/latest/how_umap_works.html>
