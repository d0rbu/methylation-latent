# Post-hoc spherical-geodesic latent UMAP visualization

## Scope

This visualization describes how the full model's normalized latent geometry changes over
`d = 16, 32, 64, 128` at `lambda = 0.1` for the completed 1 kb and 4 kb windows. It uses nested
validation loci only and is not a model-selection metric.

Every panel within a split uses the same 300 target-blind validation probes, sampled without
replacement and balanced to 75 island, shore, shelf, and open-sea loci. The normalized learned age
direction is appended as a 301st point before graph construction and is displayed as a star. No
test probe, empirical age target, or pair target enters the projection.

Each site card can recolor those same loci by genomic context, Infinium I/II chemistry, manifest
assay strand, chromosome, window-specific CpG density, or window-specific GC content. These are
probe-level inputs or manifest annotations. Sex and cell composition are sample-level properties,
so assigning either one to a probe would be a category error; they are not color modes.

## Deterministic Torch implementation

The model emits unit vectors on `S^(d-1)`. The high-dimensional distance between vectors `z_i` and
`z_j` is therefore the intrinsic great-circle distance

```text
delta_sphere(i,j) = acos(clamp(z_i dot z_j, -1, 1)).
```

This is a sphere-aware UMAP input metric, not merely Euclidean distance between unit vectors. The
two distances have the same neighbor ordering because chord distance is a monotone function of the
angle, but UMAP's local scale and fuzzy membership use distance magnitudes as well as ordering.
Consequently they need not produce the same graph weights. Unit row norms are asserted before the
geodesic is computed.

The implementation follows UMAP's fuzzy-neighborhood construction while staying entirely in Torch:

- exact intrinsic spherical-geodesic k-nearest neighbors on unit latent rows;
- locally scaled directed memberships with `n_neighbors = 15`, then fuzzy union;
- deterministic normalized-Laplacian spectral initialization;
- the fitted low-dimensional UMAP distance curve with `min_dist = 0.1` and `spread = 1`;
- exact full fuzzy Bernoulli cross-entropy over all pairs, without negative-sampling
  approximation;
- 750 deterministic CPU optimization steps.

Only the input-neighborhood geometry is sphere-intrinsic. UMAP still produces a flat two-dimensional
display. Axis orientation and global distances are not identified and must not be compared as if
they were biological coordinates. The panels are for neighborhood inspection. The age star shows
where the learned age direction lies in each independently optimized graph; it is not a fitted 2D
regression coefficient.

There are fully manifold-native analogues of PCA, including principal geodesic analysis and
principal nested spheres. Those methods summarize global axes or nested subspheres; they answer a
different question from UMAP's local-neighborhood display. Spherical geodesic UMAP is used here
because the requested visualization is neighborhood-oriented. A manifold-PCA analysis can be added
later as a separately specified ablation rather than silently changing the meaning of these panels.

## Reproduction

```bash
uv run python scripts/run_latent_umap.py \
  --config configs/protocol-v2.toml \
  --data "$RUN_ROOT/data" \
  --embeddings "$RUN_ROOT/embeddings" \
  --experiments "$RUN_ROOT/experiments" \
  --output "$RUN_ROOT/exploratory/validation-latent-spherical-umap-v2" \
  --windows 1024 4096 \
  --dimensions 16 32 64 128
```

The output records the exact sample, tuning checkpoint, embedding manifest, graph diagnostics,
optimization diagnostics, and coordinates. An existing file must match the canonical rerun exactly.

## Method references

- McInnes, Healy, and Melville, *UMAP: Uniform Manifold Approximation and Projection for Dimension
  Reduction*, 2018: <https://arxiv.org/abs/1802.03426>
- UMAP documentation, *Parameters* (including cosine/angular and precomputed input metrics):
  <https://umap-learn.readthedocs.io/en/latest/parameters.html>
- Fletcher et al., *Principal Geodesic Analysis for the Study of Nonlinear Statistics of Shape*:
  <https://proceedings.neurips.cc/paper/2013/hash/eb6fdc36b281b7d5eabf33396c2683a2-Abstract.html>
- Jung, Dryden, and Marron, *Analysis of Principal Nested Spheres*:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC3635703/>
