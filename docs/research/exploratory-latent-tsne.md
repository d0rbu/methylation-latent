# Post-hoc latent t-SNE visualization

Status: exploratory visualization added after the primary test metrics had already been
viewed. It is not a new test-set result and must not be used for model selection.

## Question and fixed comparison

The visualization asks how the full model's normalized latent geometry changes across
`d = 16, 32, 64, 128` for the already-trained `lambda_age = 0.1` tuning runs. It is
generated for both available sequence windows and both split definitions. Every panel
within a split uses the same 300 nested-validation loci and the same t-SNE
initialization, so changes across panels come from the learned latent geometry rather
than from changing which loci are displayed.

The plotted loci are selected without methylation or age targets. Sampling is balanced
exactly across island, shore, shelf, and open-sea metadata: 75 loci per context. The
test partition is not read for the projection.

## Projection contract

- Input: unit-normalized rows returned by `LatentMetric.latent` from the frozen tuning
  checkpoint for the requested `(split, window, d, lambda_age=0.1)` cell.
- Geometry: Euclidean distance between unit rows, which is equivalent to cosine
  distance because `||u-v||^2 = 2 - 2 cos(u,v)`.
- Algorithm: exact Student-t t-SNE implemented in Torch, with no NumPy or GPU path.
- Fixed settings: 300 points, perplexity 30, 60 probability-search steps, 1,000
  optimization steps, 250 early-exaggeration steps, exaggeration 12, learning rate 50,
  and a shared deterministic seed.
- Output: one provenance-bound JSON artifact per `(split, window, d)` cell, including
  source hashes, the global display-index hash, optimizer settings, final KL divergence,
  genomic metadata, and two-dimensional coordinates.

t-SNE axes, orientation, and global inter-cluster distances have no direct biological
meaning. Panels are descriptive views of local neighborhoods, not evidence that a
particular context is separable. The website therefore shows every dimension rather
than selecting the most visually appealing panel.

## Command

```console
uv run python scripts/run_latent_tsne.py \
  --config configs/gse87571-hg19-caduceus-ps-v2.yaml \
  --data "$ARTIFACT_ROOT/primary-data" \
  --embeddings "$ARTIFACT_ROOT/embeddings" \
  --experiments "$ARTIFACT_ROOT/experiments" \
  --output "$ARTIFACT_ROOT/exploratory/validation-latent-tsne-v1" \
  --windows 1024 4096 \
  --dimensions 16 32 64 128
```

An existing output is accepted only if a complete deterministic recomputation is
exactly equal to the stored JSON object. A mismatch crashes.
