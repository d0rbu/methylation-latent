# seSAMe primary environment

Protocol v2's primary processor runs in the official Bioconductor 3.23 image pinned by digest:

```text
bioconductor/bioconductor_docker@sha256:b10002b39efa30c3779ad839549806ebdbb29b3266f0d2428478b04426e55929
```

The lock records R 4.6.0, seSAMe 1.30.1, sesameData 1.30.0, and the complete
resolved dependency graph. Build it with:

```bash
docker build --tag methylation-latent-sesame:protocol-v2 environments/sesame
```

The experiment records the resulting local image digest as part of preprocessing
provenance. The ExperimentHub/seSAMe data cache is mounted separately from both
the image and the repository and is fingerprinted after `sesameDataCache()`.

`preprocess.R` accepts an exact two-column Sentrix/prefix table, an absent output
directory, and a positive worker count:

```bash
docker run --rm \
  --mount type=bind,src="$PWD",dst=/opt/methylation-latent,readonly \
  --mount type=bind,src="$DATA_ROOT",dst="$DATA_ROOT" \
  --mount type=bind,src="$SESAME_CACHE",dst=/root/.cache \
  methylation-latent-sesame:protocol-v2 \
  Rscript /opt/methylation-latent/environments/sesame/preprocess.R \
  "$DATA_ROOT/preprocessing/full/sesame-input.tsv" \
  "$DATA_ROOT/preprocessing/full/sesame-output-p005" \
  "$WORKERS"
```

The input prefixes remain absolute inside the container, so `$DATA_ROOT` is
mounted at the identical path. The script refuses an existing output path and
writes beta values, raw pOOBAH p-values, quality masks, probe/sample order, and
environment metadata exclusively.

The 12-array parity audit rejected methylprep as interchangeable. See
[`../../docs/pipelines/data-preprocessing.md`](../../docs/pipelines/data-preprocessing.md).
