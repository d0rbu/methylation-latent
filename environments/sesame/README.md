# seSAMe reference environment

The reference processor runs in the official Bioconductor 3.23 image pinned by digest:

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

