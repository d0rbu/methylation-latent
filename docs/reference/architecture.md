# Architecture

External bytes are refined into sealed scientific state before model code can consume them.

```text
GSE87571 IDATs + phenotype + GPL13534 + masks + hg19
                           |
                           v
                 pinned seSAMe outputs
                           |
                           v
       cohort + target geometry + sequence features + splits
                           |
                           v
                sealed 17-file data bundle
                    /              \
                   v                v
        frozen evaluation pairs   Caduceus shard caches
                   \                /
                    v              v
             sequence -> age-only -> full metric
                              |
                              v
                  held-out evaluation records
                              |
                              v
                      split-specific sites
```

## Modules

| Module | Responsibility |
|---|---|
| `domain` | Phantom scalar types, loci, contexts, and non-empty probe sets |
| `metadata` | Exact GEO phenotype and Sentrix identity parsing |
| `manifest` | GPL13534 and byte-pinned Chen/Zhou exclusion parsing |
| `processor_io` | Strict seSAMe binary output loading and processor comparisons |
| `preprocessing` | Sample-first and complete-case detection filtering |
| `cohort` | Static universe, GSE87571 cohort, ledgers, and probe-table persistence |
| `genome` | Indexed hg19 access, coordinate conversion, windows, and sequence statistics |
| `targets` | Standardization, semantic tensors, Gram blocks, `rho`, and rank ceilings |
| `splits` | Target-blind block/chromosome splits, buffers, and overlap assertions |
| `experiment_data` | Nested split and sequence-feature artifacts |
| `data_bundle` | Closed-world 17-file bundle verification |
| `embeddings` | Pinned Caduceus loading and centre-token inference |
| `embedding_cache` | Restart-safe shard and finalized-cache contracts |
| `batching` | Target-blind local genomic training batches |
| `model` | Bias-free latent metric and independent age vector |
| `training` | Validation-selected age-only/full tuning and refit |
| `evaluation` | Pair populations, distance classes/reference, metrics, and PCA |
| `evaluation_cache` | Immutable pair indices and exact target values |
| `distance_integration` | Post-hoc PSD distance kernels and simplex-constrained mixtures |
| `artifacts` / `storage` | Canonical JSON, hashes, and exclusive safetensors |
| `site` | Strict site-data schema and static HTML generation |
| `config` / `cli` | Frozen protocol and thin public commands |

## Dependency direction

Core model/training code does not import GEO, FASTA, R, Transformers, or site code. seSAMe ends at
typed processor exports. Caduceus ends at finalized fp16 embedding caches. Training sees only probe
metadata, target tensors, split indices, and cached embeddings.

Torch is the only array library in project code and is imported as `torch as t`. NumPy is not a
project dependency.

## Orchestration

The four primary scripts are deliberately thin:

- `prepare_gse87571.py`: build an exclusive primary-data directory;
- `seal_primary_data.py`: independently validate and close that directory;
- `run_experiments.py`: execute restart-safe ordered stages;
- `compile_results_site.py`: transform complete evaluation records into two static sites.

`run_distance_integration.py` is a separately labeled post-hoc analysis. It consumes immutable
primary artifacts and writes only the exploratory distance-integration schema.

`compile_interim_results_site.py` accepts only a strict subset of the frozen window sweep. It emits
the separate interim-site schema, exposes every completed full-model candidate only through nested
validation metrics, and labels distance integration as post-hoc. It does not relax the primary
compiler's all-window or 16-kb-projection requirements.

`run_direct_tanh_age.py` produces the corrected direct-scalar age baseline in a separate post-hoc
artifact tree. It has no latent-dimension or lambda axes and cannot replace the original frozen
age-only record in place.

Large artifacts live outside Git. Git contains protocol, code, tests, templates, and compact
validated result summaries only.
