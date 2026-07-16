# Architecture

The package separates untrusted external data, validated scientific state, and experiment
orchestration.

```text
GEO metadata + IDAT exports + GPL13534 + exclusion lists + hg19 FASTA
                              |
                              v
          validated samples, probes, beta matrix, and sequences
                              |
                 +------------+-------------+
                 |                          |
                 v                          v
        X and rho target artifact     split artifacts
                 |                          |
                 +-------------+------------+
                               |
                               v
                  frozen Caduceus embeddings
                               |
                 +-------------+-------------+
                 |             |             |
                 v             v             v
           sequence baseline  age-only    full metric
                 |             |             |
                 +-------------+-------------+
                               |
                               v
                  typed evaluation records
                               |
                               v
                       static results site
```

## Modules

| Module | Responsibility |
|---|---|
| `domain` | Phantom scalar types, probe/context records, and non-empty collections |
| `genome` | GPL coordinate conversion, indexed FASTA access, plus-strand window audits, sequence features |
| `targets` | Standardization, semantic tensor wrappers, Gram blocks, age targets, rank ceilings |
| `splits` | Target-blind diverse blocks, chromosome holdout, buffers, overlap assertions |
| `batching` | Reproducible genomic-window training batches from an allowed probe universe |
| `model` | Linear latent metric, independent age vector, loss, and zero-decay optimizer |
| `evaluation` | Pair classes, distance bins, distance-only reference, and metrics |
| `artifacts` | JSON/safetensors schemas, hashes, eligibility, and prerequisite matching |
| `site` | Static HTML generation from validated result artifacts |
| `cli` | Thin orchestration over reusable typed functions |

## Dependency direction

Core math never imports GEO, FASTA, Transformers, plotting, or methylprep. Data adapters refine
external values into domain records; model and evaluation code consume only those records and
tensors. Caduceus and methylprep run in preprocessing stages and are absent from the training loop.
The pinned Python 3.11 Caduceus environment imports the shared `embeddings` module through
`PYTHONPATH=src`; no second model-output interpretation is maintained.

## Artifact storage

Large artifacts live under a user-selected output root, not in Git. Tensor payloads use
safetensors; metadata and results use strict JSON schemas. Pickle is not a project artifact format.
