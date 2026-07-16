# Getting Started

## Prerequisites

- Python 3.13
- `uv`

Install `uv` if needed:

```bash
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup

```bash
uv sync
uv run pre-commit install
uv run pytest
```

The default Python 3.13 environment owns typed parsing, target construction, splitting, training,
evaluation, and the static site. Raw-IDAT preprocessing and Caduceus native kernels have separate
lockfiles. The Caduceus environment pins Python 3.11 but imports the same embedding implementation
from `src/`; it is not a duplicate adapter:

```bash
uv sync --project environments/methylprep --locked
uv sync --project environments/caduceus --locked
```

Invoke that shared implementation with `PYTHONPATH=src` under the isolated environment. For
example, the integration smoke in the embedding pipeline document uses this exact path.

Do not use either isolated environment until its documented platform prerequisites are satisfied.
In particular, the reviewed Caduceus stack requires a compatible NVIDIA/CUDA build.

## Daily Commands

```bash
uv run ruff check .
uv run ty check
uv run pytest
uv run pre-commit run --all-files
```

Use `uv add <package>` for runtime dependencies and `uv add --dev <package>` for
development-only tooling.

## Confirm the protocol and public inputs

```bash
uv run methylation-latent check-config --config configs/protocol-v1.toml

uv run methylation-latent audit-public-inputs \
  --config configs/protocol-v1.toml \
  --manifest /data/GPL13534_HumanMethylation450_15017482_v.1.1.csv.gz \
  --series-matrix /data/GSE40279_series_matrix.txt.gz \
  --sample-key /data/GSE40279_sample_key.txt.gz

uv run methylation-latent audit-exclusion-lists \
  --config configs/protocol-v1.toml \
  --manifest /data/GPL13534_HumanMethylation450_15017482_v.1.1.csv.gz \
  --chen /data/48639-non-specific-probes-Illumina450k.csv \
  --zhou /data/hm450.hg19.manifest.tsv.gz
```

These commands validate only the named public files. They do not satisfy the raw-IDAT gate.

After decompressing and indexing the byte-pinned UCSC reference, exhaustively audit every retained
manifest coordinate and 64 kb window:

```bash
uv run methylation-latent audit-reference \
  --config configs/protocol-v1.toml \
  --manifest /data/GPL13534_HumanMethylation450_15017482_v.1.1.csv.gz \
  --chen /data/48639-non-specific-probes-Illumina450k.csv \
  --zhou /data/hm450.hg19.manifest.tsv.gz \
  --reference-archive /data/hg19.fa.gz \
  --fasta /data/hg19.fa \
  --fasta-index /data/hg19.fa.fai
```

The command checks archive MD5, exact decompressed bytes, all one-based plus-strand CpG
coordinates, maximum-window boundaries/alphabet, and the final eligible probe-order hash.
