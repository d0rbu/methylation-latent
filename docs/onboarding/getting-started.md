# Getting started

## Environments

The typed analysis environment uses Python 3.13:

```bash
uv sync --locked
uv run pre-commit install
uv run pre-commit run --all-files
```

The pinned Caduceus native stack uses Python 3.11:

```bash
uv sync --project environments/caduceus --locked
```

`environments/methylprep` is retained only to reproduce the parity audit. It is not the primary
processor. Primary raw-IDAT preprocessing uses the pinned seSAMe container described in
[`../pipelines/data-preprocessing.md`](../pipelines/data-preprocessing.md).

## Verify the frozen protocol

```bash
uv run methylation-latent check-config --config configs/protocol-v2.toml
```

Unknown fields, source drift, an unfrozen status, illegal dimensions, or nonzero weight decay are
errors.

## Artifact root

Keep sources and large artifacts outside Git:

```bash
export PROJECT="$PWD"
export DATA_ROOT=/absolute/path/to/methylation-latent-data
export RUN_ROOT="$DATA_ROOT/artifacts/gse87571-hg19-caduceus-ps-v2"
export DATA="$RUN_ROOT/data"
export EMBEDDINGS="$RUN_ROOT/embeddings"
export EXPERIMENTS="$RUN_ROOT/experiments"
```

The repository never downloads a differently versioned source as a fallback. Populate
`$DATA_ROOT/sources` and preprocessing directories with the exact files named by
`configs/protocol-v2.toml`.

## Build and seal primary data

```bash
uv run python scripts/prepare_gse87571.py \
  --config configs/protocol-v2.toml \
  --series-matrix "$DATA_ROOT/sources/GSE87571_series_matrix.txt.gz" \
  --manifest "$DATA_ROOT/sources/GPL13534_HumanMethylation450_15017482_v.1.1.csv.gz" \
  --chen "$DATA_ROOT/sources/48639-non-specific-probes-Illumina450k.csv" \
  --zhou "$DATA_ROOT/sources/hm450.hg19.manifest.tsv.gz" \
  --reference-archive "$DATA_ROOT/sources/hg19.fa.gz" \
  --fasta "$DATA_ROOT/sources/hg19.fa" \
  --fasta-index "$DATA_ROOT/sources/hg19.fa.fai" \
  --raw-inventory "$DATA_ROOT/preprocessing/raw-inventory.json" \
  --parity-audit "$DATA_ROOT/preprocessing/parity/audit.json" \
  --additional-characteristics "$DATA_ROOT/sources/GSE87571_additional_sample_characteristics.xlsx" \
  --filelist "$DATA_ROOT/sources/GSE87571_filelist.txt" \
  --sesame-p001 "$DATA_ROOT/preprocessing/full/sesame-output-p001" \
  --sesame-p005 "$DATA_ROOT/preprocessing/full/sesame-output-p005" \
  --output "$DATA"

uv run python scripts/seal_primary_data.py \
  --config configs/protocol-v2.toml \
  --data "$DATA"
```

The sealer recomputes target and split invariants before writing `bundle.json`. It refuses an
existing bundle or any unrecognized file. Run it, the embedding entry point, the experiment
runner, and the site compiler only from a committed clean worktree: each command fails loudly on
tracked or untracked changes and records the exact producer commit.

## Freeze baseline and pair artifacts

```bash
uv run python scripts/run_experiments.py \
  --stage sequence --config configs/protocol-v2.toml \
  --data "$DATA" --embeddings "$EMBEDDINGS" \
  --output "$EXPERIMENTS" --device cuda

uv run python scripts/run_experiments.py \
  --stage pairs --config configs/protocol-v2.toml \
  --data "$DATA" --embeddings "$EMBEDDINGS" \
  --output "$EXPERIMENTS" --device cuda
```

## Precompute embeddings

Run the four commands sequentially. The protocol-pinned batch sizes correspond to increasing
window widths:

```bash
while read -r window batch; do
  PYTHONPATH=src environments/caduceus/.venv/bin/python \
    environments/caduceus/embed.py \
    --probes "$DATA/probes.tsv" \
    --fasta "$DATA_ROOT/sources/hg19.fa" \
    --model-cache "$HOME/.cache/huggingface/hub" \
    --output "$EMBEDDINGS/window-$window" \
    --window-size "$window" --batch-size "$batch" \
    --shard-size 1024 --device cuda
done <<'WINDOWS'
1024 512
4096 128
16384 32
65536 8
WINDOWS
```

Each command verifies existing shards and fills only missing ranges. A finalized manifest is
reloaded and hashed before the command exits.

## Train and evaluate

```bash
for stage in age full evaluate; do
  uv run python scripts/run_experiments.py \
    --stage "$stage" --config configs/protocol-v2.toml \
    --data "$DATA" --embeddings "$EMBEDDINGS" \
    --output "$EXPERIMENTS" --device cuda
done
```

Do not use `--stage all` to bypass observation of the ordered baseline gates. It is available for
clean automated reproduction only after all prerequisites and GPU capacity are confirmed.

Site compilation is documented in
[`../operations/results-site.md`](../operations/results-site.md).
