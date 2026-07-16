# Sequence windows and Caduceus embeddings

## Input contract

All four experiments use the same 346,628 retained probes and one UCSC hg19 plus-strand
convention. For every probe and window:

- width is exactly 1,024, 4,096, 16,384, or 65,536 bases;
- the interval is full length and unpadded;
- the probe cytosine is at index `width / 2`;
- `sequence[centre:centre+2] == "CG"`;
- every character is A, C, G, or T;
- FASTA, manifest, and ordered-probe hashes match the sealed data bundle.

Eligibility was decided using the maximum window before methylation QC, so window comparisons do
not silently use different locus universes.

## Model identity

- repository: `kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16`;
- revision: `d89eeb853136ea64da7feb3d0c8e909771b17ae6`;
- checkpoint SHA-256:
  `a3e6976fe90460ff5d90d371b457d0263dc65e156ea5651b4a452e98228daef2`;
- 7.73 million parameters;
- configured `d_model = 256`;
- RCPS backbone output width 512, represented as two 256-channel orientation halves;
- pretraining context 131,072 bases.

Remote model code is loaded only at the pinned revision. Tokenization disables special tokens and
asserts one token per input base.

## Centre-token contract

For a batch of width `w`, the pinned model must return `[batch, w, 512]`. The cache stores

```text
hidden[:, w // 2, :256]
```

as fp16. This is the plus-strand 256-channel representation at the CpG cytosine. It is not mean
pooled, reverse-complement averaged, fine-tuned, or recomputed during training. Those are separate
ablations, not fallbacks.

## Runtime and benchmark

The native stack is isolated in `environments/caduceus` under Python 3.11. A real-checkpoint
benchmark on the experiment RTX 4090 established the frozen inference batch sizes:

| Window | Batch |
|---:|---:|
| 1,024 | 512 |
| 4,096 | 128 |
| 16,384 | 32 |
| 65,536 | 8 |

Observed throughput was roughly 476–535 thousand bases per second with about 7.65 GB of model-run
GPU memory. These numbers are capacity planning, not scientific results. Never run this benchmark
or the embedding job beside another GPU benchmark.

## Restart-safe cache

`environments/caduceus/embed.py` writes immutable 1,024-probe shard directories. Each shard stores:

- the exact clean Git commit of the embedding implementation;
- its exact half-open probe-index range;
- ordered probe IDs and probe-order fingerprint;
- window size and `[count,256]` fp16 tensor;
- payload hash and strict metadata schema.

On restart, existing shards are fully validated and only absent ranges are generated. Finalization
requires exact gap-free, duplicate-free coverage of all 346,628 probes, writes `manifest.json`,
reloads the full cache, and verifies the aggregate tensor. Training reads only this finalized
cache; it never imports Transformers or reads FASTA.

The entry point fails before loading the model unless the repository is clean. Restarting from
existing shards also requires their producer commit to equal the current clean commit; shards from
different implementations cannot be silently combined.

The exact commands are in
[`../onboarding/getting-started.md`](../onboarding/getting-started.md).
