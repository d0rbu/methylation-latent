# Sequence windows and Caduceus embeddings

## Reference convention

GPL13534 is GRCh37/hg19. `MAPINFO` is interpreted as the one-based plus-strand cytosine coordinate
for the CpG locus, including probes whose assay `Strand` is `R`. Extraction always reads the hg19
plus strand; assay strand remains audit metadata only.

Before any model invocation, every retained locus/window must satisfy:

- full requested width without padding;
- centre index `width / 2`;
- `sequence[centre:centre+2] == "CG"`;
- all characters in `ACGT` after uppercasing;
- FASTA archive/build and manifest fingerprints match the protocol.

The maximum-window eligibility filter is applied once so every window experiment uses the same
probe universe.
`methylation-latent audit-reference` performs this check exhaustively on the byte-pinned reference;
unit tests alone do not satisfy the gate.

## Model identity

- repository: `kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16`
- pinned revision: `d89eeb853136ea64da7feb3d0c8e909771b17ae6`
- checkpoint SHA-256/ETag: `a3e6976fe90460ff5d90d371b457d0263dc65e156ea5651b4a452e98228daef2`
- checkpoint parameters: 7.73M
- configured channel width: 256
- raw RCPS hidden width: 512 (two 256-channel orientation halves)
- pretraining length: 131,072 bases

The pin is required because loading uses Hugging Face remote model code. `trust_remote_code=True`
without an immutable revision is forbidden.

The native stack is locked separately and exercises the shared implementation with a real
checkpoint, not a mock:

```bash
uv sync --project environments/caduceus --locked
PYTHONPATH=src uv run --project environments/caduceus --locked \
  python environments/caduceus/smoke.py \
  --cache-directory "$HOME/.cache/huggingface/hub"
```

Success requires the pinned snapshot name and a finite fp16 result of shape `[1, 256]`. This smoke
is an integration gate; it is not a throughput benchmark. Do not run it concurrently with a
profiling benchmark.

## Token and hidden-state contract

Caduceus's tokenizer appends a separator by default. Embedding explicitly sets
`add_special_tokens=False` and asserts `n_tokens == n_bases`; otherwise the artifact is rejected.
The pinned PS checkpoint has `rcps=true` and `d_model=256`. Its public final backbone state is
therefore `[batch, window, 512]`, not 256: the last axis contains two 256-channel
reverse-complement-equivariant halves. Loading asserts all three values. Following the protocol's
fixed plus-strand convention, select
`hidden[:, window // 2, :256]`, the plus-strand half at the cytosine token. The cached model input
is consequently exactly 256-wide.

The PS model is reverse-complement equivariant, but this protocol does not average the two halves
or augment strands. That would define a different feature map and is reserved for an explicit
ablation. The fixed plus-strand convention is deliberate and identical for every probe.

## Caching

Write one fp16 safetensors file per window width plus JSON metadata containing:

- ordered probe IDs and probe-order hash;
- window, centre index, build, reference hash, and manifest hash;
- model repository, revision, checkpoint hash, library versions, dtype, and device;
- batch size and complete/failed counts;
- sequence-audit fingerprint;
- tensor shape, dtype, finite check, and payload SHA-256.

Resume is allowed only through an explicit shard manifest. A shard is immutable and carries its
ordered probe slice; concatenation verifies exact, gap-free, duplicate-free coverage. Training
loads only the finalized embedding artifact and never imports Transformers or reads FASTA.

## Precision

Inference may use autocast fp16 and stores fp16. The linear metric trains in float32 after loading
embeddings. A deterministic subset is embedded in both float32 and fp16 before the full run; cosine
and downstream age-head deviations are recorded. Precision acceptance thresholds are frozen before
the test split is evaluated.
