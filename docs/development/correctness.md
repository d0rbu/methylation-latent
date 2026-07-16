# Correctness

The pipeline makes invalid scientific state unrepresentable where possible and otherwise fails
at the boundary where that state enters.

## Scalar domain types

Use `phantom-types` for bounded or structured primitives, including:

- beta values in `[0, 1]`;
- correlations in `[-1, 1]`;
- one-based genomic positions;
- even positive sequence-window widths;
- positive latent dimensions and batch sizes;
- non-negative loss weights.

Raw strings, integers, and floats are refined once. Core APIs receive refined types.

## Semantic tensor wrappers

`jaxtyping` describes tensor shape and dtype. It cannot by itself prove semantic properties, so
validated frozen wrappers own them:

- `UnitNormRows`: finite rank-2 tensor, non-empty axes, zero row means, unit row L2 norms;
- `UnitNormVector`: finite, non-empty, zero mean, unit L2 norm;
- `CorrelationVector`: finite entries bounded by one;
- `CorrelationMatrix`: square, symmetric, bounded, with unit diagonal.

Constructors validate; downstream code does not repeatedly guess whether a raw tensor satisfies
the contract.

## Correlation identity

For a raw beta row `b`, define

```text
c = b - mean(b)
x = c / ||c||_2
```

For any two nonconstant rows, Pearson correlation is exactly the normalized dot product in real
arithmetic:

```text
corr(b_i, b_j) = x_i @ x_j
```

Age is transformed by the identical operation. Production validation checks centering and norms
in float64, compares dot-product targets with `torch.corrcoef` on deterministic blocks, and stores
the tolerances and maximum observed errors. No clipping or missing-value imputation is allowed.

## Rank contract

With `n` retained samples, centering makes the all-ones vector a null direction, so
`rank(X) <= n - 1` and `rank(XX^T) <= n - 1`. For 656 samples, the target ceiling is 655.
The Caduceus embedding has width 256, which is a stricter model ceiling: dimensions above 256 do
not enlarge `M = W^T W`. Configuration rejects `d > min(256, n - 1)` and records both ceilings.

## Coordinates

GPL13534 `MAPINFO` is treated as a one-based coordinate of the plus-strand CpG cytosine. A window
of even width `w` converts it once:

```text
c0 = MAPINFO - 1
start0 = c0 - w / 2
end0 = start0 + w
centre_index = w / 2
```

The interval is zero-based and half-open. Extraction uses the reference plus strand regardless of
the assay probe's `Strand` column and asserts `sequence[centre_index:centre_index+2] == "CG"`.
Windows crossing contig bounds or containing non-ACGT bases are rejected from the common probe
universe; they are never padded.

## Runtime boundaries

Use `beartype` with `jaxtyping` at public tensor APIs and validate these untrusted boundaries:

- GEO metadata and sample-key tables;
- GPL13534 and exclusion-list rows;
- methylprep per-sample exports;
- FASTA index records and extracted sequence;
- TOML configuration;
- safetensors plus JSON artifact pairs;
- result-registry entries consumed by the site.

Do not decorate every private helper. Validate once at the boundary, then carry a strong type.

## No fallback policy

The following are errors, not alternative modes:

- missing IDAT/control data for a primary preprocessing run;
- a build label other than GRCh37/hg19;
- an unexpected manifest context or coordinate;
- a non-CG window centre;
- a constant probe or age vector;
- missing beta values after the final QC filter;
- overlapping train/test windows;
- absent prerequisite baseline artifacts;
- target or embedding fingerprints that differ across compared runs.

Processed-GEO input is a separately typed audit mode, not a fallback primary input.
