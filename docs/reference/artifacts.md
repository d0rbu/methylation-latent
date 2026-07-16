# Artifact schemas

Every artifact is an immutable payload plus canonical JSON metadata. Files are SHA-256 hashed after
write; downstream metadata names the upstream hashes it consumed.

## Common metadata

All records contain:

- schema name and integer version;
- artifact ID and creation time;
- protocol ID and Git commit;
- command and locked-environment fingerprint;
- payload path, byte size, and SHA-256;
- upstream artifact IDs and hashes;
- eligibility: `audit_only`, `primary_candidate`, or `primary_validated`;
- validation checks with explicit measured values and tolerances.

## Data artifacts

| Schema | Payload | Required facts |
|---|---|---|
| `sample_metadata.v1` | JSON | ordered GSM/Sentrix IDs, age, gender, ethnicity, source, plate |
| `probe_manifest.v1` | JSON/TSV | ordered autosomal cg IDs, hg19 cytosine coordinates, context, design, manifest hash |
| `exclusion_ledger.v1` | TSV | probe, reason, source, source version/hash; overlapping reasons retained |
| `beta_matrix.v1` | safetensors | `[probe, sample]`, float64, complete, bounded, raw-IDAT provenance |
| `target_geometry.v1` | safetensors | ordered `X` and `rho`, sample/probe hashes, centering/norm/correlation audits |
| `sequence_audit.v1` | JSON | reference/build hashes, all windows, centred-CG and alphabet counts |
| `split.v1` | JSON | target-blind inputs, anchors/test/train IDs, buffers, overlap checks |
| `embedding.v1` | safetensors | `[probe, 256]` fp16, window/model revision, probe-order hash |

## Run artifacts

| Schema | Purpose |
|---|---|
| `distance_baseline.v1` | training-only bin means/counts and trans mean |
| `sequence_age_baseline.v1` | fitted coefficients and held-out metrics for GC/CpG density |
| `caduceus_age_baseline.v1` | age-only checkpoint/config and held-out metrics |
| `latent_metric_run.v1` | full checkpoint/config, prerequisite baseline IDs, validation history |
| `evaluation_pairs.v1` | immutable seen/test and test/test pair indices plus distance classes |
| `evaluation.v1` | separate per-population/bin pair metrics and held-out age metrics |
| `latent_projection.v1` | training-fitted axes and projected held-out points/context |
| `results_registry.v1` | ordered references to validated baseline/full evaluation records |
| `site-data.v1` | complete selected-window panels plus split/data fingerprints and retained counts |

## Eligibility rules

- `processed_geo_audit` provenance can never exceed `audit_only`.
- `beta_matrix.v1` is `primary_candidate` only with verified IDAT/control provenance and completed
  detection/normalization/exclusion records.
- A target is no more eligible than its beta matrix.
- A split becomes candidate only after all window overlap checks pass.
- A full run must match both baseline upstream fingerprints exactly.
- An evaluation is `primary_validated` only if data, target, split, embedding, checkpoint, pair
  indices, and distance baseline are all candidate/validated and all checks pass.
- The site displays primary curves only from `primary_validated` evaluations.

Unknown fields and schema versions are errors. Artifacts are not upgraded implicitly.
