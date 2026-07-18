# Results site and Cloudflare tunnel

The site is a read-only static compilation of validated evaluation artifacts. It has no backend
and cannot mutate a run.

## Required panels

Each split-specific page contains:

- window-sweep curves for seen-by-held-out and held-out-by-held-out pair metrics;
- distance-class pair metrics with the training-fitted reference mean;
- age results for sequence, Caduceus age-only, and full models;
- a training-fitted 2D latent projection of every held-out probe, colored by context;
- protocol, split, retained counts, eligibility, and artifact identities.

The diverse-block and held-out-chromosome pages are separate. The compiler rejects absent windows,
partial age stages, missing pair populations, identity drift, or an incomplete projection.

## Interim site while the frozen sweep is incomplete

Do not weaken the primary compiler to publish partial runs. The separate interim compiler requires
an increasing strict subset of the four planned windows and writes
`methylation-latent.interim-site-data.v5`. It displays:

- every full-model `d × lambda` candidate using nested-validation metrics only;
- frozen-test metrics only for the already validation-selected models;
- the preregistered age and distance-stratified panels for completed windows;
- post-hoc PSD distance integration under an explicit non-confirmatory label;
- the post-hoc corrected direct scalar tanh age baseline;
- target-blind display samples of predicted versus empirical age and pair correlations, with the
  age probes switchable among six probe-level annotation colorings;
- a target-aware, post-hoc k-means audit of the two age-scatter lobes, including context,
  chromosome, probe-design, strand, sequence-feature, and sex-stratified diagnostics;
- deterministic exact-Torch UMAP constrained to an interactively rotatable unit two-sphere for the
  same context-balanced nested-validation loci plus the learned age direction at
  `d = 16, 32, 64, 128` and `lambda = 0.1`, with the same six color modes and explicit distortion
  diagnostics;
- an explicit pending state, rather than a substitute, for the 16-kb latent projection.

```bash
export DATA_ROOT=/absolute/path/to/methylation-latent-data
export RUN_ROOT="$DATA_ROOT/artifacts/gse87571-hg19-caduceus-ps-v2"

uv run python scripts/compile_interim_results_site.py \
  --config configs/protocol-v2.toml \
  --data "$RUN_ROOT/data" \
  --series-matrix "$DATA_ROOT/sources/GSE87571_series_matrix.txt.gz" \
  --experiments "$RUN_ROOT/experiments" \
  --embeddings "$RUN_ROOT/embeddings" \
  --distance-integration "$RUN_ROOT/exploratory/distance-integration-v2" \
  --direct-age "$RUN_ROOT/exploratory/direct-tanh-age-v1" \
  --prediction-scatter "$RUN_ROOT/exploratory/prediction-scatter-v1" \
  --age-cluster-audit "$RUN_ROOT/exploratory/age-scatter-cluster-audit-v1" \
  --latent-umap "$RUN_ROOT/exploratory/validation-latent-spherical-umap-v3" \
  --results-output "$RUN_ROOT/interim-site-data-v5" \
  --site-template interim-site-template \
  --root-template interim-site-root-template \
  --site-output "$RUN_ROOT/interim-site-v5" \
  --windows 1024 4096
```

Both output directories must be absent. The compiler verifies the sealed data and split identity,
all candidate hashes, the selected configuration, primary sequence-metric reproduction within an
explicit eight-ULP bound, direct-age/scatter/cluster source hashes, the complete UMAP grid and its
validation-only target-free sampling contract, unit-sphere input and output metrics, unit norms for
every three-coordinate output and the age-vector point, distortion diagnostics, exact
manifest/sequence-feature joins for every displayed probe, and the deterministic exploratory
runtime contract before writing any split page. The ULP allowance covers independently ordered
float64 reductions; counts remain exact.

### Add the post-hoc protein report

The protein extension is compiled as a linked page without weakening or rewriting the v5 site-data
schema. Start from an already validated, immutable interim site and write a new exclusive directory:

```bash
uv run python scripts/compile_protein_extension_report.py \
  --base-site "$RUN_ROOT/interim-site-v5" \
  --results "$RUN_ROOT/exploratory/protein-extension-v1/results/results.json" \
  --output "$RUN_ROOT/interim-site-v6"
```

The compiler requires all 20 records (two split families, two windows, five representations),
copies the validated base site, adds `protein-extension.html`, and links it from the existing
landing page. It never edits an existing site directory in place.

### Add the post-hoc dual-probe report

Run the frozen alpha sweep and selected-only evaluation as separate commands. The second command
cannot reconstruct selection from test metrics: it revalidates every tuning record and the
already-published validation-only selection before it may refit or read test targets.

```bash
uv run python scripts/run_dual_probe_latent.py \
  --config configs/dual-probe-latent-v1.toml \
  --parent-config configs/protocol-v2.toml \
  --data "$RUN_ROOT/data" \
  --embeddings "$RUN_ROOT/embeddings" \
  --experiments "$RUN_ROOT/experiments" \
  --output "$RUN_ROOT/exploratory/dual-probe-latent-v1" \
  --phase tune --device cuda \
  --splits diverse-blocks held-out-chromosome \
  --windows 1024 4096

uv run python scripts/run_dual_probe_latent.py \
  --config configs/dual-probe-latent-v1.toml \
  --parent-config configs/protocol-v2.toml \
  --data "$RUN_ROOT/data" \
  --embeddings "$RUN_ROOT/embeddings" \
  --experiments "$RUN_ROOT/experiments" \
  --output "$RUN_ROOT/exploratory/dual-probe-latent-v1" \
  --phase evaluate --device cuda \
  --splits diverse-blocks held-out-chromosome \
  --windows 1024 4096
```

Compile a new exclusive site directory from the currently validated site. This copies rather than
mutates the base site and adds the alpha sweep, selected-only pair and age results, target-blind
scatterplots, distance-reference curves, and initialization/alignment audits:

```bash
uv run python scripts/compile_dual_probe_report.py \
  --config configs/dual-probe-latent-v1.toml \
  --parent-config configs/protocol-v2.toml \
  --data "$RUN_ROOT/data" \
  --dual-results "$RUN_ROOT/exploratory/dual-probe-latent-v1" \
  --base-site "$RUN_ROOT/interim-site-v7" \
  --template dual-probe-site-template \
  --output "$RUN_ROOT/interim-site-v8"
```

The compiler hashes all four result records, all 84 tuning records, all 12 selected refit metadata
and model payloads, the frozen data bundle, and the dual protocol. It rejects any unselected test
record, learned validation/test row, incomplete seed/candidate axis, or display sample not marked
as target-blind.

## Compile

```bash
export DATA_ROOT=/absolute/path/to/methylation-latent-data
export RUN_ROOT="$DATA_ROOT/artifacts/gse87571-hg19-caduceus-ps-v2"

uv run python scripts/compile_results_site.py \
  --config configs/protocol-v2.toml \
  --data "$RUN_ROOT/data" \
  --experiments "$RUN_ROOT/experiments" \
  --results-output "$RUN_ROOT/site-data" \
  --site-template site-template \
  --root-template site-root-template \
  --site-output "$RUN_ROOT/site"
```

Both output directories must be absent. Exclusive generation prevents stale files from surviving
into a new site.

## Serve and verify locally

```bash
uv run python -m http.server 8080 --bind 127.0.0.1 \
  --directory "$RUN_ROOT/site"
```

From a second process:

```bash
curl --fail --silent http://127.0.0.1:8080/ | \
  grep --fixed-strings gse87571-hg19-caduceus-ps-v2
curl --fail --silent http://127.0.0.1:8080/diverse-blocks/
curl --fail --silent http://127.0.0.1:8080/held-out-chromosome/
```

Record the server PID and exact site directory.

## Temporary Cloudflare tunnel

The validated environment uses `cloudflared` 2026.7.2. After the local checks:

```bash
cloudflared tunnel --url http://127.0.0.1:8080
```

Capture the process ID, log, and generated `trycloudflare.com` URL. Fetch that URL from a separate
process and verify HTTP 200 plus the protocol marker. A quick-tunnel URL is ephemeral and must not
be described as a permanent deployment.

For an already running quick tunnel, keep its `cloudflared` process and public URL. Regenerate the
site into a new exclusive directory, validate it locally, then restart only the local HTTP origin
on the same address and port with the new directory. Do not open one quick tunnel per site refresh;
doing so creates multiple ephemeral URLs for the same report and makes provenance harder to track.

## Named tunnel

For a stable user-owned hostname, follow Cloudflare's
[locally managed tunnel documentation](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/get-started/create-local-tunnel/):

```bash
cloudflared tunnel login
cloudflared tunnel create methylation-latent
cloudflared tunnel route dns methylation-latent methylation-latent.example.org
```

Keep the tunnel UUID, account identifiers, credentials JSON, and real hostname out of Git. The
ingress service should point only at the verified local origin.

The tunnel confers no scientific validity; it merely exposes bytes already accepted by the site
compiler.
