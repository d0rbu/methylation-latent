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
`methylation-latent.interim-site-data.v1`. It displays:

- every full-model `d × lambda` candidate using nested-validation metrics only;
- frozen-test metrics only for the already validation-selected models;
- the preregistered age and distance-stratified panels for completed windows;
- post-hoc PSD distance integration under an explicit non-confirmatory label;
- an explicit pending state, rather than a substitute, for the 16-kb latent projection.

```bash
uv run python scripts/compile_interim_results_site.py \
  --config configs/protocol-v2.toml \
  --data "$RUN_ROOT/data" \
  --experiments "$RUN_ROOT/experiments" \
  --distance-integration "$RUN_ROOT/exploratory/distance-integration-v2" \
  --results-output "$RUN_ROOT/interim-site-data-v1" \
  --site-template interim-site-template \
  --root-template interim-site-root-template \
  --site-output "$RUN_ROOT/interim-site-v1" \
  --windows 1024 4096
```

Both output directories must be absent. The compiler verifies the sealed data and split identity,
all candidate hashes, the selected configuration, exact reproduction of primary sequence metrics,
and the deterministic exploratory runtime contract before writing any split page.

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
