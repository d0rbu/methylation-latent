# Results site and Cloudflare tunnel

The site is static and generated only from validated result artifacts. It has no backend and no
ability to mutate experiments.

## Required panels

- window-width sweep for seen-by-held-out and held-out-by-held-out metrics;
- distance-binned pair metrics with the training-fitted `f(|delta position|)` reference;
- held-out age correlation next to the sequence and Caduceus age-only baselines;
- deterministic 2D latent projection colored by island/shore/shelf/open-sea context;
- protocol ID, artifact hashes, split family, retained counts, and eligibility label.

An empty registry produces a visible “no eligible runs” page, not demo numbers.
For validated data, the generator additionally requires the split family, retained probe/sample
counts, and exact data/split SHA-256 fingerprints. Every selected window must contain both pair
populations, all three age stages, and at least one distance stratum per population. Duplicate or
partial panel rows are rejected before publication.

## Generate and serve locally

```bash
uv run methylation-latent build-site \
  --registry /absolute/path/to/results/registry.json \
  --output /absolute/path/to/generated-site

uv run python -m http.server 8080 --directory /absolute/path/to/generated-site
```

Verify locally at `http://127.0.0.1:8080` before exposing it.

## Named Cloudflare tunnel

Install `cloudflared` from Cloudflare's signed package or release and record its version. Then:

```bash
cloudflared tunnel login
cloudflared tunnel create methylation-latent
cloudflared tunnel route dns methylation-latent methylation-latent.example.org
```

Use `~/.cloudflared/config.yml`:

```yaml
tunnel: <tunnel-uuid>
credentials-file: /home/<user>/.cloudflared/<tunnel-uuid>.json

ingress:
  - hostname: methylation-latent.example.org
    service: http://127.0.0.1:8080
  - service: http_status:404
```

Run:

```bash
cloudflared tunnel --config ~/.cloudflared/config.yml run methylation-latent
```

Do not commit credentials, account IDs, tunnel UUIDs, or real hostnames. For a temporary preview,
`cloudflared tunnel --url http://127.0.0.1:8080` is acceptable, but its random URL is not a durable
results endpoint. See Cloudflare's
[locally managed tunnel documentation](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/get-started/create-local-tunnel/).

## Publication check

Fetch the public URL from a separate process and assert status 200, content hash, protocol ID, and
eligibility badge. The tunnel does not make an audit-only run primary.
