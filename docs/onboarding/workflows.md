# Workflows

## Change a scientific choice

1. Create a new strict protocol schema/ID.
2. Pin every new source byte, model revision, seed, grid, and selection rule.
3. Generate new artifacts under a new root; never overwrite v2 artifacts.
4. Re-run the complete validation ladder before comparing results.

Changing a threshold, split, window, feature extraction rule, pair sampler, checkpoint rule, or
test metric after seeing test results is not a v2 continuation.

## Add or replace an external source

1. Record immutable URL/revision and cryptographic hash.
2. Parse an exact schema and reject duplicates or unexpected rows.
3. Preserve original source and every overlapping exclusion reason.
4. Add malformed-schema, count, order, and hash tests.
5. Create a new protocol identity.

## Resume an embedding stage

1. Confirm the process is no longer running and the GPU has sufficient free memory.
2. Re-run the identical command.
3. The loader validates every completed shard against probe order, window, and range.
4. Missing shards are generated; a finalized cache is never modified.

Do not delete a shard merely because a run stopped. Invalid shards fail validation and require an
explicit forensic decision.

## Resume training

Re-run the identical `age`, `full`, or `evaluate` stage. Completed immutable runs are reloaded and
validated. Missing candidates continue in deterministic grid order. A partial temporary directory
is not a completed run and must never be renamed by hand.

## Change split logic

1. Keep the API limited to locus metadata and seeded Torch RNG state.
2. Add a brute-force interval oracle for the changed behavior.
3. Test primary and nested-validation buffers at the maximum window.
4. Test exact non-overlap at every configured window.
5. Regenerate split, pair, model, and evaluation artifacts under a new identity.

## Publish results

1. Require the sealed data bundle, both baseline families, final models, pair caches, and all
   evaluation records.
2. Compile the static site from artifacts, never log text.
3. Verify protocol IDs, split labels, hashes, panel completeness, and local HTTP content.
4. Start the Cloudflare tunnel only after the local origin is stable.
5. Re-fetch the public URL and verify the expected protocol marker.

## Before handoff

```bash
uv run pre-commit run --all-files
git status --short
```

Report any long-running process with its PID, output directory, and verified restart contract.
