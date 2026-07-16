# Workflows

## Add a Domain Type

1. Add the phantom type near the code that owns the domain concept.
2. Add a `parse_*` refinement function for untrusted inputs.
3. Use the refined type in public functions and dataclasses.
4. Add `st.from_type(...)` property tests when the type has a Hypothesis strategy.
5. Update `docs/development/correctness.md` if the pattern is new.

## Add an Experiment Helper

1. Put reusable code in the project module or package once one exists.
2. Keep script-only orchestration out of core logic.
3. Validate raw inputs at the boundary.
4. Return typed values, dataclasses, or explicit result objects.
5. Cover invariants with unit tests and property tests.

## Add or replace an external source

1. Add the immutable URL/revision and cryptographic hash to the protocol configuration.
2. Parse the source with an exact schema and count contract.
3. Preserve the original source name and overlapping exclusion reasons in the ledger.
4. Add malformed-schema, duplicate-ID, hash, and count tests.
5. Changing reviewed source bytes requires a new protocol ID.

## Run an experiment stage

1. Verify the protocol is frozen; committed v1 remains `draft` while IDATs and the reviewed
   optimization/checkpoint schedule are absent.
2. Verify every upstream payload against its adjacent metadata.
3. Confirm data/split/window/probe-order fingerprints match exactly.
4. Run stages in the order in `docs/pipelines/experiment-lifecycle.md`.
5. Publish the new payload and canonical metadata exclusively; never overwrite an artifact.
6. Register full-model results only after both matching baselines exist.

## Change split logic

1. Keep the API limited to locus metadata and Torch RNG state.
2. Add a brute-force interval oracle for the changed behavior.
3. Test the maximum buffer and each individual primary window.
4. Regenerate split artifacts under a new identity; old results remain attached to the old split.

## Before Handoff

```bash
uv run pre-commit run --all-files
```

If a check is intentionally skipped, document the reason in the handoff.
