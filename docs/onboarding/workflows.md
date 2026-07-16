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

## Before Handoff

```bash
uv run pre-commit run --all-files
```

If a check is intentionally skipped, document the reason in the handoff.
