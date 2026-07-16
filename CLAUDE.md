# CLAUDE.md - research-project-template

Follow [`AGENTS.md`](AGENTS.md) first. It is the source of truth for agent behavior in
this repository.

## Working style

- Read the relevant docs before changing code.
- Keep changes tightly scoped to the requested behavior.
- Prefer small, typed, tested functions over large scripts.
- Use phantom types and runtime validation to make invalid states hard to represent.
- Add or update property tests when changing invariants.
- Update docs when file purpose, commands, configuration, or workflow changes.

## Required checks

```bash
uv run pre-commit run --all-files
```

If a check cannot be run, say exactly why and what remains unverified.
