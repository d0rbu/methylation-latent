# CLAUDE.md - methylation-latent

Follow [`AGENTS.md`](AGENTS.md). It is the source of truth for agent behavior.

Read the research protocol, correctness contract, preprocessing decision, and split protocol
before editing implementation. Do not weaken a fail-loud boundary to make incomplete data run.

Run the full local gate before handoff:

```bash
uv run pre-commit run --all-files
```

Report data-dependent gates separately from unit and property tests.
