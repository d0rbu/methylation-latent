# File Reference

## Top Level

| File | Purpose |
|---|---|
| `README.md` | Project summary, quickstart, and doc links |
| `AGENTS.md` | Agent entry point and repo conventions |
| `CLAUDE.md` | Claude-specific pointer to agent conventions |
| `pyproject.toml` | Package metadata and tool configuration |
| `uv.lock` | Locked dependency graph |
| `.pre-commit-config.yaml` | Local commit hooks for lockfile, lint, type, and test checks |
| `.python-version` | Python version for local tooling |
| `.gitignore` | Local artifacts excluded from git |
| `LICENSE` | MIT license |

## Tests

| File | Purpose |
|---|---|
| `tests/test_correctness_tools.py` | Phantom type, runtime check, array contract, and property-test examples |

## Docs

| Path | Purpose |
|---|---|
| `docs/README.md` | Documentation index |
| `docs/onboarding/` | Setup and day-to-day workflows |
| `docs/development/` | Correctness and testing guidance |
| `docs/pipelines/` | Research workflow templates |
| `docs/reference/` | Architecture, configuration, and file map |

## CI

| File | Purpose |
|---|---|
| `.github/workflows/ci.yml` | Runs `ruff`, `ty`, and `pytest` on pushes and pull requests |
