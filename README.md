# research-project-template

Correctness-first Python boilerplate for research projects.

The template is intentionally small, but the repo is set up like a real project: `uv`
for packaging, `ruff` for linting, `ty` for type checking, `pytest` plus coverage and
Hypothesis for tests, and a docs hierarchy that future contributors and AI agents can
use as the source of truth.

## 1-minute quickstart

```bash
git clone https://github.com/d0rbu/research-project-template.git
cd research-project-template
uv sync
uv run pre-commit install
uv run pre-commit run --all-files
```

## What this includes

| Area | Tooling |
|---|---|
| Package management | `uv`, `pyproject.toml`, `uv.lock` |
| Local commit checks | `pre-commit` |
| Linting | `ruff` |
| Type checking | `ty` |
| Tests | `pytest`, `pytest-cov`, `hypothesis` |
| Runtime contracts | `phantom-types`, `beartype` |
| Array shape/dtype checks | `jaxtyping` |
| Agent guidance | `AGENTS.md`, `CLAUDE.md` |

## Repo layout

```
tests/              pytest suite, including property tests
docs/               project documentation
.github/workflows/  CI checks
```

## Where to go next

| You want to... | Read |
|---|---|
| Start developing | [`docs/onboarding/getting-started.md`](docs/onboarding/getting-started.md) |
| Understand the correctness model | [`docs/development/correctness.md`](docs/development/correctness.md) |
| Add a new experiment | [`docs/pipelines/experiment-lifecycle.md`](docs/pipelines/experiment-lifecycle.md) |
| See tool configuration | [`docs/reference/configuration.md`](docs/reference/configuration.md) |
| Find a file's purpose | [`docs/reference/file-reference.md`](docs/reference/file-reference.md) |

## License

MIT. See [`LICENSE`](LICENSE).
