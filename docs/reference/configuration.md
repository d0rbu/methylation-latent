# Configuration

All tool configuration lives in `pyproject.toml`.

## Package Management

Use `uv`.

```bash
uv sync
uv add numpy
uv add --dev pytest
```

## Linting

`ruff` is configured for Python 3.13 with common correctness-oriented rule families:

- `E`, `F`, `W`
- `I`
- `UP`
- `B`
- `C4`
- `SIM`
- `RET`

Run:

```bash
uv run ruff check .
```

## Pre-Commit

`pre-commit` uses local hooks that invoke the locked `uv` environment.

Install:

```bash
uv run pre-commit install
```

Run:

```bash
uv run pre-commit run --all-files
```

Configured hooks:

- `uv lock --check`
- `uv run ruff check .`
- `uv run ty check`
- `uv run pytest`

## Type Checking

`ty` is configured for Python 3.13.

Run:

```bash
uv run ty check
```

## Testing

`pytest` collects from `tests/`, runs with strict config and strict markers, and reports
coverage for the scaffold tests. When the project adds real source modules, update
`tool.coverage.run.source` and the `--cov` target.

Run:

```bash
uv run pytest
```

Markers:

- `property`: property-based tests powered by Hypothesis
- `slow`: useful but expensive tests excluded from ad hoc focused runs
