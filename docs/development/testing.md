# Testing

## Default Suite

```bash
uv run pytest
```

The default suite includes unit tests, property tests, and coverage.

## Pre-Commit

Install hooks once per clone:

```bash
uv run pre-commit install
```

Run the full configured gate manually:

```bash
uv run pre-commit run --all-files
```

The local hooks run:

- `uv lock --check`
- `uv run ruff check .`
- `uv run ty check`
- `uv run pytest`

## Focused Runs

```bash
uv run pytest tests/test_correctness_tools.py
uv run pytest tests/test_correctness_tools.py -k weighted_mean
uv run pytest -m property
```

## Coverage

Coverage is configured in `pyproject.toml` and currently fails below 95%.

Use coverage as a guardrail, not a substitute for meaningful assertions. The most useful
tests in this template check invariants: probabilities stay in range, weights normalize
to one, and invalid primitive values are rejected before they enter core code.
