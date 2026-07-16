# Architecture

This repository is a compact, package-free template for research code.

## Scaffold

The base repository intentionally starts without an importable source package. Add one
only when a research project has real reusable code.

| Module | Purpose |
|---|---|
| `tests/test_correctness_tools.py` | Executable examples for phantom types, runtime checks, array contracts, and property tests |

## Correctness Boundary

Raw values should be refined near the boundary where they enter the system. Core code
should receive domain types such as `Probability`, not broad primitive values.

Array-heavy code should use `jaxtyping` for shape and dtype expectations and ordinary
runtime checks for semantic constraints such as non-negativity or finite values.

## Tests

`tests/` contains example tests and property tests. The default suite is intentionally
fast enough to run before every handoff.
