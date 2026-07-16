# Correctness

The template bias is to make bad state unrepresentable.

## Phantom Types

Use `phantom-types` when a primitive type is too broad for a domain concept.

Examples in this repo:

- `Probability`: `float` in `[0, 1]`, demonstrated in `tests/test_correctness_tools.py`

Pattern:

1. Define the phantom type near the code that owns the domain concept.
2. Add a `parse_*` function that refines raw values.
3. Store only refined values in dataclasses and core APIs.
4. Use `st.from_type(YourType)` in property tests when a strategy exists.

## Runtime Checks

Use `beartype` at runtime boundaries and on small public functions where type violations
would otherwise become confusing downstream failures.

Do not decorate every private helper reflexively. Prefer validation at boundaries and
around domain invariants.

## Array Contracts

Use `jaxtyping` for NumPy, JAX, PyTorch, or other array-like values when shape and dtype
matter. Pair it with `beartype`:

```python
from beartype import beartype
from jaxtyping import Float64, jaxtyped

Vector = Float64[np.ndarray, "n"]

@jaxtyped(typechecker=beartype)
def normalize_weights(weights: Vector) -> Vector:
    ...
```

## Property Tests

Use Hypothesis for:

- normalization and conservation laws
- parser and serializer round trips
- shape-preserving transformations
- monotonicity and ordering invariants
- edge cases that are easy to miss with example tests

Keep generated examples bounded so the default test suite stays fast.
