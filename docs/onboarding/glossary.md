# Glossary

| Term | Meaning |
|---|---|
| Phantom type | A runtime primitive narrowed by a predicate and represented as a richer static type after validation. |
| Refinement function | A function such as `parse_probability` that validates raw input and returns a phantom type. |
| Runtime boundary | A place where untrusted values enter the system, such as CLI args, config files, data files, model outputs, or public APIs. |
| Property test | A Hypothesis test that checks an invariant across many generated examples. |
| Array contract | A dtype and shape expectation expressed with `jaxtyping` and enforced with `beartype`. |
