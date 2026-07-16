# Experiment Lifecycle

Use this as the default lifecycle for research work built from the template.

## 1. Define the Question

Write down the hypothesis, metric, and expected failure modes before adding code.

## 2. Make State Explicit

Represent raw config as validated dataclasses. Use phantom types for values that have
domain bounds such as probabilities, positive counts, feature IDs, seeds, and split
fractions.

## 3. Build Small Reusable Units

Keep reusable logic in a real module once the project has source code. Keep one-off
orchestration in scripts or notebooks that call reusable code.

## 4. Test Invariants

Add example tests for known cases and property tests for broad invariants.

## 5. Record Outputs

Keep generated artifacts out of git by default. Put durable notes in docs or experiment
reports, and make artifact paths explicit.
