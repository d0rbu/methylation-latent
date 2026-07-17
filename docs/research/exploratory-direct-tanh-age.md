# Direct tanh age baseline

Status: **post-hoc corrected baseline; not confirmatory on the existing test partitions**

The original age-only stage reused the shared-geometry model: a 256-to-`d` projection followed by
cosine similarity to a free `d`-dimensional age direction. That parameterization is meaningful for
the full model, where age deliberately occupies the learned probe geometry, but it is needlessly
indirect as an age-only sequence baseline.

The corrected baseline is exactly:

```text
rho_hat[i] = tanh(v @ embedding[i] + b)
loss       = mean_i (rho_hat[i] - rho[i])^2
```

It has one scalar output, one 256-element weight vector, one bias, no latent dimension, and no
lambda. A lambda multiplying its only loss would merely rescale the complete objective. `tanh`, not
sigmoid, enforces the target interval `[-1,1]` while permitting negative correlations.

For comparability, tuning uses the frozen batch sampler, Adam learning rate, seed, 2,000-step
budget, and 100-step validation interval. The minimum exact validation MSE selects a step; the head
is then reinitialized and refit for that many steps on the complete primary-train partition. Model
parameters start at zero, weight decay is zero, embeddings remain cached, and the producer forces a
single-thread deterministic CPU runtime.

This correction was specified after the original cosine age-only test metrics were viewed. The
mechanical split remains intact—no test target selects a step—but the same held-out results cannot
be called confirmatory. The original record is preserved rather than overwritten, and confirmation
requires a new sealed holdout or independent cohort.
