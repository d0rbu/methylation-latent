# Configuration

Tooling lives in `pyproject.toml`; scientific choices and external identities live in
`configs/protocol-v2.toml`. The protocol schema is strict and unknown fields are errors.

## Python tooling

```bash
uv sync --locked
uv run ruff check .
uv run ty check src tests scripts
uv run pytest
uv run pre-commit run --all-files
```

Python 3.13 owns parsing, targets, splitting, training, evaluation, and site compilation.
`environments/caduceus` pins the Python 3.11 native stack. `environments/methylprep` exists only
for parity reproduction. `environments/sesame` pins the primary R container and lock.

Project code uses Torch as:

```python
import torch as t
```

NumPy is not a direct dependency.

## Protocol v2 sections

`data` pins GSE87571 source bytes, raw inventory/order, phenotype sources, manifest, build, and
reference archive.

`reference_audit` pins exhaustive coordinate/window counts and the common ordered-locus
fingerprint.

`qc` pins pOOBAH/sample thresholds, seSAMe container and package identities, and the failed
methylprep parity record.

`masks` pins immutable Chen/Zhou URLs and bytes.

`model` pins Caduceus repository, revision, checkpoint, width, precision, shard size, and the four
window-specific batch sizes.

`splits` pins the window grid, block width, context anchors, chromosome 7, and primary/validation
seeds.

`targets` pins the deterministic correlation audit.

`training` pins:

- batch size 512 and 1,048,576-base neighborhood;
- Adam, learning rate 0.001, and weight decay 0;
- 2,000 tuning steps and validation every 100;
- validation pair chunking and training seed;
- separate age/full checkpoint rules and complete-train refit.

`sweep` pins dimensions `{16,32,64,128,256}`, lambdas `{0.1,1,10}`, evaluation pair caps/seeds,
and 16,384-base projection window.

## Scientific changes

The protocol status is `frozen`. Do not edit v2 in place after evaluation begins. A changed source,
threshold, split, seed, grid, optimizer, selection score, feature map, or metric requires a new
protocol ID and artifact root.

## Post-hoc protein extension

`configs/protein-extension-v1.toml` is a separate frozen protocol. It does not modify v2. It pins
the outcome-selected public protein panel audit, 52 unique HGNC/UniProt mappings, 651-subject
complete-case expectation, GRCh37 Ensembl lookup bytes, Caduceus and ESM-2 checkpoints, target-blind
protein split, representation sweep, separately averaged objective, and refit rule. Changing any
of those choices requires a new protein protocol and output root.
