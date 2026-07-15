# v0.4 implementation update

## Files changed

### `src/eigendiffusion/eigenmarkov.py`

- Keeps the original `EigenMarkovDiffusion` implementation unchanged.
- Adds `IndependentModalDiffusion(EigenMarkovDiffusion)` as the explicit name
  used by new comparisons.

### `src/eigendiffusion/correlated_modal.py`

Adds:

- `CorrelatedModalDiffusion`;
- `BankedCorrelatedModalDiffusion`;
- Gaussian nearest-neighbour transition noise with the exact conditional
  multinomial covariance;
- projection onto the nonnegative constant-mass simplex;
- compact result objects and bank diagnostics.

### `src/eigendiffusion/ensemble.py`

- Adds a model selector with the names:
  - `independent_modal`;
  - `correlated_modal`;
  - `banked_correlated_modal`.
- Keeps `run_eigenmarkov_ensemble` as a backward-compatible wrapper around
  `independent_modal`.
- Records compact bank L1 fractions for banked ensembles.

### `src/eigendiffusion/cli.py`

- Adds `--modal-model` to `validate` and `sweep-modes`.
- Defaults to `independent_modal` so old behavior is preserved.
- Adds `compare-modal-models` for one-command comparison of all variants.
- Saves model names and bank diagnostics in JSON/CSV/NPZ outputs.

### `src/eigendiffusion/plotting.py`

- Generalizes validation plots to any named modal model.
- Adds a shared model-comparison figure.

### `src/eigendiffusion/__init__.py`

Exports all three model classes, ensemble wrappers, and covariance/projection
helpers.

### Tests

Adds tests confirming:

- `independent_modal` is exactly identical to the original model for a fixed
  seed;
- correlated spatial noise is zero-sum and matches its target covariance;
- correlated modal dynamics conserve mass;
- banked outputs are nonnegative and mass conserving;
- the CLI defaults to `independent_modal` and accepts all new names.

## Important design constraint

The code does not silently replace or modify the original EigenMarkov model.
Every experimental result must name the modal formulation explicitly, and the
default remains `independent_modal`.
