# Refactor notes and original-code map

This repository was refactored from the core ideas and implementations in
[`margotwagner/diffusion-model`](https://github.com/margotwagner/diffusion-model).

| Original file | New location | Change |
|---|---|---|
| `src/models/EigenmarkovDiffusion.py` | `src/eigendiffusion/eigenmarkov.py` | Separated mathematics from plotting and I/O; uses `eigh`; makes mode truncation explicit; uses vectorized binomial transitions; preserves the zero mode exactly; removes hidden clipping. |
| `src/models/RandomWalk.py` | `src/eigendiffusion/baselines.py` | Replaced per-particle trajectory storage with equivalent node-level multinomial transitions. |
| deterministic diffusion logic in model/notebook code | `src/eigendiffusion/baselines.py` | Provides a compact exact mean solution using the same operator/eigenbasis. |
| `src/utils/EMERunMultiruns.py` and `RWRunMultiruns.py` | `src/eigendiffusion/ensemble.py` | One shared, reproducible ensemble interface using `SeedSequence`. |
| plotting and analysis utility files | `metrics.py`, `plotting.py`, and `cli.py` | Split metrics, plotting, and experiment orchestration into focused modules. |
| `notebooks/eme-scale-exp.ipynb` | CLI commands and `examples/` | Core experiments no longer depend on hidden notebook state. |

## Deliberate corrections

- The spatial spacing is consistently `length / (n_nodes - 1)` because both
  endpoints are included.
- The matrix is consistently named a positive-semidefinite diffusion operator,
  with dynamics `dn/dt = -A n`.
- Symmetric operators use `numpy.linalg.eigh`, which returns real orthonormal
  modes in sorted numerical form.
- The original `scaling_factor` is replaced by the explicit
  `modal_particle_weight`. Setting it to `2.0` corresponds to the value used in
  the original multi-run wrapper, but it is now documented as a modeling
  parameter rather than a post-hoc reconstruction factor.
- The old experimental `reflect` truncation rule is omitted because it changes
  the stochastic process without a clear physical justification.
