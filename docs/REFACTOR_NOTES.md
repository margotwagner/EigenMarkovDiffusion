# Refactor notes and original-code map

This repository was refactored from the core ideas and implementations in
[`margotwagner/diffusion-model`](https://github.com/margotwagner/diffusion-model).

| Original file | New location | Change |
|---|---|---|
| `src/models/EigenmarkovDiffusion.py` | `src/eigendiffusion/eigenmarkov.py` | Separated mathematics from plotting and I/O; uses `eigh`; makes mode truncation explicit; uses vectorized binomial transitions; preserves the zero mode exactly; removes hidden clipping. |
| `src/models/RandomWalk.py` | `src/eigendiffusion/baselines.py` | Preserves an original-style per-particle trajectory implementation and adds an exact node-level multinomial implementation with the same physical transition rule. |
| deterministic diffusion logic in model/notebook code | `src/eigendiffusion/baselines.py` | Provides a compact exact mean solution using the same operator/eigenbasis. |
| `src/utils/EMERunMultiruns.py` and `RWRunMultiruns.py` | `src/eigendiffusion/ensemble.py` | One shared, reproducible ensemble interface using `SeedSequence`; random-walk method is selected explicitly. |
| plotting and analysis utility files | `metrics.py`, `plotting.py`, and `cli.py` | Split metrics, plotting, and experiment orchestration into focused modules. |
| no dedicated random-walk benchmark | `benchmarking.py` and `benchmark-random-walk` CLI | Adds repeatable runtime and core-array memory comparisons between naive particle trajectories and multinomial node transitions. |
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
- The multinomial random walk is not presented as a different physical model.
  It is an exact aggregation of independent left/stay/right particle outcomes
  at each node and exists to separate implementation efficiency from modal
  model efficiency.
