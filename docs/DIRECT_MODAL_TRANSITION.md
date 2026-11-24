# Direct retained-modal transition

## Purpose

The direct transition removes the node-space Gaussian sampling work from the
reduced model. For retained modal counts `a_t`, it prepares

```text
a[t + 1] = D a[t] + L[t] z[t]
L[t] L[t].T = C[t + 1] - D C[t] D
```

where `C[t]` is the exact multinomial count covariance projected into the
retained cosine basis. The reflecting random-walk transition and the basis
share eigenvectors, so this process has the exact retained mean, same-time
covariance, and cross-time covariance. It is a time-inhomogeneous Gaussian
moment model, not an exact integer multinomial sampler.

`DirectModalGaussianDiffusion` batches the normal draws and innovation-factor
products, then applies only the sequential diagonal state recurrence. It can
return the modal history alone or reconstruct the full field at every saved
time.

## Direct-transition screen

The November 23, 2026 screen used 5 setup repeats, 30 balanced trajectory
repeats, one warmup, and one BLAS thread. Both fixed-step and fixed-duration
regimes were run at `N=101,201` with `M=20,35,50,75`. The table below shows the
fixed-duration workload; the fixed-step result leads to the same conclusion.

| Nodes | Modes | Setup (ms) | Raw full field (ms) | Raw / exact | Raw + simplex (ms) | Exact vectorized (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 101 | 20 | 19.14 | 0.471 | 0.069 | 6.872 | 6.807 |
| 101 | 35 | 34.58 | 0.537 | 0.080 | 6.825 | 6.707 |
| 101 | 50 | 47.85 | 0.631 | 0.094 | 6.959 | 6.727 |
| 101 | 75 | 92.11 | 0.874 | 0.123 | 7.507 | 7.124 |
| 201 | 20 | 51.03 | 1.097 | 0.062 | 17.626 | 17.832 |
| 201 | 35 | 82.12 | 1.244 | 0.071 | 17.177 | 17.461 |
| 201 | 50 | 122.98 | 1.474 | 0.084 | 17.512 | 17.638 |
| 201 | 75 | 225.43 | 2.096 | 0.119 | 18.133 | 17.565 |

All retained-algebra gates passed. Raw reconstruction does not, however,
represent the omitted stochastic field:

| Nodes | Modes | Mean truncation | Marginal-variance truncation | Full-covariance truncation |
| ---: | ---: | ---: | ---: | ---: |
| 101 | 20 | 0.434 | 0.932 | 0.971 |
| 101 | 35 | 0.324 | 0.794 | 0.900 |
| 101 | 50 | 0.261 | 0.636 | 0.807 |
| 101 | 75 | 0.177 | 0.358 | 0.605 |
| 201 | 20 | 0.442 | 0.970 | 0.987 |
| 201 | 35 | 0.342 | 0.908 | 0.957 |
| 201 | 50 | 0.292 | 0.837 | 0.920 |
| 201 | 75 | 0.243 | 0.711 | 0.849 |

The global mean metric includes the unresolved initial impulse. Mean
truncation decays quickly, but the missing variance and covariance do not.
Simplex projection enforces output constraints; it cannot recreate omitted
stochastic energy. The large raw runtime margin therefore clears the gate for
innovation-conditioned completion, while the raw field itself is rejected as
a physical model.

## Innovation-conditioned completion

For each transition, the full innovation covariance is partitioned into
retained and unresolved blocks. Given the retained innovation, the unresolved
innovation is sampled from

```text
mean = Q_UR Q_RR^+ epsilon_R
covariance = Q_UU - Q_UR Q_RR^+ Q_RU.
```

The implementation forms the conditional covariance through a square-root
null-space decomposition, avoiding an unstable Schur-complement subtraction
at early rank-deficient transitions. The unresolved deterministic initial
condition is retained exactly, so completion restores the full mean even at
low conditional rank.

The initial 201-node screen used `M=50,75`, conditional ranks 5, 10, and 20,
plus dense conditional oracles. Rank refinement then tested 25 and 30 for
`M=75`.

| Modes | Rank | Full covariance error | Worst/final covariance error | Variance error | Physical lag-1 error | Raw (ms) | Raw + simplex (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 20 | 0.0849 | 0.2201 | 0.0552 | 0.0735 | 9.47 | 25.50 |
| 75 | 20 | 0.0382 | 0.1109 | 0.0251 | 0.0169 | 10.37 | 26.44 |
| 75 | 25 | 0.0143 | 0.0483 | 0.00938 | 0.00639 | 10.64 | 26.69 |
| 75 | 30 | 0.00476 | 0.0183 | 0.00306 | 0.00217 | 10.71 | 26.70 |
| 75 | full | 1.74e-7 | 7.66e-7 | 6.38e-8 | 8.57e-8 | 13.20 | 29.23 |

The aligned exact vectorized multinomial walk took about 17.7--17.9 ms. Thus
the rank-25 completed Gaussian field is about 40% faster before physical
readout and passes a 10% first/second-moment error gate. The simplex pipeline
is about 49% slower than exact. At 101 nodes, `M=75`, rank 20 is effectively a
dense oracle because only 26 modes are unresolved: raw completion takes 2.95
ms versus 6.78 ms exact, while completion plus simplex takes 9.17 ms.

Preparation is not yet reduced-order. At 201 nodes, it takes about 3.0 seconds
and retains roughly 352--391 MiB of NumPy arrays because the prototype keeps a
full direct reference, its covariance/factor histories, and diagnostic target
arrays. The rank-25 raw pipeline amortizes this setup only after roughly 421
trajectories. A fresh-process check measured about 404 MiB peak RSS for this
prototype versus 37 MiB for one exact vectorized trajectory. The full-rank
oracle differs from the prepared full innovation
covariance by about `2e-7` relatively because the pseudoinverse deliberately
drops numerically singular directions; propagated moment errors remain below
`1.3e-6`.

## Decision

The direct transition is promoted as the stochastic core. Innovation
completion is also mathematically and computationally promising before
readout; `M=75`, conditional rank 25 is the current 201-node candidate.

The next work should target the two costs that now dominate:

1. The first readout experiment replaced most sorts with a warm exact
   active-set projection. It reduced readout time by 5--14% and prepared
   end-to-end time by 4--8%, but the modal pipeline remains 29--36% slower than
   the exact vectorized walk. Clip-and-renormalize then removed another
   39--40% of readout time and preserved moment accuracy, but remained 6.6%
   slower than exact at 201 nodes and worsened marginal histogram divergence.
   See [Warm-started exact simplex readout](WARM_SIMPLEX_READOUT.md) and
   [Clip-and-renormalize residual feedback](CLIP_RENORMALIZE_READOUT.md).
2. Stream or apply the full innovation covariance as an operator and retain
   only the selected gain/factor blocks. This should remove the prototype's
   full-reference duplication and make setup/storage genuinely reduced-order.
3. Re-run isolated RSS, amortization, and a 500-trajectory physical-output
   accuracy gate. Only after an end-to-end crossover should the method be
   scaled to larger one-dimensional grids or moved to 2D/3D.

Reproduce the screens with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 PYTHONPATH=src \
python3 -m eigendiffusion.direct_modal_benchmark \
  --output-dir outputs/direct_modal_gaussian

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 PYTHONPATH=src \
python3 -m eigendiffusion.innovation_completion_benchmark \
  --node-counts 201 --modes 50 75 --ranks 5 10 20 \
  --output-dir outputs/innovation_completion_screen
```

Generated output is ignored by Git. The study runners and schemas are tracked.
