# Compiled exact simplex readout

## Method

This implementation preserves the exact residual-feedback simplex-bank
operator while fusing its row work into an optional Numba kernel. For each
adjusted row it starts from the previous threshold, scans the proposed active
set, and certifies the exact simplex KKT conditions

```text
max(inactive input) <= threshold < min(active input).
```

The active-set update is bounded. The first row uses the historical sorting
rule inside the compiled kernel. If a numerical guard or the update bound is
hit later, the partial trajectory is discarded and the complete trajectory is
replayed through the NumPy sorting bank with a clean residual ledger. The
wrapper also rejects a replay that is not finite, nonnegative, and
mass-conserving. Missing or disabled Numba is reported explicitly rather than
silently timing interpreted Python.

The compiled path fuses threshold certification, output construction,
floating mass correction, and residual-drift correction. Compensated sums
keep large, well-conditioned rows inside the numerical equivalence tolerance.
Numba's disk cache is enabled when the configured cache or installation tree
is writable, so later processes can load the specialization instead of
rerunning the full LLVM compilation. Compilation still works without a
writable cache; only cross-process reuse is disabled.

## Experiment

The November 23, 2026 benchmark used the same workload as the warm and
clip-and-renormalize studies: 5,275 particles, fixed physical duration 100,
`M=75`, innovation-completion rank 25, and the full physical field at every
saved time. The 101-node workload has 101 saved steps and the stable 201-node
workload has 233. All BLAS, expression, and Numba thread limits were one.

- Readout-only: 32 cached completed trajectories, 2 warmups, and 30 paired
  alternating-order repeats.
- Prepared end to end: 1 warmup and 24 repeats covering all six orders of the
  NumPy warm readout, compiled exact readout, and exact vectorized multinomial
  walk. Reusable completion setup was excluded and reported separately.
- Runtime uncertainty: 10,000 paired bootstrap resamples of median ratios.
- Correctness: counts, support, adjusted inputs, residuals, mass, and residual
  ledger were compared on all 32 cached trajectories at each resolution.

| Nodes | NumPy warm readout (ms) | Compiled exact readout (ms) | Compiled/warm (95% CI) | Compiled prepared E2E (ms) | Exact walk (ms) | Compiled/exact (95% CI) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 101 | 5.893 | 0.182 | 0.0308 (0.0303--0.0311) | 3.147 | 6.814 | 0.462 (0.461--0.464) |
| 201 | 13.884 | 0.682 | 0.0491 (0.0483--0.0501) | 11.082 | 17.759 | 0.624 (0.623--0.627) |

The compiled kernel removes 96.9% and 95.1% of NumPy warm-readout time. The
prepared modal pipeline is 53.8% faster than the exact walk at 101 nodes and
37.6% faster at 201 nodes. Both predeclared crossover gates pass with narrow
paired confidence intervals.

## Numerical equivalence

No compiled fallback occurred in 3,232 tested rows at 101 nodes or 7,456 rows
at 201 nodes. Strict positive support was identical at 101 nodes. At 201 nodes,
32 entries differed only in whether a value below `1e-12` was represented as
zero; support above that tolerance was identical everywhere.

| Nodes | Maximum count difference | Maximum adjusted/residual difference | Maximum mass error | Maximum residual-sum error |
| ---: | ---: | ---: | ---: | ---: |
| 101 | `4.44e-12` | `1.48e-11` | `1.82e-12` | `8.37e-15` |
| 201 | `5.12e-12` | `2.40e-11` | `1.82e-12` | `2.14e-14` |

These differences are floating summation-order effects below the declared
`1e-10` tolerance. Because the candidate applies the same mathematical
readout to the identical completed fields, a separate stochastic ensemble is
not needed to establish the readout swap. This result establishes numerical
equivalence to the warm modal pipeline; it does not claim that the Gaussian
modal output law is identical to the integer multinomial walk.

## Compilation, setup, and break-even

A fresh isolated Numba cache took 2.59 seconds to import and compile. Loading
the resulting disk cache in a second fresh process took 0.36 seconds. Reusable
completion setup remained 0.315 seconds at 101 nodes and 3.08 seconds at 201
nodes; this experiment did not change the completion's existing retained
storage.

Ignoring compiler startup, completion setup amortizes after approximately 86
trajectories at 101 nodes and 461 at 201. For one configuration in a completely
cold process, setup plus compilation amortizes after approximately 793 and 849
trajectories. With the compiled specialization already cached, those estimates
fall to approximately 176 and 491. Compiler loading can also be shared across
multiple configurations in one process.

## Decision and next step

The compiled exact readout is promoted as the preferred prepared-trajectory
path when the optional compiler is available. It is the first exact-readout
pipeline to pass the crossover promotion gate at both tested resolutions,
without introducing the distribution-shape regression of
clip-and-renormalize. The NumPy warm and sorting paths remain the portable
reference and transactional fallback.

Completion now dominates prepared runtime: about 2.97 ms of the 3.15 ms
101-node trajectory and 10.40 ms of the 11.08 ms 201-node trajectory. The next
runtime experiment should therefore prepack the retained and conditional
innovation factors and apply all transition innovations in one batched
contraction. After that, preparation must be made genuinely reduced-order by
removing full covariance/factor histories and retaining only runtime factors.

## Reproduction

Install the optional backend and run:

```bash
python -m pip install -e ".[compiled]"

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=src \
python3 -m eigendiffusion.compiled_simplex_benchmark \
  --node-counts 101 201 \
  --modes 75 \
  --completion-rank 25 \
  --cached-trajectories 32 \
  --readout-repeats 30 \
  --end-to-end-repeats 24 \
  --bootstrap-resamples 10000 \
  --output-dir outputs/compiled_simplex_benchmark
```

Generated JSON and CSV artifacts are ignored by Git; the implementation,
runner, tests, and this result summary are tracked.

For a true cold-start measurement, point `NUMBA_CACHE_DIR` at an existing,
empty, writable directory before the first fresh-process run. Reuse that same
directory in a second fresh process to measure disk-cache loading. Without an
explicit directory, caching is best effort and depends on whether the installed
package directory is writable.
