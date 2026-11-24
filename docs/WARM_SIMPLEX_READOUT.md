# Warm-started exact simplex readout

## Purpose

The innovation-completed direct modal pipeline is faster than the exact
vectorized multinomial walk before physical readout, but its sorting-based
simplex bank removes that advantage. This experiment tests whether temporal
coherence lets the readout reuse the previous simplex threshold and avoid an
`O(N log N)` sort at most time steps without changing the mathematical
projection.

The warm path keeps the existing residual-feedback bank. The first row uses
the historical sorting projector. For every later row it:

1. forms a candidate active set from the previous threshold;
2. recomputes the threshold over that set until the strict simplex KKT support
   is a fixed point;
3. falls back to the historical sort if the candidate is empty, nonfinite, or
   does not converge within eight support updates; and
4. applies the same final floating-point mass correction as the historical
   projector.

The production counts-only path does not collect diagnostics. Its incremental
state is three Boolean node masks and one scalar threshold, so it does not add
a full trajectory history. A separate instrumented API records support updates
and fallbacks outside timed regions.

## Experiment

The November 23, 2026 experiment used the current direct modal candidate:
`M=75`, innovation-completion rank 25, 5,275 particles, and a fixed physical
duration of 100. It produced the full node-count field at every time step for
101 and 201 nodes. BLAS was restricted to one thread.

- Readout-only: 32 cached completed trajectories, 2 warmups, and 30 paired
  balanced-order repeats.
- Prepared end to end: 1 warmup and 24 position-balanced repeats of completion
  plus readout, with the exact vectorized multinomial walk as comparator. The
  reusable completion setup is measured separately and excluded from these
  per-trajectory timings.
- Uncertainty: 10,000 paired bootstrap resamples of the median-time ratio.
- Correctness and active-set diagnostics were evaluated outside timing.

| Nodes | Sorting readout (ms) | Warm readout (ms) | Warm/sort (95% CI) | Sorting prepared E2E (ms) | Warm prepared E2E (ms) | Warm/sort (95% CI) | Exact walk (ms) | Warm/exact |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 101 | 6.142 | 5.836 | 0.950 (0.935--0.963) | 9.056 | 8.737 | 0.965 (0.957--0.977) | 6.752 | 1.294 |
| 201 | 15.838 | 13.554 | 0.856 (0.843--0.877) | 26.254 | 24.092 | 0.918 (0.903--0.929) | 17.674 | 1.363 |

The warm path eliminated all post-cold-start sorts in the 10,624 eligible
rows. About 65% of rows accepted the initial candidate, 95% required at most
one support update, and the maxima were four updates at 101 nodes and five at
201 nodes.

Warm and sorting paths selected identical positive supports for every row.
Their largest count difference was `6.2e-12` at 101 nodes and `1.1e-11` at 201
nodes; maximum mass error was `9.1e-13`. The small non-bitwise differences come
from summing an active set in a different floating-point order, not from a
different simplex solution. Because the complete trajectories agree to
numerical precision, a new ensemble-accuracy study is not needed for this
optimization.

## Decision

Warm starting is a successful low-risk optimization and should remain as the
exact reference path. It cuts readout time by 5.0% at 101 nodes and 14.4% at
201 nodes, producing prepared end-to-end gains of 3.5% and 8.2%, respectively.
All warm-versus-sorting timing confidence intervals lie below 1.0.

It does not establish the target crossover: the completed modal pipeline is
still 29.4% slower than the exact walk at 101 nodes and 36.3% slower at 201
nodes. Holding the measured completion cost fixed, the readout would need to
fall by approximately another 34% and 47%, respectively, merely to reach the
exact comparator.

The follow-up tested clip-and-renormalize with temporal residual feedback. It
reduced readout time substantially and preserved moment and temporal accuracy,
but did not cross the exact comparator at 201 nodes and worsened marginal
Jensen--Shannon divergence. See
[Clip-and-renormalize residual feedback](CLIP_RENORMALIZE_READOUT.md).

A later compiled implementation fused the same exact warm projection and
residual recurrence into a guarded native loop. It removed 95--97% of
this NumPy readout time and established a prepared-runtime crossover at both
101 and 201 nodes without changing the projection law. See
[Compiled exact simplex readout](COMPILED_SIMPLEX_READOUT.md).

## Reproduction

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 PYTHONPATH=src \
python3 -m eigendiffusion.warm_simplex_benchmark \
  --node-counts 101 201 \
  --modes 75 \
  --completion-rank 25 \
  --cached-trajectories 32 \
  --readout-warmups 2 \
  --readout-repeats 30 \
  --end-to-end-warmups 1 \
  --end-to-end-repeats 24 \
  --bootstrap-resamples 10000 \
  --output-dir outputs/warm_simplex_benchmark
```

The generated JSON and CSV artifacts are ignored by Git; the benchmark runner,
schema, tests, and this result summary are tracked.
