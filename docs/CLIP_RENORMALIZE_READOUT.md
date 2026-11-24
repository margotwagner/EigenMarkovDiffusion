# Clip-and-renormalize residual-feedback readout

## Method

This experiment replaces the exact simplex projection with the linear-time
recurrence

```text
adjusted[t] = raw[t] + residual[t - 1]
positive[t] = max(adjusted[t], 0)
output[t] = N * positive[t] / sum(positive[t])
residual[t] = adjusted[t] - output[t].
```

The existing floating-point mass and residual-drift corrections are retained.
If clipping creates more than 5% excess positive mass, or a numerical guard
fails, the partial result is discarded and the complete trajectory is rerun
through the exact warm-simplex readout. This keeps fallback trajectories from
mixing two different residual recurrences.

The counts-only production kernel omits diagnostic branches. The detailed
path records clipped negative mass, normalization scale, fallback state, and
the full residual ledger.

## Experiment

The November 23, 2026 study used 5,275 particles, a fixed physical duration of
100, `M=75`, innovation-completion rank 25, and the full node-count field at
every saved time. The 101-node workload has 101 saved steps; the stable
201-node workload has 233. BLAS was restricted to one thread.

- Readout-only: 32 cached completed trajectories, 2 warmups, and 30 paired
  alternating-order repeats.
- Prepared end to end: 1 warmup and 24 repeats covering all six orders of clip,
  warm exact, and exact vectorized multinomial simulation. Reusable completion
  setup was measured separately and excluded.
- Runtime uncertainty: 10,000 paired bootstrap resamples of median ratios.
- Accuracy: 500 paired completed trajectories, an independent 500-run exact
  ensemble for analytic sampling-floor errors, and a second exact ensemble at
  selected times for distribution sampling-floor comparisons. Mean, variance,
  and covariance comparisons use 200 paired block-bootstrap resamples; the
  reported marginal Jensen--Shannon values use one frozen set of histogram
  edges for all three methods.

| Nodes | Warm readout (ms) | Clip readout (ms) | Clip/warm (95% CI) | Warm prepared E2E (ms) | Clip prepared E2E (ms) | Exact walk (ms) | Clip/exact (95% CI) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 101 | 6.014 | 3.616 | 0.601 (0.591--0.613) | 8.971 | 6.602 | 6.948 | 0.950 (0.947--0.958) |
| 201 | 14.144 | 8.609 | 0.609 (0.598--0.622) | 24.686 | 19.163 | 17.971 | 1.066 (1.063--1.070) |

Clip-and-renormalize removes about 40% of exact-readout time at 101 nodes and
39% at 201. Prepared trajectory time falls by 26% and 22%. It is statistically
faster than the exact walk at 101 nodes, but its 4.97% point margin is on the
boundary of, and just misses, the predeclared 5% promotion threshold. It
remains 6.6% slower than exact at 201 nodes. Setup is still 0.314 seconds and
3.04 seconds, respectively, so including preparation cannot create a
crossover.

## Physicality and residuals

All constraint and ledger checks passed. Maximum mass error was `9.1e-13`,
the minimum output was zero, ledger errors stayed below `2.4e-10`, and no
fallback occurred in either the 32 diagnostic trajectories or the 500-run
accuracy ensembles.

| Nodes | Warm mean residual L1/N | Clip mean residual L1/N | Warm maximum | Clip maximum | Minimum scale | Clip/warm output relative L2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 101 | 0.221% | 0.244% | 1.664% | 1.734% | 0.9914 | 0.0131% |
| 201 | 0.473% | 0.520% | 4.522% | 4.792% | 0.9766 | 0.0259% |

Across the 32 cached diagnostic trajectories, residual size rises by roughly
10%, but remains stable and below the 10% hard cap. Clip-and-renormalize retains
every positive adjusted entry, so its support differs from the exact simplex
in 14--16% of entries. The cached maximum per-row L1 output differences are
0.089% of mass at 101 nodes and 0.240% at 201 nodes; across all 500 accuracy
trajectories, the corresponding maxima are 0.154% and 0.287%.

## Accuracy

Moment and temporal screens pass at both resolutions. The paired-bootstrap
upper bounds for changes in mean, variance, and covariance error remain well
inside one quarter of their exact-ensemble sampling floors. Relative mean
errors are unchanged to about `0.14%` or better; relative variance errors
increase only `0.10%` and `0.28%`; same-time covariance and all tested
lag-covariance errors slightly improve. Normalized marginal and sliced
Wasserstein distances change by less than `0.5%`.

The one clear regression is common-bin marginal Jensen--Shannon divergence:
it rises from `0.0662` to `0.1017` at 101 nodes and from `0.0670` to `0.0990`
at 201 nodes, increases of 54% and 48%. These differences exceed the exact
ensemble sampling floors (`0.0071` and `0.0061`). The result also persists
with 15 and 60 bins. This is consistent with the operator's broader tiny-value
support and proportional attenuation of every positive node. Consequently the
moment and temporal screens pass, but the distribution screen fails.

## Decision

Clip-and-renormalize is a useful experimental kernel, but it is not a
production candidate: it establishes a statistical crossover at 101 nodes but
narrowly misses the 5% promotion margin, remains slower at 201 nodes, and
global rescaling has a detectable marginal-distribution cost. Holding the rest
of the measured 201-node pipeline fixed, the readout needs about another 14%
reduction for parity and 24% for the `clip/exact <= 0.95` promotion gate.

Do not advance directly to clip-and-redistribute-only-the-excess as the next
formal experiment. Proportional removal is the method on this page, while
equal capacity-aware water filling is the exact simplex projection already
tested. A prepared capacity-aware top-`K` donor prototype saved only about
3--5% of readout time and did not repair Jensen--Shannon divergence for
`K=1--16`; `K=32` helped only slightly at 101 nodes, cost more, and did not
help at 201 nodes. The next step should first profile and fuse the existing
kernel's repeated row reductions and mass correction. If that cannot close
the measured gap, stop this approximate-readout branch and return to making
completion genuinely reduced-order.

## Reproduction

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 PYTHONPATH=src \
python3 -m eigendiffusion.clip_renormalize_benchmark \
  --node-counts 101 201 \
  --modes 75 \
  --completion-rank 25 \
  --cached-trajectories 32 \
  --readout-repeats 30 \
  --end-to-end-repeats 24 \
  --accuracy-runs 500 \
  --accuracy-bootstrap-resamples 200 \
  --bootstrap-resamples 10000 \
  --output-dir outputs/clip_renormalize_benchmark
```

Generated JSON and CSV artifacts are ignored by Git; the runner, schemas,
tests, and this result summary are tracked.
