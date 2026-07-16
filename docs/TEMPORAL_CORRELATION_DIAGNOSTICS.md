# Temporal correlation diagnostics

Version 0.8 adds diagnostics for stochastic persistence across time. Earlier
validation compared the mean, variance, and same-time spatial covariance at
individual saved times. Those quantities do not determine whether fluctuations
persist correctly from one time point to the next.

For independent particles with one-step transition matrix `P`, single-particle
probability vector `p_t`, and `s = t + lag`, the exact cross-time covariance is

\[
\operatorname{Cov}[\mathbf n(t),\mathbf n(s)]
=
N\left[\operatorname{diag}(\mathbf p_t)P^{s-t}
-\mathbf p_t\mathbf p_s^\top\right].
\]

The new command averages this covariance over time origins in a requested
window and compares it with empirical ensembles from:

1. multinomial random walk,
2. the full correlated-modal model,
3. the raw 101-to-50 handoff,
4. the handoff plus unresolved Gaussian completion.

This specifically tests the main limitation of the current analytic completion:
its unresolved contribution is sampled independently at each output time. It
can match same-time moments while underestimating temporal persistence.

## Run the recommended comparison

```bash
mkdir -p outputs/temporal_correlation/t10_m50_completion_r10_r500

python -m eigendiffusion compare-temporal-correlation \
  --initial-modes 101 \
  --modes 50 \
  --handoff-time 10 \
  --completion-start-time 10 \
  --completion-rank 10 \
  --completion-ridge 0.01 \
  --lags 1 2 5 10 20 \
  --temporal-start-time 10 \
  --node-window-radius 10 \
  --profile-lag 5 \
  --particles 5275 \
  --nodes 101 \
  --steps 101 \
  --impulse-index 59 \
  --runs 500 \
  --random-walk-method multinomial \
  --seed 0 \
  --output outputs/temporal_correlation/t10_m50_completion_r10_r500/comparison.png \
  --csv-output outputs/temporal_correlation/t10_m50_completion_r10_r500/comparison.csv \
  --data-output outputs/temporal_correlation/t10_m50_completion_r10_r500/comparison.npz
```

## Outputs

The figure contains:

- relative cross-time covariance error versus lag,
- impulse-node temporal correlation versus lag,
- temporal correlation averaged over a central spatial window,
- the diagonal cross-time covariance profile at one selected lag.

The CSV contains one row per model and lag with:

- `cross_time_covariance_relative_frobenius_error`,
- `impulse_node_lag_correlation`,
- `central_window_lag_correlation`,
- the number of valid time origins.

## Interpretation

If unresolved completion matches same-time covariance but has substantially
lower lag correlation than the random walk, the next model should propagate a
persistent unresolved latent state instead of independently redrawing omitted
modes at every output time. A natural next closure would be an autoregressive
or linear-Gaussian unresolved process whose transition and innovation
covariance are chosen to match both same-time and cross-time modal moments.
