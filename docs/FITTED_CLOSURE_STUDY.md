# Fitted sparse closure study

The `study-fitted-closure` workflow tests whether the validated bandwidth-12
local closure can be made independent of exact multinomial covariance.

## Fitted model

Eight training configurations span 51--201 nodes, 2,000--9,000 particles,
multiple impulse locations, diffusivities, and domain lengths. One hundred
multinomial trajectories are generated per configuration. The frozen model
contains:

- frequency-binned bandwidth-12 regressions for initial and persistent
  unresolved modal deviations;
- modal fluctuations normalized by `sqrt(N)` for particle-count transfer;
- empirical innovation banks resampled on a normalized modal-frequency grid;
- five time bins for changing innovation statistics.

Inference uses deterministic mean propagation, the fitted coefficients, and
the fitted innovation banks. It does not construct analytic multinomial
covariance or conditional gains.

## Validation design

The fitted artifact is frozen before evaluation. Validation uses a separate
500-trial base configuration and five 250-trial configurations with unseen
impulse location, particle count, diffusivity, 126-node grid, and domain
length. It is compared with the analytic bandwidth-12 closure using mean,
variance, same-time covariance, lag-1 covariance, marginal Wasserstein, and
sliced Wasserstein errors.

The acceptance threshold for each metric is the analytic-local error plus an
independently measured multinomial sampling-floor error.

## Result: not accepted

The shared fitted model did not pass the accuracy or scaling gates. On the
unseen 500-trial base case:

| Metric | Fitted | Acceptance threshold |
| --- | ---: | ---: |
| Variance error | 0.230 | 0.129 |
| Covariance error | 0.412 | 0.337 |
| Lag-1 covariance error | 0.366 | 0.145 |
| Normalized marginal Wasserstein | 0.153 | 0.146 |

Variance, covariance, and temporal failures also occurred on several held-out
physical configurations. The frozen model therefore remains an experimental
baseline and the analytic bandwidth-12 local closure remains the production
default.

## Performance result

At 201 nodes, fitted setup was only 0.038 seconds versus 1.123 seconds for the
analytic local closure, and peak RSS was 83.1 versus 85.8 MiB. However, fitted
trajectory time was 0.714 seconds versus 0.162 seconds because the current
feature construction is performed in Python for every target mode and time.

## Recommended follow-up

The next model should not share one unconditional regression and innovation
bank across all physics. Use a configuration-conditioned library or mixture:

1. fit separate sparse anchors across dimensionless diffusion time, impulse
   location, domain scale, and grid resolution;
2. interpolate coefficients and innovation factors between nearby anchors;
3. replace empirical innovation-vector resampling with fitted low-rank factors;
4. vectorize feature construction before repeating the same frozen held-out
   and scaling gates.

Key outputs are under `outputs/fitted_closure/`.
