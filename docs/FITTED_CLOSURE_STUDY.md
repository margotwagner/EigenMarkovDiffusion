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

## Anchor-conditioned follow-up

The recommended successor has now been implemented and run with the same
frozen gates. It fits 12 separate physical anchors, interpolates the three
nearest models in standardized configuration space, replaces residual-vector
resampling with rank-12-plus-diagonal Gaussian innovations, and vectorizes
feature construction across target modes.

On the unseen 500-trial base case, variance/covariance/lag-1 errors improved
from 0.230/0.412/0.366 to 0.208/0.380/0.339. The normalized sliced Wasserstein
error improved from 0.110 to 0.099. These gains were insufficient: the
corresponding declared thresholds remained 0.129/0.337/0.145, and failures
also remained on all five physical held-outs.

Vectorization reduced the 201-node fitted trajectory time from 0.714 to 0.246
seconds and setup took 0.034 seconds, but analytic-local required 0.162 seconds;
the fitted runtime therefore exceeded the 1.25x gate of approximately 0.202
seconds. Peak RSS did improve to 106.6 MiB versus 116.2 MiB for analytic-local.

## Exact-anchor, persistence, and batching follow-up

The error-source ablation and proposed persistence/batching changes have now
also been implemented. The ablation used ranks 4/8/12/20 over all 12 anchors
and five time bins. At exact anchors, innovation-covariance reconstruction
errors fell from 0.091 at rank 4 to 0.0106 at rank 8, 0.00106 at rank 12, and
0.000031 at rank 20. Rank 12 is therefore already sufficient. In contrast,
leave-one-anchor-out initial/transition coefficient errors were 1.09/1.01.

A stable conditional innovation transition reduced residual-level lag-1 error
from 1.00 to 0.56, but barely changed the frozen unseen-base end-to-end errors:
variance/covariance/lag-1 became 0.206/0.381/0.336 versus
0.208/0.380/0.339 before persistence. Fresh 100-run exact-anchor checks showed
that removing interpolation helps, particularly at 101 and 201 nodes, but the
fitted closure still trails analytic-local.

Batching trajectories, precombining anchor factors, and removing repeated
feature-contraction planning reduced 201-node fitted time from 0.246 to 0.224
seconds. Analytic-local took 0.164 seconds, so the approximately 0.206-second
runtime gate still failed. The scientific bottleneck is now localized to
regression transfer across physical configurations, not innovation rank.

The analytic bandwidth-12 closure remains the production default. The next
model should condition a shared local operator directly on dimensionless modal
decay, time, and physical descriptors instead of interpolating separately
fitted ridge coefficients, which are not stable coordinates across anchors.

## Physics-conditioned regression follow-up

The next successor replaces interpolation of separately fitted coefficient
vectors with one joint regression. Local modal features are interacted with
normalized grid size, impulse fraction, and dimensionless diffusion time.
Whole-configuration three-fold cross-validation compared linear and quadratic
conditioning. Linear conditioning was selected: its fold errors were
0.885/1.062/1.306, while quadratic conditioning overfit with
6.582/2.250/2.934.

The frozen 500/250 gate showed only modest improvement. Unseen-base
variance/covariance/lag-1 errors were 0.205/0.377/0.322, compared with
0.206/0.381/0.336 for persistent anchor interpolation and declared thresholds
of 0.129/0.337/0.145. The unseen-domain marginal Wasserstein error worsened to
0.412, so the conditioned model is not accepted. At 201 nodes it required
0.223 seconds versus 0.160 seconds for analytic-local and also missed the
runtime gate.

The cross-validation error map generated four nonduplicated bridge-anchor
recommendations concentrated around 114--176 nodes, intermediate domain
lengths, and impulse fractions 0.573--0.625. These are recorded for a future
training expansion, but must be evaluated with a new lockbox suite rather than
reusing the now-inspected held-outs as an unbiased promotion test.

Original outputs are under `outputs/fitted_closure/`; conditioned outputs are
under `outputs/anchored_fitted_closure/`; persistence and ablation outputs are
under `outputs/anchored_persistent_fitted_closure/`; conditioned-regression and
anchor-design outputs are under `outputs/physics_conditioned_fitted_closure/`.
