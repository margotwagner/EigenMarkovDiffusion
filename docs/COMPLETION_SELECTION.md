# Completion-rank and physical-output selection

The completion-rank sweep now compares four output-only readouts on one shared
handoff ensemble:

- independent unresolved Gaussian completion;
- persistent unresolved completion;
- independent completion followed by a simplex bank;
- persistent completion followed by a simplex bank.

The composed simplex readouts use the same Gaussian random streams as their
unconstrained counterparts. This isolates the effect of enforcing
nonnegativity and exact mass conservation. Completion magnitude and constraint
residual are recorded separately.

The command reports mean, variance, same-time covariance, lagged cross-time
covariance, negative mass, mass conservation, completion magnitude, and
constraint residual for each rank.

## Selected operating points

All headline comparisons below use 500 trials and persistent completion
followed by the simplex bank.

| Operating point | Handoff | Completion rank | Modal work | Mean | Variance | Covariance | Lag-1 covariance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Compute-first | 101 to 35 at step 20 | 20 | 0.477 | 0.00168 | 0.0633 | 0.1097 | 0.0589 |
| Balanced | 101 to 50 at step 10 | 10 | 0.546 | 0.00180 | 0.0654 | 0.1127 | 0.0501 |
| Accuracy-first | 101 to 75 at step 30 | 5 | 0.820 | 0.00169 | 0.0610 | 0.0958 | 0.0688 |

Every constrained output had zero negative mass and conserved total mass to
floating-point precision. The simplex bank changed all reported moment errors
only minimally. For the balanced pipeline, ranks above 10 did not provide a
consistent improvement. The 35-mode compute-first pipeline required rank 20
to recover temporal covariance at the 500-trial sampling scale.

The three pipelines are statistically similar at their selected ranks. The
next decision therefore requires end-to-end runtime and setup-memory comparison
across these exact operating points; modal-update fraction alone does not
include completion-oracle cost.

Key outputs:

- `outputs/completion_selection/selected_operating_points_r500.csv`
- `outputs/completion_selection/balanced_m50_t10_r500.csv`
- `outputs/completion_selection/compute_selected_m35_t20_rank20_r500.csv`
- `outputs/completion_selection/accuracy_selected_m75_t30_r500.csv`
