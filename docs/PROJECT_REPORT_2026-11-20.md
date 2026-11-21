# EigenDiffusion project report

**Status as of November 20, 2026**

## Executive summary

EigenDiffusion tests whether stochastic diffusion can be represented in a
Laplacian eigenbasis instead of by explicitly tracking particles. The project
has progressed from the original independent-mode formulation to correlated
modal dynamics, physically constrained banking, adaptive modal reduction, and
Gaussian readouts for reconstructing unresolved modes. The package, command
line workflows, analytic references, diagnostics, and 114 tests are complete
and passing. The initial five-part presentation suite has since expanded into
the 19 experiments summarized below.

The main scientific conclusion is that mean propagation is no longer the core
problem. The full correlated model reproduces the multinomial reference's mean,
variance, and same-time covariance to within ensemble sampling error. A reduced
50-mode model can recover those same-time statistics through Gaussian
completion, and persistent completion also recovers short-lag temporal
correlation. However, the completion is an analytic oracle, can produce
noninteger counts, consumes substantial setup memory, and is not yet
faster than the aggregated multinomial baseline.

## Where we left off

The baseline presentation outputs are in `outputs/presentation/`, and
`run_presentation_results.sh` defines that pipeline. The current structured
closure decision is in `outputs/structured_closure_hardened/`; the current
resolution study is in `outputs/mode_scaling/`, with the independent finalist
confirmation in `completion_finalist_r250_fresh/`. The adaptive-rank selection,
confirmation, and timing decision are in `outputs/adaptive_rank_scaling/`.
These later artifacts supersede the earlier local-closure selection. All 114
automated tests pass.

Time-resolved errors, bootstrap uncertainty, and a 5,000-trial convergence
study, predictive handoff/rank rule, completion-rank study, exact pipeline
benchmark, and distribution-level comparison are now complete. The local gain
has been rewritten with exact restricted covariance blocks, but a statistically
hardened rerun rejects bandwidths 12 and 20 on long-domain lag covariance; dense
persistent completion is therefore the production default.
The first frozen trajectory-fitted replacement was implemented and rejected by
the predeclared accuracy and runtime gates. Its anchor-conditioned successor
was also implemented and tested; it improved accuracy and runtime but still
missed both promotion gates. Exact-anchor, innovation-rank, conditional
persistence, and batching follow-ups now localize the remaining scientific
error to cross-configuration regression transfer.
The first jointly physics-conditioned regression and configuration-level
cross-validation study are also complete; they improve some errors but do not
pass the frozen gates. The final 16-anchor target-conditioned candidate failed
its pre-lockbox CV gate, so the new unbiased lockbox remains unopened.
A resolution-aware 101/151/201-node study is also complete. Plateauing retained
mode rules fail with fixed completion rank 5. Fixed 75 retained modes becomes
noninferior with completion rank 20, but setup, retained storage, and trajectory
time are not better than the proportional rank-5 baseline.
The conditional-innovation energy rule is now implemented and validated across
the base length-4 and held-out length-6 domains. Its dense and wide-local
candidates pass fresh 250-trial accuracy confirmation, but both fail the
predeclared cost intersection, so neither advances to 2,000 trials.

## Completed work

- Refactored the research code into the `eigendiffusion` package with a CLI,
  examples, documentation, reproducible ensembles, and automated tests.
- Added exact continuous- and discrete-time references, analytic multinomial
  variance/covariance, and diagnostics for mean, covariance, negative mass,
  physical bounds, mass conservation, and readout/completion magnitude.
- Preserved the original `independent_modal` model and added
  `correlated_modal`, `banked_correlated_modal`, and full-to-reduced
  `handoff_correlated_modal` dynamics.
- Added raw, simplex-bank, Delta-Sigma, independent Gaussian completion, and
  persistent Gaussian completion readouts.
- Added handoff/rank sweeps, cross-time covariance diagnostics, and end-to-end
  runtime/setup-memory benchmarks.
- Added time-resolved mean/variance errors, reference-standardized mean error,
  bootstrap uncertainty intervals, and a nested trial-count convergence CLI.
- Added isolated peak-RSS scaling and marginal/sliced distribution metrics for
  the three selected completed pipelines.
- Added a row-sparse local persistent closure, exact restricted-block covariance
  assembly, profiling, held-out validation, and isolated scaling. The local
  closure is retained as a memory experiment; dense completion is the default.
- Added case-specific sampling floors, identical sliced projections, paired
  noninferiority intervals, worst-time gates, and an explicit runtime gate.
- Added resolution-aware proportional/fixed/spectral/mean-cutoff studies and a
  completion-rank/bandwidth follow-up at 101--201 nodes.
- Added adaptive completion rank from captured conditional-innovation energy,
  shared-CSR local gain storage, cross-domain selection/confirmation, and
  repeated setup/storage/trajectory promotion gates.
- Added a frozen trajectory-fitted sparse closure and tested it on unseen
  physics; it remains experimental because variance, covariance, temporal, and
  runtime gates failed.
- Added 12 physical fitted anchors, nearest-anchor parameter interpolation,
  rank-12-plus-diagonal Gaussian innovations, and vectorized modal feature
  construction; the frozen successor also remains experimental.
- Added exact-anchor rank ablations, leave-one-anchor-out coefficient tests,
  stable conditional residual dynamics, and batched completion; these rule out
  innovation rank as the main remaining error source.
- Added a jointly physics-conditioned local regression, whole-configuration
  linear/quadratic cross-validation, and targeted bridge-anchor design.
- Froze a new six-case lockbox, added four bridge anchors and target-level
  modal/time conditioning, and enforced a pre-lockbox CV gate; the candidate
  failed and stopped before lockbox evaluation.
- Generated the baseline presentation figures with CSV data and supporting NPZ
  arrays, plus hardened closure and mode-scaling artifacts.

## Experiments completed

All presentation experiments used 5,275 particles, 101 saved time steps, an
impulse at node 59, and fixed seeds unless stated otherwise. A **trial** below
means one independent stochastic trajectory. A timing **repeat** means one
wall-clock measurement and is not an ensemble trial.

| Experiment | Models/configuration | Modes and trials | Why it was run |
| --- | --- | --- | --- |
| 1. Core modal-model comparison | Multinomial reference; independent, correlated, and banked correlated modal models | 101 nodes; 101 modes; 500 trials per stochastic method; seed 0 | Isolate the effect of the noise formulation at full rank. Test mean, variance, covariance, nonnegativity, and mass conservation without truncation error. |
| 2. Reduced-readout comparison | 101-to-50 correlated handoff at step 10, followed by raw, simplex-bank, independent Gaussian completion, or persistent completion | 101 initial modes; 50 retained modes; completion rank 10; 500 shared base trajectories/readout trials; seed 1 | Determine what is lost by truncation and whether output-only readouts restore physical constraints, same-time moments, or temporal structure without changing the reduced dynamics. |
| 3. Handoff sweep | Raw correlated handoff over every pair of handoff step and retained rank | 101 initial modes; final modes 20, 35, 50, 75; handoff steps 1, 5, 10, 20, 50; 20 configurations; 250 trials each; seed 2 | Map the accuracy/work tradeoff and show when high-frequency modes have decayed enough to discard. This is the current empirical basis for selecting handoff time and rank. |
| 4. Temporal-correlation comparison | Multinomial reference, full correlated, raw 101-to-50 handoff, independent completion, and persistent completion; lags 1, 2, 5, 10, 20 | 101 initial modes; 50 retained; handoff/completion at step 10; completion rank 10; 500 trials; seed 3 | Test whether fluctuations persist correctly through time. Same-time mean and covariance alone cannot distinguish independent redraws from a dynamically consistent unresolved process. |
| 5. End-to-end performance | Multinomial, full correlated, raw handoff, independent completion, and persistent completion | Node sweep 31/51/101/151 with full ranks 31/51/101/151 and reduced ranks 16/26/50/76; particle sweep 1,000/5,275/10,000 at 101-to-50 modes; 5 timing repeats; seed 4 | Measure setup time, trajectory time, amortized time, and resident NumPy-array storage. Determine whether modal reduction creates an actual computational benefit rather than only reducing the nominal modal state. |
| 6. Trial-count convergence | Multinomial reference; independent, correlated, and banked correlated modal models | 101 modes; nested prefixes of 100, 250, 500, 1,000, 2,000, and 5,000 trials; 200 block-bootstrap resamples; seed 20 | Separate finite-ensemble noise from structural model error, quantify uncertainty, and make the independent model's sampled-mean error directly visible. |
| 7. Predictive handoff selection | Analytic modal-energy predictor validated against correlated handoff ensembles | 35, 50, 60, 75 retained modes; handoff steps 10, 15, 20, 30; 16 candidate handoff/rank combinations on one physical configuration; 250 trials each | Predict post-handoff mean, variance, and covariance loss before running an ensemble and identify the accuracy/work Pareto frontier. |
| 8. Completion and physical-output selection | Independent/persistent completion, with and without simplex banking | Completion-rank sweeps plus 500-trial headline checks at 35/50/75 modes | Select completion rank for compute/balanced/accuracy operating points and test whether nonnegativity and mass conservation distort stochastic moments. |
| 9. Selected-pipeline scaling and distribution comparison | Persistent completion plus simplex banking at compute-first, balanced, and accuracy-first operating points | Node sweep 51/101/151/201; 3 isolated timing repeats per point; 500 trials per point at 101 nodes; 64 sliced projections; seed 5 | Measure full-pipeline runtime, setup, retained arrays, peak RSS, marginal Jensen--Shannon/Wasserstein distance, and full-vector sliced Wasserstein distance; select a default using accuracy and measured cost. |
| 10. Structured-closure selection | Dense persistent closure versus local modal bandwidths 4/8/12/20 at the accuracy-first operating point | 500 base trials; five held-out cases with 250 trials each; node scaling 51/101/151/201; 3 timing repeats; seed 6 | Replace dense time-indexed gains, test out-of-sample robustness, and require lower setup/storage/peak RSS without distribution error beyond the sampling-floor allowance. |
| 11. Frozen fitted-closure study | Shared bandwidth-12 frequency-binned regressions and empirical innovation banks trained on eight physical configurations | 100 training trajectories per configuration; unseen base with 500 trials; five unseen physical cases with 250 trials; node scaling 51/101/151/201; seed 7 | Remove analytic covariance from inference, test cross-configuration transfer, and compare frozen fitted inference with the analytic local closure and both random-walk baselines. |
| 12. Anchor-conditioned fitted closure | Twelve separate physical anchors; three-nearest interpolation; rank-12-plus-diagonal Gaussian innovations; vectorized target-mode features | 100 training trajectories per anchor; the identical unseen 500-trial base, five 250-trial physical held-outs, and 51/101/151/201 scaling gate; seed 7 | Test whether configuration conditioning and compressed learned innovations repair the first fitted model's transfer errors while making fitted inference competitive. |
| 13. Exact-anchor and persistent-innovation ablation | Rank 4/8/12/20 innovation factors at all 12 exact anchors; leave-one-anchor-out regressions; stable conditional residual dynamics; batched completion | All anchor residuals and five time bins; fresh 100-run ensembles at representative 51/101/201-node exact anchors; repeated frozen 500/250 and scaling gates; seed 7 | Separate innovation compression, temporal persistence, and coefficient-transfer error, then test whether conditional innovations and batching close the remaining accuracy and runtime gaps. |
| 14. Physics-conditioned fitted closure | One joint local regression conditioned on normalized grid size, impulse fraction, and dimensionless diffusion time; linear versus quadratic configuration-level cross-validation | 12 anchors with 100 trials each; three complete-configuration folds; identical frozen 500-trial base, five 250-trial held-outs, and 51/101/151/201 scaling; seed 7 | Avoid interpolation of unstable coefficient coordinates, select conditioning complexity without trajectory leakage, and use the error map to design targeted bridge anchors. |
| 15. Target-conditioned pre-lockbox gate | Physics-conditioned regression plus modal index, modal decay, retained fraction, and time-since-handoff features; four targeted bridge anchors | 16 anchors with 100 trials each; four complete-configuration folds; predeclared mean/max CV gates of 0.87/1.05; lockbox frozen but not opened | Test whether targeted coverage and target-level physics produce the required 20% transfer improvement before spending the new unbiased lockbox or optimizing runtime. |
| 16. Hardened structured-closure rerun | Dense persistent closure versus every base-eligible local bandwidth, with matched projection directions and case-specific reference floors | 500 base trials; five held-outs with 250 trials each; 200 paired bootstrap resamples; 7 timing repeats; bandwidths 12 and 20 carried through held-out validation | Correct the permissive promotion logic and determine whether local memory savings survive a statistically fair temporal-transfer gate. |
| 17. Resolution-aware mode scaling | Proportional retention, fixed 75, continuum spectral cutoff, and exact discrete post-handoff mean cutoff, each followed by local persistent completion | 101/151/201 nodes; 250 trials per policy/grid; two independent references per grid; 200 paired bootstrap resamples | Test whether retained dynamical modes can plateau with spatial refinement while analytic completion restores the unresolved stochastic field. |
| 18. Completion-rank/bandwidth scaling | Fixed 75 retained modes with ranks 5/10/20 and bandwidths 12/20 against the proportional rank-5 baseline | 100-trial factorial screen with seed 20261120; rank-20/bandwidth-12 finalist independently repeated for 250 trials with seed 20261121; 7 timing repeats in physical-refinement and fixed-step regimes | Determine whether mode-count growth can be traded for unresolved completion rank and whether that trade produces an actual computational benefit. |
| 19. Adaptive-rank cross-domain scaling | Fixed 75 retained modes with dense or local persistent completion; rank selected to capture 90% of conditional-innovation variance; proportional dense rank-5 baseline; length-4 and length-6 domains | 151/201 nodes; 100-trial selection with seed 20261130; fresh 250-trial confirmation with seed 20261201; 3 setup and 7 trajectory repeats with seed 20261202; local bandwidths 12/20 on length 4 scaled to 18/30 on length 6 | Replace an empirical fixed rank with a physics-based rule, test transfer across domain length, and require accuracy, setup, storage, and trajectory cost to pass before spending 2,000 trials. |

The original presentation suite did not include the completion-rank,
selected-pipeline, or adaptive-rank scaling studies; all have now been
completed separately.

## Main results

1. **Independent modal noise is inadequate.** At full rank it produced a
   variance relative error of 2.88, covariance error of 2.84, and negative
   values in 40% of reconstructed entries. Its mean relative error was 0.0053:
   small on the scale of the full concentration profile, but about 3.2 times
   the multinomial ensemble's mean error (0.00165). Future figures should show
   this difference directly rather than visually overlaying nearly coincident
   mean curves.
2. **Correlated modal dynamics recover the full-rank first two moments.** The
   correlated model reached mean/variance/covariance relative errors of
   0.00164/0.060/0.121, essentially the multinomial ensemble's
   0.00165/0.062/0.121. The banked model eliminated negative outputs with only
   a small accuracy change and a mean bank magnitude below 0.1% of mass.
3. **Raw truncation loses stochastic structure.** A 101-to-50 handoff at step
   10 gave mean/variance/covariance errors of approximately
   0.0038/0.56/0.53 and negative spatial values.
4. **Gaussian completion restores same-time statistics.** Independent and
   persistent completions reduced variance error to 0.063 and 0.061 and
   covariance error to 0.121 and 0.124, respectively, near the particle
   reference. They still produce Gaussian, noninteger outputs with small
   negative mass.
5. **Persistence is essential across time.** At lag 1, persistent completion's
   cross-time covariance error was 0.056, close to the multinomial ensemble's
   0.054. Independent completion remained at 0.468 because it redraws omitted
   modes at every output time.
6. **There is not yet a speed advantage.** At 101 nodes, median trajectory
   times were about 20 ms for the multinomial, full correlated, and raw handoff
   pipelines and 24–25 ms for completed handoffs. Persistent completion held
   about 6.5 MiB of arrays versus 0.31 MiB for the full correlated model.
7. **The 5,000-trial study separates sampling error from model error.** At
   5,000 trials, mean/variance/covariance errors were
   0.00060/0.021/0.038 for the multinomial reference,
   0.00055/0.020/0.038 for correlated modal, and
   0.00064/0.020/0.036 for banked correlated modal. The independent model's
   errors were 0.00179/2.876/2.667. Its sampled mean error was 3.23 times the
   multinomial reference standard-error norm, largely because its fluctuations
   are overdispersed; its expected mean update remains analytically correct.
8. **The local closure is a memory result, not the production accuracy
   default.** Bandwidths 12 and 20 pass the 500-trial base gate and four of five
   held-outs, but both fail long-domain lag covariance. Their errors are 0.1545
   and 0.1176 versus 0.0912 for dense; the allowed case-specific difference is
   0.0183. Dense persistent completion is therefore the default.
9. **Mean-relevant modes nearly plateau, but stochastic completion does not.**
   The exact 0.5% worst-time post-handoff mean cutoff selects 35/34/33 modes at
   101/151/201 nodes, while the proportional rule selects 75/112/149. With
   completion rank 5, the low-mode policies fail variance and temporal gates
   as the grid is refined.
10. **Fixed retained modes can recover accuracy by increasing completion rank,
    but not current performance.** On the base length-4 domain under local
    analytic completion, fixed 75 modes with rank 20 passes an independent
    fresh-seed 250-trial noninferiority gate at 151 and 201 nodes. At 201 nodes,
    its variance/covariance/lag errors are 0.0957/0.3207/0.1003 versus
    0.0977/0.3264/0.0902 for the proportional baseline. It takes 2.80 s to set
    up, retains 16.4 MiB, and runs in 0.1429 s, versus 1.00 s, 6.8 MiB, and
    0.1403 s for proportional modes with rank 5. This is a scaling experiment,
    not a replacement for the dense production closure.
11. **The physics-based rank rule is accurate, but the exact factorization is
    not yet cheaper.** Capturing at least 90% of positive conditional-innovation
    variance selects ranks 7/10/13 and 12/17/21 at 151/201 nodes on length 4,
    and 5/7/9 and 8/11/14 on length 6. Dense adaptive and physically scaled
    wide-local completion pass independent 250-trial confirmation in all four
    case/grid combinations. At 201 nodes, dense adaptive uses 1.218--1.227 times
    baseline setup and 1.833--1.926 times resident storage. Wide local reduces
    storage to 0.435--0.553 times baseline but uses 4.270--6.606 times setup and
    1.065--1.079 times trajectory runtime. Neither passes the cost gate, so the
    2,000-trial promotion is intentionally not run.

## Completed follow-up

The first four recommendations from the November review are complete. Model
figures now show log-scale, time-resolved mean-profile error rather than only
overlaid mean curves. Aggregate, maximum-time, and final-time mean/variance
errors are saved with bootstrap uncertainty intervals. The nested convergence
study reached 5,000 trials for all four core methods and shows that the
independent variance/covariance errors plateau while the other methods continue
to converge with the multinomial sampling floor. The handoff predictor achieved
correlations of 0.999/0.9995/0.998 for mean/variance/covariance on 16 held-out
configurations. Persistent completion followed by a simplex bank now produces
nonnegative, exactly mass-conserving outputs without materially changing the
recovered moments.

Three 500-trial completed operating points are retained: compute-first
(35 modes, handoff 20, completion rank 20), balanced (50, 10, rank 10), and
accuracy-first (75, 30, rank 5). Their mean/variance/covariance errors are all
near the appropriate ensemble sampling floor. The compute-first point uses
47.7% of full modal updates, versus 54.6% for balanced and 82.0% for
accuracy-first.

The exact full-pipeline benchmark changes the cost interpretation. At 101
nodes, amortized times were 0.0581/0.0598/0.0571 s for
compute-first/balanced/accuracy-first. All three distribution scores were
within one independent-reference sampling-floor score of the best observed
score. Accuracy-first is therefore the selected default: it had the smallest
mean error (0.00176), smallest variance error (0.0615), and lowest measured
amortized runtime, despite retaining more dynamic modes. Its cheaper rank-5
completion offsets that additional modal work.

Scaling also exposes the present bottleneck. At 201 nodes all three pipelines
used approximately 433--435 MiB peak RSS, while retained NumPy arrays ranged
from 27.7 MiB for accuracy-first to 62.4 MiB for compute-first. The difference
is temporary dense linear-algebra workspace and the time-indexed analytic
completion construction, so modal rank reduction alone will not deliver a
scalable implementation.

The structured follow-up resolves most of the memory bottleneck but does not
pass the hardened transfer gate. Bandwidth 12 produced normalized sliced
Wasserstein error 0.0802 versus 0.0807 for dense completion on 500 trials. At
201 nodes it reduced retained arrays from 27.72 to 9.99 MiB and sampled peak RSS
from 435.46 to 88.23 MiB. Case-specific floors and paired intervals expose a
long-domain lag-covariance failure, however: bandwidths 12 and 20 give errors
0.1545 and 0.1176 versus 0.0912 for dense. Dense completion is therefore the
production default; local completion remains an explicit memory experiment.

The first fitted replacement did not pass. On the unseen 500-trial base case,
its variance/covariance/lag-1 errors were 0.230/0.412/0.366, compared with
acceptance thresholds 0.129/0.337/0.145. Similar temporal and covariance
failures appeared on unseen impulses, particle counts, diffusivity, grid, and
domain tests. Fitted setup at 201 nodes fell to 0.038 seconds, but trajectory
time increased to 0.714 seconds versus 0.162 seconds for analytic-local. The
dense analytic completion is now the production default after the later
hardened held-out rerun.

The anchor-conditioned successor improved the unseen-base
variance/covariance/lag-1 errors to 0.208/0.380/0.339 and sliced Wasserstein
error from 0.110 to 0.099, but the declared thresholds were
0.129/0.337/0.145. At 201 nodes, vectorization reduced fitted trajectory time
from 0.714 to 0.246 seconds and setup to 0.034 seconds; analytic-local still ran
in 0.162 seconds, so the 1.25x runtime gate was also missed. Peak RSS was 106.6
MiB for anchored-fitted versus 116.2 MiB for analytic-local.

The error-source ablation shows that innovation rank is not the limiting
factor. Exact-anchor covariance reconstruction error was 0.091/0.0106/0.00106/
0.000031 at ranks 4/8/12/20, while leave-one-anchor-out initial and transition
coefficient errors were 1.09 and 1.01. Conditional residual persistence reduced
residual lag error from 1.00 to 0.56, but unseen-base end-to-end
variance/covariance/lag-1 errors changed only to 0.206/0.381/0.336. Fresh
100-run exact-anchor ensembles were closer to analytic-local than the unseen
case, confirming that parameter transfer is the dominant error source.
Batching and precombined factors reduced 201-node runtime further to 0.224
seconds, versus 0.164 seconds for analytic-local, but still missed the 1.25x
gate.

The jointly physics-conditioned regression selected linear conditioning under
whole-configuration cross-validation; quadratic features overfit badly.
Unseen-base variance/covariance/lag-1 errors improved only to
0.205/0.377/0.322 and remained above the 0.129/0.337/0.145 thresholds. The
unseen-domain marginal Wasserstein error worsened to 0.412. At 201 nodes,
runtime was 0.223 seconds versus 0.160 seconds for analytic-local, so neither
promotion gate passed. The cross-validation map proposes four unique bridge
anchors spanning 114--176 nodes and intermediate impulse/domain parameters.

The 16-anchor target-conditioned successor did not pass its pre-lockbox gate.
Its fold errors were 0.976/1.022/1.016/1.020, for a mean of 1.008. The maximum
fold satisfied the 1.05 limit, but the mean missed the 0.87 requirement. This is
only about a 7% improvement over the previous 1.085 mean rather than the
predeclared 20%. The six-case lockbox remains unopened, and no runtime tuning
or promotion benchmark was performed for this rejected candidate.

The adaptive-rank follow-up resolves the earlier empirical choice of rank 20.
At each saved time it selects the smallest rank meeting a 90%-captured-energy
criterion for the exact conditional innovation. A 100-trial selection retained
dense adaptive completion and the physically scaled wide-local rule; the
narrower local rule failed the length-6, 201-node temporal gate. Both retained
candidates passed a disjoint-seed 250-trial confirmation at 151/201 nodes on
lengths 4 and 6.

The cost gate then produced the decisive negative result. Dense adaptive
completion keeps trajectory time within 1%, but its setup and storage exceed
the proportional rank-5 baseline. Shared CSR topology makes wide-local storage
smaller, but full-spectrum factor construction remains expensive and its
trajectory is 6--8% slower. With no candidate in the cross-domain promotion
set, stopping before 2,000 trials is the predeclared outcome rather than an
incomplete experiment.

## Ordered next steps

### 1. Implement an operator-based partial innovation eigensolver

The energy rule is validated, but setup still forms and diagonalizes the full
conditional innovation. Implement its covariance action without materializing
the full block, compute only the leading spectrum needed to meet the tolerance,
and validate captured energy, eigen-residuals, determinism, and fallback against
the current exact factors.

### 2. Optimize local gain construction without changing the physics gate

Keep the physical bandwidth scaling that repaired the held-out length-6 case,
but remove repeated dense work in local conditioning and profile packed-CSR
trajectory operations. The measured targets are setup below baseline and no
more than 5% trajectory overhead; do not loosen either margin.

### 3. Gate cost before spending another large ensemble

Repeat the three-construction and seven-trajectory timing stage first. If the
new partial factor differs numerically, also rerun the independent 250-trial
accuracy confirmation. Run 2,000 trials only after a candidate passes accuracy,
setup, resident-storage, and trajectory gates in both domain lengths.

### 4. Extend spatial scaling only after the implementation passes

Then extend analytic rank and isolated timing beyond 201 nodes before adding
Monte Carlo work. This directly tests whether fixed retained dynamics plus
adaptive stochastic rank improves scaling as the spatial grid grows.

### 5. Preserve the reference hierarchy and learned-model lockbox

Do not inspect `configs/fitted_closure_lockbox.csv` through model evaluation.
If the learned-closure effort continues, first specify a nonlinear conditional
operator, analytic-teacher distillation, or another materially different model
and pass configuration CV before opening the lockbox.
Dense analytic persistent completion remains the trusted production reference
until a cheaper candidate passes all gates.

# Appendix A: mathematical framework

## A1. Discrete diffusion and the multinomial reference

Let `n_t` be the vector of node counts, `A` the positive-semidefinite Neumann
Laplacian scaled by `D/dx^2`, and `dt` the timestep. The implemented one-step
transition is

```text
P = I - dt A,                 E[n_t] = P^t n_0.
```

If `N` independent particles start at one node and `p_t` is the single-particle
node-probability vector, then

```text
n_t ~ Multinomial(N, p_t)
Cov[n_t] = N (diag(p_t) - p_t p_t^T).
```

For `s >= t`, the exact cross-time covariance is

```text
Cov[n_t, n_s] = N (diag(p_t) P^(s-t) - p_t p_s^T).
```

This aggregated random walk is the stochastic reference. It has the same
node-count distribution as tracking particles individually but does not retain
particle identities.

## A2. Spectral representation

For the symmetric operator `A = V Lambda V^T`, modal coordinates are
`m = V^T n`, and the deterministic discrete update is

```text
E[m_(t+1)] = (I - dt Lambda) E[m_t].
```

Keeping only the first `M` slow modes gives `n_t ~= V_M m_M,t`. The constant
mode is retained, so modal evolution preserves total mass. Truncation and
signed eigenvectors do not guarantee nonnegative counts.

## A3. Error measures used in the experiments

For estimated mean `mu_hat`, variance vector `v_hat`, and covariance `C_hat`,
the principal normalized errors are

```text
mean error       = ||mu_hat - mu||_2 / ||mu||_2
variance error   = ||v_hat - v||_2 / ||v||_2
covariance error = ||C_hat - C||_F / ||C||_F.
```

The code also reports RMSE, total-mass error, minimum reconstructed count,
fraction of negative entries, negative mass divided by total mass, physical
bound/integer violations, and bank/completion residual magnitude. Cross-time
covariance uses the same relative Frobenius norm at each lag. Finite ensembles
give the multinomial reference nonzero empirical error, which is a useful noise
floor rather than a model defect. The worst-time variance score excludes saved
times whose exact variance norm is at most `1e-12` of its maximum; this removes
the deterministic initial frame and roundoff-scale denominators without
discarding physically stochastic times.

# Appendix B: model theory and experiment hypotheses

## B1. Independent modal model (original formulation)

Each nonconstant mode is represented by positive and negative state
populations, `m_k = w(q_k+ - q_k-)`. Each unit flips sign independently with
probability

```text
p_k = lambda_k dt / 2.
```

Therefore `E[m_k,t+1] = (1 - lambda_k dt) E[m_k,t]`, which gives the intended
mean. The model assumes independent modal fluctuations even though the exact
multinomial modal covariance `V^T Cov[n] V` is generally not diagonal. Experiment
1 tests the resulting hypothesis: correct modal means are not sufficient to
recover physical spatial variance and covariance.

## B2. Correlated modal model

The correlated model keeps the same mean update but samples one shared spatial
Gaussian debit-credit fluctuation with the conditional covariance of
left/stay/right particle transfers. It projects that fluctuation into the
retained basis:

```text
m_(t+1) = (I - dt Lambda_M) m_t + eta_m,t
eta_m,t = V_M^T eta_n,t.
```

Because all modal components derive from the same spatial transfers, their
noise is correlated. The Gaussian approximation matches the conditional first
two moments but does not guarantee nonnegative spatial counts. Experiment 1
tests whether shared noise repairs the independent model's moment errors at
full rank.

## B3. Banked correlated modal model

After each correlated update, reconstruct the raw spatial state, add the prior
bank balance, and project onto the fixed-mass nonnegative simplex

```text
S_N = {n : n_i >= 0 and sum_i n_i = N}.
```

The difference between the adjusted physical output and its modal
reconstruction becomes a spatial bank carried to the next step. This guarantees
nonnegative, mass-conserving output but stores one balance per node. Experiment
1 asks whether physical admissibility can be obtained without materially
distorting the recovered moments; the bank L1 fraction makes the intervention
visible.

## B4. Full-to-reduced handoff

The model evolves `H` modes during the impulse-dominated transient, then
projects into a nested `M`-mode basis at `t_h`:

```text
m_H,t+1 = (I - dt Lambda_H) m_H,t + eta_H,t        for t < t_h
m_M,t_h = V_M^T V_H m_H,t_h
m_M,t+1 = (I - dt Lambda_M) m_M,t + eta_M,t        for t >= t_h.
```

High-frequency modes decay rapidly, so a later handoff or larger `M` should
reduce projection error at greater computational cost. Experiment 3 measures
this tradeoff. The proposed predictive rule measures omitted mean and covariance
energy directly and tests whether they forecast downstream ensemble error.

# Appendix C: reduced-model readouts

Readouts transform an already generated modal trajectory. Except for the
banked correlated dynamics in B3, they do not feed corrections back into the
modal evolution. Experiment 2 applies each readout to the same type of
101-to-50 handoff so differences can be attributed to the readout.

## C1. Raw readout

```text
n_raw,t = V_M m_M,t.
```

This is the unmodified reduced reconstruction and exposes truncation error,
negative values, and moment loss. It is the baseline for all readout claims.

## C2. Simplex-bank readout

For raw output `x_t` and previous residual `e_(t-1)`, form

```text
z_t = x_t + e_(t-1)
y_t = projection of z_t onto S_N
e_t = z_t - y_t.
```

The output is continuous, nonnegative, and mass conserving. The residual carries
the correction forward rather than discarding it, but the output projection can
alter stochastic moments. Experiment 2 measures both the moment change and the
residual L1 fraction.

## C3. Unresolved Gaussian completion

Partition the full modal state into retained `r` and unresolved `u` components.
Using exact multinomial modal mean and covariance blocks, sample

```text
u | r ~ Normal(mu_u + G(r - mu_r), C_u|r)
G     = C_ur (C_rr + ridge)^(-1)
C_u|r = C_uu - G C_ru
n_tilde = V_R r + V_U u.
```

Only the leading `r_c` eigenvectors of `C_u|r` need be sampled; the presentation
uses completion rank `r_c = 10`. Because the constant mode is retained, the
completion is zero-sum and preserves mass. It is an oracle based on exact
reference moments and redraws `u` independently at every time. Experiments 2
and 4 test its expected strength (same-time moments) and weakness (temporal
persistence).

## C4. Persistent unresolved Gaussian completion

At the completion start, sample `u_t | r_t` as above. Subsequently propagate an
unresolved latent state conditioned on its own prior value and the observed
retained transition:

```text
y_t = [u_t, r_t, r_(t+1)]
u_(t+1) | y_t ~ Normal(mu_u,t+1 + K_t(y_t - mu_y,t), C_u,t+1|y)
K_t = C_(u,t+1),y (C_yy + ridge)^(-1).
```

This preserves time dependence while leaving the retained trajectory unchanged.
The innovation covariance is sampled at configurable low rank (10 in the
presentation run). Experiment 4 tests cross-time covariance and lag correlation
at lags 1, 2, 5, 10, and 20. The method remains an analytic Gaussian oracle and
does not guarantee integer or nonnegative counts.

## C5. Persistent completion with a physical simplex readout

The completed physical readout composes persistent Gaussian completion with
the temporal simplex bank. If `n_complete,t` is the completed field and
`e_(t-1)` is the constraint ledger,

```text
z_t = n_complete,t + e_(t-1)
n_physical,t = project_to_simplex(z_t, N)
e_t = z_t - n_physical,t.
```

This guarantees nonnegative output and exact mass conservation. The Gaussian
completion magnitude and simplex constraint ledger are recorded separately.
Across the selected 500-trial operating points, the constraint eliminated all
negative mass while changing mean, same-time covariance, and cross-time
covariance errors only minimally.

# Appendix D: distribution-level comparison and default selection

For scalar node-count samples `X` and `Y`, the empirical Wasserstein-1 distance
is the mean absolute separation of their quantile functions,

```text
W1(X,Y) = integral_0^1 |F_X^(-1)(q) - F_Y^(-1)(q)| dq.
```

The reported normalized value divides by the reference standard deviation
(with a one-particle lower scale bound). Common-bin Jensen--Shannon divergence
uses `M = (P + Q)/2` and

```text
JS(P,Q) = 0.5 KL(P || M) + 0.5 KL(Q || M).
```

Unlike raw KL between the original samples, this remains finite when the
continuous completed output and discrete multinomial reference do not share
support. For full spatial vectors, 64 random zero-sum unit directions `theta`
are drawn and the normalized one-dimensional Wasserstein distances of
`theta^T n` are averaged. Removing the constant direction prevents exact mass
conservation from dominating the sliced metric.

For the historical three-operating-point screen, the default-selection score
is the mean of normalized marginal and sliced
Wasserstein distance. A pipeline is accuracy-eligible if its score lies within
one independently measured multinomial-versus-multinomial sampling-floor score
of the best pipeline. The eligible point with the smallest 100-run-amortized
runtime is selected. Under the 500-trial study, all three points were eligible
and accuracy-first was fastest. The later dense-versus-local promotion decision
uses the stricter case-specific, paired noninferiority gates described next;
the earlier heuristic does not override that result.

# Appendix E: structured local persistent closure

Let `k` denote an unresolved modal index and let `J_k` contain conditioning
coordinates whose global modal index differs from `k` by at most bandwidth
`b`. The local gain row is

```text
K[k,J_k] = Cov(u_(t+1), y[J_k])
           (Cov(y[J_k], y[J_k]) + ridge)^(-1),
K[k,j]   = 0 for j not in J_k.
```

The local row systems are padded and solved in one batch per transition, then
stored as index/value arrays. Sparse gain application is therefore

```text
(K y)[k] = sum_(j in J_k) K[k,j] y[j],
```

rather than multiplication by a dense time-indexed matrix. The innovation
covariance remains `C_uu - K C_yu` and is sampled through the same configurable
low-rank factor. With `b` spanning every modal index, this construction is
numerically equivalent to the dense persistent closure. The original
Experiment 10 screen profiled `b = 12`, but the hardened Experiment 16
comparison carried both base-eligible bandwidths (`b = 12` and `b = 20`)
through all held-out cases. Both failed the long-domain lag-covariance gate, so
neither local bandwidth was selected and the dense closure remains the
production default. For each held-out physical case, a second independent
multinomial ensemble defines the finite-sample error floor. Candidate-minus-
dense differences in mean, variance, and covariance error use paired
block-bootstrap confidence intervals; temporal and distributional margins are
scaled from that same case-specific floor. Projection directions are identical
for reference, dense, and local comparisons.

# Appendix F: frozen trajectory-fitted closure

For each training trajectory, modal deviations are centered by deterministic
mean propagation and normalized by particle-number scale,

```text
z_t = (m_t - E[m_t]) / sqrt(N).
```

Targets are assigned to normalized-frequency bins. Within each bin, ridge
regression fits the same bandwidth-12 initial and transition feature maps used
by the structured analytic closure. Transition features contain local previous
unresolved coordinates and retained coordinates at the current and next time.
Regression residual vectors are interpolated to a fixed normalized-frequency
grid and stored in five time-indexed empirical innovation banks.

At inference, the fitted model uses only deterministic mean propagation,
frozen regression coefficients, and resampled innovation vectors. Exact
multinomial covariance and analytic conditional gains are not constructed.
Experiment 11 showed that this unconditional cross-configuration sharing is
too coarse: setup becomes inexpensive, but variance, covariance, and temporal
statistics do not transfer reliably. The artifact is retained as a negative
baseline for configuration-conditioned successors.

# Appendix G: anchor-conditioned fitted closure

Each anchor `a` has a standardized physical descriptor containing log grid
size, impulse fraction, log diffusivity, log domain length, log dimensionless
diffusion time, and log particle count. For query descriptor `q`, the three
nearest anchors receive inverse-square weights

```text
w_a = d(q, q_a)^(-2) / sum_b d(q, q_b)^(-2).
```

Initial and transition regression coefficients are interpolated linearly,
`B(q) = sum_a w_a B_a`. For each anchor and time bin, centered innovation
samples are compressed by SVD into a rank-12 factor `F_a` plus diagonal
remainder `D_a`, so

```text
C_a approximately F_a F_a^T + diag(D_a^2).
```

The interpolated innovation is sampled as

```text
epsilon = sum_a w_a mu_a
        + sum_a sqrt(w_a) (F_a z_a + D_a eta_a),
```

with independent standard Gaussian `z_a` and `eta_a`. Consequently its
within-anchor covariance is the weighted mixture `sum_a w_a C_a`, while its
mean and regression coefficients vary smoothly between anchors. Experiment 12
shows that this helps but does not supply enough persistent conditional
structure to match the analytic local closure.

# Appendix H: persistent innovation and error-source ablation

Let `e_t` be the fitted transition residual on the normalized unresolved-mode
grid. Within time bin `b`, the persistent successor fits

```text
e_t = mu_b + G_b (e_(t-1) - nu_b) + F_b z_t + D_b eta_t,
```

where `G_b` is a ridge-regression transition whose singular values are clipped
to 0.95, `F_b` is a low-rank conditional-innovation factor, and `D_b` is the
diagonal variance remainder. This gives a stable state-space residual process
while retaining the learned same-time marginal variance.

The ablation evaluates the covariance

```text
C_model = G_b C_previous G_b^T + F_b F_b^T + diag(D_b^2)
```

at ranks 4, 8, 12, and 20 and compares `C_model` with held-out empirical
residual covariance at the exact fitted anchor. It separately removes each
anchor and interpolates its regression coefficients from neighboring anchors.
The sharp rank convergence and order-one coefficient errors identify
coefficient transfer as the dominant failure.

Inference now precombines weighted anchor factors during setup and advances a
run-by-time-by-node ensemble with one loop over time. This removes the former
Python loop over trajectories. The production benchmark still evaluates one
trajectory, so further performance work must reduce per-step feature gathering
and sampling rather than only increasing ensemble batch size.

# Appendix I: jointly physics-conditioned local regression

For configuration coordinate

```text
q = [log(n_nodes), impulse_fraction, log(D T / L^2)],
```

the conditioned model standardizes `q` over training configurations and forms
a basis `phi(q)`. The selected linear basis is `[1, q]`; the rejected quadratic
basis additionally contains squares and pairwise products. A local modal
feature vector `x` is lifted to

```text
x_conditioned = phi(q) tensor_product x,
```

and one ridge regression is fitted jointly across every physical anchor within
each normalized-frequency bin. At inference, contracting the learned tensor
with `phi(q)` yields the local coefficient row for the requested physics. This
fits prediction behavior directly instead of interpolating separately fitted
coefficient vectors.

Cross-validation holds out complete configurations, not individual trajectory
rows. Consequently, all trajectories from a physical case are excluded
together and the score measures actual transfer. Linear conditioning reduced
some frozen errors, but fold scores remained near or above one and quadratic
conditioning extrapolated poorly. The resulting anchor-design stage places
new bridge configurations between the highest-error grid/domain regions and
their nearest covered neighbors.

# Appendix J: target-conditioned pre-lockbox gate

The final local-linear candidate augments the configuration coordinate with
target and transition information,

```text
q_target = [q_config,
            k / (n_nodes - 1),
            log(1 + lambda_k dt),
            retained_modes / n_nodes,
            (t - t_handoff) / (T - t_handoff)].
```

The linear basis `phi(q_target)` is interacted with the same local modal
features used by the sparse closure. Four bridge configurations generated from
the earlier CV error map increase the training design from 12 to 16 physical
anchors. Each anchor contributes 100 independent multinomial trajectories.

Before training changes, a six-case lockbox and all seeds were written to
`configs/fitted_closure_lockbox.csv`. The predeclared configuration-CV gate was

```text
mean fold error <= 0.87,     maximum fold error <= 1.05.
```

The observed mean/maximum were 1.008/1.022. Since both conditions were required,
the workflow returned after writing `prelockbox_summary.json`; it did not load
the lockbox manifest into validation, generate lockbox trajectories, or run a
candidate scaling benchmark. This preserves the lockbox for a materially new
future architecture and prevents repeated adaptation to the final test set.

# Appendix K: resolution-aware mode and completion scaling

For a handoff at time `t_h`, let `a_k = 1 - dt lambda_k` be the exact
discrete-time decay factor and let `mu_k(t)` denote the deterministic modal
mean. The exact mean-cutoff policy chooses the smallest retained count `M`
such that the worst post-handoff projection error satisfies

```text
max_(t >= t_h) ||V_(k>=M) mu_(k>=M)(t)||_2 / ||V mu(t)||_2 <= epsilon_mean.
```

The continuum comparison uses `exp(-D (pi k/L)^2 t_h) <= epsilon_decay`, which
gives the approximate cutoff

```text
M >= L / (pi sqrt(D t_h)) sqrt(log(1 / epsilon_decay)).
```

These rules concern deterministic mean decay only. The exact multinomial modal
covariance is

```text
C_m(t) = V^T N (diag(p_t) - p_t p_t^T) V,
```

so high modes can remain important to stochastic variance even after their
means have decayed. Experiment 17 tests whether analytic unresolved completion
can restore that covariance while the retained dynamics plateau.

Each resolution receives two independent multinomial references, so its
sampling floor matches the physical case and trial count. The candidate and
proportional baseline share projection directions, and paired block-bootstrap
intervals are applied to differences in mean, variance, and covariance error.
Worst-time, temporal, distributional, mass, and nonnegativity checks use the
same case-specific sampling-floor noninferiority margins.

Experiment 18 separates two approximation dimensions:

```text
M = number of modes advanced dynamically,
r = rank of the conditional unresolved innovation factor.
```

Holding `M = 75` while increasing `r` from 5 to 20 tests whether stochastic
resolution can move from the dynamic state into the analytic readout. The
250-trial result shows that it can recover accuracy at 151 and 201 nodes, but
not yet reduce cost. The scaling decision therefore requires both an accuracy
gate and measured setup, trajectory, and resident-storage gates; fewer dynamic
modes alone are not evidence of a faster reduced model.

Two timing regimes separate spatial algebra from timestep refinement. Physical
refinement holds domain, diffusivity, stability safety, and final physical time
fixed, so finer grids require more stable time steps. Fixed-step timing holds
the number of transitions fixed at 100 and reports seconds per transition.
Agreement between the regimes distinguishes spatial cost from the unavoidable
increase in step count of an explicit diffusion discretization.

# Appendix L: adaptive conditional-innovation rank and cross-domain gate

## L1. Conditional innovation

Let `r_t` be retained modal coordinates and `u_t` unresolved coordinates. At
the completion start, conditioning the exact multinomial modal covariance gives

```text
K_0 = C_ur (C_rr + rho s_rr I)^(-1),
S_0 = sym(C_uu - K_0 C_ru),
u_0 = mu_(u|r) + F_0 z_0,          F_0 F_0^T approx S_0.
```

Here `s_rr = tr(C_rr) / dim(C_rr)` is the covariance scale, `rho` is the
predeclared ridge, `sym(A) = (A + A^T) / 2`, and `z_0` is standard normal. At
later saved times the persistent closure conditions `u_(t+1)` on the available
persistent state
`y_t = [u_t, r_t, r_(t+1)]`. With

```text
K_t = C_(u_next,y) (C_yy + rho s_yy I)^(-1),
S_t = sym(C_(u_next,u_next) - K_t C_(y,u_next)),
```

the update is the conditional mean plus an innovation factor `F_t z_t`.
This is a physics-derived Gaussian moment closure: `C` comes from the exact
multinomial law, not a fitted trajectory model.

## L2. Captured-energy rank rule

Let the positive eigenvalues of `S_t` be ordered
`lambda_(t,1) >= ... >= lambda_(t,q) > 0`. For a predeclared residual-variance
tolerance `epsilon`, the selected rank is

```text
r_t(epsilon) = min {r :
    sum_(i=1)^r lambda_(t,i) / sum_(i=1)^q lambda_(t,i)
    >= 1 - epsilon}.
```

Experiment 19 uses `epsilon = 0.10`. The configured completion rank is only a
hard upper bound; the reported minimum/median/maximum `r_t` values are the
actual factor dimensions. If the cap prevents the inequality from being met,
the tolerance diagnostic fails and the policy cannot be promoted. When no
energy tolerance is configured, fixed-rank behavior is unchanged.

## L3. Shared sparse topology

For a local gain, each row uses the same allowed modal neighborhood at every
saved transition. Its structure is therefore stored once as CSR row pointers
`p` and column indices `j`, while only the values `g_t` vary with time:

```text
(K_t x)_i = sum_(ell=p_i)^(p_(i+1)-1) g_(t,ell) x_(j_ell).
```

This removes repeated padded indices and padded zero entries. It reduces
resident storage without approximating the gain entries, but it does not by
itself avoid construction of the dense conditional covariance or its full
eigendecomposition. That remaining setup work explains the timing result.

## L4. Cross-domain promotion rule

Selection uses 100 trials and confirmation uses a disjoint seed with 250
trials. A policy is accuracy-eligible only if it passes every noninferiority
gate at 151 and 201 nodes in both the length-4 and length-6 cases. At the
largest grid in each case, define ratios to the proportional dense rank-5
baseline:

```text
R_setup = median candidate setup / median baseline setup,
R_mem   = candidate resident arrays / baseline resident arrays,
R_run   = median candidate trajectory / median baseline trajectory.
```

The cost gate requires `R_setup < 1`, `R_mem < 1`, and `R_run <= 1.05` in both
domains. Only the intersection of the accuracy and cost sets may advance to
2,000 trials. The observed intersection is empty, so stopping is part of the
experimental design.
