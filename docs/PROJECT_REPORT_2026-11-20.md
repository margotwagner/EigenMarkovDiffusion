# EigenDiffusion project report

**Status as of November 20, 2026**

## Executive summary

EigenDiffusion tests whether stochastic diffusion can be represented in a
Laplacian eigenbasis instead of by explicitly tracking particles. The project
has progressed from the original independent-mode formulation to correlated
modal dynamics, physically constrained banking, adaptive modal reduction, and
Gaussian readouts for reconstructing unresolved modes. The package, command
line workflows, analytic references, diagnostics, and 89 tests are complete and
passing. A five-part presentation experiment suite was completed in July 2026.

The main scientific conclusion is that mean propagation is no longer the core
problem. The full correlated model reproduces the multinomial reference's mean,
variance, and same-time covariance to within ensemble sampling error. A reduced
50-mode model can recover those same-time statistics through Gaussian
completion, and persistent completion also recovers short-lag temporal
correlation. However, the completion is an analytic oracle, can produce
noninteger counts, consumes substantial setup memory, and is not yet
faster than the aggregated multinomial baseline.

## Where we left off

The latest completed study is based on commit `2138289` plus the operating-point
benchmark changes described here. The canonical presentation outputs are in
`outputs/presentation/`, and `run_presentation_results.sh` defines that
pipeline. All 89 automated tests pass.

Time-resolved errors, bootstrap uncertainty, and a 5,000-trial convergence
study, predictive handoff/rank rule, completion-rank study, exact pipeline
benchmark, and distribution-level comparison are now complete. The next phase
has replaced the dense gain with a validated bandwidth-12 structured closure.
The first frozen trajectory-fitted replacement was implemented and rejected by
the predeclared accuracy and runtime gates.

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
- Added a row-sparse local persistent closure, profiling, held-out validation,
  and isolated scaling; bandwidth 12 is now the selected-pipeline default.
- Added a frozen trajectory-fitted sparse closure and tested it on unseen
  physics; it remains experimental because variance, covariance, temporal, and
  runtime gates failed.
- Generated five presentation figures with CSV data and supporting NPZ arrays.

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
| 7. Predictive handoff selection | Analytic modal-energy predictor validated against correlated handoff ensembles | 35, 50, 60, 75 retained modes; handoff steps 10, 15, 20, 30; 16 held-out configurations; 250 trials each | Predict post-handoff mean, variance, and covariance loss before running an ensemble and identify the accuracy/work Pareto frontier. |
| 8. Completion and physical-output selection | Independent/persistent completion, with and without simplex banking | Completion-rank sweeps plus 500-trial headline checks at 35/50/75 modes | Select completion rank for compute/balanced/accuracy operating points and test whether nonnegativity and mass conservation distort stochastic moments. |
| 9. Selected-pipeline scaling and distribution comparison | Persistent completion plus simplex banking at compute-first, balanced, and accuracy-first operating points | Node sweep 51/101/151/201; 3 isolated timing repeats per point; 500 trials per point at 101 nodes; 64 sliced projections; seed 5 | Measure full-pipeline runtime, setup, retained arrays, peak RSS, marginal Jensen--Shannon/Wasserstein distance, and full-vector sliced Wasserstein distance; select a default using accuracy and measured cost. |
| 10. Structured-closure selection | Dense persistent closure versus local modal bandwidths 4/8/12/20 at the accuracy-first operating point | 500 base trials; five held-out cases with 250 trials each; node scaling 51/101/151/201; 3 timing repeats; seed 6 | Replace dense time-indexed gains, test out-of-sample robustness, and require lower setup/storage/peak RSS without distribution error beyond the sampling-floor allowance. |
| 11. Frozen fitted-closure study | Shared bandwidth-12 frequency-binned regressions and empirical innovation banks trained on eight physical configurations | 100 training trajectories per configuration; unseen base with 500 trials; five unseen physical cases with 250 trials; node scaling 51/101/151/201; seed 7 | Remove analytic covariance from inference, test cross-configuration transfer, and compare frozen fitted inference with the analytic local closure and both random-walk baselines. |

The original presentation suite did not include the completion-rank and
selected-pipeline studies; both have now been completed separately.

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

The structured follow-up resolves most of that implementation bottleneck.
Bandwidth 12 produced normalized sliced Wasserstein error 0.0802 versus 0.0807
for dense completion on 500 trials and passed all five held-out distribution
checks. At 201 nodes it reduced retained arrays from 27.72 to 9.99 MiB, sampled
peak RSS from 433.01 to 85.75 MiB, and setup time from 1.195 to 1.117 seconds.
It is therefore the selected-pipeline default. The dense closure is retained
as a reproducible scientific reference.

The first fitted replacement did not pass. On the unseen 500-trial base case,
its variance/covariance/lag-1 errors were 0.230/0.412/0.366, compared with
acceptance thresholds 0.129/0.337/0.145. Similar temporal and covariance
failures appeared on unseen impulses, particle counts, diffusivity, grid, and
domain tests. Fitted setup at 201 nodes fell to 0.038 seconds, but trajectory
time increased to 0.714 seconds versus 0.162 seconds for analytic-local. The
analytic bandwidth-12 closure therefore remains the production default.

## Ordered next steps

### 1. Condition fitted parameters on physical configuration

Replace the single shared regression and innovation bank with sparse anchor
models over dimensionless diffusion time, impulse location, domain scale, and
grid resolution. Interpolate parameters between nearby anchors.

### 2. Learn low-rank innovations instead of resampling vectors

Fit configuration-conditioned low-rank innovation factors. This should improve
variance/covariance transfer and avoid storing or interpolating empirical
innovation vectors.

### 3. Vectorize fitted inference and repeat the frozen gates

Batch feature construction across target modes and times, then repeat the same
unseen accuracy suite and 51--201-node benchmarks against analytic-local,
aggregated multinomial, and naive particle tracking. Do not promote the fitted
model unless both accuracy and runtime gates pass.

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
floor rather than a model defect.

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

The default-selection score is the mean of normalized marginal and sliced
Wasserstein distance. A pipeline is accuracy-eligible if its score lies within
one independently measured multinomial-versus-multinomial sampling-floor score
of the best pipeline. The eligible point with the smallest 100-run-amortized
runtime is selected. Under the 500-trial study, all three points were eligible
and accuracy-first was fastest.

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
numerically equivalent to the dense persistent closure. Experiment 10 selected
`b = 12` under the declared sampling-floor accuracy and scaling gates.

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
