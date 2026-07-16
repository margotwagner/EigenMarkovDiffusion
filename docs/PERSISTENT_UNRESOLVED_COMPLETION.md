# Persistent unresolved Gaussian completion

## Motivation

The rank-10 same-time unresolved completion restores the marginal mean,
variance, and same-time covariance after a 101-to-50 modal handoff. However, it
samples a fresh omitted field at every saved time. The temporal-correlation
diagnostics show that this underestimates persistence at short lags.

`persistent_unresolved_completion` keeps the existing reduced retained
trajectory fixed and adds a latent unresolved state that evolves through time.
It is an output-only analytic closure; it does not feed back into the modal
dynamics.

## Gaussian transition model

Write the full modal state as

\[
m_t = \begin{bmatrix}r_t \\ u_t\end{bmatrix},
\]

where `r_t` contains the retained modes and `u_t` contains the omitted modes.
At the completion start time, the unresolved state is sampled from the existing
same-time conditional model

\[
u_t \mid r_t \sim
\mathcal N\!\left(
\mu_{u,t} + G_t(r_t-\mu_{r,t}),
\Sigma_{u\mid r,t}
\right).
\]

For every later step, the next unresolved state is conditioned on its previous
value and the observed retained transition:

\[
u_{t+1}\mid u_t,r_t,r_{t+1}
\sim \mathcal N\!\left(
\mu_{u,t+1}+K_t(y_t-\mu_{y,t}),
\Sigma_{u_{t+1}\mid y_t}
\right),
\]

with

\[
y_t = [u_t, r_t, r_{t+1}]^\top.
\]

The gain and residual covariance are

\[
K_t = \Sigma_{u_{t+1},y_t}
\left(\Sigma_{y_t,y_t}+\gamma s_t I\right)^{-1},
\]

\[
\Sigma_{u_{t+1}\mid y_t}
= \Sigma_{u_{t+1},u_{t+1}}
- K_t\Sigma_{y_t,u_{t+1}}.
\]

All covariance blocks are computed from the exact multinomial same-time and
one-step cross-time covariance. The conditional innovation covariance is
factorized at configurable rank, so `--completion-rank 10` uses ten stochastic
innovation directions even though 51 modes are omitted.

The completed spatial output is

\[
\tilde n_t = V_R r_t + V_U u_t.
\]

Because the omitted basis excludes the constant mode, the addition is zero-sum
and total mass is preserved up to floating-point roundoff.

## Existing methods are preserved

The new readout does not replace:

- `unresolved_gaussian_completion`, which remains the independent same-time
  baseline;
- `raw`, Delta-Sigma, and bank readouts;
- any modal dynamics model.

## Recommended experiment

```bash
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
  --output outputs/persistent_completion/temporal_comparison_r500/comparison.png \
  --csv-output outputs/persistent_completion/temporal_comparison_r500/comparison.csv \
  --data-output outputs/persistent_completion/temporal_comparison_r500/comparison.npz
```

The output includes:

- multinomial random walk;
- full correlated modal dynamics;
- raw 101-to-50 handoff;
- independent same-time completion;
- persistent completion.

## Limitations

This remains an oracle closure because it uses known analytic random-walk
moments. It is Gaussian and therefore does not guarantee nonnegative integer
counts. It targets same-time and one-step temporal moments; validating longer
lags remains necessary. A future general closure would estimate these
transition statistics from simulation or derive them for reaction-diffusion
systems where analytic multinomial moments are unavailable.
