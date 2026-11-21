# Resolution-aware mode scaling

`eigendiffusion.mode_scaling` tests whether the physics-based completed modal
model can retain a grid-independent number of dynamical modes as the spatial
grid is refined.

This question is deliberately different from the raw truncation question. A
raw reduced trajectory must retain high modes to reproduce multinomial
covariance. The completed model instead retains slow dynamical modes and
restores unresolved covariance from the analytic multinomial law. The study
therefore reports both rules separately.

## Mode policies

- `proportional_75_of_101`: the existing baseline, scaled in proportion to the
  node count.
- `fixed_75`: a fixed absolute cutoff.
- `spectral_decay_1e-3`: the continuum diffusion-decay rule
  \(M \ge L [\pi\sqrt{D t_h}]^{-1}\sqrt{\log(1/\epsilon)}\).
- `discrete_mean_cutoff_0p005`: the smallest exact discrete cutoff whose
  post-handoff mean projection error is at most 0.005.
- `full_moment_predictor`: the existing raw mean/variance/covariance predictor,
  kept as a non-plateauing diagnostic baseline.

For the default physical case, the worst-time discrete mean rule selects
50/35/34/33 modes at 51/101/151/201 nodes. The full-moment predictor still
grows with resolution because high-mode stochastic covariance persists. The
ensemble stage determines whether local analytic completion restores that
covariance accurately when only the mean-selected modes are simulated.

## Results

The 250-run 101/151/201-node screen rejects every plateauing policy when
completion rank remains fixed at 5. The exact mean-cutoff policy has good
sampled-mean accuracy, but its unresolved variance and nearest-to-one-unit-lag
covariance degrade with refinement; only proportional retention passes all
grids.

A follow-up holds the dynamics at 75 modes and varies completion rank 5/10/20
and bandwidth 12/20. Rank 20 with bandwidth 12 passes an independent 250-run
noninferiority gate at both 151 and 201 nodes; selection used seed `20261120`
and confirmation used fresh seed `20261121`. Thus proportional growth in
retained dynamics can be replaced by growth in unresolved completion rank.
It does not yet produce a computational win: at 201 nodes, fixed-75/rank-20
requires 2.80 seconds of setup, 16.4 MiB of retained arrays, and 0.1429 seconds
per trajectory, versus 1.00 seconds, 6.8 MiB, and 0.1403 seconds for the
proportional/rank-5 baseline.

The current conclusion is not “the mean-selected modes are sufficient.” Retained mean
dynamics plateau, while the stochastic completion dimension must grow unless a
more efficient physics-derived innovation factor is found.

This is a controlled base-physics scaling study using the local bandwidth-12
analytic completion for both candidate and proportional baseline. Because that
closure fails the separate long-domain temporal-transfer gate, the result does
not establish transfer across domain length and does not replace dense analytic
completion as the production reference.

## Staged execution

Generate the analytic map without Monte Carlo:

```bash
PYTHONPATH=src python3 examples/run_mode_scaling.py \
  --stage analytic --nodes 51 101 151 201 \
  --output outputs/mode_scaling
```

Run a screening comparison. Each resolution receives its own two independent
multinomial ensembles, so the reported sampling floor matches the physical
case and trial count. Candidate moment errors are compared with the
proportional policy using paired block-bootstrap confidence intervals; the
remaining distribution, worst-time, temporal, and physicality diagnostics use
the same case-specific floor as a noninferiority margin.

```bash
PYTHONPATH=src python3 examples/run_mode_scaling.py \
  --stage screening --nodes 101 151 201 --runs 250 \
  --policies proportional_75_of_101 discrete_mean_cutoff_0p005 \
  --output outputs/mode_scaling
```

Only after a policy passes the 250-run accuracy gate and retains a plausible
cost advantage should it be advanced to the headline trial count:

```bash
PYTHONPATH=src python3 examples/run_mode_scaling.py \
  --stage final --nodes 101 151 201 --runs 2000 --timing \
  --policies proportional_75_of_101 <passing-policy> \
  --output outputs/mode_scaling_final
```

The mean-cutoff, spectral-cutoff, and fixed-75 rank-5 candidates did not
qualify for this 2,000-run stage.
The fixed-75/rank-20 follow-up passed the accuracy gate, but the seven-repeat
timing study showed no trajectory advantage and substantially worse setup and
storage, so spending 2,000 trials on that implementation would not change the
selection decision. A future matrix-free or lower-rank candidate should repeat
the 250-run gate, demonstrate a cost advantage, and then advance to 2,000 runs.
The accuracy stage uses fixed physical diagnostic positions and the nearest
realizable integer-step lag to a target physical lag of 1.0. `summary.json`
records the policies that pass the gate at every evaluated resolution; passing
the 250-run screen makes a policy
eligible for the 2,000-run final study, not automatically production-ready.
The timing stage writes separate rows for:

- fixed-duration physical refinement, where stable time-step refinement is
  part of the cost; and
- a fixed number of stable updates, where time per transition isolates spatial
  algebra scaling.

Do not interpret a three-repeat timing difference as a speedup. The default is
seven repeats, and close finalists should be repeated at least ten times.

## Completion-rank and bandwidth follow-up

If a low-mode policy fails because unresolved variance or lag covariance grows
with resolution, do not increase the retained dynamical modes immediately.
First hold the dynamical cutoff fixed and screen the two completion parameters:

```bash
PYTHONPATH=src python3 examples/run_completion_scaling.py \
  --nodes 151 201 --runs 100 --fixed-modes 75 \
  --ranks 5 10 20 --bandwidths 12 20 \
  --seed 20261120 \
  --output outputs/mode_scaling/completion_screen
```

The proportional policy remains the paired baseline. Variants at the same
fixed mode count reuse one raw correlated trajectory ensemble, so observed
differences isolate completion rank and conditioning bandwidth. A variant that
passes the 100-run screen must be rerun with 250 trials and a fresh seed (for
example `20261121`) so the confirmation ensemble does not contain trajectories
used for selection.

