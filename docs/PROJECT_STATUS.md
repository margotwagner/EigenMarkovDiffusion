# Current project status

**Updated November 20, 2026**

EigenDiffusion now has a validated physics-first reference hierarchy. The
aggregated multinomial random walk is the stochastic reference. Full-rank
correlated modal dynamics reproduce its first two moments to the Monte Carlo
sampling floor; the original independent-mode formulation reproduces the
expected mean update but strongly overdisperses variance and covariance.

For reduced models, dense analytic persistent completion is the trusted
default. It conditions unresolved modal fluctuations on retained dynamics and
recovers same-time and short-lag statistics. Simplex banking can make completed
outputs nonnegative and exactly mass conserving without materially changing
those moments. The method remains a Gaussian moment closure: outputs are not
integer particle counts, and setup is the current performance bottleneck.

## What is complete

- Time-resolved mean and variance diagnostics, reference-standardized mean
  error, distribution distances, and bootstrap uncertainty.
- A nested 100--5,000-trial convergence study for the multinomial,
  independent, correlated, and banked-correlated formulations.
- Predictive handoff-time/mode-count validation and completion-rank studies.
- Physical simplex-completed outputs and selected operating-point benchmarks.
- Exact restricted covariance blocks for the local persistent closure.
- Hardened comparisons with case-specific sampling floors, matched random
  projections, paired confidence intervals, worst-time gates, and an explicit
  trajectory-runtime gate.
- Resolution-aware mode scaling through 201 nodes and a completion-rank versus
  retained-mode trade study.
- Frozen fitted, anchor-conditioned, persistent-innovation, and
  physics-conditioned learned-closure studies with an unopened lockbox.

## Current conclusions

The hardened local-closure study carried bandwidths 12 and 20 through all
held-out cases. Both pass the base and four of five held-outs, but fail the
long-domain lag-covariance gate: errors are 0.1545 and 0.1176 versus 0.0912 for
dense completion, with an allowed difference of 0.0183. No local bandwidth is
promoted; dense persistent completion remains the production default.

The resolution study separates deterministic and stochastic requirements. An
exact 0.5% worst-time post-handoff mean criterion selects 35/34/33 modes at
101/151/201 nodes, but completion rank 5 cannot restore the missing variance
and temporal covariance. On the base length-4 domain, fixed 75 retained modes
with local bandwidth-12 completion and rank 20 passes an independent fresh-seed
250-trial accuracy gate at 151 and 201 nodes. At 201 nodes it requires 2.80 s
of setup, 16.4 MiB of retained arrays, and 0.1429 s per trajectory, compared
with 1.00 s, 6.8 MiB, and 0.1403 s for proportional retention with rank 5. It
is scientifically accurate but not a scaling win.

## Ordered next steps

1. Derive a physics-based completion-rank rule from the conditional innovation
   spectrum across resolution, using a predeclared captured-energy tolerance.
2. Implement a matrix-free or streaming low-rank innovation factor so a fixed
   retained-mode model reduces setup and storage as well as state dimension.
3. Repeat the 250-run accuracy gate and seven-or-more-repeat timing gate for
   that candidate. Advance to 2,000 trials only if it is accurate and cheaper.
4. Repair the local closure's long-domain temporal error using physically
   normalized neighborhoods or a low-rank nonlocal correction; do not relax
   the comparison margin.
5. Keep the learned-model lockbox closed unless a materially different
   architecture first passes whole-configuration cross-validation.

The detailed experiment inventory, results, and theory are in
`docs/PROJECT_REPORT_2026-11-20.md`; the scaling workflow is documented in
`docs/MODE_SCALING.md`.
