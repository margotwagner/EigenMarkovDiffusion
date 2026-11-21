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
- A conditional-innovation energy rank rule, shared-CSR local gain storage,
  cross-domain 100-run selection, fresh 250-run confirmation, and isolated
  setup/storage/trajectory cost gates.
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

The adaptive follow-up makes the completion dimension physics-based. The
smallest factor capturing 90% of positive conditional-innovation variance
selects ranks 7/10/13 and 12/17/21 (minimum/median/maximum) at 151/201 nodes on
the length-4 domain, and 5/7/9 and 8/11/14 on length 6. Dense adaptive and
physically scaled wide-local completion both pass the fresh 250-run accuracy
gate in all four case/grid combinations. The narrower local rule fails the
length-6, 201-node temporal gate.

Neither confirmed candidate passes the isolated cost gate. At the 201-node
physical-refinement points, dense adaptive setup is 1.218--1.227 times the
baseline and resident storage is 1.833--1.926 times the baseline, although
trajectory time is essentially unchanged. Wide-local completion reduces
resident storage to 0.435--0.553 times baseline, but setup rises to
4.270--6.606 times baseline and trajectory time to 1.065--1.079 times baseline.
The promotion set is empty, so no 2,000-run headline study is justified.

## Ordered next steps

1. Replace the full conditional-innovation eigendecomposition with a validated
   operator-based partial eigensolver. Compare its factors with the exact
   spectrum, including captured energy and Ritz residuals, before benchmarking.
2. Optimize local gain construction separately. Preserve the physically scaled
   bandwidth that repaired long-domain accuracy, and target the measured setup
   and trajectory bottlenecks without changing the statistical margins.
3. Rerun the three-setup/seven-trajectory cost gate first. If the factor changes
   numerically, repeat the fresh 250-run accuracy gate as well. Advance to 2,000
   trials only when one candidate is accurate and cheaper in both domains.
4. After that gate passes, extend analytic rank and timing scaling beyond 201
   nodes before spending additional Monte Carlo trials; this tests whether the
   fixed-dynamics benefit persists under further spatial refinement.
5. Keep dense analytic completion as the trusted production reference, and
   keep the learned-model lockbox closed unless a materially different
   architecture first passes whole-configuration cross-validation.

The detailed experiment inventory, results, and theory are in
`docs/PROJECT_REPORT_2026-11-20.md`; the scaling workflow is documented in
`docs/MODE_SCALING.md` and `docs/ADAPTIVE_RANK_SCALING.md`.
