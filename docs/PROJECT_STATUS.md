# Current project status

The code now distinguishes the exact continuous-time diffusion solution from
the exact discrete-time expectation of the implemented random walk. The current
EigenMarkov mean should be evaluated primarily against the discrete-time target,
because both use the same first-order modal decay factor.

The main open scientific issue is stochastic structure rather than mean
propagation. Independent two-state modal processes generally do not reproduce
the cross-mode covariance induced by a multinomial spatial particle process.
This appears in three diagnostics:

1. inflated or misplaced spatial variance;
2. inaccurate same-time spatial covariance;
3. negative reconstructed spatial counts.

The repository therefore reports exact multinomial variance and covariance,
negative mass fraction, fraction of entries below zero, minimum reconstructed
count, and mass-conservation error. No clipping or projection is applied.

The next model-development step should address correlated modal noise or another
constrained stochastic reduced-order representation. Refactoring or plotting
changes alone cannot resolve this mathematical limitation.

## v0.4 experimental modal variants

The original EigenMarkov formulation is now explicitly named
`independent_modal` and remains unchanged. Two research prototypes are added
alongside it:

- `correlated_modal`, which projects shared spatial Gaussian debit-credit noise
  into the retained eigenbasis;
- `banked_correlated_modal`, which additionally projects onto the nonnegative
  constant-mass simplex and carries the correction in a spatial residual bank.

These variants are hypotheses to test, not established replacements. The
comparison criteria are mean, variance, covariance, nonnegativity, mass
conservation, and—where applicable—bank magnitude.
