# Validation-analysis update

This update changes the evaluation code without changing the core EigenMarkov
transition rule.

## Implemented changes

1. Added `references.py` with:
   - the exact continuous-time mean;
   - the exact discrete-time random-walk mean;
   - the one-step transition matrix;
   - analytic multinomial variance;
   - analytic same-time spatial covariance.
2. Updated validation to score both stochastic methods against the discrete
   mean rather than conflating it with the continuous matrix-exponential mean.
3. Added ensemble variance and covariance errors.
4. Added negative mass fraction, maximum negative mass fraction, minimum count,
   fraction of entries below zero, and per-run mass error.
5. Updated the validation figure to display mean, variance, and negative mass.
6. Updated mode sweeps to report mean error after several start times, reducing
   the disproportionate influence of the initial impulse.
7. Added tests for the transition matrix, discrete reference, analytic
   multinomial statistics, and new metrics.

No clipping or residual feedback was added. The goal of this update is to expose
and quantify the current model limitation before changing the stochastic model.
