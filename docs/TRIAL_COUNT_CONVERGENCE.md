# Trial-count convergence study

The `study-trial-convergence` workflow quantifies how ensemble estimates change
as the number of independent stochastic trajectories increases. Each method is
simulated once at the largest requested count; smaller counts are deterministic
prefixes of the same ensemble. This makes every curve nested and reproducible.

The workflow reports:

- aggregate relative L2 mean and variance errors;
- same-time covariance relative Frobenius error;
- maximum and final-time relative mean/variance errors;
- sampled mean error divided by the multinomial reference's expected Monte
  Carlo standard-error norm;
- block-bootstrap confidence intervals for mean, variance, and covariance
  errors;
- time-resolved mean-profile error at the maximum trial count.

## Completed 5,000-trial study

The completed study used 5,275 particles, 101 nodes/modes, 101 saved steps, an
impulse at node 59, trial counts `100, 250, 500, 1000, 2000, 5000`, 200
bootstrap resamples, and covariance times `1, 5, 20, 50, 100`.

At 5,000 trials:

| Model | Mean error | Mean/reference SE | Variance error | Covariance error |
| --- | ---: | ---: | ---: | ---: |
| Multinomial random walk | 0.000602 | 1.09 | 0.0210 | 0.0382 |
| Independent modal | 0.001794 | 3.23 | 2.8758 | 2.6671 |
| Correlated modal | 0.000554 | 1.00 | 0.0202 | 0.0376 |
| Banked correlated modal | 0.000639 | 1.15 | 0.0199 | 0.0356 |

The multinomial, correlated, and banked errors decrease approximately with the
expected Monte Carlo trend. The independent mean estimate is visibly noisier
because the formulation has excess stochastic variance; this should not be
described as proof that its expected mean update is biased. Its variance and
covariance errors instead plateau near 2.88 and 2.67, demonstrating a structural
failure that additional trials do not resolve.

Outputs are written to:

- `outputs/convergence/core_models_r5000.png`
- `outputs/convergence/core_models_r5000.csv`
- `outputs/convergence/core_models_r5000.npz`
