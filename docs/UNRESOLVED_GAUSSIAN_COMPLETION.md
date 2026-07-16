# Unresolved Gaussian completion

## Purpose

After `handoff_correlated_modal` projects from a high-rank basis to `M` retained
modes, the mean can remain highly accurate while the variance is too small.
The missing variance lies partly in the omitted eigenmodes. Version 0.7 adds an
output-only proof-of-principle completion that samples those omitted modes
without evolving them dynamically.

## Gaussian conditional model

Let the full modal state be partitioned as

\[
\mathbf m =
\begin{bmatrix}
\mathbf m_R\\
\mathbf m_U
\end{bmatrix},
\]

where `R` denotes retained modes and `U` omitted modes. From the exact
finite-step multinomial reference, compute the full modal mean and covariance
at every saved time:

\[
\boldsymbol\mu_m = V^\top \boldsymbol\mu_n,
\qquad
C_m = V^\top C_n V.
\]

Partition `C_m` into retained and unresolved blocks. The readout samples

\[
\mathbf m_U\mid\mathbf m_R
\sim
\mathcal N\left(
\boldsymbol\mu_U + G(\mathbf m_R-\boldsymbol\mu_R),
C_{U\mid R}
\right),
\]

with a ridge-stabilized gain

\[
G = C_{UR}
\left(C_{RR}+\gamma s I\right)^{-1},
\]

where `s` is the mean retained variance and `gamma` is
`--completion-ridge`. The conditional covariance is

\[
C_{U\mid R}=C_{UU}-GC_{RU}.
\]

The output is

\[
\widetilde{\mathbf n}
=V_R\mathbf m_R+V_U\widetilde{\mathbf m}_U.
\]

Because the constant mode is retained, the added unresolved field has zero
spatial sum and total mass is preserved.

## Completion rank

The conditional covariance is eigendecomposed and only its leading `r`
directions may be sampled. `--completion-rank r` therefore controls the
stochastic readout cost. Omitting the flag uses all unresolved directions.
Rank zero is available in `sweep-completion-rank` as the unchanged raw handoff
baseline.

## Important limitation

This readout uses exact analytic moments from the linear random-walk reference.
It is an oracle diagnostic, not yet a general closure. It restores same-time
mean, variance, and covariance, but samples omitted modes independently across
saved times and therefore does not reproduce their exact temporal
correlations. A later model should replace the oracle moments with a learned,
local, or dynamically propagated closure.

## Recommended experiment

```bash
python -m eigendiffusion sweep-completion-rank \
  --initial-modes 101 \
  --modes 50 \
  --handoff-time 10 \
  --completion-ranks 0 5 10 20 30 51 \
  --completion-ridge 0.01 \
  --particles 5275 \
  --nodes 101 \
  --steps 101 \
  --impulse-index 59 \
  --runs 500 \
  --covariance-times 1 5 10 20 50 100 \
  --seed 0 \
  --output outputs/unresolved_completion/completion_rank_sweep.png \
  --csv-output outputs/unresolved_completion/completion_rank_sweep.csv
```
