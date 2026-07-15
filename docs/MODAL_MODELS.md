# Modal model variants

The repository now keeps three explicitly named modal formulations.

## 1. `independent_modal`

This is the original EigenMarkov implementation. It is preserved unchanged and
remains the default CLI model.

Each retained nonconstant eigenmode is represented by positive and negative
integer state populations. State units switch sign independently with
probability

\[
p_k = \frac{\lambda_k\Delta t}{2}.
\]

This produces the desired first-order modal mean

\[
\mathbb E[m_k(t+\Delta t)]
= (1-\lambda_k\Delta t)\mathbb E[m_k(t)],
\]

but independent modal fluctuations do not reproduce the covariance of physical
particle transfers.

## 2. `correlated_modal`

This experimental model keeps the same retained modal mean update,

\[
\mathbf m_{t+1}^{\rm mean}
= (I-\Delta t\Lambda_M)\mathbf m_t,
\]

but replaces independent mode noise with a shared spatial fluctuation field.
For each source node, it samples a Gaussian approximation to the multinomial
left/stay/right transition covariance. That single spatial debit-credit field is
then projected into the retained eigenbasis:

\[
\boldsymbol\eta_m = V_M^\top\boldsymbol\eta_n.
\]

The modal noise is therefore correlated because the same physical transfer
changes multiple modes simultaneously. The Gaussian approximation matches the
conditional first two moments of one random-walk step, but it does not guarantee
nonnegative spatial counts.

When a raw state contains negative values, a nonnegative constant-mass simplex
projection is used only as a proxy for the next noise covariance. The raw model
state itself is not clipped.

## 3. `banked_correlated_modal`

This model adds a spatial debt/profit ledger to `correlated_modal`.

At each step:

1. Update the retained modes with correlated noise.
2. Reconstruct the raw spatial state.
3. Add the previous bank balance.
4. Project onto the physical simplex

   \[
   \mathcal S_N = \{\mathbf n: n_i\ge 0,\;\sum_i n_i=N\}.
   \]

5. Carry the difference between the adjusted state and its retained-mode
   reconstruction into the next step as the new bank balance.

This guarantees nonnegative mass-conserving output. It is not a purely
reduced-order model because the bank contains one balance per spatial node.
The CLI reports the bank L1 magnitude relative to the total particle count so
that projection debt cannot remain hidden.

## What to compare

For every model, report:

- mean relative error against the exact discrete-time expectation;
- variance relative error against the analytic multinomial variance;
- covariance relative Frobenius error;
- maximum mass error;
- negative-entry and negative-mass fractions;
- for the banked model, mean and maximum bank L1 fractions.

Do not replace `independent_modal` with either experimental model until these
metrics demonstrate a genuine improvement.
