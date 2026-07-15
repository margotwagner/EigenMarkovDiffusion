# Delta-Sigma readout update

This update separates the **modal dynamics** from the **physical readout**.
The existing modal models are unchanged:

- `independent_modal`
- `correlated_modal`
- `banked_correlated_modal` (retained for backward compatibility)

A new readout layer is applied after a raw modal trajectory has been generated:

- `raw`
- `simplex_bank`
- `delta_sigma_temporal`
- `delta_sigma_neighbor`

The readouts do not feed back into `independent_modal` or `correlated_modal`.
This makes it possible to test bounded integer output constraints without
changing the stochastic modal dynamics.

## Temporal first-order Delta-Sigma

For raw spatial reconstruction `x[t]` and residual balance `e[t-1]`,

```text
z[t] = x[t] + e[t-1]
y[t] = integer_simplex_quantize(z[t], N)
e[t] = z[t] - y[t]
```

The integer simplex quantizer guarantees, at every saved time,

```text
y[i, t] is an integer
y[i, t] >= 0
y[i, t] <= N
sum_i y[i, t] = N
```

The residual is carried forward in time. The cumulative readout discrepancy
therefore telescopes into the bounded residual rather than being silently
discarded.

## Neighbor-coupled Delta-Sigma

The neighbor readout uses the same temporal residual, but routes a fraction
`alpha` of each local quantization error into the next spatial location during
the same time step. The remaining fraction `(1 - alpha)` is carried forward in
time. Scan direction alternates across time to reduce directional bias.

```text
alpha = 0      purely temporal borrowing
0 < alpha < 1  spatial and temporal borrowing
alpha = 1      maximal within-step spatial error diffusion
```

This is an experimental causal spatial error-diffusion implementation. Its
mean, variance, covariance, temporal error spectrum, and spatial error spectrum
must be measured; bounded integer outputs do not by themselves prove that the
random-walk distribution is reproduced.

## Relationship to the existing banked model

`banked_correlated_modal` changes the physical state used by the next modal
step. It is therefore a coupled dynamics-and-projection model.

`correlated_modal + simplex_bank` is different: the correlated modal trajectory
is generated first and remains unchanged, then a continuous debt/profit readout
is applied afterward.

Likewise, both Delta-Sigma variants are output-only readouts. This separation is
important for attributing any improvement or distortion to the readout rather
than to a change in the modal dynamics.

## New diagnostics

The CLI reports and saves:

- `noninteger_entry_fraction`
- `lower_bound_violation_fraction`
- `upper_bound_violation_fraction`
- `maximum_instantaneous_mass_error`
- `mean_readout_residual_l1_fraction`
- `maximum_readout_residual_l1_fraction`
- `final_mean_readout_residual_l1_fraction`

The existing mean, variance, covariance, negative-mass, and mass-conservation
metrics are retained.

## Output organization

The recommended output tree is separate from the earlier modal-model outputs:

```text
outputs/
  modal_models/                 previous results; leave unchanged
  delta_sigma_readouts/
    correlated_modal/
      raw/
      simplex_bank/
      delta_sigma_temporal/
      delta_sigma_neighbor/
    comparisons/
    sweeps/
```
