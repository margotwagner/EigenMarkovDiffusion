# Structured local persistent closure

The production selected-pipeline closure is now
`local_persistent_unresolved_completion` with modal bandwidth 12. The previous
dense persistent closure remains available as the scientific reference.

## Construction

For an unresolved target mode `k`, the local closure conditions only on modal
coordinates `j` satisfying

```text
abs(k - j) <= bandwidth.
```

This rule is applied to the previous unresolved state and to the retained
states at the current and next saved times. Conditional gains are solved in
one padded batch per transition and stored as row-sparse matrices.
Sparse gain application is vectorized. The conditional innovation covariance
retains the selected low-rank factorization, and a simplex bank still enforces
nonnegativity and exact mass conservation.

At a bandwidth spanning the full modal space, the implementation reproduces
the dense persistent closure for identical random streams.

## Selection study

The command

```bash
python -m eigendiffusion study-structured-closure
```

performs four stages:

1. profiles dense completion and bandwidth candidates 4, 8, 12, and 20;
2. compares them using 500 shared base trajectories;
3. validates the selected bandwidth with 250 trials on shifted impulse,
   particle-count, diffusivity, 151-node, and longer-domain cases;
4. measures isolated setup, runtime, retained arrays, and sampled peak RSS at
   51, 101, 151, and 201 nodes.

Bandwidth 12 passed the strict accuracy rule. On the 101-node base case, its
normalized sliced Wasserstein error was 0.0802 versus 0.0807 for the dense
closure. Its lag-1 covariance error was 0.0703 versus 0.0590 for dense. The
automatic base selector permits only one quarter of the independently measured
temporal sampling-floor error when choosing bandwidth, preventing a
distribution-only match from hiding temporal degradation.

All five held-out cases passed thresholds for mean, variance, same-time
covariance, lag-1 covariance, marginal Wasserstein, and sliced Wasserstein.

## Scaling result

At 201 nodes:

| Metric | Dense | Local bandwidth 12 |
| --- | ---: | ---: |
| Setup time | 1.195 s | 1.117 s |
| Retained arrays | 27.72 MiB | 9.99 MiB |
| Sampled peak RSS | 433.01 MiB | 85.75 MiB |

The local closure therefore passed the setup, storage, and peak-memory gates
and is the default for selected completed-pipeline studies. The dense readout
names are unchanged so earlier results remain reproducible.

## Remaining limitation

This is a structured analytic closure, not yet a learned or standalone
closure: its local coefficients are still derived from known multinomial
moments. The next scientific step is fitting the sparse coefficients from
training trajectories and validating them on the same held-out suite.
