# Structured local persistent closure

The local persistent closure remains a memory-saving research candidate. The
production selected-pipeline closure is the dense analytic persistent closure:
after statistical hardening, local bandwidths 12 and 20 both fail the
long-domain lag-covariance held-out gate.

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
3. validates every base-eligible bandwidth with 250 trials on shifted impulse,
   particle-count, diffusivity, 151-node, and longer-domain cases;
4. measures isolated setup, runtime, retained arrays, and sampled peak RSS at
   51, 101, 151, and 201 nodes.

Bandwidths 12 and 20 passed the strict base accuracy rule. On the 101-node base case,
bandwidth 12's
normalized sliced Wasserstein error was 0.0802 versus 0.0807 for the dense
closure. Its lag-1 covariance error was 0.0703 versus 0.0590 for dense. The
automatic base selector permits only one quarter of the independently measured
temporal sampling-floor error when choosing bandwidth, preventing a
distribution-only match from hiding temporal degradation.

Four of five held-out cases pass. On the long-domain case, dense lag-covariance
error is 0.0912, while bandwidths 12 and 20 give 0.1545 and 0.1176. Their
candidate-minus-dense differences are 0.0633 and 0.0263, above the case-specific
temporal margin of 0.0183. The selector therefore rejects both local candidates
and falls back to dense completion.

## Scaling result

At 201 nodes:

| Metric | Dense | Local bandwidth 12 |
| --- | ---: | ---: |
| Setup time | 1.286 s | 1.138 s |
| Retained arrays | 27.72 MiB | 9.99 MiB |
| Sampled peak RSS | 435.46 MiB | 88.23 MiB |

The local closure passes setup, trajectory-runtime, storage, and peak-memory
gates, but scientific accuracy takes precedence over those savings. Dense
persistent completion is therefore the current default. The local readout names
are unchanged so earlier results remain reproducible.

## Remaining limitation

This is a structured analytic closure, not a learned or standalone closure:
its local coefficients are still derived from known multinomial moments. A
future local candidate must improve long-domain temporal conditioning before it
can be promoted; merely widening from 12 to 20 modes is insufficient.
