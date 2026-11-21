# End-to-end performance benchmarks

The `benchmark-models` command measures the computational cost of the five
main pipelines without changing any model formulation:

1. `multinomial_random_walk`
2. `full_correlated_modal`
3. `handoff_raw`
4. `handoff_independent_completion`
5. `handoff_persistent_completion`

The final two include both the 101-to-reduced handoff simulation and their
output completion step.

## Why setup and run time are separated

The Gaussian completion models precompute analytic conditional means,
covariances, gains, and low-rank factors. Those objects can be reused for many
stochastic trajectories with the same physical parameters. The benchmark
therefore reports:

- `setup_seconds`: one-time model/completion construction;
- `median_run_seconds`: median cost of generating one complete trajectory
  after setup;
- `one_run_total_seconds`: setup plus one median run;
- `amortized_seconds_per_run`: median run time plus setup divided by
  `--amortization-runs`.

A method can be unattractive for one trajectory but competitive when setup is
reused across hundreds of runs.

## Memory metric

`resident_array_bytes` recursively sums unique NumPy arrays reachable from the
setup object and returned result. Shared arrays are counted once. This captures
large stored completion gains and factors, basis matrices, and trajectory
histories.

It is **not** peak process resident memory. It omits Python object overhead and
short-lived numerical workspaces. Use `/usr/bin/time -v` or an external process
profiler later if exact peak RSS is required.

## Stable node scaling

Increasing the number of spatial nodes decreases `dx`, so a fixed `dt=1` can
violate

```text
2 D dt / dx^2 <= 1.
```

The benchmark keeps the requested physical duration fixed and automatically
uses

```text
dt <= stability_safety * dx^2 / (2 D).
```

Consequently, high-resolution grids may require more time steps. This is the
cost of simulating the same physical interval stably, not merely a software
artifact.

## Quick run

```bash
python -m eigendiffusion benchmark-models \
  --node-counts 31 51 101 \
  --particle-counts 1000 5275 \
  --total-time 20 \
  --retained-fraction 0.5 \
  --handoff-time 10 \
  --completion-rank 10 \
  --repeats 2 \
  --output outputs/performance/quick.png \
  --csv-output outputs/performance/quick.csv
```

## Main run

```bash
python -m eigendiffusion benchmark-models \
  --node-counts 31 51 101 201 \
  --particle-counts 1000 5275 10000 \
  --base-nodes 101 \
  --base-particles 5275 \
  --total-time 100 \
  --dt 1 \
  --retained-fraction 0.5 \
  --handoff-time 10 \
  --completion-rank 10 \
  --completion-ridge 0.01 \
  --repeats 5 \
  --amortization-runs 100 \
  --seed 0 \
  --output outputs/performance/model_scaling.png \
  --csv-output outputs/performance/model_scaling.csv
```

## Exact 101-to-50 benchmark

To preserve a fixed 50-mode reduced state across node counts, pass
`--final-modes 50`. For grids with fewer than 50 nodes, the command uses all
available modes, so completion becomes a no-op. For a fair proportional
scaling comparison, prefer the default `--retained-fraction 0.5`.

## Interpreting the result

The critical comparisons are:

- whether `handoff_raw` has lower per-run time than `full_correlated_modal`;
- how much setup and per-run cost independent and persistent completion add;
- whether completion setup memory grows faster than the saved dynamic state;
- whether multinomial runtime changes with particle count in the aggregated
  implementation;
- whether modal methods scale more strongly with node count than expected due
  to full-grid noise generation and oracle covariance precomputation.

A 501-node persistent-completion benchmark can require substantial setup time
and memory because the current oracle stores large conditional gain matrices at
many time points. Run 31–201 nodes first. A steep setup-memory curve is itself
evidence that the next implementation should replace the analytic oracle with
a local, structured, or learned closure.

## Selected operating-point benchmark

`study-operating-points` benchmarks the three final persistent-completion plus
simplex-bank configurations in fresh processes. Unlike `benchmark-models`, it
therefore reports an uncontaminated process peak RSS for every configuration.
Mode count and completion rank are scaled proportionally away from the
101-node reference grid.

The November 20, 2026 headline run used 51, 101, 151, and 201 nodes, three
timing repeats, and 500 accuracy trials. At 101 nodes, amortized trajectory
times were 0.0581 s (compute-first), 0.0598 s (balanced), and 0.0571 s
(accuracy-first). Peak RSS was 124--126 MiB. At 201 nodes, peak RSS reached
433--435 MiB and setup time reached 1.12--1.69 s, demonstrating that the dense
analytic completion oracle, rather than retained modal count alone, controls
memory scaling.

The same command compares each pipeline with an independent multinomial
ensemble using:

- one-dimensional Wasserstein-1 distance and common-bin Jensen--Shannon
  divergence for selected node marginals;
- sliced Wasserstein-1 distance for the full spatial count vector;
- a second multinomial ensemble as the finite-trial sampling floor.

Raw KL divergence is intentionally not used because continuous Gaussian modal
outputs and discrete multinomial counts do not share identical support.

## Structured-closure follow-up

The subsequent `study-structured-closure` run replaced the dense time-indexed
gain with exact restricted-block, row-sparse modal gains. At 201 nodes,
bandwidth 12 reduces retained arrays from 27.72 to 9.99 MiB and sampled peak RSS
from 435.46 to 88.23 MiB in the seven-repeat hardened rerun. It passes base
accuracy and performance gates, but a
hardened rerun rejects bandwidths 12 and 20 on long-domain lag covariance.
Selected-pipeline benchmarks therefore use dense persistent completion by
default; local closure remains an explicit memory experiment.
