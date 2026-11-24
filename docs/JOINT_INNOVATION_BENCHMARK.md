# Joint batched innovation sampling

The prepared sampler packs retained noise, the corresponding conditional
mean, and residual noise into a joint factor for each transition. One batched
contraction generates the innovations for all saved timesteps. The diagonal
modal state recurrence still advances in time order, and both paths
reconstruct the full spatial field. See the conditioning formulation in
[Direct modal transition](DIRECT_MODAL_TRANSITION.md).

## Experiment

The benchmark uses 5,275 particles, a fixed physical duration of 100,
`M=75` retained modes, and conditional rank 25. The 101- and 201-node cases
save 101 and 233 full spatial fields, respectively. The workload matches
the [compiled exact simplex readout experiment](COMPILED_SIMPLEX_READOUT.md).

- Completion-only timing includes Gaussian modal sampling and full spatial
  reconstruction. It uses two warmups and 30 paired repeats, alternating
  stepwise and batched execution order.
- Prepared end-to-end timing adds the compiled exact simplex bank to both
  samplers and compares both pipelines with the exact vectorized multinomial
  walk. It uses one warmup and 30 paired repeats cycling through all six
  method orders.
- Each modal pair draws from the same seed. Exact walk timing uses a separate
  seed in the same repeat. Confidence intervals use 10,000 paired bootstrap
  resamples of the ratio of medians.
- The completion-only and end-to-end speed gates require a batched/stepwise
  median ratio of at most `0.75`. The end-to-end crossover gate requires a
  batched/exact ratio of at most `0.95`. Each speed gate also requires the
  upper 95% confidence bound to be below `1`.
- The runner records compiler startup, reusable completion setup, and lazy
  preparation of the batched factors separately. Logical array storage is
  measured before and after preparing those factors. Break-even estimates
  divide the applicable reusable setup by measured per-trajectory savings
  and exclude compiler startup.

## Runtime and storage

The November 23, 2026 run used one numerical thread. Times are medians and
intervals are the paired 95% bootstrap intervals for runtime ratios.

| Nodes | Stepwise completion (ms) | Batched completion (ms) | Batched/stepwise completion | Stepwise end to end (ms) | Batched end to end (ms) | Exact walk (ms) | Batched/exact end to end |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 101 | 3.132 | 1.035 | 0.330 (0.314–0.354) | 3.463 | 1.304 | 6.912 | 0.189 (0.180–0.190) |
| 201 | 10.513 | 4.950 | 0.471 (0.469–0.472) | 11.293 | 5.721 | 17.969 | 0.318 (0.318–0.319) |

Batched completion was 3.03× and 2.12× as fast as the stepwise sampler at
101 and 201 nodes. With physical readout, the batched pipeline was 5.30× and
3.14× as fast as the exact walk. The end-to-end batched/stepwise ratios were
0.377 (0.358–0.394) and 0.507 (0.505–0.508). The correctness and all three
speed gates passed at both resolutions.

| Nodes | Completion setup (s) | Batch preparation (ms) | Packed factors (MiB) | Total logical arrays before → after (MiB) | Break-even vs stepwise / exact (trajectories) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 101 | 0.317 | 4.90 | 7.71 | 42.61 → 50.31 | 3 / 58 |
| 201 | 3.120 | 26.85 | 35.58 | 358.00 → 393.58 | 5 / 257 |

The batch-preparation times are single measurements. The corresponding
end-to-end break-even estimates for stepwise sampling versus the exact walk
were 92 and 468 trajectories. The packed factors add persistent storage to
the full reference and conditioning arrays already held by the completer.

## Numerical equivalence

The benchmark builds the reference innovation factor by applying the
stepwise conditional gain to each unit latent column independently. It
compares the packed and reference factor Gram matrices with a relative
Frobenius tolerance of `1e-10`. The diagnostic also compares the modal path,
raw spatial path, physical counts, and final RNG state across 32 common
seeds. The absolute path tolerance is `1e-8` counts. The physical readout is
checked for finite, nonnegative, mass-conserving output and support agreement
above `1e-12` counts.

| Nodes | Innovation covariance relative Frobenius error | Max modal difference | Max raw spatial difference | Max physical count difference | Max physical mass error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 101 | `1.61e-12` | `5.90e-10` | `4.17e-10` | `4.58e-10` | `9.09e-13` |
| 201 | `1.88e-12` | `6.32e-10` | `8.80e-10` | `8.87e-10` | `1.82e-12` |

All paired RNG states and strict positive supports matched. The compiled
readout used no fallback in the diagnostic trajectories. These comparisons
measure equivalence to stepwise Gaussian sampling. The exact multinomial
walk provides the runtime reference.

## Default sampler

`InnovationConditionedModalCompleter.run()` uses the batched sampler and
prepares its joint factors once on the first call. Applications can call
`prepare_batched_sampler()` ahead of the first trajectory. The benchmark
invokes `run_stepwise()` and `run_batched()` explicitly and records the
one-time factor preparation separately from trajectory timing.

## Reproduction

Install the optional Numba backend and run with one numerical thread.

```bash
python -m pip install -e ".[compiled]"

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=src \
python3 -m eigendiffusion.joint_innovation_benchmark \
  --output-dir outputs/joint_innovation_benchmark
```

The runner writes `timing_samples.csv`, `diagnostics.csv`, `resources.csv`,
and `summary.json` in the output directory.
