# Compact prepared innovation runtime

`PreparedInnovationCompletion.from_config()` builds the innovation-conditioned
Gaussian model and extracts the state required for repeated full-field draws.
The prepared object stores the initial modal state, transition diagonal,
reconstruction basis, and packed sampler. Existing
`InnovationConditionedModalCompleter` instances can export the same state with
`prepare_runtime()`, which shares their immutable packed sampler.

```python
import numpy as np

from eigendiffusion import DiffusionConfig, PreparedInnovationCompletion

runtime = PreparedInnovationCompletion.from_config(
    DiffusionConfig(), retained_modes=75, conditional_rank=25
)
trajectory = runtime.run(np.random.default_rng(1234))
```

Each call returns the full modal trajectory and reconstructed spatial field at
every saved time. The compiled exact simplex bank can convert the raw field to
nonnegative, mass-conserving counts. The full builder remains available for
conditioning diagnostics and completion of externally generated retained paths.

## Benchmark

The benchmark uses 5,275 particles, 75 retained modes, conditional rank 25,
and a fixed duration of 100. The 101- and 201-node grids save 101 and 233
full spatial fields, respectively. The workload matches the
[joint batched innovation benchmark](JOINT_INNOVATION_BENCHMARK.md).

- Four cold setup repeats alternate between the builder and the standalone
  compact factory. The runner records builder construction, preparing the
  builder's packed sampler, converting an already prepared builder, and the
  full standalone factory call. The factory constructs the full model
  temporarily, so construction still incurs that model's setup work and
  peak allocation.
- Completion-only timing covers modal sampling and full spatial
  reconstruction. It uses two warmups and 30 paired repeats with alternating
  method order. End-to-end timing adds compiled exact simplex readout to both
  samplers and includes the exact vectorized multinomial walk. It uses one
  warmup and 30 paired repeats cycling through all six method orders.
- Each modal pair uses a common seed and the walk uses a separate seed.
  Confidence intervals use 10,000 paired bootstrap resamples of the ratio of
  medians. The compact/builder runtime gates require the upper 95% bound to
  be at most 1.10 for both timing components. The compact/exact end-to-end
  comparison requires a median ratio of at most 0.95 and an upper bound below
  1.00.
- The correctness check requires identical modal paths, raw spatial fields,
  physical counts, and final random generator states across 32 paired
  trajectories. It also checks finite output, mass conservation, and
  nonnegative physical counts.
- The storage gate requires the factory's returned runtime to retain at most
  25% of the prepared builder's reachable NumPy backing bytes. The runner also
  keeps a converted runtime alive while deleting its originating builder.
  Weak references verify that the builder, full reference, and history arrays
  are released. The converted and factory runtimes must retain the same number
  of NumPy bytes.
- On Linux the runner measures RSS once per method and grid in separate
  child processes. Each child records post-collection RSS and peak RSS
  through setup, both in absolute bytes and as increases over pre-setup RSS.
  The allocator and operating system influence RSS. The reachable backing
  bytes describe array lifetimes directly.

Break-even estimates divide median incremental setup by median per-trajectory
end-to-end savings and round up. The estimate is unavailable when the
candidate has no positive observed trajectory savings. The comparisons cover
converting an existing builder, choosing the cold compact setup instead of the
cold builder setup, and preparing either modal pipeline instead of using the
exact walk. Compiler startup is recorded separately.

## Canonical results

The November 23, 2026 run used one numerical thread with Python 3.11.2,
NumPy 1.24.2, and Numba 0.56.4. The compact runtime passed the storage,
release, correctness, and speed gates at both resolutions. All 32 paired
modal paths, raw spatial paths, physical counts, and final RNG states matched
exactly on each grid.

| Nodes | Prepared builder array storage (MiB) | Compact array storage (MiB) | Compact share | Post-setup RSS increase, builder → compact (MiB) | Peak RSS increase through setup, builder → compact (MiB) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 101 | 50.31 | 7.79 | 15.48% | 54.8 → 23.7 | 54.8 → 54.8 |
| 201 | 393.58 | 35.89 | 9.12% | 404.1 → 109.3 | 404.1 → 404.2 |

| Nodes | Builder construction (s) | Packing (ms) | Compact conversion (µs) | Full cold setup, builder → compact (s) |
| ---: | ---: | ---: | ---: | ---: |
| 101 | 0.312 | 5.03 | 33 | 0.317 → 0.316 |
| 201 | 3.049 | 27.41 | 67 | 3.077 → 3.102 |

Times are medians. The compiled readout startup took 0.368 s.

| Nodes | Completion, builder → compact (ms) | Compact/builder completion ratio (95% CI) | End to end, builder → compact (ms) | Compact/builder end-to-end ratio (95% CI) | Exact walk (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 101 | 1.075 → 1.028 | 0.956 (0.906–1.063) | 1.319 → 1.296 | 0.982 (0.942–1.059) | 6.865 |
| 201 | 4.955 → 5.006 | 1.010 (1.006–1.020) | 5.706 → 5.768 | 1.011 (1.005–1.015) | 17.778 |

The compact pipeline was 5.30× and 3.08× as fast as the exact walk end to
end at 101 and 201 nodes, respectively. With the compiled readout already
available, compact factory setup broke even after 57 and 259 trajectories
against the exact walk. Charging the measured 0.368-second compiler startup
once per fresh process raises those estimates to 123 and 289 trajectories.
At 101 nodes, the observed medians put the marginal conversion break-even at
two trajectories; that estimate exports from a live prepared builder, and its
runtime confidence interval spans parity. At 201 nodes the builder was about
1.1% faster end to end, leaving the compact-versus-builder break-even estimates
undefined.

## Decision and next step

Use `PreparedInnovationCompletion.from_config()` for repeated trajectory
sampling. Use `InnovationConditionedModalCompleter` when inspecting the
conditioning diagnostics or completing externally generated retained paths.

The compact runtime reduces steady-state storage. Its factory still incurs
the full builder's setup cost and peak memory. The next step is to construct
the packed factors by streaming transitions or applying the diffusion
operator directly, so setup can avoid allocating the full histories.

## Reproduction

Install the optional Numba backend and run with one numerical thread.

```bash
python -m pip install -e ".[compiled]"

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=src \
python3 -m eigendiffusion.compact_runtime_benchmark \
  --output-dir outputs/compact_runtime_benchmark
```

The runner writes `setup_samples.csv`, `timing_samples.csv`, `diagnostics.csv`,
`resources.csv`, and `summary.json`.
