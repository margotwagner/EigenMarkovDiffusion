# Streaming compact innovation setup

`PreparedInnovationCompletion.from_config()` constructs the packed sampler by
processing one conditioning transition at a time. Both constructors return
the same compact runtime for repeated full-field draws.
`PreparedInnovationCompletion.from_full_history_config()` retains the original
setup path as a numerical reference. The full `InnovationConditionedModalCompleter`
provides conditioning histories for diagnostics and externally generated retained paths.

## Benchmark protocol

The paired benchmark uses 5,275 particles, 75 retained modes, conditional rank
25, and a fixed duration of 100. The 101- and 201-node grids save 101 and 233
full spatial fields, respectively. It uses the same compiled exact simplex
readout and vectorized multinomial walk as the earlier
[compact runtime benchmark](COMPACT_INNOVATION_RUNTIME.md).

- **Cold setup.** Four paired repeats alternate construction order. Each
  timed call prepares a new runtime. The runner releases each result before
  preparing the next and reports the paired ratio of medians and its 95%
  bootstrap confidence interval. The setup gate requires a median ratio at
  most 1.10 and an upper confidence bound at most 1.25.
- **Memory.** The retained NumPy backing bytes of both compact runtimes must
  match each other and their reported `nbytes` totals. Each method and grid
  also runs in a separate child process on Linux. The child records baseline
  RSS, post-collection setup RSS, and high-water RSS through setup. At 201
  nodes the streaming peak must be below 150 MiB in absolute terms and at most
  40% of the full-history factory's absolute peak.
- **Correctness.** The constructors must produce identical packed factors,
  rank and latent layout, initial state, transition factors, and reconstruction
  basis. For each of 32 paired seeds, modal paths, reconstructed fields,
  physical readouts, and final random generator states must match exactly.
  The runner validates finite paths, mass conservation, and nonnegative
  physical counts.
- **Trajectory runtime.** Completion-only measures prepared sampling and
  spatial reconstruction. End-to-end adds compiled exact simplex readout and
  compares both modal pipelines with the exact multinomial walk. Each
  component has 30 paired repeats, balanced across all method orders, with
  two completion warmups and one end-to-end warmup. Modal methods share a
  seed within each repeat. The walk uses a separate seed. Both streaming to
  full-history runtime ratios require a 95% upper confidence bound at most
  1.10. Streaming end to end must reach a median ratio at most 0.95 against
  the exact walk and an upper bound below 1.00.

The confidence intervals use 10,000 paired bootstrap resamples. Constructor
and compiler startup are measured separately from prepared trajectory runtime.
Break-even estimates against the walk use median setup divided by the observed
per-trajectory savings, rounded up. The runner reports those estimates both
with and without the one-time compiled readout startup.

## Canonical results

The November 23, 2026 one-thread run used Python 3.11.2, NumPy 1.24.2, and
Numba 0.56.4. Every correctness, storage, setup, prepared-runtime, exact-walk,
and peak-RSS gate passed. Across 32 seeds per grid, the packed factors, modal
paths, spatial fields, physical readouts, and final random-generator states
matched the full-history constructor exactly.

Both constructors retain the same compact runtime: 7.79 MiB at 101 nodes and
35.89 MiB at 201 nodes. Streaming changes only construction lifetime.

| Nodes | Full-history setup (s) | Streaming setup (s) | Ratio (95% CI) | Peak RSS, full history -> streaming (MiB) | Peak ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 101 | 0.324 | 0.293 | 0.906 (0.887-0.940) | 93.11 -> 49.61 | 0.533 |
| 201 | 3.058 | 2.840 | 0.928 (0.914-0.963) | 442.52 -> 80.97 | 0.183 |

The baseline-subtracted peak at 201 nodes fell from 404.90 MiB to 43.49 MiB.
This is the main result: the old factory briefly retained probability,
covariance, innovation, and conditioning histories even though the returned
runtime needed only packed factors. The streamed factory retains one
transition's dense scratch arrays at a time. It also reduced median setup by
9.4% at 101 nodes and 7.2% at 201 nodes.

Prepared trajectory cost remained at parity, as expected for identical runtime
objects. Completion-only medians were 1.022 and 4.928 ms at 101 and 201 nodes;
end-to-end medians including compiled exact simplex readout were 1.298 and
5.690 ms. The streaming/full-history end-to-end ratios were 0.999 and 1.003.

| Nodes | Streaming end to end (ms) | Exact walk (ms) | Streaming speedup | Break-even trajectories | Break-even with compiler startup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 101 | 1.298 | 6.851 | 5.28x | 53 | 120 |
| 201 | 5.690 | 17.870 | 3.14x | 234 | 264 |

Streaming therefore solves the constructor's transient-memory problem and
slightly improves setup time, but it does not change the asymptotic factorization
work or packed-factor storage. At 201 nodes, a one-thread profile attributed
about half of setup to the two conditioning SVDs and about one third to the
full innovation eigendecomposition. Those factorizations are the next setup
bottleneck.

## Reproduction

Install the optional Numba backend and run with one numerical thread.

```bash
python -m pip install -e ".[compiled]"

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=src \
python3 -m eigendiffusion.streaming_constructor_benchmark \
  --output-dir outputs/streaming_constructor_benchmark
```

The runner writes `setup_samples.csv`, `timing_samples.csv`, `diagnostics.csv`,
`resources.csv`, and `summary.json`. It exits with status 1 when a gate fails.
