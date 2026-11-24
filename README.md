# EigenDiffusion

EigenDiffusion is a research package for simulating stochastic diffusion in a
Laplacian eigenbasis. It compares reduced modal models with particle random
walks and exact moment references, with particular attention to mean,
variance, covariance, nonnegativity, and mass conservation.

The project is experimental: matching the mean is straightforward, while
reproducing the full stochastic structure of a spatial particle process with a
reduced modal model remains the central research question.

## Models

- `independent_modal` — the original EigenMarkov two-state model and default.
- `correlated_modal` — Gaussian moment closure with shared spatial
  debit-credit noise projected into the eigenbasis.
- `banked_correlated_modal` — correlated modal dynamics followed by a
  nonnegative, mass-conserving spatial bank.
- `handoff_correlated_modal` — starts at high modal rank and switches to a
  reduced basis at a chosen time.
- `direct_modal_gaussian` — a prepared retained-dimensional Gaussian
  transition with exact projected first and second temporal moments.
- `innovation_conditioned_modal_completion` — completes a direct retained
  trajectory by conditioning unresolved transition innovations.

Output-only readouts can transform a modal trajectory without changing its
dynamics:

- `raw`
- `simplex_bank`
- `warm_simplex_bank` (specialized exact counts/readout API)
- `compiled_warm_simplex_bank` (optional compiled exact counts/readout API)
- `clip_renormalize_bank` (specialized residual-feedback counts/readout API)
- `delta_sigma_temporal`
- `delta_sigma_neighbor`
- `unresolved_gaussian_completion`
- `persistent_unresolved_completion`
- `local_persistent_unresolved_completion`
- `local_persistent_unresolved_completion_simplex_bank`

See [Modal models](docs/MODAL_MODELS.md) and
[Mathematical details](docs/MATH.md) for definitions and assumptions. The
direct-transition prototype, benchmark, and current decision are documented in
[Direct retained-modal transition](docs/DIRECT_MODAL_TRANSITION.md). The first
physical-readout optimization and timing decision are documented in
[Warm-started exact simplex readout](docs/WARM_SIMPLEX_READOUT.md).
The subsequent approximate-readout experiment is documented in
[Clip-and-renormalize residual feedback](docs/CLIP_RENORMALIZE_READOUT.md).
The compiled exact kernel and first exact-readout path to clear the crossover
gate at both resolutions are documented in
[Compiled exact simplex readout](docs/COMPILED_SIMPLEX_READOUT.md).

## Installation

Python 3.10 or newer is required.

```bash
git clone <repository-url>
cd EigenMarkovDiffusion
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install the optional compiled exact readout with
`python -m pip install -e ".[compiled]"`.

Confirm the installation:

```bash
python -m eigendiffusion --help
pytest
ruff check .
```

The module form (`python -m eigendiffusion`) is used below, although installing
the package also provides an `eigendiffusion` command.

## Quick start

Run a small validation first:

```bash
python -m eigendiffusion validate \
  --modal-model independent_modal \
  --particles 1000 \
  --nodes 31 \
  --steps 40 \
  --runs 20 \
  --modes 31 \
  --seed 0 \
  --output outputs/quick_validation.png
```

For a minimal Python example:

```bash
python examples/basic_api.py
```

Or use the API directly:

```python
import numpy as np

from eigendiffusion import DiffusionConfig, IndependentModalDiffusion

config = DiffusionConfig(
    n_particles=1_000,
    n_nodes=31,
    n_steps=40,
    impulse_index=15,
)
result = IndependentModalDiffusion(config, n_modes=31).run(
    rng=np.random.default_rng(0)
)

print(result.spatial_counts.shape)
```

## Command-line workflows

List all options for any command with `--help`:

```bash
python -m eigendiffusion validate --help
```

The available workflows are:

| Command | Purpose |
| --- | --- |
| `validate` | Compare one modal model with analytic and random-walk references |
| `sweep-modes` | Measure sensitivity to the number of retained modes |
| `compare-modal-models` | Compare independent, correlated, and banked models |
| `study-trial-convergence` | Measure error convergence from 100–5,000 trials with bootstrap intervals |
| `study-handoff-selection` | Predict and validate handoff-time/rank accuracy-work tradeoffs |
| `compare-readouts` | Apply several readouts to the same modal ensemble |
| `sweep-handoff` | Sweep handoff times and reduced mode counts |
| `sweep-completion-rank` | Sweep unresolved Gaussian completion rank |
| `compare-temporal-correlation` | Compare empirical and exact cross-time covariance |
| `benchmark-models` | Benchmark full, reduced, and completed model pipelines |
| `benchmark-components` | Time setup, dynamics, completion, and full-field readouts separately |
| `benchmark-component-scaling` | Scale component timing and isolated memory across node counts |
| `python -m eigendiffusion.direct_modal_benchmark` | Screen the direct retained transition across nodes and modes |
| `python -m eigendiffusion.innovation_completion_benchmark` | Screen low-rank conditional innovation completion |
| `python -m eigendiffusion.warm_simplex_benchmark` | Compare warm-started and sorting-based exact simplex readouts |
| `python -m eigendiffusion.clip_renormalize_benchmark` | Compare clip-and-renormalize feedback with the warm exact readout |
| `python -m eigendiffusion.compiled_simplex_benchmark` | Compare the compiled exact readout with NumPy warm exact and the vectorized walk |
| `study-operating-points` | Benchmark selected completed pipelines, peak RSS, and distribution error |
| `study-structured-closure` | Select, validate, and scale the local persistent closure |
| `study-fitted-closure` | Train and test a frozen trajectory-fitted sparse closure |
| `benchmark-random-walk` | Compare particle and aggregated random-walk baselines |

Break the canonical accuracy-first pipeline into construction, raw dynamics,
persistent completion, and full-field physical-readout costs with:

```bash
python -m eigendiffusion benchmark-components
```

The command independently times continuous simplex and temporal Delta-Sigma
integer readouts on the same completed workload and writes long-format samples
plus a JSON summary under `outputs/component_benchmark/`.

Sweep the same component breakdown across spatial resolutions with:

```bash
python -m eigendiffusion benchmark-component-scaling
```

The `fixed_steps` regime holds the number of updates fixed and reduces `dt` as
needed for stability, isolating spatial work per update while allowing the
physical duration to change. The `fixed_duration` regime holds physical time
fixed, so finer grids may require more stable updates. Timing samples and
fresh-process RSS measurements are written separately to
`scaling_samples.csv` and `scaling_memory.csv`; `scaling_summary.json` combines
their scaling slopes without adding memory-sampling overhead to timed runs.
By default, the study also interleaves three full-state baselines on every
configuration. The original node-loop `aggregated_multinomial` method and the
batched conditional-binomial `aggregated_multinomial_vectorized` method sample
the same exact integer stochastic process. The `deterministic_markov` method
instead propagates the exact finite-step expected field with the matrix-free
Markov stencil; it returns nonnegative, mass-conserving real counts but has no
stochastic variance. CSV rows are identified by `pipeline` and `method`. The
vectorized stochastic method remains the primary like-for-like comparator for
runtime, memory, amortization, and break-even ratios; deterministic ratios only
compare full-field computational work. Pass
`--no-include-standard-comparator` to disable all three baselines.

Three staged scaling drivers live under `examples/`:

- `run_mode_scaling.py` compares proportional, fixed, spectral-decay, and exact
  mean-cutoff retained-mode policies.
- `run_completion_scaling.py` screens completion rank and modal bandwidth after
  a retained-mode policy fails.
- `run_adaptive_rank_scaling.py` selects rank from conditional-innovation
  energy, validates across two domain lengths, and applies measured-cost gates.

For example, compare the main modal formulations:

```bash
python -m eigendiffusion compare-modal-models \
  --models independent_modal correlated_modal banked_correlated_modal \
  --particles 1000 \
  --nodes 31 \
  --steps 40 \
  --runs 20 \
  --modes 31 \
  --seed 0 \
  --output outputs/model_comparison.png \
  --csv-output outputs/model_comparison.csv
```

Generated results belong under `outputs/`, which is ignored by Git except for
the placeholder that keeps the directory present.

To reproduce the high-trial-count convergence study:

```bash
python -m eigendiffusion study-trial-convergence \
  --models independent_modal correlated_modal banked_correlated_modal \
  --particles 5275 \
  --nodes 101 \
  --steps 101 \
  --impulse-index 59 \
  --modes 101 \
  --trial-counts 100 250 500 1000 2000 5000 \
  --bootstrap-resamples 200 \
  --covariance-times 1 5 20 50 100 \
  --output outputs/convergence/core_models_r5000.png \
  --csv-output outputs/convergence/core_models_r5000.csv
```

The comparison figure shows time-resolved relative mean-profile error rather
than nearly overlapping mean curves. CSV diagnostics additionally include the
maximum and final-time mean/variance errors and the sampled mean error relative
to the multinomial reference's expected Monte Carlo standard error.

Predict handoff error and validate a candidate grid with stochastic ensembles:

```bash
python -m eigendiffusion study-handoff-selection \
  --initial-modes 101 \
  --final-modes 35 50 60 75 \
  --handoff-times 10 15 20 30 \
  --runs 250 \
  --covariance-times 1 5 20 50 100 \
  --output outputs/handoff_selection/validated_grid_r250.png \
  --csv-output outputs/handoff_selection/validated_grid_r250.csv
```

The completion-rank sweep accepts independent, persistent, and simplex-banked
completion readouts and reports both same-time and cross-time covariance error.
See [predictive handoff selection](docs/HANDOFF_SELECTION.md) and
[completion selection](docs/COMPLETION_SELECTION.md) for the completed studies.

The selected-pipeline study reproduces the final accuracy/cost decision:

```bash
python -m eigendiffusion study-operating-points \
  --node-counts 51 101 151 201 \
  --accuracy-nodes 101 \
  --accuracy-runs 500 \
  --repeats 3
```

It measures setup and trajectory time, retained arrays, isolated process peak
RSS, selected-node Jensen--Shannon and Wasserstein distances, and sliced
Wasserstein distance for full spatial vectors.

Selected completed-pipeline workflows use the dense analytic persistent closure
by default. The hardened local-closure rerun uses case-specific sampling floors,
paired confidence intervals, and identical distribution projections; bandwidths
12 and 20 both miss the long-domain lag-covariance gate despite retaining a
large memory advantage. See [structured local closure](docs/STRUCTURED_LOCAL_CLOSURE.md).

The resolution-aware mode study compares proportional retention with fixed,
continuum-decay, and exact discrete mean cutoffs. A 250-run screen shows that a
fixed rank-5 completion cannot support plateauing retained modes. Under the
local analytic completion used for this scaling screen, fixed 75 modes becomes
noninferior at 151 and 201 nodes when completion rank is increased to 20, but
current setup time and retained storage are worse than the proportional rank-5
baseline. See [mode scaling](docs/MODE_SCALING.md).

The adaptive follow-up selects the minimum conditional-innovation rank that
captures 90% of positive residual variance. Dense and physically scaled
wide-local candidates pass fresh 250-run confirmation at 151/201 nodes on both
length-4 and length-6 domains. Neither passes the setup, resident-storage, and
trajectory cost intersection, so the predeclared 2,000-run promotion is
intentionally not run. See
[adaptive rank scaling](docs/ADAPTIVE_RANK_SCALING.md).

The first cross-configuration fitted closure was also implemented and tested,
but it did not pass the declared variance, covariance, temporal, or runtime
gates. A 12-anchor, low-rank-innovation successor improved all four measures
but also missed the frozen promotion gates. Exact-anchor rank and persistence
ablations identify regression transfer—not innovation rank—as the remaining
scientific bottleneck. A jointly physics-conditioned regression improves lag
error modestly but also fails the frozen gates; cross-validation now supplies
a targeted bridge-anchor design. These models remain experimental; see
[fitted closure study](docs/FITTED_CLOSURE_STUDY.md).

The subsequent 16-anchor target-conditioned model failed its predeclared
configuration-CV gate, so the newly frozen lockbox was not opened. This avoids
spending an unbiased test set on an architecture that has not demonstrated
adequate transfer.

## Repository layout

```text
src/eigendiffusion/   package implementation and CLI
tests/                unit, numerical, and CLI tests
examples/             small runnable API examples
docs/                 model notes, mathematics, and experiment guides
outputs/              generated figures and data (not versioned)
pyproject.toml        package metadata and development configuration
```

Useful starting points:

- [Basic API example](examples/basic_api.py)
- [Modal-model comparison example](examples/compare_modal_models.py)
- [Readout comparison example](examples/compare_readouts.py)
- [Performance benchmarks](docs/PERFORMANCE_BENCHMARKS.md)
- [Adaptive handoff](docs/ADAPTIVE_HANDOFF.md)
- [Persistent unresolved completion](docs/PERSISTENT_UNRESOLVED_COMPLETION.md)
- [Temporal correlation diagnostics](docs/TEMPORAL_CORRELATION_DIAGNOSTICS.md)

## Scientific interpretation

After spatial discretization, the diffusion operator is diagonalized as

\[
A = V\Lambda V^\top,\qquad \mathbf m = V^\top\mathbf n.
\]

The discrete expected update is

\[
\mathbf n_{t+1} = (I-\Delta t A)\mathbf n_t.
\]

For independent particles with node probabilities \(\mathbf p_t\), the spatial
reference distribution is multinomial:

\[
\operatorname{Cov}[\mathbf n_t]
=N\left(\operatorname{diag}(\mathbf p_t)-\mathbf p_t\mathbf p_t^\top\right).
\]

When evaluating a model, do not rely on mean agreement alone. Check spatial
variance and covariance, cross-time covariance, negative mass, total-mass
error, and bank or completion magnitude where applicable. The correlated and
banked variants are research prototypes, not established replacements for the
independent baseline.

## Development

Run the full local checks before committing:

```bash
pytest
ruff check .
```

The tests use the `src/` tree directly, so an editable install is convenient
but not required for `pytest`.
