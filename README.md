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

Output-only readouts can transform a modal trajectory without changing its
dynamics:

- `raw`
- `simplex_bank`
- `delta_sigma_temporal`
- `delta_sigma_neighbor`
- `unresolved_gaussian_completion`
- `persistent_unresolved_completion`

See [Modal models](docs/MODAL_MODELS.md) and
[Mathematical details](docs/MATH.md) for definitions and assumptions.

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
| `study-operating-points` | Benchmark selected completed pipelines, peak RSS, and distribution error |
| `benchmark-random-walk` | Compare particle and aggregated random-walk baselines |

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
