# EigenDiffusion

A small, reproducible research codebase for **stochastic diffusion using Markov
dynamics in a diffusion eigenbasis**.

This is a clean refactor of the core diffusion work in
[`margotwagner/diffusion-model`](https://github.com/margotwagner/diffusion-model).
It keeps the EigenMarkov model, two physical random-walk baselines,
deterministic spectral diffusion, ensemble validation, mode sweeps, and a
runtime/memory benchmark. Notebook-only analysis, duplicate plotting code,
generated results, and unfinished reaction-diffusion extensions are
intentionally excluded.

## Scientific idea

A one-dimensional diffusion model can be written as

\[
\frac{d\mathbf n}{dt}=-A\mathbf n,
\qquad A=V\Lambda V^\top,
\qquad \mathbf m=V^\top\mathbf n.
\]

The eigenmodes evolve independently. EigenMarkov represents every nonconstant
mode as the difference between positive and negative Markov-state populations.
A state unit switches sign with probability

\[
p_k=\frac{\lambda_k\Delta t}{2},
\]

so its expected modal update is

\[
\mathbb E[m_k(t+\Delta t)]=(1-\lambda_k\Delta t)\mathbb E[m_k(t)].
\]

The spatial distribution is reconstructed with \(\mathbf n=V\mathbf m\). Using
only the first \(M\) slow modes produces a reduced-order model. A fuller
explanation is in [`docs/MATH.md`](docs/MATH.md).

> **Research status:** the ensemble mean is the main proof of concept. Signed
> stochastic eigenmodes can reconstruct negative spatial values, and the current
> method does not yet guarantee the complete covariance of a physical particle
> process. The code measures and exposes this limitation instead of silently
> clipping it. See [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md).

## Random-walk baselines

The repository deliberately includes two random-walk implementations:

1. **Naive particle trajectories** — simulates each particle separately and
   stores its full position history in an array shaped
   `(n_particles, n_steps)`. This closely follows the implementation in the
   original repository.
2. **Multinomial node transitions** — stores node counts and jointly samples the
   numbers moving left, staying, and moving right at each node. Under the model's
   independent-particle assumption, this gives the same node-count process in
   distribution without preserving particle identities.

Both implementations use the same grid, jump probability, reflecting-boundary
rule, and output convention. This makes it possible to distinguish improvements
from simple aggregation from improvements due to modal reduction.

## Installation

```bash
git clone <YOUR-NEW-REPOSITORY-URL>
cd EigenDiffusion

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest
```

## Benchmark the two random walks

```bash
eigendiffusion benchmark-random-walk \
  --particle-counts 100 500 1000 5275 \
  --nodes 101 \
  --steps 101 \
  --repeats 3 \
  --output outputs/random_walk_benchmark.png \
  --csv-output outputs/random_walk_benchmark.csv
```

The command reports median runtime, speedup, and estimated memory occupied by
the principal NumPy arrays. The memory values are implementation-level array
estimates, not measurements of total process resident memory.

The benchmark compares:

- `naive`: complete per-particle trajectories plus node-count history;
- `multinomial`: node-count history plus one temporary node-count vector.

Do not interpret the benchmark as proving an asymptotic EigenMarkov advantage.
Its purpose is to establish how much speedup comes from aggregating an ordinary
random walk before comparing either baseline with EigenMarkov.

## Quick validation

Compare EigenMarkov against the multinomial random walk and the exact mean of
the discretized diffusion equation:

```bash
eigendiffusion validate \
  --particles 5275 \
  --nodes 101 \
  --steps 101 \
  --impulse-index 59 \
  --runs 100 \
  --modes 101 \
  --random-walk-method multinomial \
  --output outputs/validation.png
```

To validate against the original-style particle implementation instead:

```bash
eigendiffusion validate \
  --particles 5275 \
  --nodes 101 \
  --steps 101 \
  --runs 20 \
  --random-walk-method naive \
  --output outputs/validation_naive.png
```

The command prints a JSON report and writes:

- the requested validation figure;
- a matching `.npz` file containing deterministic values and all stochastic
  runs.

For a faster smoke test:

```bash
eigendiffusion validate --particles 1000 --nodes 31 --steps 40 --runs 10
```

The equivalent module form is:

```bash
python -m eigendiffusion validate --runs 10 --nodes 31 --steps 40
```

## Sweep the number of retained modes

```bash
eigendiffusion sweep-modes \
  --modes 2 5 10 20 50 101 \
  --runs 100 \
  --output outputs/mode_sweep.png \
  --csv-output outputs/mode_sweep.csv
```

This measures the ensemble-mean error against full deterministic diffusion and
reports the fraction of reconstructed values below zero.

## Python API

```python
import numpy as np

from eigendiffusion import (
    DiffusionConfig,
    EigenMarkovDiffusion,
    multinomial_random_walk_diffusion,
    naive_random_walk_diffusion,
)

config = DiffusionConfig(
    n_particles=5275,
    n_nodes=101,
    n_steps=101,
    impulse_index=59,
    dt=1.0,
    length=4.0,
    diffusion_coefficient=2.20e-4,
)

naive_counts = naive_random_walk_diffusion(
    config,
    rng=np.random.default_rng(1),
)
multinomial_counts = multinomial_random_walk_diffusion(
    config,
    rng=np.random.default_rng(2),
)

model = EigenMarkovDiffusion(config, n_modes=50)
eigenmarkov = model.run(rng=np.random.default_rng(3))

print(naive_counts.shape)               # (time, space)
print(multinomial_counts.shape)         # (time, space)
print(eigenmarkov.spatial_counts.shape) # (time, space)
```

Use `naive_random_walk_trajectories` directly when individual particle paths are
needed. See [`examples/basic_api.py`](examples/basic_api.py) and
[`examples/quick_validation.py`](examples/quick_validation.py).

## Important parameters

| Parameter | Meaning |
|---|---|
| `n_nodes` | Number of spatial grid points, including both endpoints. |
| `n_steps` | Number of saved time points, including the initial state. |
| `n_modes` | Number of slow diffusion eigenmodes retained. `None` uses all modes. |
| `modal_particle_weight` | Physical amplitude represented by one positive/negative modal state unit. Smaller values use more state units and reduce integerization error. The original multi-run wrapper effectively used `2.0`. |
| `initialization` | `nearest` gives deterministic rounding; `stochastic` gives unbiased stochastic rounding of initial modal state units. |

The timestep must satisfy the nearest-neighbour condition
\(2D\Delta t/\Delta x^2\leq1\). Invalid configurations fail immediately with a
clear error.

## Repository layout

```text
src/eigendiffusion/
    config.py          shared physical and numerical parameters
    operators.py       Neumann diffusion matrix and eigentransforms
    eigenmarkov.py     core stochastic modal model
    baselines.py       deterministic, naive, and multinomial references
    ensemble.py        reproducible repeated simulations
    benchmarking.py    random-walk timing and core-array memory estimates
    metrics.py         accuracy and physical diagnostics
    plotting.py        validation and benchmark figures
    cli.py             validation, sweep, and benchmark commands

examples/              minimal direct API examples
tests/                 conservation and mathematical invariants
docs/                  math, project status, and refactor decisions
outputs/               generated files; ignored by Git
```

## What was intentionally left out

The original repository also contains calcium-calbindin reaction-diffusion
models, historical implementations, large notebooks, paper assets, and several
overlapping plotting utilities. Those are not required to reproduce the core
EigenMarkov diffusion experiment and are not included here. The mapping from old
to new code is documented in [`docs/REFACTOR_NOTES.md`](docs/REFACTOR_NOTES.md).

## Suggested first experiments

1. Benchmark the naive and multinomial random walks over particle count, node
   count, and simulation length.
2. Run full-mode validation and confirm that both random-walk ensemble means
   approach the deterministic solution as the number of runs increases.
3. Sweep `n_modes` to quantify the accuracy/computation tradeoff.
4. Sweep `modal_particle_weight` to measure its effect on stochastic variance,
   initialization error, and negative spatial reconstruction.
5. Compare full spatial covariance matrices, not only pointwise means and
   standard deviations, before making a strong equivalence claim.
