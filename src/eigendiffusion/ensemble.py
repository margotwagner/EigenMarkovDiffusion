"""Utilities for repeated stochastic simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray

from .baselines import (
    multinomial_random_walk_diffusion,
    naive_random_walk_diffusion,
)
from .config import DiffusionConfig
from .eigenmarkov import EigenMarkovDiffusion

FloatArray = NDArray[np.float64]
RandomWalkMethod = Literal["naive", "multinomial"]


@dataclass(frozen=True, slots=True)
class EnsembleResult:
    """A stack of runs and its summary statistics."""

    runs: FloatArray

    @property
    def mean(self) -> FloatArray:
        return np.mean(self.runs, axis=0)

    @property
    def std(self) -> FloatArray:
        return np.std(self.runs, axis=0, ddof=0)


def _run_ensemble(
    simulate: Callable[[np.random.Generator], FloatArray],
    n_runs: int,
    seed: int,
) -> EnsembleResult:
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")

    child_sequences = np.random.SeedSequence(seed).spawn(n_runs)
    runs = [simulate(np.random.default_rng(sequence)) for sequence in child_sequences]
    return EnsembleResult(runs=np.stack(runs).astype(float, copy=False))


def run_random_walk_ensemble(
    config: DiffusionConfig,
    n_runs: int,
    seed: int = 0,
    method: RandomWalkMethod = "multinomial",
) -> EnsembleResult:
    """Run either random-walk baseline repeatedly."""

    simulators = {
        "naive": naive_random_walk_diffusion,
        "multinomial": multinomial_random_walk_diffusion,
    }
    try:
        simulator = simulators[method]
    except KeyError as error:
        raise ValueError("method must be 'naive' or 'multinomial'") from error

    return _run_ensemble(
        lambda rng: simulator(config, rng=rng),
        n_runs=n_runs,
        seed=seed,
    )


def run_naive_random_walk_ensemble(
    config: DiffusionConfig,
    n_runs: int,
    seed: int = 0,
) -> EnsembleResult:
    return run_random_walk_ensemble(config, n_runs=n_runs, seed=seed, method="naive")


def run_multinomial_random_walk_ensemble(
    config: DiffusionConfig,
    n_runs: int,
    seed: int = 0,
) -> EnsembleResult:
    return run_random_walk_ensemble(
        config,
        n_runs=n_runs,
        seed=seed,
        method="multinomial",
    )


def run_eigenmarkov_ensemble(
    config: DiffusionConfig,
    n_runs: int,
    n_modes: int | None = None,
    modal_particle_weight: float = 1.0,
    initialization: str = "nearest",
    seed: int = 0,
) -> EnsembleResult:
    model = EigenMarkovDiffusion(
        config=config,
        n_modes=n_modes,
        modal_particle_weight=modal_particle_weight,
        initialization=initialization,
    )
    return _run_ensemble(
        lambda rng: model.run(rng=rng).spatial_counts,
        n_runs=n_runs,
        seed=seed,
    )
