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
from .correlated_modal import (
    BankedCorrelatedModalDiffusion,
    CorrelatedModalDiffusion,
)
from .eigenmarkov import IndependentModalDiffusion

FloatArray = NDArray[np.float64]
RandomWalkMethod = Literal["naive", "multinomial"]
ModalModelName = Literal[
    "independent_modal",
    "correlated_modal",
    "banked_correlated_modal",
]
MODAL_MODEL_NAMES: tuple[ModalModelName, ...] = (
    "independent_modal",
    "correlated_modal",
    "banked_correlated_modal",
)


@dataclass(frozen=True, slots=True)
class EnsembleResult:
    """A stack of runs and its summary statistics.

    ``auxiliary`` contains compact diagnostics that are specific to a model.
    The banked model, for example, stores the L1 magnitude of its spatial
    debt/profit ledger at each time rather than retaining every full ledger.
    """

    runs: FloatArray
    model_name: str | None = None
    auxiliary: dict[str, FloatArray] | None = None

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
    *,
    model_name: str | None = None,
) -> EnsembleResult:
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")

    child_sequences = np.random.SeedSequence(seed).spawn(n_runs)
    runs = [simulate(np.random.default_rng(sequence)) for sequence in child_sequences]
    return EnsembleResult(
        runs=np.stack(runs).astype(float, copy=False),
        model_name=model_name,
    )



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
        model_name=f"random_walk_{method}",
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



def run_modal_ensemble(
    config: DiffusionConfig,
    n_runs: int,
    *,
    model: ModalModelName = "independent_modal",
    n_modes: int | None = None,
    modal_particle_weight: float = 1.0,
    initialization: str = "nearest",
    seed: int = 0,
) -> EnsembleResult:
    """Run one named modal model repeatedly.

    ``independent_modal`` is the original EigenMarkov implementation and is
    intentionally preserved without changing its dynamics. The other two
    options are experimental correlated-noise variants.
    """

    if n_runs <= 0:
        raise ValueError("n_runs must be positive")

    if model == "independent_modal":
        simulator = IndependentModalDiffusion(
            config=config,
            n_modes=n_modes,
            modal_particle_weight=modal_particle_weight,
            initialization=initialization,
        )
    elif model == "correlated_modal":
        simulator = CorrelatedModalDiffusion(config=config, n_modes=n_modes)
    elif model == "banked_correlated_modal":
        simulator = BankedCorrelatedModalDiffusion(config=config, n_modes=n_modes)
    else:
        allowed = ", ".join(MODAL_MODEL_NAMES)
        raise ValueError(f"unknown modal model {model!r}; choose one of: {allowed}")

    child_sequences = np.random.SeedSequence(seed).spawn(n_runs)
    spatial_runs: list[FloatArray] = []
    bank_l1_fraction_runs: list[FloatArray] = []

    for sequence in child_sequences:
        result = simulator.run(rng=np.random.default_rng(sequence))
        spatial_runs.append(result.spatial_counts)
        if model == "banked_correlated_modal":
            bank_l1_fraction_runs.append(
                np.abs(result.bank_balances).sum(axis=-1) / float(config.n_particles)
            )

    auxiliary: dict[str, FloatArray] | None = None
    if bank_l1_fraction_runs:
        auxiliary = {
            "bank_l1_fraction": np.stack(bank_l1_fraction_runs).astype(float, copy=False)
        }

    return EnsembleResult(
        runs=np.stack(spatial_runs).astype(float, copy=False),
        model_name=model,
        auxiliary=auxiliary,
    )



def run_eigenmarkov_ensemble(
    config: DiffusionConfig,
    n_runs: int,
    n_modes: int | None = None,
    modal_particle_weight: float = 1.0,
    initialization: str = "nearest",
    seed: int = 0,
) -> EnsembleResult:
    """Backward-compatible wrapper for ``independent_modal``."""

    return run_modal_ensemble(
        config,
        n_runs=n_runs,
        model="independent_modal",
        n_modes=n_modes,
        modal_particle_weight=modal_particle_weight,
        initialization=initialization,
        seed=seed,
    )



def run_correlated_modal_ensemble(
    config: DiffusionConfig,
    n_runs: int,
    n_modes: int | None = None,
    seed: int = 0,
) -> EnsembleResult:
    return run_modal_ensemble(
        config,
        n_runs=n_runs,
        model="correlated_modal",
        n_modes=n_modes,
        seed=seed,
    )



def run_banked_correlated_modal_ensemble(
    config: DiffusionConfig,
    n_runs: int,
    n_modes: int | None = None,
    seed: int = 0,
) -> EnsembleResult:
    return run_modal_ensemble(
        config,
        n_runs=n_runs,
        model="banked_correlated_modal",
        n_modes=n_modes,
        seed=seed,
    )
