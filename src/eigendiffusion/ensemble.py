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
from .completion import UnresolvedGaussianCompleter
from .config import DiffusionConfig
from .correlated_modal import (
    BankedCorrelatedModalDiffusion,
    CorrelatedModalDiffusion,
    HandoffCorrelatedModalDiffusion,
)
from .eigenmarkov import IndependentModalDiffusion
from .readouts import ReadoutName, apply_readout

FloatArray = NDArray[np.float64]
RandomWalkMethod = Literal["naive", "multinomial"]
ModalModelName = Literal[
    "independent_modal",
    "correlated_modal",
    "banked_correlated_modal",
    "handoff_correlated_modal",
]
MODAL_MODEL_NAMES: tuple[ModalModelName, ...] = (
    "independent_modal",
    "correlated_modal",
    "banked_correlated_modal",
    "handoff_correlated_modal",
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
    initial_n_modes: int | None = None,
    handoff_time: float = 10.0,
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
    elif model == "handoff_correlated_modal":
        final_n_modes = config.n_nodes if n_modes is None else int(n_modes)
        simulator = HandoffCorrelatedModalDiffusion(
            config=config,
            final_n_modes=final_n_modes,
            initial_n_modes=initial_n_modes,
            handoff_time=handoff_time,
        )
    else:
        allowed = ", ".join(MODAL_MODEL_NAMES)
        raise ValueError(f"unknown modal model {model!r}; choose one of: {allowed}")

    child_sequences = np.random.SeedSequence(seed).spawn(n_runs)
    spatial_runs: list[FloatArray] = []
    bank_l1_fraction_runs: list[FloatArray] = []
    handoff_projection_errors: list[float] = []
    handoff_modal_work_fractions: list[float] = []

    for sequence in child_sequences:
        result = simulator.run(rng=np.random.default_rng(sequence))
        spatial_runs.append(result.spatial_counts)
        if model == "banked_correlated_modal":
            bank_l1_fraction_runs.append(
                np.abs(result.bank_balances).sum(axis=-1) / float(config.n_particles)
            )
        elif model == "handoff_correlated_modal":
            handoff_projection_errors.append(result.handoff_projection_relative_l2)
            handoff_modal_work_fractions.append(result.modal_update_fraction_of_full)

    auxiliary: dict[str, FloatArray] | None = None
    if bank_l1_fraction_runs:
        auxiliary = {
            "bank_l1_fraction": np.stack(bank_l1_fraction_runs).astype(float, copy=False)
        }
    if handoff_projection_errors:
        if auxiliary is None:
            auxiliary = {}
        auxiliary["handoff_projection_relative_l2"] = np.asarray(
            handoff_projection_errors, dtype=float
        )
        auxiliary["handoff_modal_update_fraction_of_full"] = np.asarray(
            handoff_modal_work_fractions, dtype=float
        )

    return EnsembleResult(
        runs=np.stack(spatial_runs).astype(float, copy=False),
        model_name=model,
        auxiliary=auxiliary,
    )



def apply_readout_ensemble(
    ensemble: EnsembleResult,
    config: DiffusionConfig,
    *,
    readout: ReadoutName = "raw",
    spatial_error_fraction: float = 0.5,
    retained_modes: int | None = None,
    completion_start_time: float = 0.0,
    completion_rank: int | None = None,
    completion_ridge: float = 1.0e-2,
    seed: int = 0,
) -> EnsembleResult:
    """Apply an output-only readout independently to every ensemble run.

    ``unresolved_gaussian_completion`` precomputes one analytic completion
    plan and uses independent child random-number streams for each run. The
    original modal trajectories are never modified.
    """

    if readout == "unresolved_gaussian_completion":
        if retained_modes is None:
            raise ValueError(
                "unresolved_gaussian_completion requires retained_modes"
            )
        completer = UnresolvedGaussianCompleter(
            config,
            retained_modes=retained_modes,
            completion_start_time=completion_start_time,
            completion_rank=completion_rank,
            ridge=completion_ridge,
        )
    else:
        completer = None

    child_sequences = np.random.SeedSequence(seed).spawn(ensemble.runs.shape[0])
    processed: list[FloatArray] = []
    residual_l1_fraction: list[FloatArray] = []
    for run, sequence in zip(ensemble.runs, child_sequences, strict=True):
        rng = np.random.default_rng(sequence)
        if completer is not None:
            counts, additions = completer.complete(run, rng=rng)
            processed.append(counts)
            residual_l1_fraction.append(
                np.abs(additions).sum(axis=-1) / float(config.n_particles)
            )
            continue

        result = apply_readout(
            run,
            config.n_particles,
            readout=readout,
            spatial_error_fraction=spatial_error_fraction,
            config=config,
            retained_modes=retained_modes,
            completion_start_time=completion_start_time,
            completion_rank=completion_rank,
            completion_ridge=completion_ridge,
            rng=rng,
        )
        processed.append(result.counts)
        residual_l1_fraction.append(
            result.residual_l1 / float(config.n_particles)
        )

    auxiliary = {} if ensemble.auxiliary is None else dict(ensemble.auxiliary)
    stacked_residual = np.stack(residual_l1_fraction).astype(float, copy=False)
    auxiliary["readout_residual_l1_fraction"] = stacked_residual
    if readout == "unresolved_gaussian_completion":
        auxiliary["completion_l1_fraction"] = stacked_residual

    return EnsembleResult(
        runs=np.stack(processed).astype(float, copy=False),
        model_name=f"{ensemble.model_name}+{readout}",
        auxiliary=auxiliary,
    )



def run_handoff_correlated_modal_ensemble(
    config: DiffusionConfig,
    n_runs: int,
    *,
    final_n_modes: int,
    handoff_time: float,
    initial_n_modes: int | None = None,
    seed: int = 0,
) -> EnsembleResult:
    """Run the high-rank-to-reduced correlated modal model repeatedly."""

    return run_modal_ensemble(
        config,
        n_runs=n_runs,
        model="handoff_correlated_modal",
        n_modes=final_n_modes,
        initial_n_modes=initial_n_modes,
        handoff_time=handoff_time,
        seed=seed,
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
