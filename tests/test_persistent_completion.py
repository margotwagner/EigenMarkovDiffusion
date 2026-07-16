import numpy as np

from eigendiffusion.completion import PersistentUnresolvedGaussianCompleter
from eigendiffusion.config import DiffusionConfig
from eigendiffusion.ensemble import apply_readout_ensemble, run_modal_ensemble
from eigendiffusion.temporal import (
    mean_node_lag_correlation_from_runs,
    multinomial_mean_node_lag_correlation,
    valid_time_origins,
)


def persistent_config() -> DiffusionConfig:
    return DiffusionConfig(
        n_particles=1000,
        n_nodes=31,
        n_steps=40,
        impulse_index=15,
        dt=1.0,
        length=4.0,
        diffusion_coefficient=2.20e-4,
    )


def test_persistent_completion_is_full_rank_noop() -> None:
    config = persistent_config()
    raw = run_modal_ensemble(
        config,
        n_runs=2,
        model="correlated_modal",
        n_modes=config.n_nodes,
        seed=0,
    )
    completed = apply_readout_ensemble(
        raw,
        config,
        readout="persistent_unresolved_completion",
        retained_modes=config.n_nodes,
        completion_start_time=0.0,
        seed=1,
    )
    np.testing.assert_array_equal(completed.runs, raw.runs)


def test_persistent_completion_preserves_mass_and_pre_handoff_values() -> None:
    config = persistent_config()
    raw = run_modal_ensemble(
        config,
        n_runs=3,
        model="handoff_correlated_modal",
        n_modes=15,
        initial_n_modes=config.n_nodes,
        handoff_time=10.0,
        seed=2,
    )
    completed = apply_readout_ensemble(
        raw,
        config,
        readout="persistent_unresolved_completion",
        retained_modes=15,
        completion_start_time=10.0,
        completion_rank=5,
        seed=3,
    )
    np.testing.assert_array_equal(completed.runs[:, :10], raw.runs[:, :10])
    np.testing.assert_allclose(
        completed.runs.sum(axis=-1),
        config.n_particles,
        atol=1e-9,
    )


def test_persistent_completion_is_reproducible() -> None:
    config = persistent_config()
    raw = run_modal_ensemble(
        config,
        n_runs=2,
        model="handoff_correlated_modal",
        n_modes=15,
        initial_n_modes=config.n_nodes,
        handoff_time=10.0,
        seed=4,
    )
    first = apply_readout_ensemble(
        raw,
        config,
        readout="persistent_unresolved_completion",
        retained_modes=15,
        completion_start_time=10.0,
        completion_rank=5,
        seed=5,
    )
    second = apply_readout_ensemble(
        raw,
        config,
        readout="persistent_unresolved_completion",
        retained_modes=15,
        completion_start_time=10.0,
        completion_rank=5,
        seed=5,
    )
    np.testing.assert_array_equal(first.runs, second.runs)


def test_persistent_completion_improves_one_step_temporal_correlation() -> None:
    config = persistent_config()
    raw = run_modal_ensemble(
        config,
        n_runs=120,
        model="handoff_correlated_modal",
        n_modes=15,
        initial_n_modes=config.n_nodes,
        handoff_time=10.0,
        seed=6,
    )
    independent = apply_readout_ensemble(
        raw,
        config,
        readout="unresolved_gaussian_completion",
        retained_modes=15,
        completion_start_time=10.0,
        completion_rank=5,
        seed=7,
    )
    persistent = apply_readout_ensemble(
        raw,
        config,
        readout="persistent_unresolved_completion",
        retained_modes=15,
        completion_start_time=10.0,
        completion_rank=5,
        seed=8,
    )
    origins = valid_time_origins(config.n_steps, 1, start_step=10)
    reference = multinomial_mean_node_lag_correlation(
        config,
        1,
        [config.impulse_index],
        start_indices=origins,
    )
    independent_value = mean_node_lag_correlation_from_runs(
        independent.runs,
        1,
        [config.impulse_index],
        start_indices=origins,
    )
    persistent_value = mean_node_lag_correlation_from_runs(
        persistent.runs,
        1,
        [config.impulse_index],
        start_indices=origins,
    )
    assert abs(persistent_value - reference) < abs(independent_value - reference)


def test_persistent_completion_rank_is_capped() -> None:
    config = persistent_config()
    completer = PersistentUnresolvedGaussianCompleter(
        config,
        retained_modes=30,
        completion_rank=100,
    )
    assert completer.completion_rank == 1
