import numpy as np

from eigendiffusion.completion import UnresolvedGaussianCompleter
from eigendiffusion.config import DiffusionConfig
from eigendiffusion.ensemble import apply_readout_ensemble, run_modal_ensemble
from eigendiffusion.metrics import relative_l2_error
from eigendiffusion.references import multinomial_marginal_variance


def small_config() -> DiffusionConfig:
    return DiffusionConfig(
        n_particles=1000,
        n_nodes=21,
        n_steps=31,
        impulse_index=10,
        dt=1.0,
        length=4.0,
        diffusion_coefficient=2.20e-4,
    )


def test_full_rank_completion_is_noop() -> None:
    config = small_config()
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
        readout="unresolved_gaussian_completion",
        retained_modes=config.n_nodes,
        completion_start_time=0.0,
        seed=1,
    )
    np.testing.assert_array_equal(completed.runs, raw.runs)


def test_completion_preserves_pre_handoff_state_and_mass() -> None:
    config = small_config()
    raw = run_modal_ensemble(
        config,
        n_runs=3,
        model="handoff_correlated_modal",
        n_modes=10,
        initial_n_modes=config.n_nodes,
        handoff_time=8.0,
        seed=2,
    )
    completed = apply_readout_ensemble(
        raw,
        config,
        readout="unresolved_gaussian_completion",
        retained_modes=10,
        completion_start_time=8.0,
        completion_rank=5,
        seed=3,
    )
    np.testing.assert_array_equal(completed.runs[:, :8], raw.runs[:, :8])
    np.testing.assert_allclose(
        completed.runs.sum(axis=-1),
        config.n_particles,
        atol=1e-9,
    )
    assert completed.auxiliary is not None
    assert "completion_l1_fraction" in completed.auxiliary


def test_completion_is_reproducible_with_seed() -> None:
    config = small_config()
    raw = run_modal_ensemble(
        config,
        n_runs=2,
        model="handoff_correlated_modal",
        n_modes=10,
        initial_n_modes=config.n_nodes,
        handoff_time=8.0,
        seed=4,
    )
    first = apply_readout_ensemble(
        raw,
        config,
        readout="unresolved_gaussian_completion",
        retained_modes=10,
        completion_start_time=8.0,
        seed=5,
    )
    second = apply_readout_ensemble(
        raw,
        config,
        readout="unresolved_gaussian_completion",
        retained_modes=10,
        completion_start_time=8.0,
        seed=5,
    )
    np.testing.assert_array_equal(first.runs, second.runs)


def test_completion_reduces_missing_variance_error() -> None:
    config = small_config()
    raw = run_modal_ensemble(
        config,
        n_runs=160,
        model="handoff_correlated_modal",
        n_modes=10,
        initial_n_modes=config.n_nodes,
        handoff_time=8.0,
        seed=6,
    )
    completed = apply_readout_ensemble(
        raw,
        config,
        readout="unresolved_gaussian_completion",
        retained_modes=10,
        completion_start_time=8.0,
        completion_ridge=1.0e-2,
        seed=7,
    )
    target = multinomial_marginal_variance(config)[8:]
    raw_error = relative_l2_error(target, np.var(raw.runs[:, 8:], axis=0))
    completed_error = relative_l2_error(
        target,
        np.var(completed.runs[:, 8:], axis=0),
    )
    assert completed_error < raw_error


def test_completion_rank_is_capped_by_omitted_dimension() -> None:
    config = small_config()
    completer = UnresolvedGaussianCompleter(
        config,
        retained_modes=20,
        completion_rank=100,
    )
    assert completer.completion_rank == 1
