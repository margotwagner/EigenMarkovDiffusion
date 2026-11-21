from __future__ import annotations

import numpy as np

from eigendiffusion.config import DiffusionConfig
from eigendiffusion.ensemble import run_modal_ensemble, run_random_walk_ensemble
from eigendiffusion.fitted_closure import (
    FittedClosureTrainingCase,
    FittedLocalClosureModel,
    FittedLocalPersistentCompleter,
)


def fitted_config(impulse_index: int = 5) -> DiffusionConfig:
    return DiffusionConfig(
        n_particles=200,
        n_nodes=11,
        n_steps=8,
        impulse_index=impulse_index,
        dt=1.0,
        length=4.0,
        diffusion_coefficient=2.2e-4,
    )


def test_fitted_closure_trains_saves_and_completes(tmp_path) -> None:
    config = fitted_config()
    reference = run_random_walk_ensemble(config, n_runs=12, seed=1)
    model = FittedLocalClosureModel.fit(
        [
            FittedClosureTrainingCase(
                config=config,
                runs=reference.runs,
                retained_modes=6,
                handoff_time=2.0,
            )
        ],
        bandwidth=2,
        n_frequency_bins=2,
        n_time_bins=2,
        residual_grid_size=4,
        max_residual_samples=20,
        seed=2,
    )
    path = model.save(tmp_path / "model.npz")
    loaded = FittedLocalClosureModel.load(path)
    np.testing.assert_allclose(loaded.transition_coefficients, model.transition_coefficients)

    raw = run_modal_ensemble(
        config,
        n_runs=1,
        model="handoff_correlated_modal",
        n_modes=6,
        initial_n_modes=11,
        handoff_time=2.0,
        seed=3,
    ).runs[0]
    completer = FittedLocalPersistentCompleter(
        config,
        loaded,
        retained_modes=6,
        completion_start_time=2.0,
    )
    first, _ = completer.complete(raw, rng=np.random.default_rng(4))
    second, _ = completer.complete(raw, rng=np.random.default_rng(4))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[:2], raw[:2])
    np.testing.assert_allclose(first.sum(axis=1), config.n_particles, atol=1.0e-9)


def test_fitted_model_accepts_unseen_impulse() -> None:
    training_config = fitted_config(4)
    training = run_random_walk_ensemble(training_config, n_runs=10, seed=5)
    model = FittedLocalClosureModel.fit(
        [
            FittedClosureTrainingCase(
                config=training_config,
                runs=training.runs,
                retained_modes=6,
                handoff_time=2.0,
            )
        ],
        bandwidth=2,
        n_frequency_bins=2,
        n_time_bins=2,
        residual_grid_size=4,
        max_residual_samples=20,
        seed=6,
    )
    unseen = fitted_config(7)
    raw = run_modal_ensemble(
        unseen,
        n_runs=1,
        model="handoff_correlated_modal",
        n_modes=6,
        initial_n_modes=11,
        handoff_time=2.0,
        seed=7,
    ).runs[0]
    completed, _ = FittedLocalPersistentCompleter(
        unseen,
        model,
        retained_modes=6,
        completion_start_time=2.0,
    ).complete(raw, rng=np.random.default_rng(8))
    assert np.all(np.isfinite(completed))
