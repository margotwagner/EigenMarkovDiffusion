from __future__ import annotations

import numpy as np

from eigendiffusion.anchored_closure import AnchoredFittedClosureModel
from eigendiffusion.conditioned_closure import (
    PhysicsConditionedClosureModel,
    PhysicsConditionedLocalCompleter,
    PhysicsConditionedRegressionModel,
    cross_validate_conditioned_regression,
    recommend_anchor_expansion,
)
from eigendiffusion.config import DiffusionConfig
from eigendiffusion.ensemble import run_modal_ensemble, run_random_walk_ensemble
from eigendiffusion.fitted_closure import FittedClosureTrainingCase


def conditioned_config(impulse: int, diffusion: float) -> DiffusionConfig:
    return DiffusionConfig(
        n_particles=200,
        n_nodes=11,
        n_steps=8,
        impulse_index=impulse,
        dt=1.0,
        length=4.0,
        diffusion_coefficient=diffusion,
    )


def test_conditioned_model_cross_validates_and_completes(tmp_path) -> None:
    cases = []
    for index, config in enumerate(
        (
            conditioned_config(3, 1.7e-4),
            conditioned_config(5, 2.2e-4),
            conditioned_config(7, 2.9e-4),
        )
    ):
        reference = run_random_walk_ensemble(config, n_runs=10, seed=10 + index)
        cases.append(
            FittedClosureTrainingCase(
                config=config,
                runs=reference.runs,
                retained_modes=6,
                handoff_time=2.0,
            )
        )

    regression = PhysicsConditionedRegressionModel.fit(
        cases,
        bandwidth=2,
        n_frequency_bins=2,
        degree=1,
        max_regression_samples_per_target=100,
        seed=20,
    )
    initial, transition = regression.target_coefficients(
        conditioned_config(4, 2.0e-4),
        retained_modes=6,
        time_fraction=0.5,
    )
    assert initial.shape == (5, 5)
    assert transition.shape == (5, 15)
    innovations = AnchoredFittedClosureModel.fit(
        cases,
        bandwidth=2,
        n_frequency_bins=2,
        n_time_bins=2,
        residual_grid_size=5,
        innovation_rank=2,
        nearest_anchors=2,
        persistent_innovations=True,
        max_residual_samples=30,
        max_regression_samples_per_target=100,
        seed=30,
    )
    model = PhysicsConditionedClosureModel(regression, innovations)
    assert model.save(tmp_path / "conditioned.npz").exists()

    query = conditioned_config(4, 2.0e-4)
    raw = run_modal_ensemble(
        query,
        n_runs=1,
        model="handoff_correlated_modal",
        n_modes=6,
        initial_n_modes=11,
        handoff_time=2.0,
        seed=40,
    ).runs[0]
    completer = PhysicsConditionedLocalCompleter(
        query,
        model,
        retained_modes=6,
        completion_start_time=2.0,
    )
    completed, _ = completer.complete(raw, rng=np.random.default_rng(50))
    assert np.all(np.isfinite(completed))
    np.testing.assert_allclose(completed.sum(axis=1), query.n_particles, atol=1.0e-9)

    path, selected = cross_validate_conditioned_regression(
        cases,
        tmp_path / "cv.csv",
        degrees=[1],
        folds=3,
        bandwidth=2,
        n_frequency_bins=2,
        ridge=1.0e-2,
        max_regression_samples_per_target=100,
        seed=60,
    )
    assert path.exists()
    assert selected == 1
    assert len(path.read_text(encoding="utf-8").splitlines()) == 4
    recommendations = recommend_anchor_expansion(
        cases,
        path,
        tmp_path / "anchors.csv",
        count=2,
    )
    assert len(recommendations.read_text(encoding="utf-8").splitlines()) == 3
