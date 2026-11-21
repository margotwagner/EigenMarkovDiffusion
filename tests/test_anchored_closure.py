from __future__ import annotations

import numpy as np

from eigendiffusion.anchored_closure import (
    AnchoredFittedClosureModel,
    AnchoredFittedLocalCompleter,
)
from eigendiffusion.config import DiffusionConfig
from eigendiffusion.ensemble import run_modal_ensemble, run_random_walk_ensemble
from eigendiffusion.fitted_closure import FittedClosureTrainingCase
from eigendiffusion.fitted_ablation import (
    run_anchor_innovation_ablation,
    run_exact_anchor_ensemble_validation,
)


def anchor_config(impulse_index: int, diffusion: float = 2.2e-4) -> DiffusionConfig:
    return DiffusionConfig(
        n_particles=200,
        n_nodes=11,
        n_steps=8,
        impulse_index=impulse_index,
        dt=1.0,
        length=4.0,
        diffusion_coefficient=diffusion,
    )


def test_anchored_model_interpolates_saves_and_completes(tmp_path) -> None:
    configs = (anchor_config(3, 1.8e-4), anchor_config(7, 2.8e-4))
    cases = []
    for index, config in enumerate(configs):
        reference = run_random_walk_ensemble(config, n_runs=12, seed=10 + index)
        cases.append(
            FittedClosureTrainingCase(
                config=config,
                runs=reference.runs,
                retained_modes=6,
                handoff_time=2.0,
            )
        )
    model = AnchoredFittedClosureModel.fit(
        cases,
        bandwidth=2,
        n_frequency_bins=2,
        n_time_bins=2,
        residual_grid_size=5,
        innovation_rank=2,
        nearest_anchors=2,
        persistent_innovations=True,
        max_residual_samples=30,
        seed=20,
    )
    loaded = AnchoredFittedClosureModel.load(model.save(tmp_path / "anchors.npz"))
    assert loaded.persistent_innovations
    query = anchor_config(5, 2.3e-4)
    indices, weights = loaded.interpolation_weights(query)
    assert indices.shape == weights.shape == (2,)
    np.testing.assert_allclose(np.sum(weights), 1.0)

    raw = run_modal_ensemble(
        query,
        n_runs=1,
        model="handoff_correlated_modal",
        n_modes=6,
        initial_n_modes=11,
        handoff_time=2.0,
        seed=30,
    ).runs[0]
    completer = AnchoredFittedLocalCompleter(
        query,
        loaded,
        retained_modes=6,
        completion_start_time=2.0,
    )
    first, _ = completer.complete(raw, rng=np.random.default_rng(40))
    second, _ = completer.complete(raw, rng=np.random.default_rng(40))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[:2], raw[:2])
    np.testing.assert_allclose(first.sum(axis=1), query.n_particles, atol=1.0e-9)
    assert np.all(np.isfinite(first))

    batch, batch_additions = completer.complete_batch(
        np.stack((raw, raw)), rng=np.random.default_rng(50)
    )
    assert batch.shape == batch_additions.shape == (2, query.n_steps, query.n_nodes)
    np.testing.assert_allclose(batch.sum(axis=2), query.n_particles, atol=1.0e-9)

    ablation = run_anchor_innovation_ablation(
        cases,
        tmp_path / "ablation.csv",
        ranks=[1, 2],
        bandwidth=2,
        n_frequency_bins=2,
        n_time_bins=2,
        residual_grid_size=5,
        nearest_anchors=1,
        ridge=1.0e-3,
        max_residual_samples=30,
        max_regression_samples_per_target=100,
        seed=60,
    )
    lines = ablation.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    assert "persistent_conditional" in lines[-1]

    exact_validation = run_exact_anchor_ensemble_validation(
        cases,
        loaded,
        tmp_path / "exact.csv",
        n_runs=4,
        bandwidth=2,
        seed=70,
    )
    exact_lines = exact_validation.read_text(encoding="utf-8").splitlines()
    assert len(exact_lines) == 5
    assert "anchored_persistent_fitted" in exact_lines[-1]
