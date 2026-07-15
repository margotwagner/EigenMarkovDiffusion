"""Minimal Python comparison of all three modal formulations."""

import numpy as np

from eigendiffusion import DiffusionConfig, run_modal_ensemble

config = DiffusionConfig(
    n_particles=1_000,
    n_nodes=31,
    n_steps=40,
    impulse_index=15,
)

for seed, model_name in enumerate(
    (
        "independent_modal",
        "correlated_modal",
        "banked_correlated_modal",
    ),
    start=1,
):
    ensemble = run_modal_ensemble(
        config,
        n_runs=20,
        model=model_name,
        n_modes=31,
        seed=seed,
    )
    minimum = float(np.min(ensemble.runs))
    mass_error = float(
        np.max(np.abs(ensemble.runs.sum(axis=-1) - config.n_particles))
    )
    print(f"{model_name:26s} minimum={minimum:9.3f} mass_error={mass_error:.3e}")
