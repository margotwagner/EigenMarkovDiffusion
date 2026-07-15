"""Compare output-only readouts on one correlated-modal ensemble."""

from eigendiffusion import (
    DiffusionConfig,
    apply_readout_ensemble,
    run_modal_ensemble,
)

config = DiffusionConfig(
    n_particles=1_000,
    n_nodes=31,
    n_steps=40,
    impulse_index=15,
)
raw = run_modal_ensemble(
    config,
    n_runs=20,
    model="correlated_modal",
    n_modes=31,
    seed=0,
)

for readout in (
    "raw",
    "simplex_bank",
    "delta_sigma_temporal",
    "delta_sigma_neighbor",
):
    result = apply_readout_ensemble(
        raw,
        config,
        readout=readout,
        spatial_error_fraction=0.5,
    )
    print(readout, result.runs.shape, result.runs.min())
