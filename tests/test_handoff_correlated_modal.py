import numpy as np

from eigendiffusion import (
    CorrelatedModalDiffusion,
    DiffusionConfig,
    HandoffCorrelatedModalDiffusion,
    run_handoff_correlated_modal_ensemble,
)


def small_config() -> DiffusionConfig:
    return DiffusionConfig(
        n_particles=500,
        n_nodes=21,
        n_steps=21,
        impulse_index=10,
        dt=1.0,
        length=4.0,
        diffusion_coefficient=2.20e-4,
    )


def test_handoff_with_equal_mode_counts_matches_correlated_model() -> None:
    config = small_config()
    seed = 123
    standard = CorrelatedModalDiffusion(config, n_modes=config.n_nodes).run(
        rng=np.random.default_rng(seed)
    )
    handoff = HandoffCorrelatedModalDiffusion(
        config,
        initial_n_modes=config.n_nodes,
        final_n_modes=config.n_nodes,
        handoff_time=5.0,
    ).run(rng=np.random.default_rng(seed))

    np.testing.assert_allclose(handoff.spatial_counts, standard.spatial_counts, atol=1e-12)
    assert handoff.handoff_projection_relative_l2 < 1e-12


def test_handoff_at_zero_matches_fixed_reduced_correlated_model() -> None:
    config = small_config()
    seed = 456
    reduced = CorrelatedModalDiffusion(config, n_modes=8).run(
        rng=np.random.default_rng(seed)
    )
    handoff = HandoffCorrelatedModalDiffusion(
        config,
        initial_n_modes=config.n_nodes,
        final_n_modes=8,
        handoff_time=0.0,
    ).run(rng=np.random.default_rng(seed))

    np.testing.assert_allclose(handoff.spatial_counts, reduced.spatial_counts, atol=1e-12)


def test_handoff_conserves_mass() -> None:
    config = small_config()
    result = HandoffCorrelatedModalDiffusion(
        config,
        initial_n_modes=config.n_nodes,
        final_n_modes=8,
        handoff_time=5.0,
    ).run(rng=np.random.default_rng(7))

    np.testing.assert_allclose(
        result.spatial_counts.sum(axis=1),
        config.n_particles,
        atol=1e-9,
    )
    assert np.allclose(result.modal_amplitudes[6:, 8:], 0.0)


def test_handoff_ensemble_reports_projection_error() -> None:
    config = small_config()
    ensemble = run_handoff_correlated_modal_ensemble(
        config,
        n_runs=3,
        initial_n_modes=config.n_nodes,
        final_n_modes=8,
        handoff_time=5.0,
        seed=0,
    )

    assert ensemble.runs.shape == (3, config.n_steps, config.n_nodes)
    assert ensemble.auxiliary is not None
    errors = ensemble.auxiliary["handoff_projection_relative_l2"]
    assert errors.shape == (3,)
    assert np.all(errors >= 0.0)
