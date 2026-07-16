import numpy as np

from eigendiffusion.baselines import multinomial_random_walk_diffusion
from eigendiffusion.config import DiffusionConfig
from eigendiffusion.metrics import relative_frobenius_error
from eigendiffusion.references import multinomial_marginal_covariance
from eigendiffusion.temporal import (
    empirical_mean_cross_time_covariance,
    mean_node_lag_correlation_from_runs,
    multinomial_mean_cross_time_covariance,
    multinomial_mean_node_lag_correlation,
    valid_time_origins,
)


def test_valid_time_origins_respects_lag_and_end_step():
    origins = valid_time_origins(10, 3, start_step=2, end_step=8)
    assert np.array_equal(origins, np.array([2, 3, 4, 5]))


def test_lag_zero_matches_mean_same_time_covariance():
    config = DiffusionConfig(n_particles=100, n_nodes=7, n_steps=6, impulse_index=3)
    origins = np.array([1, 2, 3])
    observed = multinomial_mean_cross_time_covariance(
        config,
        0,
        start_indices=origins,
    )
    expected = np.mean(multinomial_marginal_covariance(config)[origins], axis=0)
    assert np.allclose(observed, expected, atol=1e-10)


def test_cross_time_covariance_conserves_total_mass():
    config = DiffusionConfig(n_particles=100, n_nodes=7, n_steps=8, impulse_index=3)
    covariance = multinomial_mean_cross_time_covariance(config, 2)
    assert np.allclose(covariance.sum(axis=0), 0.0, atol=1e-10)
    assert np.allclose(covariance.sum(axis=1), 0.0, atol=1e-10)


def test_empirical_cross_time_covariance_zero_for_identical_runs():
    trajectory = np.arange(24, dtype=float).reshape(4, 6)
    runs = np.repeat(trajectory[None, :, :], 5, axis=0)
    covariance = empirical_mean_cross_time_covariance(runs, 1)
    assert np.allclose(covariance, 0.0)


def test_multinomial_random_walk_matches_analytic_cross_time_covariance():
    config = DiffusionConfig(
        n_particles=200,
        n_nodes=7,
        n_steps=8,
        impulse_index=3,
        dt=0.5,
    )
    sequences = np.random.SeedSequence(0).spawn(1200)
    runs = np.stack(
        [
            multinomial_random_walk_diffusion(
                config,
                rng=np.random.default_rng(sequence),
            )
            for sequence in sequences
        ]
    ).astype(float)
    origins = np.array([2, 3, 4])
    analytic = multinomial_mean_cross_time_covariance(
        config,
        2,
        start_indices=origins,
    )
    empirical = empirical_mean_cross_time_covariance(
        runs,
        2,
        start_indices=origins,
    )
    assert relative_frobenius_error(analytic, empirical) < 0.18


def test_lag_correlation_is_finite_and_bounded():
    config = DiffusionConfig(n_particles=100, n_nodes=7, n_steps=8, impulse_index=3)
    analytic = multinomial_mean_node_lag_correlation(config, 1, [2, 3, 4])
    assert np.isfinite(analytic)
    assert -1.0 <= analytic <= 1.0

    rng = np.random.default_rng(1)
    runs = rng.normal(size=(50, 8, 7))
    empirical = mean_node_lag_correlation_from_runs(runs, 1, [2, 3, 4])
    assert np.isfinite(empirical)
    assert -1.0 <= empirical <= 1.0
