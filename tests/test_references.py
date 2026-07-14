import numpy as np

from eigendiffusion import DiffusionConfig
from eigendiffusion.operators import impulse_initial_condition
from eigendiffusion.references import (
    continuous_expected_diffusion,
    discrete_expected_diffusion,
    multinomial_marginal_covariance,
    multinomial_marginal_variance,
    random_walk_transition_matrix,
)


def test_discrete_reference_matches_repeated_transition_matrix():
    config = DiffusionConfig(
        n_particles=100,
        n_nodes=9,
        n_steps=8,
        diffusion_coefficient=5.0e-4,
    )
    transition = random_walk_transition_matrix(config)
    current = impulse_initial_condition(
        config.n_nodes,
        config.n_particles,
        config.impulse_index,
    )
    direct = [current.copy()]
    for _ in range(config.n_steps - 1):
        current = transition @ current
        direct.append(current.copy())

    spectral = discrete_expected_diffusion(config)
    assert np.allclose(spectral, np.stack(direct), atol=1e-10)


def test_continuous_and_discrete_references_share_initial_condition_and_mass():
    config = DiffusionConfig(n_particles=200, n_nodes=11, n_steps=10)
    continuous = continuous_expected_diffusion(config)
    discrete = discrete_expected_diffusion(config)
    assert np.allclose(continuous[0], discrete[0], atol=1e-10)
    assert np.allclose(continuous.sum(axis=1), config.n_particles, atol=1e-10)
    assert np.allclose(discrete.sum(axis=1), config.n_particles, atol=1e-10)


def test_multinomial_variance_and_covariance_are_consistent():
    config = DiffusionConfig(n_particles=100, n_nodes=9, n_steps=8)
    variance = multinomial_marginal_variance(config)
    covariance = multinomial_marginal_covariance(config)
    assert covariance.shape == (config.n_steps, config.n_nodes, config.n_nodes)
    assert np.allclose(np.diagonal(covariance, axis1=1, axis2=2), variance)
    assert np.allclose(covariance, np.swapaxes(covariance, 1, 2))
    # Fixed total particle count implies zero covariance with the total.
    assert np.allclose(covariance.sum(axis=2), 0.0, atol=1e-10)
    assert np.allclose(variance[0], 0.0, atol=1e-12)
