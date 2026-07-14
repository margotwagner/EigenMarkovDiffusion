import numpy as np

from eigendiffusion import DiffusionConfig
from eigendiffusion.baselines import (
    deterministic_diffusion,
    multinomial_random_walk_diffusion,
    naive_random_walk_diffusion,
    naive_random_walk_trajectories,
    trajectories_to_counts,
)
from eigendiffusion.ensemble import run_random_walk_ensemble


def test_deterministic_full_basis_reconstructs_impulse_and_mass():
    config = DiffusionConfig(n_particles=100, n_nodes=15, n_steps=10)
    values = deterministic_diffusion(config)
    expected = np.zeros(config.n_nodes)
    expected[config.impulse_index] = config.n_particles
    assert np.allclose(values[0], expected, atol=1e-10)
    assert np.allclose(values.sum(axis=1), config.n_particles, atol=1e-10)


def _assert_physical_random_walk(values, config):
    assert values.shape == (config.n_steps, config.n_nodes)
    assert np.issubdtype(values.dtype, np.integer)
    assert np.all(values >= 0)
    assert np.all(values.sum(axis=1) == config.n_particles)


def test_naive_random_walk_is_nonnegative_integer_and_mass_conserving():
    config = DiffusionConfig(n_particles=100, n_nodes=15, n_steps=20)
    values = naive_random_walk_diffusion(config, rng=np.random.default_rng(2))
    _assert_physical_random_walk(values, config)


def test_multinomial_random_walk_is_nonnegative_integer_and_mass_conserving():
    config = DiffusionConfig(n_particles=100, n_nodes=15, n_steps=20)
    values = multinomial_random_walk_diffusion(config, rng=np.random.default_rng(2))
    _assert_physical_random_walk(values, config)


def test_naive_trajectory_postprocessing_matches_returned_counts():
    config = DiffusionConfig(n_particles=50, n_nodes=9, n_steps=12)
    seed = 7
    trajectories = naive_random_walk_trajectories(
        config,
        rng=np.random.default_rng(seed),
    )
    counts = trajectories_to_counts(trajectories, config.n_nodes)
    direct = naive_random_walk_diffusion(
        config,
        rng=np.random.default_rng(seed),
    )
    assert trajectories.shape == (config.n_particles, config.n_steps)
    assert np.array_equal(counts, direct)


def test_naive_and_multinomial_ensemble_means_are_statistically_consistent():
    config = DiffusionConfig(
        n_particles=80,
        n_nodes=9,
        n_steps=8,
        diffusion_coefficient=5.0e-4,
    )
    naive = run_random_walk_ensemble(
        config,
        n_runs=600,
        seed=10,
        method="naive",
    )
    multinomial = run_random_walk_ensemble(
        config,
        n_runs=600,
        seed=20,
        method="multinomial",
    )
    # The implementations are distributionally, not pathwise, equivalent.
    # This tolerance compares independent Monte Carlo estimates.
    assert np.max(np.abs(naive.mean - multinomial.mean)) < 0.75
