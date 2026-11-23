import numpy as np

from eigendiffusion import DiffusionConfig
from eigendiffusion.baselines import (
    deterministic_diffusion,
    multinomial_random_walk_diffusion,
    naive_random_walk_diffusion,
    naive_random_walk_trajectories,
    trajectories_to_counts,
    vectorized_multinomial_random_walk_diffusion,
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


def test_vectorized_multinomial_random_walk_is_reproducible_and_physical():
    config = DiffusionConfig(n_particles=100, n_nodes=15, n_steps=20)
    first = vectorized_multinomial_random_walk_diffusion(
        config,
        rng=np.random.default_rng(3),
    )
    second = vectorized_multinomial_random_walk_diffusion(
        config,
        rng=np.random.default_rng(3),
    )

    assert np.array_equal(first, second)
    _assert_physical_random_walk(first, config)


def test_vectorized_multinomial_handles_boundaries_and_maximal_probability():
    config = DiffusionConfig(
        n_particles=101,
        n_nodes=3,
        n_steps=2,
        impulse_index=1,
        dt=1.0,
        length=2.0,
        diffusion_coefficient=0.5,
    )
    values = vectorized_multinomial_random_walk_diffusion(
        config,
        rng=np.random.default_rng(4),
    )

    _assert_physical_random_walk(values, config)
    assert values[1, 1] == 0
    assert values[1, 0] + values[1, 2] == config.n_particles

    two_node = DiffusionConfig(
        n_particles=101,
        n_nodes=2,
        n_steps=4,
        impulse_index=0,
        dt=1.0,
        length=1.0,
        diffusion_coefficient=0.5,
    )
    _assert_physical_random_walk(
        vectorized_multinomial_random_walk_diffusion(
            two_node,
            rng=np.random.default_rng(5),
        ),
        two_node,
    )


def test_vectorized_and_loop_multinomial_have_exact_one_step_moments():
    config = DiffusionConfig(
        n_particles=200,
        n_nodes=5,
        n_steps=2,
        impulse_index=2,
        dt=1.0,
        length=4.0,
        diffusion_coefficient=0.2,
    )
    n_runs = 4_000
    loop = np.stack(
        [
            multinomial_random_walk_diffusion(
                config,
                rng=np.random.default_rng(10_000 + run),
            )[-1, 1:4]
            for run in range(n_runs)
        ]
    )
    vectorized = np.stack(
        [
            vectorized_multinomial_random_walk_diffusion(
                config,
                rng=np.random.default_rng(20_000 + run),
            )[-1, 1:4]
            for run in range(n_runs)
        ]
    )

    probabilities = np.asarray([0.2, 0.6, 0.2])
    expected_mean = config.n_particles * probabilities
    expected_covariance = config.n_particles * (
        np.diag(probabilities) - np.outer(probabilities, probabilities)
    )
    for samples in (loop, vectorized):
        assert np.max(np.abs(samples.mean(axis=0) - expected_mean)) < 0.5
        assert np.max(
            np.abs(np.cov(samples, rowvar=False, ddof=1) - expected_covariance)
        ) < 3.5


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
