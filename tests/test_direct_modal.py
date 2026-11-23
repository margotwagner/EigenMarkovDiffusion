import numpy as np
import pytest

from eigendiffusion.config import DiffusionConfig
from eigendiffusion.direct_modal import DirectModalGaussianDiffusion


def _dense_transition(config: DiffusionConfig) -> np.ndarray:
    probability = config.jump_probability
    transition = np.zeros((config.n_nodes, config.n_nodes), dtype=float)
    for source in range(config.n_nodes):
        transition[source, source] = (
            1.0 - probability if source in (0, config.n_nodes - 1) else 1.0 - 2.0 * probability
        )
        if source > 0:
            transition[source, source - 1] = probability
        if source + 1 < config.n_nodes:
            transition[source, source + 1] = probability
    return transition


def _dense_probabilities(config: DiffusionConfig, transition: np.ndarray) -> np.ndarray:
    probabilities = np.zeros((config.n_steps, config.n_nodes), dtype=float)
    probabilities[0, config.impulse_index] = 1.0
    for step in range(config.n_steps - 1):
        probabilities[step + 1] = probabilities[step] @ transition
    return probabilities


def test_exact_projected_means_same_time_and_all_pairwise_covariances():
    config = DiffusionConfig(
        n_particles=37,
        n_nodes=6,
        n_steps=7,
        impulse_index=0,
        dt=0.4,
    )
    simulator = DirectModalGaussianDiffusion(config, n_modes=4)
    transition = _dense_transition(config)
    probabilities = _dense_probabilities(config, transition)
    vectors = simulator.basis.eigenvectors

    expected_mean = config.n_particles * probabilities @ vectors
    assert np.allclose(simulator.transition_matrix, transition, atol=1.0e-14)
    assert np.allclose(simulator.modal_means, expected_mean, atol=1.0e-12)

    all_cross = simulator.all_modal_cross_covariances()
    for first in range(config.n_steps):
        for second in range(config.n_steps):
            if first <= second:
                transition_power = np.linalg.matrix_power(transition, second - first)
                spatial_cross = config.n_particles * (
                    np.diag(probabilities[first]) @ transition_power
                    - np.outer(probabilities[first], probabilities[second])
                )
                expected = vectors.T @ spatial_cross @ vectors
            else:
                expected = all_cross[second, first].T
            assert np.allclose(all_cross[first, second], expected, atol=2.0e-12)
            assert np.allclose(
                simulator.modal_cross_covariance(first, second),
                expected,
                atol=2.0e-12,
            )

    assert np.allclose(
        simulator.modal_covariances,
        all_cross[np.arange(config.n_steps), np.arange(config.n_steps)],
    )


def test_cached_singular_innovation_factors_satisfy_exact_recursion():
    config = DiffusionConfig(
        n_particles=113,
        n_nodes=8,
        n_steps=9,
        impulse_index=3,
        dt=0.3,
    )
    simulator = DirectModalGaussianDiffusion(config, n_modes=6)
    covariance = simulator.modal_covariances
    diagonal = simulator.one_step_factors
    expected = (
        covariance[1:]
        - diagonal[None, :, None]
        * covariance[:-1]
        * diagonal[None, None, :]
    )
    factors = simulator.innovation_factors

    assert np.allclose(simulator.innovation_covariances, expected, atol=1.0e-11)
    assert np.allclose(
        factors @ np.swapaxes(factors, -1, -2),
        simulator.innovation_covariances,
        atol=1.0e-13,
    )
    assert np.all(np.linalg.eigvalsh(simulator.innovation_covariances) >= -1.0e-12)
    assert np.all(simulator.innovation_covariances[:, 0] == 0.0)
    assert np.all(simulator.innovation_covariances[:, :, 0] == 0.0)


def test_seed_reproducibility_optional_reconstruction_and_mass_pin():
    config = DiffusionConfig(
        n_particles=211,
        n_nodes=9,
        n_steps=12,
        impulse_index=2,
        dt=0.25,
    )
    simulator = DirectModalGaussianDiffusion(config, n_modes=5)
    first = simulator.run(np.random.default_rng(19))
    second = simulator.run(np.random.default_rng(19))
    compact = simulator.run(
        np.random.default_rng(19),
        reconstruct_spatial=False,
    )

    assert np.array_equal(first.modal_amplitudes, second.modal_amplitudes)
    assert np.array_equal(first.spatial_counts, second.spatial_counts)
    assert np.array_equal(first.modal_amplitudes, compact.modal_amplitudes)
    assert compact.spatial_counts is None
    assert not compact.has_spatial_reconstruction
    assert first.has_spatial_reconstruction
    assert np.all(first.modal_amplitudes[:, 0] == first.modal_amplitudes[0, 0])
    assert np.allclose(first.spatial_counts.sum(axis=1), config.n_particles, atol=1.0e-11)


@pytest.mark.parametrize("impulse_index", [0, 6])
def test_full_basis_edge_impulse_matches_dense_spatial_moments(impulse_index):
    config = DiffusionConfig(
        n_particles=59,
        n_nodes=7,
        n_steps=6,
        impulse_index=impulse_index,
        dt=0.5,
    )
    simulator = DirectModalGaussianDiffusion(config, n_modes=config.n_nodes)
    transition = _dense_transition(config)
    probabilities = _dense_probabilities(config, transition)
    vectors = simulator.basis.eigenvectors
    spatial_mean = simulator.modal_means @ vectors.T
    spatial_covariance = np.einsum(
        "im,tmn,jn->tij",
        vectors,
        simulator.modal_covariances,
        vectors,
    )
    expected_covariance = config.n_particles * (
        np.einsum("ti,ij->tij", probabilities, np.eye(config.n_nodes))
        - np.einsum("ti,tj->tij", probabilities, probabilities)
    )

    assert np.allclose(spatial_mean, config.n_particles * probabilities, atol=2.0e-12)
    assert np.allclose(spatial_covariance, expected_covariance, atol=2.0e-12)
    result = simulator.run(np.random.default_rng(7))
    assert np.allclose(result.spatial_counts[0], simulator.initial_spatial_counts())


def test_one_mode_model_is_deterministic_uniform_mass():
    config = DiffusionConfig(
        n_particles=101,
        n_nodes=7,
        n_steps=8,
        impulse_index=0,
    )
    simulator = DirectModalGaussianDiffusion(config, n_modes=1)
    result = simulator.run(np.random.default_rng(5))

    assert np.all(simulator.modal_covariances == 0.0)
    assert np.all(simulator.innovation_factors == 0.0)
    assert np.allclose(result.spatial_counts, config.n_particles / config.n_nodes)
    assert np.allclose(result.spatial_counts.sum(axis=1), config.n_particles)
