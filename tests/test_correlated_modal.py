import numpy as np

from eigendiffusion import (
    BankedCorrelatedModalDiffusion,
    CorrelatedModalDiffusion,
    DiffusionConfig,
    EigenMarkovDiffusion,
    IndependentModalDiffusion,
    nearest_neighbour_transition_covariance,
    project_to_mass_simplex,
    run_modal_ensemble,
    sample_nearest_neighbour_gaussian_noise,
)
from eigendiffusion.correlated_modal import _PreparedNearestNeighbourGaussianSampler


def _legacy_nearest_neighbour_gaussian_noise(
    source_counts: np.ndarray,
    jump_probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Reference the allocation-heavy sampler used before preparation."""

    counts = np.clip(np.asarray(source_counts, dtype=float), 0.0, None)
    p = float(jump_probability)
    noise = np.zeros_like(counts)

    def local_noise(mass: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(probabilities, dtype=float)
        if mass.ndim == 0:
            z = rng.standard_normal(probabilities.size)
            weighted = np.sqrt(probabilities) * z
            return np.sqrt(float(mass)) * (
                weighted - probabilities * float(weighted.sum())
            )

        z = rng.standard_normal((mass.size, probabilities.size))
        weighted = np.sqrt(probabilities)[None, :] * z
        return np.sqrt(mass)[:, None] * (
            weighted - probabilities[None, :] * weighted.sum(axis=1, keepdims=True)
        )

    left = local_noise(np.asarray(counts[0]), np.asarray([1.0 - p, p]))
    noise[0] += left[0]
    noise[1] += left[1]

    if counts.size > 2:
        interior = local_noise(
            counts[1:-1],
            np.asarray([p, 1.0 - 2.0 * p, p]),
        )
        noise[:-2] += interior[:, 0]
        noise[1:-1] += interior[:, 1]
        noise[2:] += interior[:, 2]

    right = local_noise(np.asarray(counts[-1]), np.asarray([p, 1.0 - p]))
    noise[-2] += right[0]
    noise[-1] += right[1]
    noise[-1] -= float(noise.sum())
    return noise


def test_independent_modal_alias_preserves_original_model_exactly():
    config = DiffusionConfig(n_particles=200, n_nodes=15, n_steps=12)
    original = EigenMarkovDiffusion(config, n_modes=10).run(
        rng=np.random.default_rng(7)
    )
    renamed = IndependentModalDiffusion(config, n_modes=10).run(
        rng=np.random.default_rng(7)
    )
    assert np.array_equal(original.positive_states, renamed.positive_states)
    assert np.array_equal(original.negative_states, renamed.negative_states)
    assert np.allclose(original.spatial_counts, renamed.spatial_counts)


def test_simplex_projection_is_nonnegative_and_mass_conserving():
    values = np.array([-3.0, 1.0, 4.0, 10.0])
    projected = project_to_mass_simplex(values, total_mass=8.0)
    assert np.all(projected >= 0.0)
    assert np.isclose(projected.sum(), 8.0)


def test_shared_spatial_noise_is_zero_sum():
    counts = np.array([10.0, 20.0, 30.0, 40.0])
    noise = sample_nearest_neighbour_gaussian_noise(
        counts,
        jump_probability=0.1,
        rng=np.random.default_rng(4),
    )
    assert abs(noise.sum()) < 1e-12


def test_public_noise_sampler_preserves_legacy_bits_and_rng_state():
    counts = np.array([-5.0e-13, 10.0])
    expected_rng = np.random.default_rng(41)
    actual_rng = np.random.default_rng(41)

    expected = _legacy_nearest_neighbour_gaussian_noise(counts, 0.5, expected_rng)
    actual = sample_nearest_neighbour_gaussian_noise(counts, 0.5, actual_rng)

    assert np.array_equal(actual, expected)
    np.testing.assert_equal(actual_rng.bit_generator.state, expected_rng.bit_generator.state)


def test_prepared_noise_sampler_reuses_workspace_without_changing_stream():
    jump_probability = 0.12
    sampler = _PreparedNearestNeighbourGaussianSampler(21, jump_probability)
    workspace = sampler.new_workspace()
    expected_rng = np.random.default_rng(73)
    actual_rng = np.random.default_rng(73)

    for counts in (
        np.linspace(0.0, 100.0, 21),
        np.linspace(100.0, 1.0, 21),
        np.square(np.linspace(0.0, 5.0, 21)),
    ):
        expected = _legacy_nearest_neighbour_gaussian_noise(
            counts,
            jump_probability,
            expected_rng,
        )
        actual = sampler.sample(counts, actual_rng, workspace)
        assert np.array_equal(actual, expected)

    np.testing.assert_equal(actual_rng.bit_generator.state, expected_rng.bit_generator.state)


def test_shared_spatial_noise_matches_conditional_covariance():
    counts = np.array([20.0, 40.0, 30.0, 10.0])
    jump_probability = 0.12
    expected = nearest_neighbour_transition_covariance(counts, jump_probability)
    rng = np.random.default_rng(11)
    samples = np.stack(
        [
            sample_nearest_neighbour_gaussian_noise(
                counts,
                jump_probability=jump_probability,
                rng=rng,
            )
            for _ in range(12_000)
        ]
    )
    observed = np.cov(samples, rowvar=False, bias=True)
    relative_error = np.linalg.norm(observed - expected) / np.linalg.norm(expected)
    assert relative_error < 0.06


def test_correlated_modal_full_basis_preserves_mass():
    config = DiffusionConfig(n_particles=200, n_nodes=15, n_steps=20)
    result = CorrelatedModalDiffusion(config).run(rng=np.random.default_rng(5))
    assert np.allclose(result.spatial_counts.sum(axis=1), config.n_particles, atol=1e-9)


def test_banked_correlated_modal_is_nonnegative_and_mass_conserving():
    config = DiffusionConfig(n_particles=200, n_nodes=15, n_steps=20)
    result = BankedCorrelatedModalDiffusion(config, n_modes=8).run(
        rng=np.random.default_rng(5)
    )
    assert np.min(result.spatial_counts) >= -1e-12
    assert np.allclose(result.spatial_counts.sum(axis=1), config.n_particles, atol=1e-9)
    assert result.bank_balances.shape == result.spatial_counts.shape


def test_banked_ensemble_exposes_compact_bank_diagnostics():
    config = DiffusionConfig(n_particles=100, n_nodes=11, n_steps=8)
    ensemble = run_modal_ensemble(
        config,
        n_runs=3,
        model="banked_correlated_modal",
        n_modes=6,
        seed=2,
    )
    assert ensemble.model_name == "banked_correlated_modal"
    assert ensemble.auxiliary is not None
    assert ensemble.auxiliary["bank_l1_fraction"].shape == (3, config.n_steps)
