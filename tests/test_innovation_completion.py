import numpy as np
import pytest

from eigendiffusion.config import DiffusionConfig
from eigendiffusion.direct_modal import DirectModalGaussianDiffusion
from eigendiffusion.innovation_completion import (
    InnovationConditionedModalCompleter,
)


def _config(*, impulse_index: int = 0) -> DiffusionConfig:
    return DiffusionConfig(
        n_particles=173,
        n_nodes=7,
        n_steps=8,
        impulse_index=impulse_index,
        dt=1.0,
        length=4.0,
        diffusion_coefficient=0.1,
    )


def test_full_conditional_rank_reconstructs_every_full_innovation_covariance():
    config = _config()
    retained = 3
    completer = InnovationConditionedModalCompleter(
        config,
        retained_modes=retained,
    )
    target = completer.target_innovation_covariances

    for time_index, step in enumerate(completer.conditioning_steps):
        q_rr = target[time_index, :retained, :retained]
        q_ur = target[time_index, retained:, :retained]
        expected_gain = q_ur @ np.linalg.pinv(
            q_rr,
            rcond=completer.pseudoinverse_tolerance,
            hermitian=True,
        )
        expected_conditional = target[time_index, retained:, retained:] - (
            expected_gain @ q_ur.T
        )

        assert np.allclose(step.conditional_gain, expected_gain, atol=2.0e-11)
        assert np.allclose(
            step.conditional_covariance,
            expected_conditional,
            atol=2.0e-11,
        )
        assert np.allclose(
            step.sampled_conditional_covariance,
            step.conditional_covariance,
            atol=2.0e-11,
        )

    assert np.allclose(
        completer.effective_innovation_covariances(),
        target,
        atol=3.0e-11,
    )


def test_full_rank_completion_has_exact_full_mean_covariance_and_temporal_moments():
    config = _config(impulse_index=6)
    completer = InnovationConditionedModalCompleter(
        config,
        retained_modes=4,
    )
    reference = completer.full_reference
    diagonal = completer.one_step_factors
    effective_innovations = completer.effective_innovation_covariances()

    propagated_mean = reference.modal_means[0].copy()
    propagated_covariance = np.zeros((config.n_nodes, config.n_nodes), dtype=float)
    propagated_covariances = np.empty_like(reference.modal_covariances)
    propagated_covariances[0] = propagated_covariance
    for time_index in range(config.n_steps - 1):
        propagated_mean = diagonal * propagated_mean
        propagated_covariance = (
            diagonal[:, None]
            * propagated_covariance
            * diagonal[None, :]
            + effective_innovations[time_index]
        )
        propagated_covariances[time_index + 1] = propagated_covariance
        assert np.allclose(
            propagated_mean,
            reference.modal_means[time_index + 1],
            atol=2.0e-11,
        )

    assert np.allclose(
        propagated_covariances,
        reference.modal_covariances,
        atol=3.0e-11,
    )
    for first in range(config.n_steps):
        for second in range(config.n_steps):
            if first <= second:
                expected = propagated_covariances[first] * (
                    diagonal ** (second - first)
                )[None, :]
            else:
                expected = (
                    propagated_covariances[second]
                    * (diagonal ** (first - second))[None, :]
                ).T
            assert np.allclose(
                expected,
                reference.modal_cross_covariance(first, second),
                atol=3.0e-11,
            )


def test_run_and_external_completion_are_reproducible_and_preserve_mass():
    config = _config()
    retained = 3
    completer = InnovationConditionedModalCompleter(
        config,
        retained_modes=retained,
    )
    first = completer.run(np.random.default_rng(17))
    second = completer.run(np.random.default_rng(17))
    exact_initial = completer.basis.to_modal(
        completer.full_reference.initial_spatial_counts()
    )

    assert np.array_equal(first.modal_amplitudes, second.modal_amplitudes)
    assert np.array_equal(first.spatial_counts, second.spatial_counts)
    assert np.allclose(first.modal_amplitudes[0], exact_initial, atol=2.0e-12)
    assert np.allclose(
        first.spatial_counts[0],
        completer.full_reference.initial_spatial_counts(),
        atol=2.0e-12,
    )
    assert np.allclose(first.spatial_counts.sum(axis=1), config.n_particles)

    retained_path = DirectModalGaussianDiffusion(config, n_modes=retained).run(
        np.random.default_rng(21),
        reconstruct_spatial=False,
    ).modal_amplitudes
    completed_first = completer.complete(
        retained_path,
        rng=np.random.default_rng(22),
    )
    completed_second = completer.complete(
        retained_path,
        rng=np.random.default_rng(22),
    )
    assert np.array_equal(
        completed_first.modal_amplitudes,
        completed_second.modal_amplitudes,
    )
    assert np.array_equal(
        completed_first.retained_modal_amplitudes,
        retained_path,
    )
    assert np.allclose(
        completed_first.modal_amplitudes[0, retained:],
        exact_initial[retained:],
        atol=2.0e-12,
    )
    assert np.allclose(
        completed_first.spatial_counts.sum(axis=1),
        config.n_particles,
    )


def test_conditional_rank_cap_and_no_unresolved_mode_edge_case():
    config = _config(impulse_index=3)
    capped = InnovationConditionedModalCompleter(
        config,
        retained_modes=3,
        conditional_rank=1,
    )
    full = InnovationConditionedModalCompleter(
        config,
        retained_modes=3,
    )
    assert capped.conditional_rank == 1
    assert capped.conditional_factors.shape == (
        config.n_steps - 1,
        config.n_nodes - 3,
        1,
    )
    assert np.all(capped.captured_variance_fractions <= 1.0 + 1.0e-12)
    assert np.any(capped.captured_variance_fractions < 0.95)
    assert not np.allclose(
        capped.effective_innovation_covariances(),
        full.target_innovation_covariances,
    )

    capped_to_dimension = InnovationConditionedModalCompleter(
        config,
        retained_modes=5,
        conditional_rank=100,
    )
    assert capped_to_dimension.conditional_rank == 2

    no_unresolved = InnovationConditionedModalCompleter(
        config,
        retained_modes=config.n_nodes,
    )
    result = no_unresolved.run(np.random.default_rng(9))
    assert no_unresolved.conditional_rank == 0
    assert no_unresolved.conditional_factors.shape == (config.n_steps - 1, 0, 0)
    assert result.unresolved_modal_amplitudes.shape == (config.n_steps, 0)
    assert np.allclose(
        no_unresolved.effective_innovation_covariances(),
        no_unresolved.target_innovation_covariances,
        atol=2.0e-11,
    )

    with pytest.raises(ValueError, match="conditional_rank"):
        InnovationConditionedModalCompleter(
            config,
            retained_modes=3,
            conditional_rank=0,
        )
