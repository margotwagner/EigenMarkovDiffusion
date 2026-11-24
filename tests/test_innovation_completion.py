import gc
import weakref
from dataclasses import replace

import numpy as np
import pytest

import eigendiffusion.innovation_completion as innovation_completion
from eigendiffusion.config import DiffusionConfig
from eigendiffusion.direct_modal import DirectModalGaussianDiffusion
from eigendiffusion.innovation_completion import (
    InnovationConditionedModalCompleter,
    PreparedInnovationCompletion,
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


@pytest.mark.parametrize("retained, rank", [(3, None), (3, 1), (1, None), (7, None)])
def test_batched_sampler_matches_stepwise_paths_and_innovation_covariance(retained, rank):
    config = _config()
    model = InnovationConditionedModalCompleter(
        config, retained_modes=retained, conditional_rank=rank
    )
    reference_rng = np.random.default_rng(431)
    expected = model.run_stepwise(reference_rng)
    assert model._batched_sampler is None
    batch_rng = np.random.default_rng(431)
    actual = model.run_batched(batch_rng)

    np.testing.assert_allclose(
        actual.modal_amplitudes, expected.modal_amplitudes, rtol=0, atol=2e-11
    )
    np.testing.assert_allclose(actual.spatial_counts, expected.spatial_counts, rtol=0, atol=2e-11)
    assert np.array_equal(actual.modal_amplitudes[:, 0], expected.modal_amplitudes[:, 0])
    np.testing.assert_allclose(actual.spatial_counts.sum(axis=1), config.n_particles)
    assert batch_rng.standard_normal() == reference_rng.standard_normal()

    prepared = model.prepare_batched_sampler()
    assert model.prepare_batched_sampler() is prepared
    factors = prepared.joint_factors
    np.testing.assert_allclose(
        factors @ factors.transpose(0, 2, 1),
        model.effective_innovation_covariances(),
        rtol=0,
        atol=2e-11,
    )
    assert prepared.latent_count == (config.n_steps - 1) * (retained + model.conditional_rank)
    assert prepared.nbytes == factors.nbytes + prepared.selected_ranks.nbytes


def test_batched_sampler_skips_padding_when_step_ranks_vary():
    model = InnovationConditionedModalCompleter(_config(), retained_modes=3)
    # Exercise the draw layout with a different residual rank at each step.
    ranks = [0, 4, 1, 3, 2, 0, 4]
    model._conditioning_steps = tuple(
        replace(step, selected_rank=rank, conditional_factor=step.conditional_factor[:, :rank])
        for step, rank in zip(model.conditioning_steps, ranks)
    )
    prepared = model.prepare_batched_sampler()
    assert np.array_equal(prepared.selected_ranks, ranks)
    assert prepared.latent_count == len(ranks) * model.retained_modes + sum(ranks)
    assert prepared._active_columns is not None
    assert not prepared._active_columns.flags.writeable
    assert prepared.nbytes == (
        prepared.joint_factors.nbytes
        + prepared.selected_ranks.nbytes
        + prepared._active_columns.nbytes
    )

    reference_rng = np.random.default_rng(805)
    expected = model.run_stepwise(reference_rng)
    batch_rng = np.random.default_rng(805)
    actual = model.run_batched(batch_rng)
    np.testing.assert_allclose(
        actual.modal_amplitudes, expected.modal_amplitudes, rtol=0, atol=2e-11
    )
    next_draw = batch_rng.standard_normal()
    assert next_draw == reference_rng.standard_normal()

    compact = model.prepare_runtime()
    compact_rng = np.random.default_rng(805)
    compact_result = compact.run(compact_rng)
    np.testing.assert_array_equal(compact_result.modal_amplitudes, actual.modal_amplitudes)
    np.testing.assert_array_equal(compact_result.spatial_counts, actual.spatial_counts)
    assert compact_rng.standard_normal() == next_draw
    assert compact.nbytes == (
        compact.initial_modal.nbytes
        + compact.one_step_factors.nbytes
        + compact.basis.eigenvalues.nbytes
        + compact.basis.eigenvectors.nbytes
        + prepared.nbytes
    )


@pytest.mark.parametrize("retained,rank", [(3, None), (3, 1), (1, None), (7, None), (5, 100)])
def test_streaming_runtime_matches_full_history_factors_paths_and_rng(retained, rank):
    config = _config(impulse_index=3)
    settings = dict(
        retained_modes=retained,
        conditional_rank=rank,
        pseudoinverse_tolerance=1e-9,
        psd_tolerance=2e-10,
    )
    oracle = PreparedInnovationCompletion.from_full_history_config(config, **settings)
    compact = PreparedInnovationCompletion.from_config(config, **settings)
    np.testing.assert_array_equal(compact.sampler.joint_factors, oracle.sampler.joint_factors)
    np.testing.assert_array_equal(compact.sampler.selected_ranks, oracle.sampler.selected_ranks)
    assert compact.sampler._active_columns is oracle.sampler._active_columns is None
    assert compact.sampler.latent_count == oracle.sampler.latent_count
    assert compact.nbytes == oracle.nbytes
    batch_rng = np.random.default_rng(431)
    compact_rng = np.random.default_rng(431)
    for _ in range(2):
        expected = oracle.run(batch_rng)
        actual = compact.run(compact_rng)
        np.testing.assert_array_equal(actual.modal_amplitudes, expected.modal_amplitudes)
        np.testing.assert_array_equal(actual.spatial_counts, expected.spatial_counts)
        assert actual.basis is compact.basis
        assert actual.retained_modes == expected.retained_modes
        assert compact_rng.bit_generator.state == batch_rng.bit_generator.state


def test_streaming_build_uses_only_transition_local_full_factors(monkeypatch):
    def history_forbidden(*args, **kwargs):
        pytest.fail("streaming setup constructed a diagnostic model or conditioning record")

    monkeypatch.setattr(InnovationConditionedModalCompleter, "__init__", history_forbidden)
    monkeypatch.setattr(DirectModalGaussianDiffusion, "__init__", history_forbidden)
    monkeypatch.setattr(innovation_completion, "InnovationConditioningStep", history_forbidden)

    covariances = []
    probabilities = []
    modal_covariance = innovation_completion._modal_covariance

    def check_moment_lifetimes(probability, vectors, particle_count):
        assert all(ref() is None for ref in covariances[:-1])
        assert all(ref() is None for ref in probabilities)
        current = modal_covariance(probability, vectors, particle_count)
        covariances.append(weakref.ref(current))
        probabilities.append(weakref.ref(probability))
        return current

    monkeypatch.setattr(innovation_completion, "_modal_covariance", check_moment_lifetimes)
    previous_factor = None
    pack = innovation_completion._pack_conditioned_innovation

    def check_factor_lifetime(packed, full_factor, *settings):
        nonlocal previous_factor
        assert previous_factor is None or previous_factor() is None
        previous_factor = weakref.ref(full_factor)
        pack(packed, full_factor, *settings)

    monkeypatch.setattr(
        innovation_completion, "_pack_conditioned_innovation", check_factor_lifetime
    )
    compact = PreparedInnovationCompletion.from_config(_config(), retained_modes=3)
    assert previous_factor is not None and previous_factor() is None
    assert len(probabilities) == len(covariances) == _config().n_steps - 1
    assert all(ref() is None for ref in covariances + probabilities)
    arrays = (
        compact.initial_modal,
        compact.one_step_factors,
        compact.basis.eigenvalues,
        compact.basis.eigenvectors,
        compact.sampler.joint_factors,
        compact.sampler.selected_ranks,
    )
    assert all(a.flags.owndata and not a.flags.writeable for a in arrays)
    assert compact.nbytes == sum(a.nbytes for a in arrays)


def test_compact_runtime_releases_setup_histories_and_keeps_owned_readonly_arrays():
    model = InnovationConditionedModalCompleter(_config(), retained_modes=3)
    expected = model.run_batched(np.random.default_rng(74))
    compact = model.prepare_runtime()
    assert model._initial_modal.base is None
    assert compact.sampler is model.prepare_batched_sampler()
    for original, copied in (
        (model._initial_modal, compact.initial_modal),
        (model._one_step_factors, compact.one_step_factors),
        (model.basis.eigenvalues, compact.basis.eigenvalues),
        (model.basis.eigenvectors, compact.basis.eigenvectors),
    ):
        assert copied.base is None
        assert not np.shares_memory(copied, original)
        assert not copied.flags.writeable
    assert not compact.sampler.joint_factors.flags.writeable
    assert not compact.sampler.selected_ranks.flags.writeable
    references = [
        weakref.ref(model),
        weakref.ref(model.full_reference),
        weakref.ref(model._target_innovation_covariances),
        weakref.ref(model.conditioning_steps[0].conditional_factor),
    ]
    del model
    gc.collect()
    assert all(reference() is None for reference in references)
    actual = compact.run(np.random.default_rng(74))
    np.testing.assert_array_equal(actual.modal_amplitudes, expected.modal_amplitudes)
    np.testing.assert_array_equal(actual.spatial_counts, expected.spatial_counts)


def test_conditioning_arrays_are_immutable_after_batched_preparation():
    model = InnovationConditionedModalCompleter(_config(), retained_modes=3)
    model.prepare_batched_sampler()

    with pytest.raises(ValueError, match="read-only"):
        model.conditioning_steps[-1].conditional_factor[0, 0] = 0.0


def test_run_uses_the_prepared_batched_path():
    model = InnovationConditionedModalCompleter(_config(), retained_modes=3)

    default = model.run(np.random.default_rng(37))
    explicit = model.run_batched(np.random.default_rng(37))

    assert model._batched_sampler is not None
    assert np.array_equal(default.modal_amplitudes, explicit.modal_amplitudes)
    assert np.array_equal(default.spatial_counts, explicit.spatial_counts)


def test_batched_sampler_matches_stepwise_at_representative_dimension():
    config = DiffusionConfig(
        n_particles=5_275,
        n_nodes=101,
        n_steps=30,
        impulse_index=59,
        dt=1.0,
        length=4.0,
        diffusion_coefficient=2.2e-4,
    )
    model = InnovationConditionedModalCompleter(
        config,
        retained_modes=75,
        conditional_rank=25,
    )
    reference_rng = np.random.default_rng(431)
    expected = model.run_stepwise(reference_rng)
    batch_rng = np.random.default_rng(431)
    actual = model.run_batched(batch_rng)

    np.testing.assert_allclose(
        actual.modal_amplitudes,
        expected.modal_amplitudes,
        rtol=0.0,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        actual.spatial_counts,
        expected.spatial_counts,
        rtol=0.0,
        atol=1.0e-8,
    )
    sampler = model.prepare_batched_sampler()
    compact = PreparedInnovationCompletion.from_config(
        config, retained_modes=75, conditional_rank=25
    )
    np.testing.assert_array_equal(compact.sampler.joint_factors, sampler.joint_factors)
    stream_rng = np.random.default_rng(431)
    streamed = compact.run(stream_rng)
    np.testing.assert_array_equal(streamed.modal_amplitudes, actual.modal_amplitudes)
    np.testing.assert_array_equal(streamed.spatial_counts, actual.spatial_counts)
    assert stream_rng.bit_generator.state == batch_rng.bit_generator.state
    packed_covariances = np.stack(
        [factor @ factor.T for factor in sampler.joint_factors]
    )
    np.testing.assert_allclose(
        model.effective_innovation_covariances(),
        packed_covariances,
        rtol=1.0e-13,
        atol=1.0e-9,
    )
    assert batch_rng.standard_normal() == reference_rng.standard_normal()
