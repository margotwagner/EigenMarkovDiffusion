import numpy as np

from eigendiffusion import DiffusionConfig, EigenMarkovDiffusion


def test_transition_probabilities_are_valid():
    config = DiffusionConfig(n_particles=200, n_nodes=21, n_steps=20)
    model = EigenMarkovDiffusion(config)
    probabilities = model.transition_probabilities
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    assert probabilities[0] < 1e-12


def test_zero_mode_preserves_total_mass_exactly():
    config = DiffusionConfig(n_particles=200, n_nodes=21, n_steps=20)
    model = EigenMarkovDiffusion(config, modal_particle_weight=2.0)
    result = model.run(rng=np.random.default_rng(4))
    assert np.allclose(result.spatial_counts.sum(axis=1), config.n_particles, atol=1e-9)


def test_modal_state_totals_are_conserved_per_mode():
    config = DiffusionConfig(n_particles=500, n_nodes=21, n_steps=30)
    model = EigenMarkovDiffusion(config)
    result = model.run(rng=np.random.default_rng(5))
    totals = result.positive_states + result.negative_states
    assert np.all(totals == totals[0])


def test_truncated_model_has_requested_shape():
    config = DiffusionConfig(n_particles=100, n_nodes=21, n_steps=10)
    model = EigenMarkovDiffusion(config, n_modes=5)
    result = model.run(rng=np.random.default_rng(6))
    assert result.modal_amplitudes.shape == (config.n_steps, 5)
    assert result.spatial_counts.shape == (config.n_steps, config.n_nodes)
