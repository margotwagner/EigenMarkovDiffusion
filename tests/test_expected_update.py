import numpy as np

from eigendiffusion import DiffusionConfig, EigenMarkovDiffusion


def test_markov_transition_matches_first_order_modal_decay():
    config = DiffusionConfig(n_particles=100, n_nodes=11, n_steps=2)
    model = EigenMarkovDiffusion(config)
    amplitudes = np.linspace(-5.0, 5.0, model.n_modes)
    probabilities = model.transition_probabilities
    expected_from_markov = (1.0 - 2.0 * probabilities) * amplitudes
    expected_euler = (1.0 - model.basis.eigenvalues * config.dt) * amplitudes
    assert np.allclose(expected_from_markov, expected_euler)
