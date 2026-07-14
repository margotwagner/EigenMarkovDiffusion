import numpy as np

from eigendiffusion.metrics import (
    empirical_covariance,
    relative_l2_error_from_time,
    summarize_ensemble,
)


def test_relative_l2_error_from_time_excludes_early_samples():
    reference = np.ones((4, 2))
    estimate = reference.copy()
    estimate[0] = 10.0
    times = np.arange(4, dtype=float)
    assert relative_l2_error_from_time(reference, estimate, times, 1.0) == 0.0


def test_ensemble_diagnostics_negative_mass_and_mass_error():
    reference_mean = np.array([[2.0, 0.0], [1.0, 1.0]])
    reference_variance = np.ones_like(reference_mean)
    runs = np.array(
        [
            [[2.0, 0.0], [1.5, 0.5]],
            [[2.0, 0.0], [-0.5, 2.5]],
        ]
    )
    diagnostics = summarize_ensemble(
        reference_mean,
        reference_variance,
        runs,
        total_particles=2,
    )
    assert diagnostics.maximum_mass_error == 0.0
    assert diagnostics.minimum_reconstructed_count == -0.5
    assert diagnostics.negative_entry_fraction > 0.0
    assert diagnostics.mean_negative_mass_fraction > 0.0


def test_empirical_covariance_shape():
    runs = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [1.0, 0.0]],
        ]
    )
    covariance = empirical_covariance(runs, [1])
    assert covariance.shape == (1, 2, 2)
    assert np.allclose(covariance, np.swapaxes(covariance, 1, 2))
