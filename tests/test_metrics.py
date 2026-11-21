import numpy as np

from eigendiffusion.metrics import (
    bootstrap_error_intervals,
    empirical_covariance,
    paired_bootstrap_error_difference_intervals,
    relative_l2_error_from_time,
    summarize_ensemble,
    time_resolved_relative_l2_error,
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


def test_time_resolved_error_handles_zero_reference_norm():
    reference = np.array([[0.0, 0.0], [1.0, 1.0]])
    estimate = np.array([[0.0, 0.0], [2.0, 1.0]])
    error = time_resolved_relative_l2_error(reference, estimate)
    assert error[0] == 0.0
    assert np.isclose(error[1], 1.0 / np.sqrt(2.0))


def test_time_resolved_error_can_exclude_negligible_reference_norms():
    reference = np.array([[1.0e-15, 0.0], [1.0, 1.0]])
    estimate = np.array([[1.0, 0.0], [1.0, 1.0]])

    unfiltered = time_resolved_relative_l2_error(reference, estimate)
    filtered = time_resolved_relative_l2_error(
        reference,
        estimate,
        minimum_reference_norm_fraction=1.0e-12,
    )

    assert np.isfinite(unfiltered[0])
    assert np.isnan(filtered[0])
    assert filtered[1] == 0.0


def test_ensemble_diagnostics_include_time_and_sampling_metrics():
    reference_mean = np.ones((3, 2))
    reference_variance = np.ones((3, 2))
    runs = np.stack([reference_mean - 1.0, reference_mean + 1.0])
    diagnostics = summarize_ensemble(
        reference_mean,
        reference_variance,
        runs,
        total_particles=2,
    )
    assert diagnostics.mean_max_time_relative_l2_error == 0.0
    assert diagnostics.mean_final_time_relative_l2_error == 0.0
    assert diagnostics.mean_error_to_reference_se_ratio == 0.0
    assert diagnostics.variance_relative_l2_error == 0.0


def test_ensemble_variance_max_excludes_roundoff_scale_initial_variance():
    reference_mean = np.ones((3, 2))
    reference_variance = np.ones((3, 2))
    reference_variance[0] = 1.0e-15
    runs = np.stack([reference_mean - 1.0, reference_mean + 1.0])

    diagnostics = summarize_ensemble(
        reference_mean,
        reference_variance,
        runs,
        total_particles=2,
    )

    assert diagnostics.variance_max_time_relative_l2_error == 0.0
    assert diagnostics.variance_final_time_relative_l2_error == 0.0


def test_bootstrap_error_intervals_are_ordered_and_reproducible():
    rng = np.random.default_rng(5)
    reference_mean = np.ones((3, 2))
    reference_variance = np.ones((3, 2))
    runs = rng.normal(1.0, 1.0, size=(30, 3, 2))
    first = bootstrap_error_intervals(
        reference_mean,
        reference_variance,
        runs,
        n_resamples=20,
        n_blocks=5,
        seed=7,
    )
    second = bootstrap_error_intervals(
        reference_mean,
        reference_variance,
        runs,
        n_resamples=20,
        n_blocks=5,
        seed=7,
    )
    assert first == second
    for lower, upper in first.values():
        assert lower <= upper


def test_paired_bootstrap_difference_is_zero_for_identical_runs():
    rng = np.random.default_rng(11)
    reference_mean = np.ones((3, 2))
    reference_variance = np.ones((3, 2))
    runs = rng.normal(1.0, 1.0, size=(30, 3, 2))
    covariance = np.stack([np.eye(2) for _ in range(3)])
    intervals = paired_bootstrap_error_difference_intervals(
        reference_mean,
        reference_variance,
        runs,
        runs,
        reference_covariance=covariance,
        covariance_time_indices=[1, 2],
        n_resamples=20,
        n_blocks=5,
        seed=9,
    )
    assert set(intervals) == {
        "mean_relative_l2_error",
        "variance_relative_l2_error",
        "covariance_relative_frobenius_error",
    }
    for lower, upper in intervals.values():
        assert lower == 0.0
        assert upper == 0.0


def test_paired_bootstrap_difference_validates_matching_shapes():
    reference_mean = np.ones((3, 2))
    reference_variance = np.ones((3, 2))
    baseline = np.ones((4, 3, 2))
    candidate = np.ones((5, 3, 2))
    with np.testing.assert_raises_regex(ValueError, "same shape"):
        paired_bootstrap_error_difference_intervals(
            reference_mean,
            reference_variance,
            baseline,
            candidate,
        )
