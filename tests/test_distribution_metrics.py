from __future__ import annotations

import numpy as np

from eigendiffusion.distribution_metrics import (
    histogram_jensen_shannon_divergence,
    summarize_distribution_ensemble,
    wasserstein_distance_1d,
)


def test_scalar_distribution_metrics_are_zero_for_identical_samples() -> None:
    samples = np.asarray([0.0, 1.0, 1.0, 3.0])
    assert wasserstein_distance_1d(samples, samples) == 0.0
    assert histogram_jensen_shannon_divergence(samples, samples) == 0.0


def test_distribution_summary_detects_shift() -> None:
    rng = np.random.default_rng(4)
    reference = rng.normal(size=(30, 3, 4))
    shifted = reference.copy()
    shifted[:, 1:, 0] += 2.0
    identical = summarize_distribution_ensemble(
        reference,
        reference,
        time_indices=[1, 2],
        node_indices=[0, 1],
        n_projections=8,
        seed=2,
    )
    different = summarize_distribution_ensemble(
        reference,
        shifted,
        time_indices=[1, 2],
        node_indices=[0, 1],
        n_projections=8,
        seed=2,
    )
    assert identical.marginal_wasserstein == 0.0
    assert identical.sliced_wasserstein == 0.0
    assert different.marginal_wasserstein > 0.0
    assert different.sliced_wasserstein > 0.0
