import numpy as np

from eigendiffusion.config import DiffusionConfig
from eigendiffusion.handoff import pareto_handoff_indices, predict_handoff_grid


def small_config() -> DiffusionConfig:
    return DiffusionConfig(
        n_particles=1000,
        n_nodes=21,
        n_steps=21,
        impulse_index=10,
        dt=1.0,
        length=4.0,
        diffusion_coefficient=2.20e-4,
    )


def test_full_rank_handoff_has_zero_truncation_error():
    config = small_config()
    prediction = predict_handoff_grid(
        config,
        final_modes=[config.n_nodes],
        handoff_times=[5.0],
        covariance_time_indices=[1, 5, 10, 20],
    )[0]
    assert np.isclose(prediction.predicted_mean_relative_l2_error, 0.0)
    assert np.isclose(prediction.predicted_variance_relative_l2_error, 0.0)
    assert np.isclose(
        prediction.predicted_covariance_relative_frobenius_error,
        0.0,
    )


def test_more_modes_and_later_handoff_reduce_error_but_increase_work():
    config = small_config()
    predictions = predict_handoff_grid(
        config,
        final_modes=[5, 10],
        handoff_times=[2.0, 10.0],
        covariance_time_indices=[1, 5, 10, 20],
    )
    by_pair = {
        (row.final_n_modes, row.handoff_time): row for row in predictions
    }
    early_small = by_pair[(5, 2.0)]
    late_small = by_pair[(5, 10.0)]
    early_large = by_pair[(10, 2.0)]
    assert late_small.predicted_mean_relative_l2_error < early_small.predicted_mean_relative_l2_error
    assert late_small.modal_update_fraction_of_full > early_small.modal_update_fraction_of_full
    assert early_large.predicted_covariance_relative_frobenius_error < early_small.predicted_covariance_relative_frobenius_error
    assert early_large.modal_update_fraction_of_full > early_small.modal_update_fraction_of_full


def test_pareto_frontier_is_nonempty():
    config = small_config()
    predictions = predict_handoff_grid(
        config,
        final_modes=[5, 10, 15],
        handoff_times=[2.0, 5.0, 10.0],
        covariance_time_indices=[1, 5, 10, 20],
    )
    frontier = pareto_handoff_indices(predictions)
    assert frontier
    assert frontier <= set(range(len(predictions)))
