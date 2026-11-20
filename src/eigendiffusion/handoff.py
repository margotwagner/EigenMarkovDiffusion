"""Analytic error predictors for full-to-reduced modal handoff."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .config import DiffusionConfig
from .operators import eigendecompose, neumann_laplacian_1d
from .references import (
    discrete_expected_diffusion,
    multinomial_marginal_covariance,
    multinomial_marginal_variance,
)


@dataclass(frozen=True, slots=True)
class HandoffPrediction:
    """Predicted truncation error and modal work for one handoff choice."""

    final_n_modes: int
    requested_handoff_time: float
    handoff_step: int
    handoff_time: float
    predicted_mean_relative_l2_error: float
    predicted_post_handoff_mean_relative_l2_error: float
    predicted_variance_relative_l2_error: float
    predicted_covariance_relative_frobenius_error: float
    instantaneous_mean_projection_relative_l2: float
    instantaneous_covariance_projection_relative_frobenius: float
    modal_update_fraction_of_full: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def predict_handoff_grid(
    config: DiffusionConfig,
    final_modes: list[int],
    handoff_times: list[float],
    *,
    covariance_time_indices: list[int] | None = None,
    initial_n_modes: int | None = None,
) -> list[HandoffPrediction]:
    """Predict errors for every handoff-time/mode-count combination.

    Before handoff, the prediction is the full analytic reference. At and after
    handoff, the exact mean and covariance are orthogonally projected onto the
    retained eigenbasis. This isolates the irreducible truncation error; it does
    not include finite-ensemble sampling noise or Gaussian-closure error.
    """

    initial_modes = config.n_nodes if initial_n_modes is None else int(initial_n_modes)
    if not 1 <= initial_modes <= config.n_nodes:
        raise ValueError("initial_n_modes must lie in [1, n_nodes]")
    modes = sorted(set(int(value) for value in final_modes))
    if not modes or any(value < 1 or value > initial_modes for value in modes):
        raise ValueError("final_modes must lie in [1, initial_n_modes]")
    requested_times = sorted(set(float(value) for value in handoff_times))
    if not requested_times:
        raise ValueError("handoff_times must be nonempty")
    if requested_times[0] < config.times[0] - 1.0e-12:
        raise ValueError("handoff time precedes the simulated interval")
    if requested_times[-1] > config.times[-1] + 1.0e-12:
        raise ValueError("handoff time exceeds the simulated interval")

    operator = neumann_laplacian_1d(
        config.n_nodes,
        config.length,
        config.diffusion_coefficient,
    )
    basis = eigendecompose(operator)
    vectors = basis.eigenvectors
    mean = discrete_expected_diffusion(config)
    variance = multinomial_marginal_variance(config)
    covariance = multinomial_marginal_covariance(config)
    modal_mean = mean @ vectors
    modal_covariance = np.einsum(
        "ia,tij,jb->tab",
        vectors,
        covariance,
        vectors,
        optimize=True,
    )

    if covariance_time_indices is None:
        covariance_indices = np.arange(config.n_steps, dtype=int)
    else:
        covariance_indices = np.asarray(covariance_time_indices, dtype=int)
        if covariance_indices.ndim != 1 or covariance_indices.size == 0:
            raise ValueError("covariance_time_indices must be nonempty")
        if np.min(covariance_indices) < 0 or np.max(covariance_indices) >= config.n_steps:
            raise ValueError("covariance time index outside the simulated interval")

    mean_energy = np.sum(modal_mean**2, axis=1)
    covariance_energy = np.sum(modal_covariance**2, axis=(1, 2))
    transition_count = max(config.n_steps - 1, 1)
    predictions: list[HandoffPrediction] = []

    for mode_count in modes:
        retained_vectors = vectors[:, :mode_count]
        retained_covariance = modal_covariance[:, :mode_count, :mode_count]
        omitted_mean_energy = np.sum(modal_mean[:, mode_count:] ** 2, axis=1)
        omitted_covariance_energy = np.maximum(
            covariance_energy - np.sum(retained_covariance**2, axis=(1, 2)),
            0.0,
        )
        projected_variance = np.einsum(
            "ia,tab,ib->ti",
            retained_vectors,
            retained_covariance,
            retained_vectors,
            optimize=True,
        )

        for requested_time in requested_times:
            handoff_step = int(np.argmin(np.abs(config.times - requested_time)))
            handoff_time = float(config.times[handoff_step])
            post = np.arange(config.n_steps) >= handoff_step
            covariance_post = covariance_indices >= handoff_step

            mean_numerator = float(np.sum(omitted_mean_energy[post]))
            mean_denominator = float(np.sum(mean_energy))
            post_mean_denominator = float(np.sum(mean_energy[post]))
            predicted_mean = np.sqrt(mean_numerator / mean_denominator)
            predicted_post_mean = np.sqrt(mean_numerator / post_mean_denominator)

            hybrid_variance = variance.copy()
            hybrid_variance[post] = projected_variance[post]
            variance_denominator = float(np.sum(variance**2))
            predicted_variance = float(
                np.linalg.norm(hybrid_variance - variance) / np.sqrt(variance_denominator)
            )

            selected_omitted = omitted_covariance_energy[covariance_indices].copy()
            selected_omitted[~covariance_post] = 0.0
            covariance_denominator = float(
                np.sum(covariance_energy[covariance_indices])
            )
            predicted_covariance = np.sqrt(
                float(np.sum(selected_omitted)) / covariance_denominator
            )

            mean_denominator = float(mean_energy[handoff_step])
            covariance_denominator_at_handoff = float(
                covariance_energy[handoff_step]
            )
            instant_mean = (
                0.0
                if mean_denominator == 0.0
                else np.sqrt(
                    float(omitted_mean_energy[handoff_step]) / mean_denominator
                )
            )
            instant_covariance = (
                0.0
                if covariance_denominator_at_handoff == 0.0
                else np.sqrt(
                    float(omitted_covariance_energy[handoff_step])
                    / covariance_denominator_at_handoff
                )
            )
            modal_work = (
                handoff_step * initial_modes
                + (transition_count - handoff_step) * mode_count
            )
            full_work = transition_count * config.n_nodes
            predictions.append(
                HandoffPrediction(
                    final_n_modes=mode_count,
                    requested_handoff_time=requested_time,
                    handoff_step=handoff_step,
                    handoff_time=handoff_time,
                    predicted_mean_relative_l2_error=float(predicted_mean),
                    predicted_post_handoff_mean_relative_l2_error=float(
                        predicted_post_mean
                    ),
                    predicted_variance_relative_l2_error=predicted_variance,
                    predicted_covariance_relative_frobenius_error=float(
                        predicted_covariance
                    ),
                    instantaneous_mean_projection_relative_l2=float(instant_mean),
                    instantaneous_covariance_projection_relative_frobenius=float(
                        instant_covariance
                    ),
                    modal_update_fraction_of_full=float(modal_work / full_work),
                )
            )
    return predictions


def pareto_handoff_indices(predictions: list[HandoffPrediction]) -> set[int]:
    """Return nondominated indices for work, mean, variance, and covariance."""

    values = np.asarray(
        [
            (
                row.modal_update_fraction_of_full,
                row.predicted_mean_relative_l2_error,
                row.predicted_variance_relative_l2_error,
                row.predicted_covariance_relative_frobenius_error,
            )
            for row in predictions
        ],
        dtype=float,
    )
    frontier: set[int] = set()
    for index, candidate in enumerate(values):
        dominated = np.any(
            np.all(values <= candidate, axis=1)
            & np.any(values < candidate, axis=1)
        )
        if not dominated:
            frontier.add(index)
    return frontier
