"""Exact-anchor diagnostics for fitted regression and innovation error sources."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .anchored_closure import (
    AnchoredFittedClosureModel,
    _conditional_innovation_statistics,
    _low_rank_factor,
    _residual_sequences,
    configuration_descriptor,
    complete_anchored_ensemble,
)
from .ensemble import apply_readout_ensemble, run_modal_ensemble
from .fitted_closure import FittedClosureTrainingCase, FittedLocalClosureModel
from .metrics import relative_frobenius_error, summarize_ensemble
from .references import (
    discrete_expected_diffusion,
    multinomial_marginal_covariance,
    multinomial_marginal_variance,
)
from .temporal import (
    empirical_mean_cross_time_covariance,
    multinomial_mean_cross_time_covariance,
    valid_time_origins,
)


def _relative_frobenius(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(
        np.linalg.norm(candidate - reference)
        / max(np.linalg.norm(reference), np.finfo(float).tiny)
    )


def _relative_diagonal(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference_diagonal = np.diag(reference)
    candidate_diagonal = np.diag(candidate)
    return float(
        np.linalg.norm(candidate_diagonal - reference_diagonal)
        / max(np.linalg.norm(reference_diagonal), np.finfo(float).tiny)
    )


def _covariance(values: np.ndarray) -> np.ndarray:
    centered = values - np.mean(values, axis=0)
    return centered.T @ centered / max(values.shape[0] - 1, 1)


def run_anchor_innovation_ablation(
    cases: list[FittedClosureTrainingCase],
    output_path: str | Path,
    *,
    ranks: list[int],
    bandwidth: int,
    n_frequency_bins: int,
    n_time_bins: int,
    residual_grid_size: int,
    nearest_anchors: int,
    ridge: float,
    max_residual_samples: int,
    max_regression_samples_per_target: int,
    seed: int,
) -> Path:
    """Measure compression, persistence, and leave-one-anchor interpolation errors."""

    if not ranks or any(rank <= 0 for rank in ranks):
        raise ValueError("ranks must contain positive integers")
    models = [
        FittedLocalClosureModel.fit(
            [case],
            bandwidth=bandwidth,
            n_frequency_bins=n_frequency_bins,
            n_time_bins=n_time_bins,
            residual_grid_size=residual_grid_size,
            ridge=ridge,
            max_residual_samples=max_residual_samples,
            max_regression_samples_per_target=max_regression_samples_per_target,
            seed=seed + index,
        )
        for index, case in enumerate(cases)
    ]
    sequences = [
        _residual_sequences(case, model)
        for case, model in zip(cases, models, strict=True)
    ]

    descriptors = np.stack([configuration_descriptor(case.config) for case in cases])
    center = np.mean(descriptors, axis=0)
    scale = np.std(descriptors, axis=0, ddof=0)
    scale[scale < 1.0e-12] = 1.0
    normalized = (descriptors - center) / scale
    initial_coefficient_errors: list[float] = []
    transition_coefficient_errors: list[float] = []
    for anchor_index, model in enumerate(models):
        distances = np.linalg.norm(normalized - normalized[anchor_index], axis=1)
        distances[anchor_index] = np.inf
        selected = np.argsort(distances)[: min(nearest_anchors, len(models) - 1)]
        weights = 1.0 / np.maximum(distances[selected], 5.0e-2) ** 2
        weights /= np.sum(weights)
        initial_candidate = np.tensordot(
            weights,
            np.stack([models[index].initial_coefficients for index in selected]),
            axes=(0, 0),
        )
        transition_candidate = np.tensordot(
            weights,
            np.stack([models[index].transition_coefficients for index in selected]),
            axes=(0, 0),
        )
        initial_coefficient_errors.append(
            _relative_frobenius(model.initial_coefficients, initial_candidate)
        )
        transition_coefficient_errors.append(
            _relative_frobenius(model.transition_coefficients, transition_candidate)
        )

    rows: list[dict[str, object]] = []
    for rank in sorted(set(ranks)):
        independent_variance: list[float] = []
        independent_covariance: list[float] = []
        independent_lag: list[float] = []
        persistent_variance: list[float] = []
        persistent_covariance: list[float] = []
        persistent_lag: list[float] = []
        for case_index, (initial, transitions, time_bins) in enumerate(sequences):
            previous = np.concatenate(
                (initial[:, None, :], transitions[:, :-1, :]), axis=1
            )
            persistent = _conditional_innovation_statistics(
                initial,
                transitions,
                time_bins,
                n_time_bins=n_time_bins,
                rank=rank,
                ridge=ridge,
                max_samples=max_residual_samples,
                seed=seed + 20_000 + 100 * rank + case_index,
            )
            _, previous_means, gains, factors, diagonals = persistent
            for time_bin in range(n_time_bins):
                selected_times = np.flatnonzero(time_bins == time_bin)
                current_values = transitions[:, selected_times].reshape(
                    -1, residual_grid_size
                )
                previous_values = previous[:, selected_times].reshape(
                    -1, residual_grid_size
                )
                current_covariance = _covariance(current_values)
                previous_covariance = _covariance(previous_values)
                previous_centered = previous_values - previous_means[time_bin]
                current_centered = current_values - np.mean(current_values, axis=0)
                lag_covariance = (
                    previous_centered.T @ current_centered
                    / max(current_values.shape[0] - 1, 1)
                )

                _, factor, diagonal = _low_rank_factor(current_values, rank)
                independent_model = factor @ factor.T + np.diag(diagonal**2)
                independent_variance.append(
                    _relative_diagonal(current_covariance, independent_model)
                )
                independent_covariance.append(
                    _relative_frobenius(current_covariance, independent_model)
                )
                independent_lag.append(
                    _relative_frobenius(lag_covariance, np.zeros_like(lag_covariance))
                )

                conditional_covariance = (
                    factors[time_bin] @ factors[time_bin].T
                    + np.diag(diagonals[time_bin] ** 2)
                )
                persistent_model = (
                    gains[time_bin]
                    @ previous_covariance
                    @ gains[time_bin].T
                    + conditional_covariance
                )
                persistent_lag_model = previous_covariance @ gains[time_bin].T
                persistent_variance.append(
                    _relative_diagonal(current_covariance, persistent_model)
                )
                persistent_covariance.append(
                    _relative_frobenius(current_covariance, persistent_model)
                )
                persistent_lag.append(
                    _relative_frobenius(lag_covariance, persistent_lag_model)
                )

        common = {
            "rank": rank,
            "anchors": len(cases),
            "time_bins": n_time_bins,
            "leave_one_out_initial_coefficient_error": float(
                np.mean(initial_coefficient_errors)
            ),
            "leave_one_out_transition_coefficient_error": float(
                np.mean(transition_coefficient_errors)
            ),
        }
        rows.extend(
            (
                {
                    **common,
                    "dynamics": "independent",
                    "exact_anchor_variance_error": float(np.mean(independent_variance)),
                    "exact_anchor_covariance_error": float(
                        np.mean(independent_covariance)
                    ),
                    "exact_anchor_lag1_error": float(np.mean(independent_lag)),
                },
                {
                    **common,
                    "dynamics": "persistent_conditional",
                    "exact_anchor_variance_error": float(np.mean(persistent_variance)),
                    "exact_anchor_covariance_error": float(np.mean(persistent_covariance)),
                    "exact_anchor_lag1_error": float(np.mean(persistent_lag)),
                },
            )
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def run_exact_anchor_ensemble_validation(
    cases: list[FittedClosureTrainingCase],
    model: AnchoredFittedClosureModel,
    output_path: str | Path,
    *,
    n_runs: int,
    bandwidth: int,
    seed: int,
    anchor_indices: tuple[int, ...] | None = None,
) -> Path:
    """Evaluate fresh completed ensembles at exact fitted physical anchors."""

    if n_runs <= 0:
        raise ValueError("n_runs must be positive")
    if anchor_indices is None:
        anchor_indices = tuple(sorted({0, len(cases) // 2, len(cases) - 1}))
    rows: list[dict[str, object]] = []
    for position, anchor_index in enumerate(anchor_indices):
        case = cases[anchor_index]
        config = case.config
        retained = case.retained_modes
        completion_rank = min(
            config.n_nodes - retained,
            max(1, int(round(config.n_nodes * 5 / 101))),
        )
        raw = run_modal_ensemble(
            config,
            n_runs=n_runs,
            model="handoff_correlated_modal",
            n_modes=retained,
            initial_n_modes=config.n_nodes,
            handoff_time=case.handoff_time,
            seed=seed + 1_000 * position,
        )
        analytic = apply_readout_ensemble(
            raw,
            config,
            readout="local_persistent_unresolved_completion_simplex_bank",
            retained_modes=retained,
            completion_start_time=case.handoff_time,
            completion_rank=completion_rank,
            completion_bandwidth=bandwidth,
            seed=seed + 10_000 + position,
        )
        fitted = complete_anchored_ensemble(
            raw.runs,
            config,
            model,
            retained_modes=retained,
            completion_start_time=case.handoff_time,
            seed=seed + 20_000 + position,
            apply_simplex_bank=True,
        )
        time_indices = sorted(
            {
                min(config.n_steps - 1, max(1, int(round(fraction * (config.n_steps - 1)))))
                for fraction in (0.2, 0.5, 1.0)
            }
        )
        exact_mean = discrete_expected_diffusion(config)
        exact_variance = multinomial_marginal_variance(config)
        exact_covariance = multinomial_marginal_covariance(config)
        start = int(np.argmin(np.abs(config.times - case.handoff_time)))
        origins = valid_time_origins(config.n_steps, 1, start_step=start)
        exact_cross = multinomial_mean_cross_time_covariance(
            config, 1, start_indices=origins
        )
        for model_name, runs in (
            ("analytic_local", analytic.runs),
            ("anchored_persistent_fitted", fitted),
        ):
            moments = summarize_ensemble(
                exact_mean,
                exact_variance,
                runs,
                config.n_particles,
                reference_covariance=exact_covariance,
                covariance_time_indices=time_indices,
            )
            empirical_cross = empirical_mean_cross_time_covariance(
                runs, 1, start_indices=origins
            )
            rows.append(
                {
                    "anchor_index": anchor_index,
                    "model": model_name,
                    "nodes": config.n_nodes,
                    "particles": config.n_particles,
                    "impulse_index": config.impulse_index,
                    "diffusion": config.diffusion_coefficient,
                    "length": config.length,
                    "runs": n_runs,
                    "mean_relative_l2_error": moments.mean_relative_l2_error,
                    "variance_relative_l2_error": moments.variance_relative_l2_error,
                    "covariance_relative_frobenius_error": (
                        moments.covariance_relative_frobenius_error
                    ),
                    "lag1_covariance_relative_frobenius_error": (
                        relative_frobenius_error(exact_cross, empirical_cross)
                    ),
                }
            )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output
