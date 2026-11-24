"""Benchmark clip-and-renormalize residual feedback against exact readout."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np

from .baselines import vectorized_multinomial_random_walk_diffusion
from .benchmarking import stable_diffusion_config
from .config import DiffusionConfig
from .distribution_metrics import summarize_distribution_ensemble
from .innovation_completion import InnovationConditionedModalCompleter
from .metrics import (
    empirical_covariance,
    paired_bootstrap_error_difference_intervals,
    relative_frobenius_error,
    relative_l2_error,
    summarize_ensemble,
)
from .readouts import (
    clip_renormalize_bank_counts,
    clip_renormalize_bank_counts_with_diagnostics,
    clip_renormalize_bank_readout,
    warm_simplex_bank_counts,
    warm_simplex_bank_readout,
)
from .references import (
    discrete_expected_diffusion,
    multinomial_marginal_variance,
    multinomial_node_probabilities,
)
from .temporal import (
    empirical_mean_cross_time_covariance,
    multinomial_mean_cross_time_covariance,
)

WARM_METHOD = "warm_simplex_bank"
CLIP_METHOD = "clip_renormalize_bank"
EXACT_METHOD = "exact_vectorized_multinomial"


@dataclass(frozen=True, slots=True)
class ClipRenormalizeBenchmarkCase:
    """One completed-Gaussian workload for readout comparison."""

    n_nodes: int
    retained_modes: int = 75
    completion_rank: int = 25


@dataclass(frozen=True, slots=True)
class ClipRenormalizeBenchmarkSpec:
    """Protocol for timing, residual, and ensemble-accuracy experiments."""

    cases: tuple[ClipRenormalizeBenchmarkCase, ...] = (
        ClipRenormalizeBenchmarkCase(101),
        ClipRenormalizeBenchmarkCase(201),
    )
    n_particles: int = 5_275
    total_time: float = 100.0
    requested_dt: float = 1.0
    length: float = 4.0
    diffusion_coefficient: float = 2.20e-4
    impulse_fraction: float = 0.59
    stability_safety: float = 0.95
    cached_trajectories: int = 32
    readout_warmups: int = 2
    readout_repeats: int = 30
    end_to_end_warmups: int = 1
    end_to_end_repeats: int = 24
    bootstrap_resamples: int = 10_000
    accuracy_runs: int = 500
    accuracy_bootstrap_resamples: int = 200
    accuracy_bootstrap_blocks: int = 20
    accuracy_confidence_level: float = 0.95
    covariance_times: tuple[float, ...] = (1.0, 5.0, 20.0, 50.0, 100.0)
    distribution_times: tuple[float, ...] = (20.0, 50.0, 100.0)
    temporal_lags: tuple[float, ...] = (1.0, 5.0, 20.0)
    temporal_origins: int = 5
    distribution_projections: int = 32
    histogram_bins: int = 30
    seed: int = 20_261_123

    def __post_init__(self) -> None:
        if not self.cases:
            raise ValueError("cases must not be empty")
        nodes = [case.n_nodes for case in self.cases]
        if len(set(nodes)) != len(nodes):
            raise ValueError("cases must have unique node counts")
        for case in self.cases:
            if case.n_nodes < 2:
                raise ValueError("case node counts must be at least two")
            if not 1 <= case.retained_modes <= case.n_nodes:
                raise ValueError("retained_modes must lie in [1, n_nodes]")
            if case.completion_rank <= 0:
                raise ValueError("completion_rank must be positive")
        if self.readout_warmups < 0 or self.end_to_end_warmups < 0:
            raise ValueError("warmup counts must be nonnegative")
        if self.readout_repeats <= 0 or self.end_to_end_repeats <= 0:
            raise ValueError("timing repeat counts must be positive")
        if self.readout_repeats % 2 or self.end_to_end_repeats % 6:
            raise ValueError("timing repeats must balance the method orders")
        if self.cached_trajectories < self.readout_warmups + self.readout_repeats:
            raise ValueError("cached_trajectories must cover readout timing")
        if self.bootstrap_resamples <= 0 or self.accuracy_runs < 2:
            raise ValueError("bootstrap_resamples and accuracy_runs are too small")
        if self.accuracy_bootstrap_resamples <= 0:
            raise ValueError("accuracy_bootstrap_resamples must be positive")
        if self.accuracy_bootstrap_blocks <= 1:
            raise ValueError("accuracy_bootstrap_blocks must exceed one")
        if not 0.0 < self.accuracy_confidence_level < 1.0:
            raise ValueError("accuracy_confidence_level must lie in (0, 1)")
        if self.temporal_origins <= 0 or self.distribution_projections <= 0:
            raise ValueError("diagnostic sample counts must be positive")
        if self.histogram_bins <= 1:
            raise ValueError("histogram_bins must exceed one")


def _configuration(
    spec: ClipRenormalizeBenchmarkSpec,
    case: ClipRenormalizeBenchmarkCase,
) -> DiffusionConfig:
    return stable_diffusion_config(
        n_particles=spec.n_particles,
        n_nodes=case.n_nodes,
        total_time=spec.total_time,
        requested_dt=spec.requested_dt,
        length=spec.length,
        diffusion_coefficient=spec.diffusion_coefficient,
        impulse_fraction=spec.impulse_fraction,
        stability_safety=spec.stability_safety,
    )


def _columns(
    config: DiffusionConfig,
    case: ClipRenormalizeBenchmarkCase,
) -> dict[str, int | float]:
    return {
        "n_particles": config.n_particles,
        "n_nodes": config.n_nodes,
        "n_steps": config.n_steps,
        "dt": config.dt,
        "total_time": float(config.times[-1]),
        "retained_modes": case.retained_modes,
        "completion_rank": min(
            case.completion_rank,
            config.n_nodes - case.retained_modes,
        ),
    }


def _validate_output(values: np.ndarray, config: DiffusionConfig) -> None:
    if values.shape != (config.n_steps, config.n_nodes):
        raise RuntimeError("readout returned the wrong output shape")
    if not np.all(np.isfinite(values)):
        raise RuntimeError("readout returned nonfinite output")
    if float(np.min(values)) < -1.0e-12:
        raise RuntimeError("readout returned negative output")
    if not np.allclose(
        values.sum(axis=1),
        config.n_particles,
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise RuntimeError("readout failed mass conservation")


def _paired_ratio_interval(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, float]:
    if baseline.shape != candidate.shape or baseline.ndim != 1:
        raise ValueError("paired samples must be equal-length vectors")
    observed = float(np.median(candidate) / np.median(baseline))
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        baseline.size,
        size=(n_resamples, baseline.size),
    )
    ratios = np.median(candidate[indices], axis=1) / np.median(
        baseline[indices], axis=1
    )
    lower, upper = np.quantile(ratios, [0.025, 0.975])
    return {
        "median_ratio": observed,
        "ratio_ci_lower": float(lower),
        "ratio_ci_upper": float(upper),
    }


def _nearest_indices(
    config: DiffusionConfig,
    requested_times: Sequence[float],
) -> list[int]:
    return sorted(
        {
            int(np.argmin(np.abs(config.times - float(time))))
            for time in requested_times
        }
    )


def _exact_covariances(
    probabilities: np.ndarray,
    indices: Sequence[int],
    total_mass: int,
) -> np.ndarray:
    result = []
    for index in indices:
        probability = probabilities[index]
        covariance = -total_mass * np.outer(probability, probability)
        diagonal = np.arange(probability.size)
        covariance[diagonal, diagonal] += total_mass * probability
        result.append(covariance)
    return np.stack(result)


def _lag_origins(
    config: DiffusionConfig,
    lag_steps: int,
    count: int,
) -> np.ndarray:
    last = config.n_steps - lag_steps - 1
    if last < 0:
        raise ValueError("temporal lag exceeds the available trajectory")
    return np.unique(
        np.rint(np.linspace(0, last, min(count, last + 1))).astype(np.int64)
    )


def _fixed_bin_marginal_js(
    reference: np.ndarray,
    estimates: dict[str, np.ndarray],
    *,
    node_indices: Sequence[int],
    bins: int,
) -> dict[str, float]:
    """Return marginal JS scores using one frozen edge set per marginal."""

    totals = {method: 0.0 for method in estimates}
    marginal_count = reference.shape[1] * len(node_indices)
    for time_index in range(reference.shape[1]):
        for node_index in node_indices:
            reference_samples = reference[:, time_index, node_index]
            estimate_samples = {
                method: values[:, time_index, node_index]
                for method, values in estimates.items()
            }
            lower = min(
                float(np.min(reference_samples)),
                *(float(np.min(values)) for values in estimate_samples.values()),
            )
            upper = max(
                float(np.max(reference_samples)),
                *(float(np.max(values)) for values in estimate_samples.values()),
            )
            if lower == upper:
                continue
            edges = np.linspace(lower, upper, bins + 1)
            reference_counts = np.histogram(reference_samples, bins=edges)[0]
            reference_probability = reference_counts / reference_counts.sum()
            for method, values in estimate_samples.items():
                estimate_counts = np.histogram(values, bins=edges)[0]
                estimate_probability = estimate_counts / estimate_counts.sum()
                midpoint = 0.5 * (
                    reference_probability + estimate_probability
                )
                divergence = 0.0
                for probability in (
                    reference_probability,
                    estimate_probability,
                ):
                    mask = probability > 0.0
                    divergence += float(
                        np.sum(
                            probability[mask]
                            * np.log(probability[mask] / midpoint[mask])
                        )
                    )
                totals[method] += 0.5 * divergence
    return {
        method: total / marginal_count for method, total in totals.items()
    }


def _evaluate_ensemble(
    config: DiffusionConfig,
    runs: np.ndarray,
    *,
    method: str,
    exact_mean: np.ndarray,
    exact_variance: np.ndarray,
    exact_probabilities: np.ndarray,
    covariance_indices: list[int],
    temporal_lags: Sequence[float],
    temporal_origins: int,
) -> dict[str, int | float | str | None]:
    moments = summarize_ensemble(
        exact_mean,
        exact_variance,
        runs,
        config.n_particles,
    )
    exact_covariance = _exact_covariances(
        exact_probabilities,
        covariance_indices,
        config.n_particles,
    )
    observed_covariance = empirical_covariance(runs, covariance_indices)
    row: dict[str, int | float | str | None] = {
        "method": method,
        "runs": runs.shape[0],
        "mean_relative_l2_error": moments.mean_relative_l2_error,
        "mean_max_time_relative_l2_error": moments.mean_max_time_relative_l2_error,
        "mean_final_time_relative_l2_error": moments.mean_final_time_relative_l2_error,
        "mean_error_to_reference_se_ratio": (
            moments.mean_error_to_reference_se_ratio
        ),
        "variance_relative_l2_error": moments.variance_relative_l2_error,
        "variance_max_time_relative_l2_error": (
            moments.variance_max_time_relative_l2_error
        ),
        "variance_final_time_relative_l2_error": (
            moments.variance_final_time_relative_l2_error
        ),
        "covariance_relative_frobenius_error": relative_frobenius_error(
            exact_covariance,
            observed_covariance,
        ),
        "maximum_mass_error": moments.maximum_mass_error,
        "minimum_output": moments.minimum_reconstructed_count,
        "negative_entry_fraction": moments.negative_entry_fraction,
    }
    for duration in temporal_lags:
        lag_steps = min(
            max(1, int(round(float(duration) / config.dt))),
            config.n_steps - 1,
        )
        origins = _lag_origins(config, lag_steps, temporal_origins)
        exact_cross = multinomial_mean_cross_time_covariance(
            config,
            lag_steps,
            start_indices=origins,
        )
        observed_cross = empirical_mean_cross_time_covariance(
            runs,
            lag_steps,
            start_indices=origins,
        )
        label = str(duration).replace(".", "p")
        row[f"lag_{label}_steps"] = lag_steps
        row[f"lag_{label}_covariance_relative_frobenius_error"] = (
            relative_frobenius_error(exact_cross, observed_cross)
        )
    return row


def _paired_output_diagnostics(
    warm_runs: np.ndarray,
    clip_runs: np.ndarray,
    total_mass: int,
) -> dict[str, float]:
    squared_difference = 0.0
    squared_reference = 0.0
    row_l1: list[float] = []
    for warm, clip in zip(warm_runs, clip_runs, strict=True):
        difference = clip - warm
        squared_difference += float(np.sum(difference**2))
        squared_reference += float(np.sum(warm**2))
        row_l1.extend(
            float(value)
            for value in np.sum(np.abs(difference), axis=1) / total_mass
        )
    l1 = np.asarray(row_l1)
    return {
        "clip_warm_trajectory_relative_l2": float(
            np.sqrt(squared_difference / squared_reference)
        ),
        "clip_warm_row_l1_fraction_mean": float(np.mean(l1)),
        "clip_warm_row_l1_fraction_p95": float(np.percentile(l1, 95.0)),
        "clip_warm_row_l1_fraction_maximum": float(np.max(l1)),
        "clip_warm_ensemble_mean_relative_l2": relative_l2_error(
            np.mean(warm_runs, axis=0),
            np.mean(clip_runs, axis=0),
        ),
    }


def _accuracy_study(
    spec: ClipRenormalizeBenchmarkSpec,
    case: ClipRenormalizeBenchmarkCase,
    case_index: int,
    config: DiffusionConfig,
    completer: InnovationConditionedModalCompleter,
) -> tuple[
    list[dict[str, int | float | str | None]],
    dict[str, int | float | bool],
    int,
]:
    shape = (spec.accuracy_runs, config.n_steps, config.n_nodes)
    warm_runs = np.empty(shape, dtype=float)
    clip_runs = np.empty(shape, dtype=float)
    fallback_trajectories = 0
    sequences = np.random.SeedSequence(
        [spec.seed, case_index, 3]
    ).spawn(spec.accuracy_runs)
    for run_index, sequence in enumerate(sequences):
        raw = completer.run_stepwise(np.random.default_rng(sequence)).spatial_counts
        warm_runs[run_index] = warm_simplex_bank_counts(raw, config.n_particles)
        fast_clip_counts = clip_renormalize_bank_counts(
            raw,
            config.n_particles,
        )
        diagnostic_clip_counts, clip_diagnostics = (
            clip_renormalize_bank_counts_with_diagnostics(
                raw,
                config.n_particles,
            )
        )
        if not np.array_equal(fast_clip_counts, diagnostic_clip_counts):
            raise RuntimeError("fast and instrumented clip paths differ")
        clip_runs[run_index] = fast_clip_counts
        fallback_trajectories += int(clip_diagnostics.used_fallback)

    covariance_indices = _nearest_indices(config, spec.covariance_times)
    distribution_indices = _nearest_indices(config, spec.distribution_times)
    exact_mean = discrete_expected_diffusion(config)
    exact_variance = multinomial_marginal_variance(config)
    exact_probabilities = multinomial_node_probabilities(config)
    moment_intervals = paired_bootstrap_error_difference_intervals(
        exact_mean,
        exact_variance,
        warm_runs,
        clip_runs,
        n_resamples=spec.accuracy_bootstrap_resamples,
        confidence_level=spec.accuracy_confidence_level,
        n_blocks=spec.accuracy_bootstrap_blocks,
        seed=spec.seed + case_index * 100 + 20,
    )
    covariance_intervals = paired_bootstrap_error_difference_intervals(
        exact_mean[covariance_indices],
        exact_variance[covariance_indices],
        warm_runs[:, covariance_indices],
        clip_runs[:, covariance_indices],
        reference_covariance=_exact_covariances(
            exact_probabilities,
            covariance_indices,
            config.n_particles,
        ),
        covariance_time_indices=list(range(len(covariance_indices))),
        n_resamples=spec.accuracy_bootstrap_resamples,
        confidence_level=spec.accuracy_confidence_level,
        n_blocks=spec.accuracy_bootstrap_blocks,
        seed=spec.seed + case_index * 100 + 21,
    )
    moment_intervals["covariance_relative_frobenius_error"] = (
        covariance_intervals["covariance_relative_frobenius_error"]
    )
    accuracy_rows = [
        _evaluate_ensemble(
            config,
            warm_runs,
            method=WARM_METHOD,
            exact_mean=exact_mean,
            exact_variance=exact_variance,
            exact_probabilities=exact_probabilities,
            covariance_indices=covariance_indices,
            temporal_lags=spec.temporal_lags,
            temporal_origins=spec.temporal_origins,
        ),
        _evaluate_ensemble(
            config,
            clip_runs,
            method=CLIP_METHOD,
            exact_mean=exact_mean,
            exact_variance=exact_variance,
            exact_probabilities=exact_probabilities,
            covariance_indices=covariance_indices,
            temporal_lags=spec.temporal_lags,
            temporal_origins=spec.temporal_origins,
        ),
    ]
    paired = _paired_output_diagnostics(
        warm_runs,
        clip_runs,
        config.n_particles,
    )
    warm_selected = warm_runs[:, distribution_indices, :].copy()
    clip_selected = clip_runs[:, distribution_indices, :].copy()
    ensemble_bytes = int(warm_runs.nbytes + clip_runs.nbytes)
    del warm_runs, clip_runs
    gc.collect()

    exact_runs = np.empty(shape, dtype=float)
    exact_sequences = np.random.SeedSequence(
        [spec.seed, case_index, 4]
    ).spawn(spec.accuracy_runs)
    for run_index, sequence in enumerate(exact_sequences):
        exact_runs[run_index] = vectorized_multinomial_random_walk_diffusion(
            config,
            np.random.default_rng(sequence),
        )
    accuracy_rows.append(
        _evaluate_ensemble(
            config,
            exact_runs,
            method="exact_sampling_floor",
            exact_mean=exact_mean,
            exact_variance=exact_variance,
            exact_probabilities=exact_probabilities,
            covariance_indices=covariance_indices,
            temporal_lags=spec.temporal_lags,
            temporal_origins=spec.temporal_origins,
        )
    )
    exact_selected = exact_runs[:, distribution_indices, :].copy()
    del exact_runs
    gc.collect()

    second_exact_selected = np.empty_like(exact_selected)
    second_sequences = np.random.SeedSequence(
        [spec.seed, case_index, 5]
    ).spawn(spec.accuracy_runs)
    for run_index, sequence in enumerate(second_sequences):
        output = vectorized_multinomial_random_walk_diffusion(
            config,
            np.random.default_rng(sequence),
        )
        second_exact_selected[run_index] = output[distribution_indices]

    node_indices = sorted(
        {
            int(round(fraction * (config.n_nodes - 1)))
            for fraction in (0.25, 0.5, spec.impulse_fraction, 0.75)
        }
    )
    distribution_inputs = {
        WARM_METHOD: warm_selected,
        CLIP_METHOD: clip_selected,
        "exact_sampling_floor": second_exact_selected,
    }
    fixed_js = _fixed_bin_marginal_js(
        exact_selected,
        distribution_inputs,
        node_indices=node_indices,
        bins=spec.histogram_bins,
    )
    rows_by_method = {str(row["method"]): row for row in accuracy_rows}
    for method, values in distribution_inputs.items():
        diagnostics = summarize_distribution_ensemble(
            exact_selected,
            values,
            time_indices=list(range(len(distribution_indices))),
            node_indices=node_indices,
            n_projections=spec.distribution_projections,
            histogram_bins=spec.histogram_bins,
            seed=spec.seed + case_index * 100,
        )
        rows_by_method[method].update(diagnostics.as_dict())
        rows_by_method[method]["marginal_jensen_shannon"] = fixed_js[method]

    warm = rows_by_method[WARM_METHOD]
    clip = rows_by_method[CLIP_METHOD]
    floor = rows_by_method["exact_sampling_floor"]
    moment_metrics = (
        "mean_relative_l2_error",
        "variance_relative_l2_error",
        "covariance_relative_frobenius_error",
    )
    distribution_metrics = (
        "marginal_normalized_wasserstein",
        "marginal_jensen_shannon",
        "sliced_normalized_wasserstein",
    )
    comparison: dict[str, int | float | bool] = {
        "n_nodes": config.n_nodes,
        "accuracy_fallback_trajectories": fallback_trajectories,
        "accuracy_fallback_trajectory_rate": (
            fallback_trajectories / spec.accuracy_runs
        ),
        **paired,
    }
    moment_passes = []
    distribution_passes = []
    for metric in (*moment_metrics, *distribution_metrics):
        warm_value = float(warm[metric])
        clip_value = float(clip[metric])
        floor_value = float(floor[metric])
        allowance = (
            0.25 * floor_value if metric in moment_metrics else floor_value
        )
        comparison[f"{metric}_clip_over_warm"] = clip_value / max(
            warm_value,
            np.finfo(float).tiny,
        )
        comparison[f"{metric}_clip_minus_warm"] = clip_value - warm_value
        comparison[f"{metric}_allowance"] = allowance
        if metric in moment_metrics:
            lower, upper = moment_intervals[metric]
            comparison[f"{metric}_difference_ci_lower"] = lower
            comparison[f"{metric}_difference_ci_upper"] = upper
            passed = upper <= allowance
            moment_passes.append(passed)
        else:
            passed = clip_value - warm_value <= allowance
            distribution_passes.append(passed)
    temporal_passes = []
    for duration in spec.temporal_lags:
        label = str(duration).replace(".", "p")
        metric = f"lag_{label}_covariance_relative_frobenius_error"
        warm_value = float(warm[metric])
        clip_value = float(clip[metric])
        allowance = 0.25 * float(floor[metric])
        comparison[f"{metric}_clip_over_warm"] = clip_value / max(
            warm_value,
            np.finfo(float).tiny,
        )
        comparison[f"{metric}_clip_minus_warm"] = clip_value - warm_value
        comparison[f"{metric}_allowance"] = allowance
        temporal_passes.append(clip_value - warm_value <= allowance)
    comparison["moment_noninferiority_pass"] = bool(all(moment_passes))
    comparison["temporal_noninferiority_pass"] = bool(all(temporal_passes))
    comparison["distribution_noninferiority_pass"] = bool(
        all(distribution_passes)
    )
    comparison["accuracy_noninferiority_pass"] = bool(
        all(moment_passes)
        and all(temporal_passes)
        and all(distribution_passes)
        and fallback_trajectories / spec.accuracy_runs <= 0.001
    )
    return accuracy_rows, comparison, ensemble_bytes


def _cached_diagnostics(
    raw_fields: np.ndarray,
    config: DiffusionConfig,
) -> dict[str, int | float | bool | dict[str, int]]:
    row_l1: list[float] = []
    warm_residual_l1: list[float] = []
    clip_residual_l1: list[float] = []
    warm_final: list[float] = []
    clip_final: list[float] = []
    maximum_ratios: list[float] = []
    scales: list[float] = []
    negative_mass_fractions: list[float] = []
    failure_reasons: dict[str, int] = {}
    fallback_trajectories = 0
    support_disagreement_entries = 0
    total_entries = 0
    squared_difference = 0.0
    squared_reference = 0.0
    maximum_mass_error = 0.0
    minimum_output = float("inf")
    maximum_residual_coordinate_fraction = 0.0
    maximum_clip_ledger_error = 0.0
    maximum_warm_ledger_error = 0.0
    maximum_joint_ledger_error = 0.0
    maximum_raw_mass_drift = 0.0

    for raw in raw_fields:
        warm = warm_simplex_bank_readout(raw, config.n_particles)
        clip = clip_renormalize_bank_readout(raw, config.n_particles)
        diagnostic_counts, diagnostics = (
            clip_renormalize_bank_counts_with_diagnostics(
                raw,
                config.n_particles,
            )
        )
        if not np.array_equal(diagnostic_counts, clip.counts):
            raise RuntimeError("instrumented and detailed clip paths differ")
        fast_counts = clip_renormalize_bank_counts(raw, config.n_particles)
        if not np.array_equal(fast_counts, diagnostic_counts):
            raise RuntimeError("fast and instrumented clip paths differ")
        difference = clip.counts - warm.counts
        squared_difference += float(np.sum(difference**2))
        squared_reference += float(np.sum(warm.counts**2))
        row_l1.extend(
            float(value)
            for value in np.sum(np.abs(difference), axis=1) / config.n_particles
        )
        support_disagreement_entries += int(
            np.count_nonzero((clip.counts > 0.0) != (warm.counts > 0.0))
        )
        total_entries += clip.counts.size

        warm_l1 = np.sum(np.abs(warm.residuals), axis=1) / config.n_particles
        clip_l1 = np.sum(np.abs(clip.residuals), axis=1) / config.n_particles
        warm_residual_l1.extend(float(value) for value in warm_l1)
        clip_residual_l1.extend(float(value) for value in clip_l1)
        warm_final.append(float(warm_l1[-1]))
        clip_final.append(float(clip_l1[-1]))
        maximum_ratios.append(
            float(np.max(clip_l1) / max(np.max(warm_l1), np.finfo(float).tiny))
        )
        maximum_residual_coordinate_fraction = max(
            maximum_residual_coordinate_fraction,
            float(np.max(np.abs(clip.residuals)) / config.n_particles),
        )

        valid_scales = diagnostics.scale_factors[
            np.isfinite(diagnostics.scale_factors)
        ]
        scales.extend(float(value) for value in valid_scales)
        valid_negative_mass = diagnostics.negative_masses[
            np.isfinite(diagnostics.negative_masses)
        ]
        negative_mass_fractions.extend(
            float(value / config.n_particles) for value in valid_negative_mass
        )
        if diagnostics.used_fallback:
            fallback_trajectories += 1
            reason = diagnostics.failure_reason or "unknown"
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

        maximum_mass_error = max(
            maximum_mass_error,
            float(
                np.max(
                    np.abs(clip.counts.sum(axis=1) - config.n_particles)
                )
            ),
        )
        minimum_output = min(minimum_output, float(np.min(clip.counts)))
        maximum_raw_mass_drift = max(
            maximum_raw_mass_drift,
            float(np.max(np.abs(raw.sum(axis=1) - config.n_particles))),
        )
        clip_ledger = np.cumsum(raw - clip.counts, axis=0)
        warm_ledger = np.cumsum(raw - warm.counts, axis=0)
        joint_ledger = np.cumsum(clip.counts - warm.counts, axis=0)
        maximum_clip_ledger_error = max(
            maximum_clip_ledger_error,
            float(np.max(np.abs(clip_ledger - clip.residuals))),
        )
        maximum_warm_ledger_error = max(
            maximum_warm_ledger_error,
            float(np.max(np.abs(warm_ledger - warm.residuals))),
        )
        maximum_joint_ledger_error = max(
            maximum_joint_ledger_error,
            float(
                np.max(
                    np.abs(
                        joint_ledger - (warm.residuals - clip.residuals)
                    )
                )
            ),
        )

    warm_values = np.asarray(warm_residual_l1)
    clip_values = np.asarray(clip_residual_l1)
    l1_values = np.asarray(row_l1)
    scale_values = np.asarray(scales)
    negative_values = np.asarray(negative_mass_fractions)
    max_ratio_values = np.asarray(maximum_ratios)
    fallback_rate = fallback_trajectories / raw_fields.shape[0]
    correctness_pass = bool(
        minimum_output >= -1.0e-12
        and maximum_mass_error <= 1.0e-10
        and maximum_clip_ledger_error <= 1.0e-9
        and maximum_joint_ledger_error <= 1.0e-9
    )
    residual_stability_pass = bool(
        float(np.mean(clip_values)) <= 1.25 * float(np.mean(warm_values))
        and float(np.percentile(clip_values, 95.0))
        <= 1.25 * float(np.percentile(warm_values, 95.0))
        and float(np.mean(clip_final)) <= 1.25 * float(np.mean(warm_final))
        and float(np.percentile(max_ratio_values, 95.0)) <= 1.25
        and float(np.max(clip_values)) <= 0.10
        and fallback_rate <= 0.001
    )
    return {
        "cached_trajectories": raw_fields.shape[0],
        "clip_warm_relative_l2": float(
            np.sqrt(squared_difference / squared_reference)
        ),
        "clip_warm_row_l1_fraction_mean": float(np.mean(l1_values)),
        "clip_warm_row_l1_fraction_p95": float(np.percentile(l1_values, 95.0)),
        "clip_warm_row_l1_fraction_maximum": float(np.max(l1_values)),
        "support_disagreement_entry_fraction": (
            support_disagreement_entries / total_entries
        ),
        "warm_residual_l1_fraction_mean": float(np.mean(warm_values)),
        "warm_residual_l1_fraction_p95": float(
            np.percentile(warm_values, 95.0)
        ),
        "warm_residual_l1_fraction_maximum": float(np.max(warm_values)),
        "warm_final_residual_l1_fraction_mean": float(np.mean(warm_final)),
        "clip_residual_l1_fraction_mean": float(np.mean(clip_values)),
        "clip_residual_l1_fraction_p95": float(
            np.percentile(clip_values, 95.0)
        ),
        "clip_residual_l1_fraction_maximum": float(np.max(clip_values)),
        "clip_final_residual_l1_fraction_mean": float(np.mean(clip_final)),
        "clip_maximum_residual_coordinate_fraction": (
            maximum_residual_coordinate_fraction
        ),
        "clip_over_warm_per_trajectory_maximum_residual_p95": float(
            np.percentile(max_ratio_values, 95.0)
        ),
        "scale_minimum": float(np.min(scale_values)),
        "scale_p05": float(np.percentile(scale_values, 5.0)),
        "scale_mean": float(np.mean(scale_values)),
        "negative_mass_fraction_mean": float(np.mean(negative_values)),
        "negative_mass_fraction_p95": float(
            np.percentile(negative_values, 95.0)
        ),
        "negative_mass_fraction_maximum": float(np.max(negative_values)),
        "fallback_trajectories": fallback_trajectories,
        "fallback_trajectory_rate": fallback_rate,
        "fallback_reasons": failure_reasons,
        "maximum_mass_error": maximum_mass_error,
        "minimum_output": minimum_output,
        "maximum_raw_mass_drift": maximum_raw_mass_drift,
        "maximum_clip_ledger_error": maximum_clip_ledger_error,
        "maximum_warm_ledger_error": maximum_warm_ledger_error,
        "maximum_joint_ledger_error": maximum_joint_ledger_error,
        "correctness_pass": correctness_pass,
        "residual_stability_pass": residual_stability_pass,
    }


def _run_method(
    method: str,
    config: DiffusionConfig,
    completer: InnovationConditionedModalCompleter,
    rng: np.random.Generator,
) -> np.ndarray:
    if method == EXACT_METHOD:
        return vectorized_multinomial_random_walk_diffusion(config, rng)
    raw = completer.run_stepwise(rng).spatial_counts
    if method == WARM_METHOD:
        return warm_simplex_bank_counts(raw, config.n_particles)
    if method == CLIP_METHOD:
        return clip_renormalize_bank_counts(raw, config.n_particles)
    raise ValueError(f"unknown benchmark method: {method}")


def _benchmark_case(
    spec: ClipRenormalizeBenchmarkSpec,
    case: ClipRenormalizeBenchmarkCase,
    case_index: int,
) -> tuple[
    list[dict[str, object]],
    dict[str, object],
    list[dict[str, int | float | str | None]],
    dict[str, int | float | bool],
    dict[str, object],
]:
    config = _configuration(spec, case)
    columns = _columns(config, case)
    gc.collect()
    start = perf_counter()
    completer = InnovationConditionedModalCompleter(
        config,
        retained_modes=case.retained_modes,
        conditional_rank=case.completion_rank,
    )
    setup_seconds = perf_counter() - start
    sequences = np.random.SeedSequence(
        [spec.seed, case_index, 0]
    ).spawn(spec.cached_trajectories)
    raw_fields = np.stack(
        [
            completer.run_stepwise(np.random.default_rng(sequence)).spatial_counts
            for sequence in sequences
        ]
    )

    for index in range(spec.readout_warmups):
        raw = raw_fields[index]
        warm_simplex_bank_counts(raw, config.n_particles)
        clip_renormalize_bank_counts(raw, config.n_particles)

    timing_rows: list[dict[str, object]] = []
    was_enabled = gc.isenabled()
    if was_enabled:
        gc.disable()
    try:
        for repeat in range(spec.readout_repeats):
            raw = raw_fields[spec.readout_warmups + repeat]
            order = (
                (WARM_METHOD, CLIP_METHOD)
                if repeat % 2 == 0
                else (CLIP_METHOD, WARM_METHOD)
            )
            for position, method in enumerate(order):
                start = perf_counter()
                output = (
                    warm_simplex_bank_counts(raw, config.n_particles)
                    if method == WARM_METHOD
                    else clip_renormalize_bank_counts(raw, config.n_particles)
                )
                seconds = perf_counter() - start
                _validate_output(output, config)
                timing_rows.append(
                    {
                        **columns,
                        "component": "readout_only",
                        "method": method,
                        "repeat": repeat,
                        "execution_position": position,
                        "seconds": seconds,
                    }
                )
    finally:
        if was_enabled:
            gc.enable()

    for warmup in range(spec.end_to_end_warmups):
        for method_index, method in enumerate(
            (WARM_METHOD, CLIP_METHOD, EXACT_METHOD)
        ):
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [spec.seed, case_index, 1, warmup, method_index]
                )
            )
            _validate_output(_run_method(method, config, completer, rng), config)

    orders = (
        (WARM_METHOD, CLIP_METHOD, EXACT_METHOD),
        (EXACT_METHOD, CLIP_METHOD, WARM_METHOD),
        (CLIP_METHOD, EXACT_METHOD, WARM_METHOD),
        (WARM_METHOD, EXACT_METHOD, CLIP_METHOD),
        (EXACT_METHOD, WARM_METHOD, CLIP_METHOD),
        (CLIP_METHOD, WARM_METHOD, EXACT_METHOD),
    )
    was_enabled = gc.isenabled()
    if was_enabled:
        gc.disable()
    try:
        for repeat in range(spec.end_to_end_repeats):
            for position, method in enumerate(orders[repeat % len(orders)]):
                channel = 1 if method == EXACT_METHOD else 0
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [spec.seed, case_index, 2, repeat, channel]
                    )
                )
                start = perf_counter()
                output = _run_method(method, config, completer, rng)
                seconds = perf_counter() - start
                _validate_output(output, config)
                timing_rows.append(
                    {
                        **columns,
                        "component": "end_to_end",
                        "method": method,
                        "repeat": repeat,
                        "execution_position": position,
                        "seconds": seconds,
                    }
                )
    finally:
        if was_enabled:
            gc.enable()

    diagnostics = {**columns, **_cached_diagnostics(raw_fields, config)}
    accuracy_rows, accuracy_comparison, ensemble_bytes = _accuracy_study(
        spec,
        case,
        case_index,
        config,
        completer,
    )
    accuracy_rows = [{**columns, **row} for row in accuracy_rows]
    accuracy_comparison = {**columns, **accuracy_comparison}
    resources = {
        **columns,
        "setup_seconds": setup_seconds,
        "cached_trajectory_bytes": int(raw_fields.nbytes),
        "accuracy_pair_peak_array_bytes": ensemble_bytes,
    }
    return (
        timing_rows,
        diagnostics,
        accuracy_rows,
        accuracy_comparison,
        resources,
    )


def _timing_report(
    rows: Sequence[dict[str, object]],
    *,
    bootstrap_resamples: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    groups: dict[tuple[int, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (int(row["n_nodes"]), str(row["component"]), str(row["method"]))
        groups.setdefault(key, []).append(row)
    summaries: list[dict[str, object]] = []
    samples: dict[tuple[int, str, str], np.ndarray] = {}
    for key, selected in groups.items():
        selected = sorted(selected, key=lambda row: int(row["repeat"]))
        values = np.asarray([float(row["seconds"]) for row in selected])
        samples[key] = values
        n_nodes, component, method = key
        summaries.append(
            {
                "n_nodes": n_nodes,
                "component": component,
                "method": method,
                "samples": values.size,
                "median_seconds": float(np.median(values)),
                "mean_seconds": float(np.mean(values)),
                "std_seconds": float(np.std(values, ddof=0)),
                "min_seconds": float(np.min(values)),
                "max_seconds": float(np.max(values)),
            }
        )
    comparisons: list[dict[str, object]] = []
    specs = (
        ("readout_only", WARM_METHOD, CLIP_METHOD, "clip_over_warm"),
        ("end_to_end", WARM_METHOD, CLIP_METHOD, "clip_over_warm"),
        ("end_to_end", EXACT_METHOD, CLIP_METHOD, "clip_over_exact"),
        ("end_to_end", EXACT_METHOD, WARM_METHOD, "warm_over_exact"),
    )
    for n_nodes in sorted({int(row["n_nodes"]) for row in rows}):
        for offset, (component, baseline, candidate, label) in enumerate(specs):
            interval = _paired_ratio_interval(
                samples[(n_nodes, component, baseline)],
                samples[(n_nodes, component, candidate)],
                n_resamples=bootstrap_resamples,
                seed=seed + n_nodes * 100 + offset,
            )
            comparisons.append(
                {
                    "n_nodes": n_nodes,
                    "component": component,
                    "comparison": label,
                    "baseline_method": baseline,
                    "candidate_method": candidate,
                    **interval,
                }
            )
    return summaries, comparisons


def _write_csv(rows: Sequence[dict[str, object]], path: Path) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_clip_renormalize_benchmark(
    spec: ClipRenormalizeBenchmarkSpec,
    output_dir: str | Path,
) -> dict[str, object]:
    """Run the paired clip-versus-warm experiment and write artifacts."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    timing_rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    accuracy_rows: list[dict[str, object]] = []
    accuracy_comparisons: list[dict[str, object]] = []
    resources: list[dict[str, object]] = []
    for case_index, case in enumerate(spec.cases):
        result = _benchmark_case(spec, case, case_index)
        timing_rows.extend(result[0])
        diagnostics.append(result[1])
        accuracy_rows.extend(result[2])
        accuracy_comparisons.append(result[3])
        resources.append(result[4])

    summaries, comparisons = _timing_report(
        timing_rows,
        bootstrap_resamples=spec.bootstrap_resamples,
        seed=spec.seed,
    )
    comparison_lookup = {
        (int(row["n_nodes"]), str(row["component"]), str(row["comparison"])): row
        for row in comparisons
    }
    accuracy_lookup = {
        int(row["n_nodes"]): row for row in accuracy_comparisons
    }
    gates = []
    for diagnostic in diagnostics:
        n_nodes = int(diagnostic["n_nodes"])
        readout = comparison_lookup[(n_nodes, "readout_only", "clip_over_warm")]
        end_to_end = comparison_lookup[(n_nodes, "end_to_end", "clip_over_warm")]
        exact = comparison_lookup[(n_nodes, "end_to_end", "clip_over_exact")]
        gates.append(
            {
                "n_nodes": n_nodes,
                "correctness_pass": bool(diagnostic["correctness_pass"]),
                "residual_stability_pass": bool(
                    diagnostic["residual_stability_pass"]
                ),
                "accuracy_noninferiority_pass": bool(
                    accuracy_lookup[n_nodes]["accuracy_noninferiority_pass"]
                ),
                "moment_noninferiority_pass": bool(
                    accuracy_lookup[n_nodes]["moment_noninferiority_pass"]
                ),
                "temporal_noninferiority_pass": bool(
                    accuracy_lookup[n_nodes]["temporal_noninferiority_pass"]
                ),
                "distribution_noninferiority_pass": bool(
                    accuracy_lookup[n_nodes]["distribution_noninferiority_pass"]
                ),
                "readout_speed_pass": bool(
                    float(readout["median_ratio"]) <= 0.60
                    and float(readout["ratio_ci_upper"]) < 1.0
                ),
                "end_to_end_improvement_pass": bool(
                    float(end_to_end["median_ratio"]) <= 0.80
                    and float(end_to_end["ratio_ci_upper"]) < 1.0
                ),
                "exact_crossover_pass": bool(
                    float(exact["median_ratio"]) <= 0.95
                    and float(exact["ratio_ci_upper"]) < 1.0
                ),
            }
        )

    paths = {
        "timing_samples_csv": destination / "timing_samples.csv",
        "diagnostics_csv": destination / "diagnostics.csv",
        "accuracy_csv": destination / "accuracy.csv",
        "accuracy_comparisons_csv": destination / "accuracy_comparisons.csv",
        "summary_json": destination / "summary.json",
    }
    _write_csv(timing_rows, paths["timing_samples_csv"])
    _write_csv(diagnostics, paths["diagnostics_csv"])
    _write_csv(accuracy_rows, paths["accuracy_csv"])
    _write_csv(accuracy_comparisons, paths["accuracy_comparisons_csv"])
    report: dict[str, object] = {
        "schema_version": 1,
        "study_contract": {
            "workload": "full_field_every_timestep",
            "regime": "fixed_duration",
            "model_setup_in_end_to_end_timing": False,
            "model_setup_reused_across_trajectories": True,
            "exact_readout_control": WARM_METHOD,
            "candidate": CLIP_METHOD,
            "end_to_end_comparator": EXACT_METHOD,
            "paired_completed_fields": True,
            "position_balanced_timing": True,
            "diagnostics_outside_timing": True,
            "accuracy_reference": (
                "analytic multinomial moments and two exact ensembles"
            ),
        },
        "specification": asdict(spec),
        "runtime_environment": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "thread_limits": {
                name: os.environ.get(name)
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        },
        "timing_summaries": summaries,
        "paired_timing_comparisons": comparisons,
        "diagnostics": diagnostics,
        "accuracy": accuracy_rows,
        "accuracy_comparisons": accuracy_comparisons,
        "resources": resources,
        "gates": gates,
        "artifacts": {name: str(path) for name, path in paths.items()},
    }
    paths["summary_json"].write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark clip-and-renormalize residual feedback."
    )
    parser.add_argument("--node-counts", type=int, nargs="+", default=[101, 201])
    parser.add_argument("--modes", type=int, default=75)
    parser.add_argument("--completion-rank", type=int, default=25)
    parser.add_argument("--particles", type=int, default=5_275)
    parser.add_argument("--total-time", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--length", type=float, default=4.0)
    parser.add_argument("--diffusion", type=float, default=2.20e-4)
    parser.add_argument("--impulse-fraction", type=float, default=0.59)
    parser.add_argument("--stability-safety", type=float, default=0.95)
    parser.add_argument("--cached-trajectories", type=int, default=32)
    parser.add_argument("--readout-warmups", type=int, default=2)
    parser.add_argument("--readout-repeats", type=int, default=30)
    parser.add_argument("--end-to-end-warmups", type=int, default=1)
    parser.add_argument("--end-to-end-repeats", type=int, default=24)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--accuracy-runs", type=int, default=500)
    parser.add_argument(
        "--accuracy-bootstrap-resamples",
        type=int,
        default=200,
    )
    parser.add_argument("--accuracy-bootstrap-blocks", type=int, default=20)
    parser.add_argument(
        "--accuracy-confidence-level",
        type=float,
        default=0.95,
    )
    parser.add_argument("--distribution-projections", type=int, default=32)
    parser.add_argument("--histogram-bins", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20_261_123)
    parser.add_argument(
        "--output-dir",
        default="outputs/clip_renormalize_benchmark",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = ClipRenormalizeBenchmarkSpec(
        cases=tuple(
            ClipRenormalizeBenchmarkCase(
                n_nodes=n_nodes,
                retained_modes=args.modes,
                completion_rank=args.completion_rank,
            )
            for n_nodes in args.node_counts
        ),
        n_particles=args.particles,
        total_time=args.total_time,
        requested_dt=args.dt,
        length=args.length,
        diffusion_coefficient=args.diffusion,
        impulse_fraction=args.impulse_fraction,
        stability_safety=args.stability_safety,
        cached_trajectories=args.cached_trajectories,
        readout_warmups=args.readout_warmups,
        readout_repeats=args.readout_repeats,
        end_to_end_warmups=args.end_to_end_warmups,
        end_to_end_repeats=args.end_to_end_repeats,
        bootstrap_resamples=args.bootstrap_resamples,
        accuracy_runs=args.accuracy_runs,
        accuracy_bootstrap_resamples=args.accuracy_bootstrap_resamples,
        accuracy_bootstrap_blocks=args.accuracy_bootstrap_blocks,
        accuracy_confidence_level=args.accuracy_confidence_level,
        distribution_projections=args.distribution_projections,
        histogram_bins=args.histogram_bins,
        seed=args.seed,
    )
    report = run_clip_renormalize_benchmark(spec, args.output_dir)
    print(
        json.dumps(
            {"gates": report["gates"], "artifacts": report["artifacts"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
