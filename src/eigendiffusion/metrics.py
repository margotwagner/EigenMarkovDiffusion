"""Accuracy, variance, covariance, and physicality diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import NormalDist

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ValidationMetrics:
    """Backward-compatible metrics for comparing two mean trajectories."""

    rmse: float
    relative_l2_error: float
    maximum_mass_error: float
    minimum_reconstructed_count: float
    negative_value_fraction: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EnsembleDiagnostics:
    """Diagnostics for a stochastic ensemble against exact marginal targets."""

    mean_rmse: float
    mean_relative_l2_error: float
    mean_max_time_relative_l2_error: float
    mean_final_time_relative_l2_error: float
    mean_error_to_reference_se_ratio: float
    variance_rmse: float
    variance_relative_l2_error: float
    variance_max_time_relative_l2_error: float
    variance_final_time_relative_l2_error: float
    maximum_mass_error: float
    minimum_reconstructed_count: float
    negative_entry_fraction: float
    mean_negative_mass_fraction: float
    maximum_negative_mass_fraction: float
    covariance_relative_frobenius_error: float | None = None

    def as_dict(self) -> dict[str, float | None]:
        return asdict(self)


def root_mean_square_error(reference: FloatArray, estimate: FloatArray) -> float:
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    if reference.shape != estimate.shape:
        raise ValueError("reference and estimate must have the same shape")
    return float(np.sqrt(np.mean((reference - estimate) ** 2)))


def relative_l2_error(reference: FloatArray, estimate: FloatArray) -> float:
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    if reference.shape != estimate.shape:
        raise ValueError("reference and estimate must have the same shape")
    denominator = np.linalg.norm(reference.ravel())
    if denominator == 0:
        raise ValueError("reference norm is zero")
    return float(np.linalg.norm((reference - estimate).ravel()) / denominator)


def relative_l2_error_from_time(
    reference: FloatArray,
    estimate: FloatArray,
    times: FloatArray,
    start_time: float,
) -> float:
    """Compute relative L2 error using only samples with ``time >= start_time``."""

    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    times = np.asarray(times, dtype=float)
    if reference.shape != estimate.shape:
        raise ValueError("reference and estimate must have the same shape")
    if reference.shape[0] != times.size:
        raise ValueError("times must match the first data dimension")
    mask = times >= start_time - 1e-12
    if not np.any(mask):
        raise ValueError("start_time is after the final saved time")
    return relative_l2_error(reference[mask], estimate[mask])


def time_resolved_relative_l2_error(
    reference: FloatArray,
    estimate: FloatArray,
) -> FloatArray:
    """Return a relative L2 error for every sample along the first axis.

    A time with zero reference norm is reported as NaN unless the estimate is
    also exactly zero, in which case its error is zero. This convention is
    useful for multinomial variance, which is identically zero initially.
    """

    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    if reference.shape != estimate.shape:
        raise ValueError("reference and estimate must have the same shape")
    if reference.ndim < 2:
        raise ValueError("reference and estimate must have at least two dimensions")

    axes = tuple(range(1, reference.ndim))
    numerator = np.sqrt(np.sum((reference - estimate) ** 2, axis=axes))
    denominator = np.sqrt(np.sum(reference**2, axis=axes))
    result = np.full(denominator.shape, np.nan, dtype=float)
    np.divide(numerator, denominator, out=result, where=denominator > 0.0)
    result[(denominator == 0.0) & (numerator == 0.0)] = 0.0
    return result


def _finite_max(values: FloatArray) -> float:
    finite = np.asarray(values, dtype=float)[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else float("nan")


def _mean_sampling_diagnostics(
    reference_mean: FloatArray,
    sample_mean: FloatArray,
    reference_variance: FloatArray,
    n_runs: int,
) -> float:
    """Compare sampled mean error with the reference Monte Carlo noise floor."""

    standard_error = np.sqrt(np.maximum(reference_variance, 0.0) / float(n_runs))
    bias = np.abs(sample_mean - reference_mean)
    noise_norm = float(np.linalg.norm(standard_error.ravel()))
    bias_norm = float(np.linalg.norm(bias.ravel()))
    if noise_norm == 0.0:
        ratio = 0.0 if bias_norm == 0.0 else float("inf")
    else:
        ratio = bias_norm / noise_norm

    return float(ratio)


def empirical_covariance(
    runs: FloatArray,
    time_indices: list[int] | NDArray[np.integer],
) -> FloatArray:
    """Compute same-time spatial covariance from an ensemble of runs."""

    values = np.asarray(runs, dtype=float)
    if values.ndim != 3:
        raise ValueError("runs must have shape (runs, time, space)")
    indices = np.asarray(time_indices, dtype=int)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("time_indices must be a nonempty one-dimensional sequence")
    if np.min(indices) < 0 or np.max(indices) >= values.shape[1]:
        raise ValueError("time index outside the available range")

    covariance = []
    for time_index in indices:
        samples = values[:, time_index, :]
        centered = samples - samples.mean(axis=0, keepdims=True)
        covariance.append(centered.T @ centered / samples.shape[0])
    return np.stack(covariance)


def relative_frobenius_error(reference: FloatArray, estimate: FloatArray) -> float:
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    if reference.shape != estimate.shape:
        raise ValueError("reference and estimate must have the same shape")
    denominator = np.linalg.norm(reference.ravel())
    if denominator == 0:
        raise ValueError("reference Frobenius norm is zero")
    return float(np.linalg.norm((reference - estimate).ravel()) / denominator)


def summarize(reference: FloatArray, estimate: FloatArray) -> ValidationMetrics:
    """Summarize two mean trajectories; retained for API compatibility."""

    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    if reference.shape != estimate.shape:
        raise ValueError("reference and estimate must have the same shape")

    initial_mass = float(np.sum(estimate[0]))
    mass = np.sum(estimate, axis=1)
    return ValidationMetrics(
        rmse=root_mean_square_error(reference, estimate),
        relative_l2_error=relative_l2_error(reference, estimate),
        maximum_mass_error=float(np.max(np.abs(mass - initial_mass))),
        minimum_reconstructed_count=float(np.min(estimate)),
        negative_value_fraction=float(np.mean(estimate < 0.0)),
    )


def summarize_ensemble(
    reference_mean: FloatArray,
    reference_variance: FloatArray,
    runs: FloatArray,
    total_particles: int,
    *,
    reference_covariance: FloatArray | None = None,
    covariance_time_indices: list[int] | None = None,
) -> EnsembleDiagnostics:
    """Compare an ensemble with exact mean, variance, and optional covariance."""

    values = np.asarray(runs, dtype=float)
    reference_mean = np.asarray(reference_mean, dtype=float)
    reference_variance = np.asarray(reference_variance, dtype=float)
    if values.ndim != 3:
        raise ValueError("runs must have shape (runs, time, space)")
    if values.shape[1:] != reference_mean.shape:
        raise ValueError("run dimensions must match reference_mean")
    if reference_variance.shape != reference_mean.shape:
        raise ValueError("reference_variance must match reference_mean")
    if total_particles <= 0:
        raise ValueError("total_particles must be positive")

    mean = np.mean(values, axis=0)
    variance = np.var(values, axis=0, ddof=0)
    mean_error_by_time = time_resolved_relative_l2_error(reference_mean, mean)
    variance_error_by_time = time_resolved_relative_l2_error(
        reference_variance,
        variance,
    )
    mean_error_to_reference_se = _mean_sampling_diagnostics(
        reference_mean,
        mean,
        reference_variance,
        values.shape[0],
    )
    masses = np.sum(values, axis=-1)
    negative_mass_fraction = np.clip(-values, 0.0, None).sum(axis=-1) / total_particles

    covariance_error: float | None = None
    if reference_covariance is not None or covariance_time_indices is not None:
        if reference_covariance is None or covariance_time_indices is None:
            raise ValueError(
                "reference_covariance and covariance_time_indices must be provided together"
            )
        indices = np.asarray(covariance_time_indices, dtype=int)
        expected_covariance = np.asarray(reference_covariance, dtype=float)[indices]
        observed_covariance = empirical_covariance(values, indices)
        covariance_error = relative_frobenius_error(
            expected_covariance,
            observed_covariance,
        )

    return EnsembleDiagnostics(
        mean_rmse=root_mean_square_error(reference_mean, mean),
        mean_relative_l2_error=relative_l2_error(reference_mean, mean),
        mean_max_time_relative_l2_error=_finite_max(mean_error_by_time),
        mean_final_time_relative_l2_error=float(mean_error_by_time[-1]),
        mean_error_to_reference_se_ratio=mean_error_to_reference_se,
        variance_rmse=root_mean_square_error(reference_variance, variance),
        variance_relative_l2_error=relative_l2_error(reference_variance, variance),
        variance_max_time_relative_l2_error=_finite_max(variance_error_by_time),
        variance_final_time_relative_l2_error=float(variance_error_by_time[-1]),
        maximum_mass_error=float(np.max(np.abs(masses - total_particles))),
        minimum_reconstructed_count=float(np.min(values)),
        negative_entry_fraction=float(np.mean(values < -1e-12)),
        mean_negative_mass_fraction=float(np.mean(negative_mass_fraction)),
        maximum_negative_mass_fraction=float(np.max(negative_mass_fraction)),
        covariance_relative_frobenius_error=covariance_error,
    )


def bootstrap_error_intervals(
    reference_mean: FloatArray,
    reference_variance: FloatArray,
    runs: FloatArray,
    *,
    reference_covariance: FloatArray | None = None,
    covariance_time_indices: list[int] | None = None,
    n_resamples: int = 200,
    confidence_level: float = 0.95,
    n_blocks: int = 20,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """Estimate confidence intervals for ensemble moment errors.

    Independent trajectories are partitioned into blocks. Bootstrap samples
    resample those blocks and aggregate sufficient statistics, avoiding copies
    of the full run tensor. Normal intervals centered on the observed metric
    use the bootstrap standard error to quantify Monte Carlo uncertainty in
    the reported mean, variance, and optional covariance errors.
    """

    values = np.asarray(runs, dtype=float)
    reference_mean = np.asarray(reference_mean, dtype=float)
    reference_variance = np.asarray(reference_variance, dtype=float)
    if values.ndim != 3:
        raise ValueError("runs must have shape (runs, time, space)")
    if values.shape[1:] != reference_mean.shape:
        raise ValueError("run dimensions must match reference_mean")
    if reference_variance.shape != reference_mean.shape:
        raise ValueError("reference_variance must match reference_mean")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie between zero and one")
    if n_blocks <= 1:
        raise ValueError("n_blocks must be greater than one")

    covariance_indices: NDArray[np.int_] | None = None
    expected_covariance: FloatArray | None = None
    if reference_covariance is not None or covariance_time_indices is not None:
        if reference_covariance is None or covariance_time_indices is None:
            raise ValueError(
                "reference_covariance and covariance_time_indices must be provided together"
            )
        covariance_indices = np.asarray(covariance_time_indices, dtype=int)
        expected_covariance = np.asarray(reference_covariance, dtype=float)[
            covariance_indices
        ]

    blocks = np.array_split(np.arange(values.shape[0]), min(n_blocks, values.shape[0]))
    if len(blocks) <= 1:
        raise ValueError("at least two runs are required for bootstrap intervals")
    block_counts = np.asarray([block.size for block in blocks], dtype=int)
    block_sums = np.stack([values[block].sum(axis=0) for block in blocks])
    block_sum_squares = np.stack(
        [np.square(values[block]).sum(axis=0) for block in blocks]
    )

    block_cross_products: FloatArray | None = None
    if covariance_indices is not None:
        cross_products = []
        for block in blocks:
            selected = values[block][:, covariance_indices, :]
            cross_products.append(np.einsum("rti,rtj->tij", selected, selected))
        block_cross_products = np.stack(cross_products)

    rng = np.random.default_rng(seed)
    metric_samples: dict[str, list[float]] = {
        "mean_relative_l2_error": [],
        "variance_relative_l2_error": [],
    }
    if expected_covariance is not None:
        metric_samples["covariance_relative_frobenius_error"] = []

    n_available_blocks = len(blocks)
    for _ in range(n_resamples):
        selected_blocks = rng.integers(
            0,
            n_available_blocks,
            size=n_available_blocks,
        )
        count = int(np.sum(block_counts[selected_blocks]))
        sample_sum = np.sum(block_sums[selected_blocks], axis=0)
        sample_mean = sample_sum / count
        sample_second_moment = np.sum(
            block_sum_squares[selected_blocks], axis=0
        ) / count
        sample_variance = np.maximum(sample_second_moment - sample_mean**2, 0.0)
        metric_samples["mean_relative_l2_error"].append(
            relative_l2_error(reference_mean, sample_mean)
        )
        metric_samples["variance_relative_l2_error"].append(
            relative_l2_error(reference_variance, sample_variance)
        )

        if expected_covariance is not None and block_cross_products is not None:
            second_cross = np.sum(
                block_cross_products[selected_blocks], axis=0
            ) / count
            selected_mean = sample_mean[covariance_indices]
            sample_covariance = second_cross - np.einsum(
                "ti,tj->tij",
                selected_mean,
                selected_mean,
            )
            metric_samples["covariance_relative_frobenius_error"].append(
                relative_frobenius_error(expected_covariance, sample_covariance)
            )

    observed_mean = np.mean(values, axis=0)
    observed_variance = np.var(values, axis=0, ddof=0)
    observed: dict[str, float] = {
        "mean_relative_l2_error": relative_l2_error(reference_mean, observed_mean),
        "variance_relative_l2_error": relative_l2_error(
            reference_variance,
            observed_variance,
        ),
    }
    if expected_covariance is not None and covariance_indices is not None:
        observed_covariance = empirical_covariance(values, covariance_indices)
        observed["covariance_relative_frobenius_error"] = relative_frobenius_error(
            expected_covariance,
            observed_covariance,
        )

    z_score = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    intervals: dict[str, tuple[float, float]] = {}
    for name, samples in metric_samples.items():
        standard_error = float(np.std(samples, ddof=1))
        center = observed[name]
        intervals[name] = (
            max(0.0, center - z_score * standard_error),
            center + z_score * standard_error,
        )
    return intervals
