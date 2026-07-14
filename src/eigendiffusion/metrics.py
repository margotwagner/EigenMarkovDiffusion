"""Accuracy, variance, covariance, and physicality diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

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
    variance_rmse: float
    variance_relative_l2_error: float
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
        variance_rmse=root_mean_square_error(reference_variance, variance),
        variance_relative_l2_error=relative_l2_error(reference_variance, variance),
        maximum_mass_error=float(np.max(np.abs(masses - total_particles))),
        minimum_reconstructed_count=float(np.min(values)),
        negative_entry_fraction=float(np.mean(values < -1e-12)),
        mean_negative_mass_fraction=float(np.mean(negative_mass_fraction)),
        maximum_negative_mass_fraction=float(np.max(negative_mass_fraction)),
        covariance_relative_frobenius_error=covariance_error,
    )
