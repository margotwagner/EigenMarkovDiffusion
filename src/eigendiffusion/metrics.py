"""Validation metrics for comparing diffusion simulations."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ValidationMetrics:
    rmse: float
    relative_l2_error: float
    maximum_mass_error: float
    minimum_reconstructed_count: float
    negative_value_fraction: float

    def as_dict(self) -> dict[str, float]:
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
    denominator = np.linalg.norm(reference.ravel())
    if denominator == 0:
        raise ValueError("reference norm is zero")
    return float(np.linalg.norm((reference - estimate).ravel()) / denominator)


def summarize(reference: FloatArray, estimate: FloatArray) -> ValidationMetrics:
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
