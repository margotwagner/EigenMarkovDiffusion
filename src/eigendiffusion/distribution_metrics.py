"""Distribution-level diagnostics for stochastic spatial ensembles.

The modal outputs are continuous while the multinomial reference is discrete,
so raw KL divergence is generally undefined or infinite.  These diagnostics
use common-bin Jensen--Shannon divergence for scalar marginals and sliced
Wasserstein distance for full spatial vectors.  Both remain finite when the
two models do not have identical support.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class DistributionDiagnostics:
    """Average distribution errors across selected times and spatial nodes."""

    marginal_wasserstein: float
    marginal_normalized_wasserstein: float
    marginal_jensen_shannon: float
    sliced_wasserstein: float
    sliced_normalized_wasserstein: float
    n_marginals: int
    n_projections: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def wasserstein_distance_1d(reference: FloatArray, estimate: FloatArray) -> float:
    """Return the empirical one-dimensional Wasserstein-1 distance.

    Quantile interpolation supports unequal ensemble sizes without SciPy.
    """

    reference = np.asarray(reference, dtype=float).ravel()
    estimate = np.asarray(estimate, dtype=float).ravel()
    if reference.size == 0 or estimate.size == 0:
        raise ValueError("samples must be nonempty")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(estimate)):
        raise ValueError("samples must be finite")

    quantiles = (np.arange(max(reference.size, estimate.size)) + 0.5) / max(
        reference.size,
        estimate.size,
    )
    reference_quantiles = np.quantile(reference, quantiles, method="linear")
    estimate_quantiles = np.quantile(estimate, quantiles, method="linear")
    return float(np.mean(np.abs(reference_quantiles - estimate_quantiles)))


def histogram_jensen_shannon_divergence(
    reference: FloatArray,
    estimate: FloatArray,
    *,
    bins: int = 30,
) -> float:
    """Return a finite common-bin Jensen--Shannon divergence in nats."""

    reference = np.asarray(reference, dtype=float).ravel()
    estimate = np.asarray(estimate, dtype=float).ravel()
    if reference.size == 0 or estimate.size == 0:
        raise ValueError("samples must be nonempty")
    if bins < 2:
        raise ValueError("bins must be at least two")
    lower = float(min(np.min(reference), np.min(estimate)))
    upper = float(max(np.max(reference), np.max(estimate)))
    if lower == upper:
        return 0.0
    edges = np.linspace(lower, upper, bins + 1)
    reference_counts = np.histogram(reference, bins=edges)[0].astype(float)
    estimate_counts = np.histogram(estimate, bins=edges)[0].astype(float)
    reference_probability = reference_counts / reference_counts.sum()
    estimate_probability = estimate_counts / estimate_counts.sum()
    midpoint = 0.5 * (reference_probability + estimate_probability)

    def divergence(probability: FloatArray) -> float:
        mask = probability > 0.0
        return float(
            np.sum(probability[mask] * np.log(probability[mask] / midpoint[mask]))
        )

    return 0.5 * (
        divergence(reference_probability) + divergence(estimate_probability)
    )


def summarize_distribution_ensemble(
    reference_runs: FloatArray,
    estimate_runs: FloatArray,
    *,
    time_indices: list[int] | NDArray[np.integer],
    node_indices: list[int] | NDArray[np.integer],
    n_projections: int = 64,
    histogram_bins: int = 30,
    seed: int = 0,
) -> DistributionDiagnostics:
    """Compare selected marginals and full spatial-vector distributions."""

    reference = np.asarray(reference_runs, dtype=float)
    estimate = np.asarray(estimate_runs, dtype=float)
    if reference.ndim != 3 or estimate.ndim != 3:
        raise ValueError("runs must have shape (runs, time, space)")
    if reference.shape[1:] != estimate.shape[1:]:
        raise ValueError("reference and estimate time/space shapes must match")
    if n_projections <= 0:
        raise ValueError("n_projections must be positive")
    times = np.asarray(time_indices, dtype=int)
    nodes = np.asarray(node_indices, dtype=int)
    if times.ndim != 1 or times.size == 0 or nodes.ndim != 1 or nodes.size == 0:
        raise ValueError("time_indices and node_indices must be nonempty vectors")
    if np.min(times) < 0 or np.max(times) >= reference.shape[1]:
        raise ValueError("time index outside available range")
    if np.min(nodes) < 0 or np.max(nodes) >= reference.shape[2]:
        raise ValueError("node index outside available range")

    marginal_wasserstein: list[float] = []
    marginal_normalized: list[float] = []
    marginal_jensen_shannon: list[float] = []
    for time_index in times:
        for node_index in nodes:
            reference_samples = reference[:, time_index, node_index]
            estimate_samples = estimate[:, time_index, node_index]
            distance = wasserstein_distance_1d(reference_samples, estimate_samples)
            scale = max(float(np.std(reference_samples, ddof=0)), 1.0)
            marginal_wasserstein.append(distance)
            marginal_normalized.append(distance / scale)
            marginal_jensen_shannon.append(
                histogram_jensen_shannon_divergence(
                    reference_samples,
                    estimate_samples,
                    bins=histogram_bins,
                )
            )

    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n_projections, reference.shape[2]))
    # Total mass is fixed, so remove the uninformative constant direction.
    directions -= directions.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions /= np.maximum(norms, np.finfo(float).eps)

    sliced: list[float] = []
    sliced_normalized: list[float] = []
    for time_index in times:
        reference_projection = reference[:, time_index, :] @ directions.T
        estimate_projection = estimate[:, time_index, :] @ directions.T
        for projection_index in range(n_projections):
            reference_samples = reference_projection[:, projection_index]
            estimate_samples = estimate_projection[:, projection_index]
            distance = wasserstein_distance_1d(reference_samples, estimate_samples)
            scale = max(float(np.std(reference_samples, ddof=0)), 1.0)
            sliced.append(distance)
            sliced_normalized.append(distance / scale)

    return DistributionDiagnostics(
        marginal_wasserstein=float(np.mean(marginal_wasserstein)),
        marginal_normalized_wasserstein=float(np.mean(marginal_normalized)),
        marginal_jensen_shannon=float(np.mean(marginal_jensen_shannon)),
        sliced_wasserstein=float(np.mean(sliced)),
        sliced_normalized_wasserstein=float(np.mean(sliced_normalized)),
        n_marginals=len(marginal_wasserstein),
        n_projections=n_projections,
    )
