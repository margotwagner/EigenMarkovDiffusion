"""Cross-time covariance and temporal-correlation diagnostics.

The existing validation metrics compare distributions independently at each saved
physical time.  This module adds diagnostics for whether stochastic fluctuations
persist correctly from one time to another.

For independent particles with one-step transition matrix ``P`` and single-particle
probabilities ``p_t``, the exact cross-time node-count covariance for ``s = t + lag`` is

``Cov[n(t), n(s)] = N (diag(p_t) P**lag - p_t p_s.T)``.

The functions below average this covariance over a selected set of time origins.  The
same averaging is applied to empirical simulation ensembles, allowing a stable and
computationally manageable comparison of temporal statistics.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import DiffusionConfig
from .references import multinomial_node_probabilities, random_walk_transition_matrix

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def valid_time_origins(
    n_steps: int,
    lag_steps: int,
    *,
    start_step: int = 0,
    end_step: int | None = None,
) -> IntArray:
    """Return valid time-origin indices for a nonnegative lag.

    ``end_step`` is inclusive and refers to the latest allowed destination time.
    """

    if n_steps < 2:
        raise ValueError("n_steps must be at least 2")
    if lag_steps < 0:
        raise ValueError("lag_steps must be nonnegative")
    if start_step < 0 or start_step >= n_steps:
        raise ValueError("start_step must lie within the saved trajectory")

    final_destination = n_steps - 1 if end_step is None else int(end_step)
    if final_destination < start_step or final_destination >= n_steps:
        raise ValueError("end_step must lie between start_step and n_steps - 1")

    final_origin = final_destination - lag_steps
    if final_origin < start_step:
        return np.empty(0, dtype=np.int64)
    return np.arange(start_step, final_origin + 1, dtype=np.int64)


def multinomial_mean_cross_time_covariance(
    config: DiffusionConfig,
    lag_steps: int,
    *,
    start_indices: IntArray | list[int] | None = None,
) -> FloatArray:
    """Return exact cross-time covariance averaged over time origins.

    Parameters
    ----------
    config:
        Diffusion configuration.
    lag_steps:
        Nonnegative integer lag measured in simulation steps.
    start_indices:
        Optional origin indices.  Every ``start + lag_steps`` must be valid.  When
        omitted, all valid origins are used.
    """

    if lag_steps < 0:
        raise ValueError("lag_steps must be nonnegative")
    if start_indices is None:
        origins = valid_time_origins(config.n_steps, lag_steps)
    else:
        origins = np.asarray(start_indices, dtype=np.int64)
        if origins.ndim != 1 or origins.size == 0:
            raise ValueError("start_indices must be a nonempty one-dimensional sequence")
        if np.min(origins) < 0 or np.max(origins) + lag_steps >= config.n_steps:
            raise ValueError("start_indices and lag_steps exceed the saved trajectory")

    probabilities = multinomial_node_probabilities(config)
    transition = random_walk_transition_matrix(config)
    transition_power = np.linalg.matrix_power(transition, int(lag_steps))

    p_start = probabilities[origins]
    p_end = probabilities[origins + lag_steps]
    # For one particle, E[X_t X_s^T] = diag(p_t) P^(s-t).
    joint = p_start[:, :, None] * transition_power[None, :, :]
    covariance = config.n_particles * (
        joint - p_start[:, :, None] * p_end[:, None, :]
    )
    return np.mean(covariance, axis=0)


def empirical_mean_cross_time_covariance(
    runs: FloatArray,
    lag_steps: int,
    *,
    start_indices: IntArray | list[int] | None = None,
) -> FloatArray:
    """Estimate cross-time spatial covariance, averaged over time origins.

    ``runs`` must have shape ``(n_runs, n_steps, n_nodes)``.  Samples are centered
    separately at each time origin before covariance is averaged across origins.
    """

    values = np.asarray(runs, dtype=float)
    if values.ndim != 3:
        raise ValueError("runs must have shape (runs, time, space)")
    if values.shape[0] <= 0:
        raise ValueError("runs must contain at least one simulation")
    if lag_steps < 0:
        raise ValueError("lag_steps must be nonnegative")

    if start_indices is None:
        origins = valid_time_origins(values.shape[1], lag_steps)
    else:
        origins = np.asarray(start_indices, dtype=np.int64)
        if origins.ndim != 1 or origins.size == 0:
            raise ValueError("start_indices must be a nonempty one-dimensional sequence")
        if np.min(origins) < 0 or np.max(origins) + lag_steps >= values.shape[1]:
            raise ValueError("start_indices and lag_steps exceed the saved trajectory")

    start = values[:, origins, :]
    end = values[:, origins + lag_steps, :]
    start_centered = start - np.mean(start, axis=0, keepdims=True)
    end_centered = end - np.mean(end, axis=0, keepdims=True)

    # Collapse run and time-origin dimensions after centering each origin separately.
    start_flat = start_centered.reshape(-1, values.shape[2])
    end_flat = end_centered.reshape(-1, values.shape[2])
    return start_flat.T @ end_flat / float(start_flat.shape[0])


def mean_node_lag_correlation_from_runs(
    runs: FloatArray,
    lag_steps: int,
    node_indices: IntArray | list[int],
    *,
    start_indices: IntArray | list[int] | None = None,
    variance_tolerance: float = 1.0e-12,
) -> float:
    """Average Pearson lag correlation over selected nodes and time origins."""

    values = np.asarray(runs, dtype=float)
    if values.ndim != 3:
        raise ValueError("runs must have shape (runs, time, space)")
    nodes = np.asarray(node_indices, dtype=np.int64)
    if nodes.ndim != 1 or nodes.size == 0:
        raise ValueError("node_indices must be a nonempty one-dimensional sequence")
    if np.min(nodes) < 0 or np.max(nodes) >= values.shape[2]:
        raise ValueError("node index outside the spatial grid")

    if start_indices is None:
        origins = valid_time_origins(values.shape[1], lag_steps)
    else:
        origins = np.asarray(start_indices, dtype=np.int64)
    if origins.size == 0:
        return float("nan")

    start = values[:, origins, :][:, :, nodes]
    end = values[:, origins + lag_steps, :][:, :, nodes]
    start_centered = start - np.mean(start, axis=0, keepdims=True)
    end_centered = end - np.mean(end, axis=0, keepdims=True)

    covariance = np.mean(start_centered * end_centered, axis=0)
    variance_start = np.mean(start_centered**2, axis=0)
    variance_end = np.mean(end_centered**2, axis=0)
    denominator = np.sqrt(variance_start * variance_end)
    valid = denominator > variance_tolerance
    if not np.any(valid):
        return float("nan")
    return float(np.mean(covariance[valid] / denominator[valid]))


def multinomial_mean_node_lag_correlation(
    config: DiffusionConfig,
    lag_steps: int,
    node_indices: IntArray | list[int],
    *,
    start_indices: IntArray | list[int] | None = None,
    variance_tolerance: float = 1.0e-12,
) -> float:
    """Return exact mean lag correlation for selected nodes and origins."""

    nodes = np.asarray(node_indices, dtype=np.int64)
    if nodes.ndim != 1 or nodes.size == 0:
        raise ValueError("node_indices must be a nonempty one-dimensional sequence")
    if np.min(nodes) < 0 or np.max(nodes) >= config.n_nodes:
        raise ValueError("node index outside the spatial grid")

    if start_indices is None:
        origins = valid_time_origins(config.n_steps, lag_steps)
    else:
        origins = np.asarray(start_indices, dtype=np.int64)
    if origins.size == 0:
        return float("nan")

    probabilities = multinomial_node_probabilities(config)
    transition = random_walk_transition_matrix(config)
    transition_power = np.linalg.matrix_power(transition, int(lag_steps))

    p_start = probabilities[origins][:, nodes]
    p_end = probabilities[origins + lag_steps][:, nodes]
    stay_probability = np.diag(transition_power)[nodes]
    covariance = config.n_particles * (
        p_start * stay_probability[None, :] - p_start * p_end
    )
    variance_start = config.n_particles * p_start * (1.0 - p_start)
    variance_end = config.n_particles * p_end * (1.0 - p_end)
    denominator = np.sqrt(variance_start * variance_end)
    valid = denominator > variance_tolerance
    if not np.any(valid):
        return float("nan")
    return float(np.mean(covariance[valid] / denominator[valid]))
