"""Analytic reference solutions for the one-dimensional diffusion model.

The stochastic simulators in this repository advance in finite timesteps. Two
reference means are therefore useful and should not be conflated:

``continuous_expected_diffusion``
    The matrix-exponential solution of the spatially discretized ODE.

``discrete_expected_diffusion``
    The exact ensemble mean of the finite-step nearest-neighbour random walk,
    equivalently repeated multiplication by ``P = I - dt * A``.

For independent particles that all begin at the same node, the marginal node
counts at each time are multinomial. The functions below also return the exact
pointwise variance and same-time spatial covariance of that process.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import DiffusionConfig
from .operators import eigendecompose, impulse_initial_condition, neumann_laplacian_1d

FloatArray = NDArray[np.float64]


def random_walk_transition_matrix(config: DiffusionConfig) -> FloatArray:
    """Return the one-step node transition matrix ``P = I - dt * A``.

    ``P`` uses the same reflecting-boundary convention as both random-walk
    simulators. It is symmetric and each row/column sums to one.
    """

    operator = neumann_laplacian_1d(
        config.n_nodes,
        config.length,
        config.diffusion_coefficient,
    )
    transition = np.eye(config.n_nodes, dtype=float) - config.dt * operator
    if np.min(transition) < -1e-12:
        raise ValueError(
            "The timestep creates negative transition probabilities. "
            "Decrease dt or use fewer spatial nodes."
        )
    transition[np.abs(transition) < 1e-15] = 0.0
    return transition


def _modal_reference(
    config: DiffusionConfig,
    *,
    n_modes: int | None,
    discrete: bool,
) -> FloatArray:
    operator = neumann_laplacian_1d(
        config.n_nodes,
        config.length,
        config.diffusion_coefficient,
    )
    basis = eigendecompose(operator, n_modes=n_modes)
    initial = impulse_initial_condition(
        config.n_nodes,
        config.n_particles,
        config.impulse_index,
    )
    modal_initial = basis.to_modal(initial)

    if discrete:
        one_step_factors = 1.0 - config.dt * basis.eigenvalues
        step_numbers = np.arange(config.n_steps, dtype=int)
        propagation = one_step_factors[None, :] ** step_numbers[:, None]
    else:
        propagation = np.exp(-np.outer(config.times, basis.eigenvalues))

    modal_over_time = propagation * modal_initial[None, :]
    return modal_over_time @ basis.eigenvectors.T


def continuous_expected_diffusion(
    config: DiffusionConfig,
    n_modes: int | None = None,
) -> FloatArray:
    """Return the matrix-exponential mean of the discretized diffusion ODE."""

    return _modal_reference(config, n_modes=n_modes, discrete=False)


def discrete_expected_diffusion(
    config: DiffusionConfig,
    n_modes: int | None = None,
) -> FloatArray:
    """Return the exact mean of the finite-step random-walk process.

    With all modes retained, this is exactly ``P**t @ n0`` for
    ``P = I - dt * A``. This is the appropriate mean reference for both random
    walk implementations and for the current first-order EigenMarkov update.
    """

    return _modal_reference(config, n_modes=n_modes, discrete=True)


def multinomial_node_probabilities(config: DiffusionConfig) -> FloatArray:
    """Return exact single-particle node probabilities at every saved time."""

    probabilities = discrete_expected_diffusion(config) / float(config.n_particles)
    # Remove harmless eigendecomposition round-off and enforce normalization.
    probabilities = np.clip(probabilities, 0.0, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def multinomial_marginal_variance(config: DiffusionConfig) -> FloatArray:
    """Return exact node-count variances for independent particles.

    At each time, ``n(t) ~ Multinomial(N, p(t))`` marginally, so
    ``Var[n_i(t)] = N p_i(t) (1 - p_i(t))``.
    """

    probabilities = multinomial_node_probabilities(config)
    return config.n_particles * probabilities * (1.0 - probabilities)


def multinomial_marginal_covariance(config: DiffusionConfig) -> FloatArray:
    """Return exact same-time spatial covariance matrices.

    The returned array has shape ``(n_steps, n_nodes, n_nodes)`` and uses

    ``Cov[n(t)] = N (diag(p(t)) - p(t) p(t)^T)``.
    """

    probabilities = multinomial_node_probabilities(config)
    covariance = -config.n_particles * np.einsum(
        "ti,tj->tij",
        probabilities,
        probabilities,
    )
    diagonal = np.arange(config.n_nodes)
    covariance[:, diagonal, diagonal] += config.n_particles * probabilities
    return covariance
