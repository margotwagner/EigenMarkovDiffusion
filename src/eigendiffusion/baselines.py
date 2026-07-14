"""Deterministic and stochastic diffusion baselines.

Two random-walk implementations are provided deliberately:

``naive_random_walk_diffusion``
    Mirrors the original research implementation by simulating and storing the
    complete trajectory of every particle before converting trajectories to
    node counts.

``multinomial_random_walk_diffusion``
    Samples the same independent particle transitions jointly at each node.
    It produces the same node-count process in distribution without retaining
    particle identities or trajectories.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import DiffusionConfig
from .operators import eigendecompose, impulse_initial_condition, neumann_laplacian_1d

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def deterministic_diffusion(
    config: DiffusionConfig,
    n_modes: int | None = None,
) -> FloatArray:
    """Compute the deterministic diffusion mean by eigenmode propagation.

    Returns an array with shape ``(n_steps, n_nodes)``. With every mode
    retained this is the exact solution of the spatially discretized linear
    system at the requested time points.
    """

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
    decay = np.exp(-np.outer(config.times, basis.eigenvalues))
    modal_over_time = decay * modal_initial[None, :]
    return modal_over_time @ basis.eigenvectors.T


def naive_random_walk_trajectories(
    config: DiffusionConfig,
    rng: np.random.Generator | None = None,
) -> IntArray:
    """Simulate and store every particle trajectory.

    The returned array has shape ``(n_particles, n_steps)``. This intentionally
    follows the structure of the original repository's ``RandomWalk`` model so
    that the cost of a particle-by-particle implementation can be benchmarked.
    Reflecting boundaries are implemented by rejecting outward moves.
    """

    generator = np.random.default_rng() if rng is None else rng
    trajectories = np.empty(
        (config.n_particles, config.n_steps),
        dtype=np.int64,
    )
    p = config.jump_probability
    final_node = config.n_nodes - 1

    for particle_index in range(config.n_particles):
        particle_path = trajectories[particle_index]
        particle_path[0] = config.impulse_index
        random_values = generator.random(config.n_steps - 1)

        for time_index, random_value in enumerate(random_values):
            position = int(particle_path[time_index])
            if random_value < p and position > 0:
                position -= 1
            elif random_value > 1.0 - p and position < final_node:
                position += 1
            particle_path[time_index + 1] = position

    return trajectories


def trajectories_to_counts(
    trajectories: IntArray,
    n_nodes: int,
) -> IntArray:
    """Convert particle trajectories to node counts over time.

    Parameters
    ----------
    trajectories:
        Integer array shaped ``(n_particles, n_steps)``.
    n_nodes:
        Number of spatial nodes.

    Returns
    -------
    numpy.ndarray
        Integer counts shaped ``(n_steps, n_nodes)``.
    """

    if trajectories.ndim != 2:
        raise ValueError("trajectories must have shape (n_particles, n_steps)")
    if n_nodes < 2:
        raise ValueError("n_nodes must be at least 2")
    if trajectories.size and (
        np.min(trajectories) < 0 or np.max(trajectories) >= n_nodes
    ):
        raise ValueError("trajectory positions must lie within the spatial grid")

    n_steps = trajectories.shape[1]
    counts = np.empty((n_steps, n_nodes), dtype=np.int64)
    for time_index in range(n_steps):
        counts[time_index] = np.bincount(
            trajectories[:, time_index],
            minlength=n_nodes,
        )
    return counts


def naive_random_walk_diffusion(
    config: DiffusionConfig,
    rng: np.random.Generator | None = None,
) -> IntArray:
    """Run the naive particle-trajectory random walk and return node counts.

    Internally, this stores an ``(n_particles, n_steps)`` trajectory matrix,
    then post-processes it into an ``(n_steps, n_nodes)`` count matrix. Particle
    identities are not returned, but their storage cost is intentionally paid.
    """

    trajectories = naive_random_walk_trajectories(config, rng=rng)
    return trajectories_to_counts(trajectories, n_nodes=config.n_nodes)


def multinomial_random_walk_diffusion(
    config: DiffusionConfig,
    rng: np.random.Generator | None = None,
) -> IntArray:
    """Simulate an exact aggregated nearest-neighbour random walk.

    At each node, one multinomial draw jointly samples how many particles move
    left, stay, or move right. Under the model assumption that particles move
    independently with identical node-specific probabilities, this is exactly
    the same node-count process in distribution as the naive implementation.
    Particle identities and complete trajectories are not represented.
    """

    generator = np.random.default_rng() if rng is None else rng
    counts = np.zeros((config.n_steps, config.n_nodes), dtype=np.int64)
    counts[0, config.impulse_index] = config.n_particles
    p = config.jump_probability

    for time_index in range(config.n_steps - 1):
        current = counts[time_index]
        updated = np.zeros(config.n_nodes, dtype=np.int64)

        for node_index, node_count in enumerate(current):
            if node_count == 0:
                continue

            p_left = p if node_index > 0 else 0.0
            p_right = p if node_index < config.n_nodes - 1 else 0.0
            p_stay = 1.0 - p_left - p_right
            left, stay, right = generator.multinomial(
                int(node_count),
                [p_left, p_stay, p_right],
            )

            if node_index > 0:
                updated[node_index - 1] += left
            updated[node_index] += stay
            if node_index < config.n_nodes - 1:
                updated[node_index + 1] += right

        counts[time_index + 1] = updated

    return counts


def random_walk_diffusion(
    config: DiffusionConfig,
    rng: np.random.Generator | None = None,
) -> IntArray:
    """Backward-compatible alias for the multinomial implementation.

    New code should call ``multinomial_random_walk_diffusion`` explicitly so
    benchmark reports and methods sections are unambiguous.
    """

    return multinomial_random_walk_diffusion(config, rng=rng)
