"""Diffusion operators and spatial/modal transformations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Eigenbasis:
    """Sorted eigendecomposition of a symmetric diffusion operator."""

    eigenvalues: FloatArray
    eigenvectors: FloatArray

    @property
    def n_modes(self) -> int:
        return int(self.eigenvalues.size)

    def to_modal(self, spatial_values: FloatArray) -> FloatArray:
        """Project spatial values into the retained eigenbasis."""

        values = np.asarray(spatial_values, dtype=float)
        return self.eigenvectors.T @ values

    def to_spatial(self, modal_values: FloatArray) -> FloatArray:
        """Reconstruct spatial values from retained modal coefficients."""

        values = np.asarray(modal_values, dtype=float)
        return self.eigenvectors @ values


def neumann_laplacian_1d(
    n_nodes: int,
    length: float,
    diffusion_coefficient: float,
) -> FloatArray:
    """Return the positive-semidefinite 1D diffusion operator.

    The deterministic dynamics use

        d n / dt = -A n,

    where ``A`` is this matrix. Reflecting (zero-flux/Neumann) boundaries are
    represented by endpoint diagonal entries of ``k`` rather than ``2k``.
    """

    if n_nodes < 2:
        raise ValueError("n_nodes must be at least 2")
    if length <= 0 or diffusion_coefficient <= 0:
        raise ValueError("length and diffusion_coefficient must be positive")

    dx = length / (n_nodes - 1)
    k = diffusion_coefficient / dx**2

    diagonal = np.full(n_nodes, 2.0 * k, dtype=float)
    diagonal[[0, -1]] = k
    off_diagonal = np.full(n_nodes - 1, -k, dtype=float)

    return (
        np.diag(diagonal)
        + np.diag(off_diagonal, k=1)
        + np.diag(off_diagonal, k=-1)
    )


def eigendecompose(
    operator: FloatArray,
    n_modes: int | None = None,
) -> Eigenbasis:
    """Diagonalize a symmetric operator and retain the slowest modes.

    Modes are ordered by increasing eigenvalue. For diffusion, the first mode
    is the constant mass-conserving mode and higher modes decay more quickly.
    """

    matrix = np.asarray(operator, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("operator must be a square matrix")
    if not np.allclose(matrix, matrix.T, atol=1e-12):
        raise ValueError("operator must be symmetric")

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, 0.0, None)

    total_modes = matrix.shape[0]
    retained = total_modes if n_modes is None else int(n_modes)
    if not 1 <= retained <= total_modes:
        raise ValueError(f"n_modes must be between 1 and {total_modes}")

    return Eigenbasis(
        eigenvalues=eigenvalues[:retained],
        eigenvectors=eigenvectors[:, :retained],
    )


def impulse_initial_condition(
    n_nodes: int,
    n_particles: int,
    impulse_index: int,
) -> FloatArray:
    """Create a spatial impulse containing all particles at one node."""

    if not 0 <= impulse_index < n_nodes:
        raise ValueError("impulse_index is outside the spatial grid")
    values = np.zeros(n_nodes, dtype=float)
    values[impulse_index] = float(n_particles)
    return values
