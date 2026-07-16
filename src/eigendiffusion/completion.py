"""Analytic unresolved-mode completion for reduced modal trajectories.

This module provides an output-only proof-of-principle closure for variance
lost when diffusion dynamics are evolved in a truncated eigenbasis. It does
not change the retained modal dynamics. Instead, it samples the omitted modal
coordinates at requested output times from a ridge-regularized Gaussian
conditional model built from the exact finite-step multinomial moments.

The method is intentionally labelled analytic: it uses the known mean and
same-time covariance of the linear random-walk reference. It is useful for
asking whether omitted-mode stochastic completion can recover full-rank
statistics without dynamically evolving every mode, but it is not yet a
standalone closure for nonlinear or unknown systems.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import DiffusionConfig
from .operators import Eigenbasis, eigendecompose, neumann_laplacian_1d
from .references import discrete_expected_diffusion, multinomial_marginal_covariance

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class CompletionStep:
    """Precomputed Gaussian conditional parameters for one saved time."""

    retained_mean: FloatArray
    omitted_mean: FloatArray
    conditional_gain: FloatArray
    conditional_factor: FloatArray


class UnresolvedGaussianCompleter:
    """Sample omitted diffusion modes conditional on the retained state.

    Let the full modal vector be partitioned into retained and unresolved
    components, ``m = [m_R, m_U]``. Using the exact multinomial mean and
    covariance at each time, this class constructs the regularized Gaussian
    approximation

    ``m_U | m_R ~ N(mu_U + G (m_R - mu_R), C_U|R)``.

    The gain is stabilized with ridge regularization,

    ``G = C_UR (C_RR + ridge * scale * I)^-1``.

    The unresolved covariance is then sampled through a possibly low-rank
    eigendecomposition. The retained coordinates are never modified, and the
    omitted basis excludes the constant mode, so total mass is preserved up to
    floating-point roundoff.
    """

    def __init__(
        self,
        config: DiffusionConfig,
        *,
        retained_modes: int,
        completion_start_time: float = 0.0,
        completion_rank: int | None = None,
        ridge: float = 1.0e-2,
        eigenvalue_tolerance: float = 1.0e-12,
    ) -> None:
        if not 1 <= retained_modes <= config.n_nodes:
            raise ValueError("retained_modes must lie in [1, n_nodes]")
        if not config.times[0] - 1e-12 <= completion_start_time <= config.times[-1] + 1e-12:
            raise ValueError("completion_start_time must lie within the simulation")
        if completion_rank is not None and completion_rank <= 0:
            raise ValueError("completion_rank must be positive when provided")
        if ridge < 0.0:
            raise ValueError("ridge must be nonnegative")
        if eigenvalue_tolerance < 0.0:
            raise ValueError("eigenvalue_tolerance must be nonnegative")

        self.config = config
        self.retained_modes = int(retained_modes)
        self.omitted_modes = config.n_nodes - self.retained_modes
        self.completion_start_step = int(
            np.argmin(np.abs(config.times - float(completion_start_time)))
        )
        self.completion_start_time = float(config.times[self.completion_start_step])
        self.ridge = float(ridge)
        self.eigenvalue_tolerance = float(eigenvalue_tolerance)

        max_rank = self.omitted_modes
        requested_rank = max_rank if completion_rank is None else int(completion_rank)
        self.completion_rank = min(requested_rank, max_rank)

        operator = neumann_laplacian_1d(
            config.n_nodes,
            config.length,
            config.diffusion_coefficient,
        )
        self.full_basis = eigendecompose(operator, n_modes=config.n_nodes)
        self.retained_basis = Eigenbasis(
            eigenvalues=self.full_basis.eigenvalues[: self.retained_modes].copy(),
            eigenvectors=self.full_basis.eigenvectors[:, : self.retained_modes].copy(),
        )
        self.omitted_eigenvectors = self.full_basis.eigenvectors[:, self.retained_modes :]

        self._steps = self._precompute_steps()

    def _precompute_steps(self) -> list[CompletionStep | None]:
        if self.omitted_modes == 0:
            return [None] * self.config.n_steps

        reference_mean = discrete_expected_diffusion(self.config)
        reference_covariance = multinomial_marginal_covariance(self.config)
        vectors = self.full_basis.eigenvectors
        steps: list[CompletionStep | None] = []

        for time_index in range(self.config.n_steps):
            if time_index < self.completion_start_step:
                steps.append(None)
                continue

            modal_mean = vectors.T @ reference_mean[time_index]
            modal_covariance = vectors.T @ reference_covariance[time_index] @ vectors
            modal_covariance = 0.5 * (modal_covariance + modal_covariance.T)

            retained_mean = modal_mean[: self.retained_modes]
            omitted_mean = modal_mean[self.retained_modes :]
            c_rr = modal_covariance[: self.retained_modes, : self.retained_modes]
            c_ur = modal_covariance[self.retained_modes :, : self.retained_modes]
            c_ru = modal_covariance[: self.retained_modes, self.retained_modes :]
            c_uu = modal_covariance[self.retained_modes :, self.retained_modes :]

            scale = float(np.trace(c_rr) / max(self.retained_modes, 1))
            if scale <= 1e-15 or not np.any(c_ur):
                gain = np.zeros_like(c_ur)
            else:
                stabilized = c_rr + self.ridge * scale * np.eye(self.retained_modes)
                gain = np.linalg.solve(stabilized.T, c_ur.T).T

            conditional_covariance = c_uu - gain @ c_ru
            conditional_covariance = 0.5 * (
                conditional_covariance + conditional_covariance.T
            )
            factor = self._low_rank_psd_factor(conditional_covariance)
            steps.append(
                CompletionStep(
                    retained_mean=retained_mean,
                    omitted_mean=omitted_mean,
                    conditional_gain=gain,
                    conditional_factor=factor,
                )
            )

        return steps

    def _low_rank_psd_factor(self, covariance: FloatArray) -> FloatArray:
        if covariance.size == 0 or self.completion_rank == 0:
            return np.zeros((self.omitted_modes, 0), dtype=float)

        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        scale = max(float(np.max(eigenvalues)), 1.0)
        positive = eigenvalues > self.eigenvalue_tolerance * scale
        keep = min(self.completion_rank, int(np.count_nonzero(positive)))
        if keep == 0:
            return np.zeros((self.omitted_modes, 0), dtype=float)
        return eigenvectors[:, :keep] * np.sqrt(eigenvalues[:keep])[None, :]

    def complete(
        self,
        raw_counts: FloatArray,
        rng: np.random.Generator | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        """Return completed counts and the added unresolved spatial field."""

        values = np.asarray(raw_counts, dtype=float)
        if values.shape != (self.config.n_steps, self.config.n_nodes):
            raise ValueError(
                "raw_counts must have shape (config.n_steps, config.n_nodes)"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("raw_counts must be finite")

        if self.omitted_modes == 0:
            return values.copy(), np.zeros_like(values)

        generator = np.random.default_rng() if rng is None else rng
        completed = values.copy()
        additions = np.zeros_like(values)
        v_r = self.retained_basis.eigenvectors
        v_u = self.omitted_eigenvectors

        for time_index in range(self.completion_start_step, self.config.n_steps):
            parameters = self._steps[time_index]
            if parameters is None:
                continue
            retained_state = v_r.T @ values[time_index]
            conditional_mean = parameters.omitted_mean + (
                parameters.conditional_gain
                @ (retained_state - parameters.retained_mean)
            )
            if parameters.conditional_factor.shape[1]:
                latent = generator.standard_normal(
                    parameters.conditional_factor.shape[1]
                )
                unresolved_state = (
                    conditional_mean + parameters.conditional_factor @ latent
                )
            else:
                unresolved_state = conditional_mean

            addition = v_u @ unresolved_state
            # The unresolved eigenvectors are orthogonal to the constant mode.
            # Remove only floating-point mass drift.
            addition -= float(addition.sum()) / self.config.n_nodes
            additions[time_index] = addition
            completed[time_index] = values[time_index] + addition

        return completed, additions
