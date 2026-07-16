"""Analytic unresolved-mode completion for reduced modal trajectories.

Two output-only Gaussian closures are provided:

``UnresolvedGaussianCompleter``
    Samples omitted modes independently at each saved time from the exact
    same-time Gaussian conditional distribution. It restores marginal mean and
    covariance, but does not enforce temporal persistence.

``PersistentUnresolvedGaussianCompleter``
    Maintains an unresolved latent state through time. The next unresolved
    state is sampled from the exact Gaussian conditional distribution given the
    previous unresolved state and the retained states at the current and next
    saved times. This preserves much more of the cross-time covariance while
    leaving the retained modal trajectory unchanged.

Both methods use the known finite-step multinomial moments of the linear random
walk. They are therefore analytic proof-of-principle closures, not standalone
closures for unknown or nonlinear systems.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import DiffusionConfig
from .operators import Eigenbasis, eigendecompose, neumann_laplacian_1d
from .references import (
    discrete_expected_diffusion,
    multinomial_marginal_covariance,
    multinomial_node_probabilities,
    random_walk_transition_matrix,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class CompletionStep:
    """Precomputed same-time Gaussian conditional parameters."""

    retained_mean: FloatArray
    omitted_mean: FloatArray
    conditional_gain: FloatArray
    conditional_factor: FloatArray


@dataclass(frozen=True, slots=True)
class PersistentTransitionStep:
    """Conditional parameters for one unresolved-state transition.

    The target is the omitted modal state at ``t + 1``. The conditioning vector
    concatenates ``[u_t, r_t, r_{t+1}]``.
    """

    conditioning_mean: FloatArray
    target_mean: FloatArray
    conditional_gain: FloatArray
    conditional_factor: FloatArray


class _GaussianCompletionBase:
    """Shared basis construction and stable Gaussian conditioning helpers."""

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
        if not (
            config.times[0] - 1e-12
            <= completion_start_time
            <= config.times[-1] + 1e-12
        ):
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
        self.omitted_eigenvectors = self.full_basis.eigenvectors[
            :, self.retained_modes :
        ]

    def _stable_gain(
        self,
        target_conditioning_covariance: FloatArray,
        conditioning_covariance: FloatArray,
    ) -> FloatArray:
        if target_conditioning_covariance.size == 0:
            return np.zeros_like(target_conditioning_covariance)
        scale = float(
            np.trace(conditioning_covariance)
            / max(conditioning_covariance.shape[0], 1)
        )
        if scale <= 1e-15 or not np.any(target_conditioning_covariance):
            return np.zeros_like(target_conditioning_covariance)
        stabilized = conditioning_covariance + self.ridge * scale * np.eye(
            conditioning_covariance.shape[0]
        )
        return np.linalg.solve(
            stabilized.T,
            target_conditioning_covariance.T,
        ).T

    def _low_rank_psd_factor(self, covariance: FloatArray) -> FloatArray:
        if covariance.size == 0 or self.completion_rank == 0:
            return np.zeros((self.omitted_modes, 0), dtype=float)

        symmetric = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        scale = max(float(np.max(eigenvalues)), 1.0)
        positive = eigenvalues > self.eigenvalue_tolerance * scale
        keep = min(self.completion_rank, int(np.count_nonzero(positive)))
        if keep == 0:
            return np.zeros((self.omitted_modes, 0), dtype=float)
        return eigenvectors[:, :keep] * np.sqrt(eigenvalues[:keep])[None, :]

    def _same_time_step(
        self,
        modal_mean: FloatArray,
        modal_covariance: FloatArray,
    ) -> CompletionStep:
        retained_mean = modal_mean[: self.retained_modes]
        omitted_mean = modal_mean[self.retained_modes :]
        c_rr = modal_covariance[: self.retained_modes, : self.retained_modes]
        c_ur = modal_covariance[self.retained_modes :, : self.retained_modes]
        c_ru = modal_covariance[: self.retained_modes, self.retained_modes :]
        c_uu = modal_covariance[self.retained_modes :, self.retained_modes :]

        gain = self._stable_gain(c_ur, c_rr)
        conditional_covariance = c_uu - gain @ c_ru
        factor = self._low_rank_psd_factor(conditional_covariance)
        return CompletionStep(
            retained_mean=retained_mean,
            omitted_mean=omitted_mean,
            conditional_gain=gain,
            conditional_factor=factor,
        )

    def _validate_counts(self, raw_counts: FloatArray) -> FloatArray:
        values = np.asarray(raw_counts, dtype=float)
        if values.shape != (self.config.n_steps, self.config.n_nodes):
            raise ValueError(
                "raw_counts must have shape (config.n_steps, config.n_nodes)"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("raw_counts must be finite")
        return values

    def _sample_gaussian(
        self,
        mean: FloatArray,
        factor: FloatArray,
        rng: np.random.Generator,
    ) -> FloatArray:
        if factor.shape[1] == 0:
            return mean.copy()
        latent = rng.standard_normal(factor.shape[1])
        return mean + factor @ latent

    def _spatial_addition(self, unresolved_state: FloatArray) -> FloatArray:
        addition = self.omitted_eigenvectors @ unresolved_state
        # Omitted eigenvectors exclude the constant mode. Remove only numerical
        # roundoff in the total mass.
        addition -= float(addition.sum()) / self.config.n_nodes
        return addition


class UnresolvedGaussianCompleter(_GaussianCompletionBase):
    """Sample omitted diffusion modes independently at each saved time.

    Let ``m = [m_R, m_U]``. At each time, this class uses the exact multinomial
    mean and covariance to construct

    ``m_U | m_R ~ N(mu_U + G (m_R - mu_R), C_U|R)``.

    This restores same-time moments but redraws the unresolved state at every
    output time, so short-lag temporal persistence can be underestimated.
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
        super().__init__(
            config,
            retained_modes=retained_modes,
            completion_start_time=completion_start_time,
            completion_rank=completion_rank,
            ridge=ridge,
            eigenvalue_tolerance=eigenvalue_tolerance,
        )
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
            modal_covariance = (
                vectors.T @ reference_covariance[time_index] @ vectors
            )
            modal_covariance = 0.5 * (modal_covariance + modal_covariance.T)
            steps.append(self._same_time_step(modal_mean, modal_covariance))
        return steps

    def complete(
        self,
        raw_counts: FloatArray,
        rng: np.random.Generator | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        """Return completed counts and the added unresolved spatial field."""

        values = self._validate_counts(raw_counts)
        if self.omitted_modes == 0:
            return values.copy(), np.zeros_like(values)

        generator = np.random.default_rng() if rng is None else rng
        completed = values.copy()
        additions = np.zeros_like(values)
        v_r = self.retained_basis.eigenvectors

        for time_index in range(self.completion_start_step, self.config.n_steps):
            parameters = self._steps[time_index]
            if parameters is None:
                continue
            retained_state = v_r.T @ values[time_index]
            conditional_mean = parameters.omitted_mean + (
                parameters.conditional_gain
                @ (retained_state - parameters.retained_mean)
            )
            unresolved_state = self._sample_gaussian(
                conditional_mean,
                parameters.conditional_factor,
                generator,
            )
            addition = self._spatial_addition(unresolved_state)
            additions[time_index] = addition
            completed[time_index] = values[time_index] + addition

        return completed, additions


class PersistentUnresolvedGaussianCompleter(_GaussianCompletionBase):
    """Propagate a latent unresolved state with analytic Gaussian transitions.

    At the completion start time, ``u_t`` is sampled from the same-time
    conditional distribution given ``r_t``. For each subsequent step, the next
    omitted state is sampled from

    ``u_(t+1) | [u_t, r_t, r_(t+1)]``.

    The joint Gaussian moments are computed from the exact multinomial
    same-time and one-step cross-time covariance. Conditioning on the observed
    retained transition couples the unresolved innovation to the retained
    innovation, while conditioning on ``u_t`` preserves short-lag persistence.
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
        super().__init__(
            config,
            retained_modes=retained_modes,
            completion_start_time=completion_start_time,
            completion_rank=completion_rank,
            ridge=ridge,
            eigenvalue_tolerance=eigenvalue_tolerance,
        )
        self._initial_step, self._transition_steps = self._precompute_persistent_steps()

    def _precompute_persistent_steps(
        self,
    ) -> tuple[CompletionStep | None, list[PersistentTransitionStep | None]]:
        if self.omitted_modes == 0:
            return None, [None] * max(self.config.n_steps - 1, 0)

        reference_mean = discrete_expected_diffusion(self.config)
        spatial_covariance = multinomial_marginal_covariance(self.config)
        probabilities = multinomial_node_probabilities(self.config)
        transition = random_walk_transition_matrix(self.config)
        vectors = self.full_basis.eigenvectors

        modal_means = reference_mean @ vectors
        modal_covariances = np.einsum(
            "ia,tij,jb->tab",
            vectors,
            spatial_covariance,
            vectors,
            optimize=True,
        )
        modal_covariances = 0.5 * (
            modal_covariances + np.swapaxes(modal_covariances, 1, 2)
        )

        start = self.completion_start_step
        initial_step = self._same_time_step(
            modal_means[start],
            modal_covariances[start],
        )

        transitions: list[PersistentTransitionStep | None] = [
            None
        ] * (self.config.n_steps - 1)
        r = self.retained_modes

        for time_index in range(start, self.config.n_steps - 1):
            p_t = probabilities[time_index]
            p_next = probabilities[time_index + 1]
            cross_spatial = self.config.n_particles * (
                p_t[:, None] * transition
                - p_t[:, None] * p_next[None, :]
            )
            cross_modal = vectors.T @ cross_spatial @ vectors

            c_tt = modal_covariances[time_index]
            c_nn = modal_covariances[time_index + 1]

            # Conditioning vector y = [u_t, r_t, r_(t+1)].
            c_ut_ut = c_tt[r:, r:]
            c_ut_rt = c_tt[r:, :r]
            c_ut_rn = cross_modal[r:, :r]
            c_rt_rt = c_tt[:r, :r]
            c_rt_rn = cross_modal[:r, :r]
            c_rn_rn = c_nn[:r, :r]

            conditioning_covariance = np.block(
                [
                    [c_ut_ut, c_ut_rt, c_ut_rn],
                    [c_ut_rt.T, c_rt_rt, c_rt_rn],
                    [c_ut_rn.T, c_rt_rn.T, c_rn_rn],
                ]
            )
            conditioning_covariance = 0.5 * (
                conditioning_covariance + conditioning_covariance.T
            )

            # Cov(u_(t+1), [u_t, r_t, r_(t+1)]).
            target_conditioning_covariance = np.concatenate(
                [
                    cross_modal[r:, r:].T,
                    cross_modal[:r, r:].T,
                    c_nn[r:, :r],
                ],
                axis=1,
            )
            gain = self._stable_gain(
                target_conditioning_covariance,
                conditioning_covariance,
            )
            target_covariance = c_nn[r:, r:]
            conditional_covariance = target_covariance - (
                gain @ target_conditioning_covariance.T
            )
            factor = self._low_rank_psd_factor(conditional_covariance)

            conditioning_mean = np.concatenate(
                [
                    modal_means[time_index, r:],
                    modal_means[time_index, :r],
                    modal_means[time_index + 1, :r],
                ]
            )
            transitions[time_index] = PersistentTransitionStep(
                conditioning_mean=conditioning_mean,
                target_mean=modal_means[time_index + 1, r:],
                conditional_gain=gain,
                conditional_factor=factor,
            )

        return initial_step, transitions

    def complete(
        self,
        raw_counts: FloatArray,
        rng: np.random.Generator | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        """Return persistently completed counts and the added unresolved field."""

        values = self._validate_counts(raw_counts)
        if self.omitted_modes == 0:
            return values.copy(), np.zeros_like(values)

        generator = np.random.default_rng() if rng is None else rng
        completed = values.copy()
        additions = np.zeros_like(values)
        v_r = self.retained_basis.eigenvectors
        start = self.completion_start_step

        retained_states = values @ v_r
        initial = self._initial_step
        if initial is None:
            return completed, additions

        initial_mean = initial.omitted_mean + (
            initial.conditional_gain
            @ (retained_states[start] - initial.retained_mean)
        )
        unresolved_state = self._sample_gaussian(
            initial_mean,
            initial.conditional_factor,
            generator,
        )
        addition = self._spatial_addition(unresolved_state)
        additions[start] = addition
        completed[start] = values[start] + addition

        for time_index in range(start, self.config.n_steps - 1):
            parameters = self._transition_steps[time_index]
            if parameters is None:
                continue
            conditioning_state = np.concatenate(
                [
                    unresolved_state,
                    retained_states[time_index],
                    retained_states[time_index + 1],
                ]
            )
            conditional_mean = parameters.target_mean + (
                parameters.conditional_gain
                @ (conditioning_state - parameters.conditioning_mean)
            )
            unresolved_state = self._sample_gaussian(
                conditional_mean,
                parameters.conditional_factor,
                generator,
            )
            addition = self._spatial_addition(unresolved_state)
            additions[time_index + 1] = addition
            completed[time_index + 1] = values[time_index + 1] + addition

        return completed, additions
