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


def _modal_multinomial_covariance_block(
    probability: FloatArray,
    vectors: FloatArray,
    n_particles: int,
    row_modes: NDArray[np.integer],
    column_modes: NDArray[np.integer],
    *,
    modal_probability: FloatArray | None = None,
) -> FloatArray:
    """Return an exact same-time modal covariance block.

    This evaluates the multinomial identity directly for the requested modal
    index sets instead of constructing either the spatial covariance or the
    full modal covariance:

    ``N (V_I.T diag(p) V_J - (V_I.T p) (V_J.T p).T)``.
    """

    probability = np.asarray(probability, dtype=float)
    vectors = np.asarray(vectors, dtype=float)
    rows = np.asarray(row_modes, dtype=np.int64)
    columns = np.asarray(column_modes, dtype=np.int64)
    q = probability @ vectors if modal_probability is None else modal_probability
    left = vectors[:, rows]
    right = vectors[:, columns]
    return n_particles * (
        left.T @ (probability[:, None] * right)
        - np.outer(q[rows], q[columns])
    )


def _modal_multinomial_cross_covariance_block(
    probability: FloatArray,
    next_probability: FloatArray,
    vectors: FloatArray,
    one_step_factors: FloatArray,
    n_particles: int,
    row_modes: NDArray[np.integer],
    column_modes: NDArray[np.integer],
    *,
    modal_probability: FloatArray | None = None,
    next_modal_probability: FloatArray | None = None,
) -> FloatArray:
    """Return ``Cov(m_t[I], m_(t+1)[J])`` without a dense transition.

    The random-walk transition and diffusion operator share ``vectors``, so
    ``P V_J = V_J diag(1 - dt * lambda_J)``.  The requested block is therefore

    ``N (V_I.T diag(p_t) P V_J - q_t[I] q_(t+1)[J].T)``.
    """

    probability = np.asarray(probability, dtype=float)
    next_probability = np.asarray(next_probability, dtype=float)
    vectors = np.asarray(vectors, dtype=float)
    factors = np.asarray(one_step_factors, dtype=float)
    rows = np.asarray(row_modes, dtype=np.int64)
    columns = np.asarray(column_modes, dtype=np.int64)
    q_t = probability @ vectors if modal_probability is None else modal_probability
    q_next = (
        next_probability @ vectors
        if next_modal_probability is None
        else next_modal_probability
    )
    left = vectors[:, rows]
    propagated_right = vectors[:, columns] * factors[columns][None, :]
    return n_particles * (
        left.T @ (probability[:, None] * propagated_right)
        - np.outer(q_t[rows], q_next[columns])
    )


@dataclass(frozen=True, slots=True)
class CompletionFactorDiagnostics:
    """Captured-variance diagnostics for one conditional innovation factor."""

    selected_rank: int
    captured_variance_fraction: float
    tolerance_met: bool


@dataclass(frozen=True, slots=True)
class CompletionStep:
    """Precomputed same-time Gaussian conditional parameters."""

    retained_mean: FloatArray
    omitted_mean: FloatArray
    conditional_gain: FloatArray
    conditional_factor: FloatArray
    factor_diagnostics: CompletionFactorDiagnostics


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
    factor_diagnostics: CompletionFactorDiagnostics


@dataclass(frozen=True, slots=True)
class SparseRowTopology:
    """Shared compressed-row topology for a family of sparse gains."""

    indptr: NDArray[np.int64]
    indices: NDArray[np.int64]
    row_indices: NDArray[np.int64]
    n_columns: int

    @property
    def n_rows(self) -> int:
        return self.indptr.size - 1

    @property
    def counts(self) -> NDArray[np.int64]:
        return np.diff(self.indptr)

    def reduce_rows(self, values: FloatArray) -> FloatArray:
        """Sum packed values by CSR row without materializing padding."""

        values = np.asarray(values, dtype=float)
        if values.ndim == 0 or values.shape[0] != self.indices.size:
            raise ValueError("values must have one entry per sparse index")
        result = np.zeros((self.n_rows, *values.shape[1:]), dtype=float)
        nonempty_rows = np.flatnonzero(self.counts)
        if nonempty_rows.size:
            result[nonempty_rows] = np.add.reduceat(
                values,
                self.indptr[:-1][nonempty_rows],
                axis=0,
            )
        return result


@dataclass(frozen=True, slots=True)
class SparseRowGain:
    """CSR gain whose immutable topology can be shared across time."""

    topology: SparseRowTopology
    values: FloatArray

    @property
    def n_columns(self) -> int:
        return self.topology.n_columns

    @property
    def counts(self) -> NDArray[np.int64]:
        return self.topology.counts

    @property
    def indices(self) -> NDArray[np.int64]:
        return self.topology.indices

    def matvec(self, vector: FloatArray) -> FloatArray:
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (self.n_columns,):
            raise ValueError("vector has the wrong size for sparse gain")
        if self.values.size == 0:
            return np.zeros(self.topology.n_rows, dtype=float)
        return self.topology.reduce_rows(
            self.values * vector[self.indices],
        )

    def matmat(self, matrix: FloatArray) -> FloatArray:
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != self.n_columns:
            raise ValueError("matrix has the wrong leading size for sparse gain")
        return self.topology.reduce_rows(
            self.values[:, None] * matrix[self.indices]
        )

    def rmatvec(self, vector: FloatArray) -> FloatArray:
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (self.topology.n_rows,):
            raise ValueError("vector has the wrong size for transposed sparse gain")
        if self.values.size == 0:
            return np.zeros(self.n_columns, dtype=float)
        return np.bincount(
            self.indices,
            weights=self.values * vector[self.topology.row_indices],
            minlength=self.n_columns,
        )

    def to_dense(self) -> FloatArray:
        dense = np.zeros((self.topology.n_rows, self.n_columns), dtype=float)
        if self.values.size:
            np.add.at(
                dense,
                (self.topology.row_indices, self.indices),
                self.values,
            )
        return dense


@dataclass(frozen=True, slots=True)
class LocalCompletionStep:
    """Same-time conditional parameters with a row-sparse modal gain."""

    retained_mean: FloatArray
    omitted_mean: FloatArray
    conditional_gain: SparseRowGain
    conditional_factor: FloatArray
    factor_diagnostics: CompletionFactorDiagnostics


@dataclass(frozen=True, slots=True)
class LocalPersistentTransitionStep:
    """Persistent transition parameters with local modal conditioning."""

    conditioning_mean: FloatArray
    target_mean: FloatArray
    conditional_gain: SparseRowGain
    conditional_factor: FloatArray
    factor_diagnostics: CompletionFactorDiagnostics


class _GaussianCompletionBase:
    """Shared basis construction and stable Gaussian conditioning helpers."""

    def __init__(
        self,
        config: DiffusionConfig,
        *,
        retained_modes: int,
        completion_start_time: float = 0.0,
        completion_rank: int | None = None,
        residual_variance_tolerance: float | None = None,
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
        if residual_variance_tolerance is not None and not (
            0.0 <= residual_variance_tolerance < 1.0
        ):
            raise ValueError("residual_variance_tolerance must lie in [0, 1)")
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
        self.residual_variance_tolerance = residual_variance_tolerance

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

    def _low_rank_psd_factor_with_diagnostics(
        self,
        covariance: FloatArray,
    ) -> tuple[FloatArray, CompletionFactorDiagnostics]:
        if covariance.size == 0 or self.completion_rank == 0:
            return (
                np.zeros((self.omitted_modes, 0), dtype=float),
                CompletionFactorDiagnostics(
                    selected_rank=0,
                    captured_variance_fraction=1.0,
                    tolerance_met=True,
                ),
            )

        symmetric = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        scale = max(float(np.max(eigenvalues)), 1.0)
        positive = eigenvalues > self.eigenvalue_tolerance * scale
        positive_count = int(np.count_nonzero(positive))
        positive_variance = float(np.sum(eigenvalues[:positive_count]))
        if self.residual_variance_tolerance is None:
            requested = positive_count
        elif positive_count == 0:
            requested = 0
        else:
            target_fraction = 1.0 - self.residual_variance_tolerance
            cumulative_fraction = (
                np.cumsum(eigenvalues[:positive_count]) / positive_variance
            )
            requested = int(
                np.searchsorted(
                    cumulative_fraction,
                    target_fraction,
                    side="left",
                )
                + 1
            )
        keep = min(self.completion_rank, requested)
        if keep == 0:
            factor = np.zeros((self.omitted_modes, 0), dtype=float)
        else:
            factor = eigenvectors[:, :keep] * np.sqrt(eigenvalues[:keep])[None, :]
        captured_fraction = (
            1.0
            if positive_variance <= 0.0
            else float(np.sum(eigenvalues[:keep]) / positive_variance)
        )
        tolerance_met = (
            self.residual_variance_tolerance is None
            or 1.0 - captured_fraction
            <= self.residual_variance_tolerance + 10.0 * np.finfo(float).eps
        )
        return (
            factor,
            CompletionFactorDiagnostics(
                selected_rank=keep,
                captured_variance_fraction=captured_fraction,
                tolerance_met=tolerance_met,
            ),
        )

    def _low_rank_psd_factor(self, covariance: FloatArray) -> FloatArray:
        factor, _ = self._low_rank_psd_factor_with_diagnostics(covariance)
        return factor

    def factor_diagnostics(self) -> tuple[CompletionFactorDiagnostics, ...]:
        """Return innovation-factor diagnostics in saved-time order."""

        if hasattr(self, "_steps"):
            steps = getattr(self, "_steps")
            return tuple(
                step.factor_diagnostics for step in steps if step is not None
            )
        diagnostics: list[CompletionFactorDiagnostics] = []
        initial = getattr(self, "_initial_step", None)
        if initial is not None:
            diagnostics.append(initial.factor_diagnostics)
        transitions = getattr(self, "_transition_steps", ())
        diagnostics.extend(
            step.factor_diagnostics for step in transitions if step is not None
        )
        return tuple(diagnostics)

    def factor_diagnostics_summary(self) -> dict[str, int | float | bool]:
        """Summarize adaptive ranks and captured conditional variance."""

        diagnostics = self.factor_diagnostics()
        if not diagnostics:
            return {
                "factor_count": 0,
                "selected_rank_min": 0,
                "selected_rank_median": 0.0,
                "selected_rank_max": 0,
                "minimum_captured_variance_fraction": 1.0,
                "tolerance_met": True,
            }
        ranks = np.asarray(
            [diagnostic.selected_rank for diagnostic in diagnostics],
            dtype=np.int64,
        )
        return {
            "factor_count": len(diagnostics),
            "selected_rank_min": int(np.min(ranks)),
            "selected_rank_median": float(np.median(ranks)),
            "selected_rank_max": int(np.max(ranks)),
            "minimum_captured_variance_fraction": float(
                min(
                    diagnostic.captured_variance_fraction
                    for diagnostic in diagnostics
                )
            ),
            "tolerance_met": all(
                diagnostic.tolerance_met for diagnostic in diagnostics
            ),
        }

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
        factor, diagnostics = self._low_rank_psd_factor_with_diagnostics(
            conditional_covariance
        )
        return CompletionStep(
            retained_mean=retained_mean,
            omitted_mean=omitted_mean,
            conditional_gain=gain,
            conditional_factor=factor,
            factor_diagnostics=diagnostics,
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
        residual_variance_tolerance: float | None = None,
        ridge: float = 1.0e-2,
        eigenvalue_tolerance: float = 1.0e-12,
    ) -> None:
        super().__init__(
            config,
            retained_modes=retained_modes,
            completion_start_time=completion_start_time,
            completion_rank=completion_rank,
            residual_variance_tolerance=residual_variance_tolerance,
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
        residual_variance_tolerance: float | None = None,
        ridge: float = 1.0e-2,
        eigenvalue_tolerance: float = 1.0e-12,
    ) -> None:
        super().__init__(
            config,
            retained_modes=retained_modes,
            completion_start_time=completion_start_time,
            completion_rank=completion_rank,
            residual_variance_tolerance=residual_variance_tolerance,
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
            factor, diagnostics = self._low_rank_psd_factor_with_diagnostics(
                conditional_covariance
            )

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
                factor_diagnostics=diagnostics,
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


class LocalPersistentUnresolvedGaussianCompleter(_GaussianCompletionBase):
    """Persistent completion with band-limited conditioning in modal index.

    For target mode ``k``, only conditioning modes whose global modal index is
    within ``modal_bandwidth`` of ``k`` are used. All local row systems are
    padded and solved in one batch per transition, then stored in a row-sparse
    representation. This replaces the dense time-indexed conditional gains
    while retaining the same Gaussian persistent-state construction.
    """

    def __init__(
        self,
        config: DiffusionConfig,
        *,
        retained_modes: int,
        completion_start_time: float = 0.0,
        completion_rank: int | None = None,
        residual_variance_tolerance: float | None = None,
        modal_bandwidth: int = 12,
        ridge: float = 1.0e-2,
        eigenvalue_tolerance: float = 1.0e-12,
    ) -> None:
        if modal_bandwidth < 0:
            raise ValueError("modal_bandwidth must be nonnegative")
        self.modal_bandwidth = int(modal_bandwidth)
        super().__init__(
            config,
            retained_modes=retained_modes,
            completion_start_time=completion_start_time,
            completion_rank=completion_rank,
            residual_variance_tolerance=residual_variance_tolerance,
            ridge=ridge,
            eigenvalue_tolerance=eigenvalue_tolerance,
        )
        self._initial_step, self._transition_steps = self._precompute_local_steps()

    def _solve_sparse_gain(
        self,
        target_conditioning_covariance: FloatArray,
        conditioning_covariance: FloatArray,
        column_indices: list[NDArray[np.int64]],
        topology: SparseRowTopology | None = None,
    ) -> SparseRowGain:
        width = max((indices.size for indices in column_indices), default=0)
        indices_array = np.zeros((self.omitted_modes, width), dtype=np.int64)
        values = np.zeros((self.omitted_modes, width), dtype=float)
        counts = np.zeros(self.omitted_modes, dtype=np.int64)
        for row, indices in enumerate(column_indices):
            count = indices.size
            counts[row] = count
            if count == 0:
                continue
            indices_array[row, :count] = indices
        if width:
            active = np.arange(width)[None, :] < counts[:, None]
            covariance = conditioning_covariance[
                indices_array[:, :, None],
                indices_array[:, None, :],
            ]
            active_pairs = active[:, :, None] & active[:, None, :]
            covariance = np.where(active_pairs, covariance, 0.0)
            cross = target_conditioning_covariance[
                np.arange(self.omitted_modes)[:, None],
                indices_array,
            ]
            cross = np.where(active, cross, 0.0)
            active_count = np.maximum(counts, 1)
            scales = np.trace(covariance, axis1=1, axis2=2) / active_count
            diagonal = np.arange(width)
            covariance[:, diagonal, diagonal] += np.where(
                active,
                self.ridge * scales[:, None],
                1.0,
            )
            solved = np.linalg.solve(covariance, cross[..., None])[..., 0]
            values = np.where(active, solved, 0.0)
        if topology is None:
            topology = self._sparse_topology(
                column_indices,
                n_columns=conditioning_covariance.shape[0],
            )
        packed_indices = (
            indices_array[active] if width else np.empty(0, dtype=np.int64)
        )
        packed_values = values[active] if width else np.empty(0, dtype=float)
        if (
            topology.n_rows != self.omitted_modes
            or topology.n_columns != conditioning_covariance.shape[0]
            or not np.array_equal(packed_indices, topology.indices)
        ):
            raise ValueError("sparse topology does not match column_indices")
        return SparseRowGain(topology=topology, values=packed_values)

    @staticmethod
    def _sparse_topology(
        column_indices: list[NDArray[np.int64]],
        *,
        n_columns: int,
    ) -> SparseRowTopology:
        """Pack row-local column indices into an immutable CSR topology."""

        counts = np.asarray(
            [indices.size for indices in column_indices],
            dtype=np.int64,
        )
        indptr = np.empty(counts.size + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(counts, out=indptr[1:])
        nonempty = [indices for indices in column_indices if indices.size]
        indices = (
            np.concatenate(nonempty).astype(np.int64, copy=False)
            if nonempty
            else np.empty(0, dtype=np.int64)
        )
        if indices.size and (np.min(indices) < 0 or np.max(indices) >= n_columns):
            raise ValueError("sparse column index lies outside n_columns")
        row_indices = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
        return SparseRowTopology(
            indptr=indptr,
            indices=indices,
            row_indices=row_indices,
            n_columns=n_columns,
        )

    def _initial_columns(self) -> list[NDArray[np.int64]]:
        r = self.retained_modes
        bandwidth = self.modal_bandwidth
        return [
            np.flatnonzero(np.abs(np.arange(r) - (r + row)) <= bandwidth).astype(
                np.int64
            )
            for row in range(self.omitted_modes)
        ]

    def _transition_columns(self) -> list[NDArray[np.int64]]:
        r = self.retained_modes
        u = self.omitted_modes
        bandwidth = self.modal_bandwidth
        retained_indices = np.arange(r)
        omitted_indices = r + np.arange(u)
        rows: list[NDArray[np.int64]] = []
        for row in range(u):
            target_mode = r + row
            previous_u = np.flatnonzero(
                np.abs(omitted_indices - target_mode) <= bandwidth
            )
            nearby_r = np.flatnonzero(
                np.abs(retained_indices - target_mode) <= bandwidth
            )
            columns = np.concatenate(
                [previous_u, u + nearby_r, u + r + nearby_r]
            ).astype(np.int64)
            rows.append(columns)
        return rows

    @staticmethod
    def _active_columns(
        column_indices: list[NDArray[np.int64]],
    ) -> tuple[NDArray[np.int64], list[NDArray[np.int64]]]:
        """Return the union of row-local columns and compact row indices."""

        nonempty = [indices for indices in column_indices if indices.size]
        if not nonempty:
            empty = np.empty(0, dtype=np.int64)
            return empty, [empty.copy() for _ in column_indices]
        active = np.unique(np.concatenate(nonempty)).astype(np.int64)
        compact = [np.searchsorted(active, indices) for indices in column_indices]
        return active, compact

    def _solve_restricted_gain(
        self,
        conditioning_to_target_covariance: FloatArray,
        conditioning_covariance: FloatArray,
        compact_column_indices: list[NDArray[np.int64]],
        active_columns: NDArray[np.int64],
        *,
        n_columns: int,
        compact_topology: SparseRowTopology | None = None,
        stored_topology: SparseRowTopology | None = None,
    ) -> tuple[SparseRowGain, FloatArray]:
        """Solve local rows and return their Schur-complement correction.

        The covariance inputs live only on the union of columns used by at
        least one local row.  Stored indices are mapped back to the full
        conditioning vector so trajectory evaluation is unchanged.
        """

        compact_gain = self._solve_sparse_gain(
            conditioning_to_target_covariance.T,
            conditioning_covariance,
            compact_column_indices,
            topology=compact_topology,
        )
        if compact_gain.values.size == 0:
            correction = np.zeros(
                (self.omitted_modes, self.omitted_modes),
                dtype=float,
            )
        else:
            counts = compact_gain.counts
            width = int(np.max(counts))
            padded_indices = np.zeros(
                (self.omitted_modes, width),
                dtype=np.int64,
            )
            active = np.arange(width)[None, :] < counts[:, None]
            padded_indices[active] = compact_gain.indices
            padded_values = np.zeros_like(padded_indices, dtype=float)
            padded_values[active] = compact_gain.values
            selected_covariance = conditioning_to_target_covariance[
                padded_indices
            ]
            correction = np.einsum(
                "rw,rwu->ru",
                padded_values,
                selected_covariance,
                optimize=True,
            )
        if stored_topology is None:
            stored_columns = [
                active_columns[indices] for indices in compact_column_indices
            ]
            stored_topology = self._sparse_topology(
                stored_columns,
                n_columns=n_columns,
            )
        gain = SparseRowGain(
            topology=stored_topology,
            values=compact_gain.values,
        )
        return gain, correction

    def _transition_restricted_covariances(
        self,
        current_covariance: FloatArray,
        next_covariance: FloatArray,
        cross_covariance: FloatArray,
        active_columns: NDArray[np.int64],
        mode_lookup: NDArray[np.int64],
        target_indices: NDArray[np.int64],
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Assemble only the active conditioning blocks for one transition."""

        r = self.retained_modes
        u = self.omitted_modes
        active_size = active_columns.size
        conditioning_covariance = np.zeros(
            (active_size, active_size),
            dtype=float,
        )
        conditioning_to_target = np.zeros(
            (active_size, u),
            dtype=float,
        )

        current_positions = np.flatnonzero(active_columns < u + r)
        next_positions = np.flatnonzero(active_columns >= u + r)
        current_columns = active_columns[current_positions]
        current_modes = np.where(
            current_columns < u,
            r + current_columns,
            current_columns - u,
        ).astype(np.int64)
        next_modes = (active_columns[next_positions] - u - r).astype(np.int64)
        current_indices = mode_lookup[current_modes]
        next_indices = mode_lookup[next_modes]

        if current_positions.size:
            conditioning_covariance[np.ix_(current_positions, current_positions)] = (
                current_covariance[np.ix_(current_indices, current_indices)]
            )
            conditioning_to_target[current_positions] = (
                cross_covariance[np.ix_(current_indices, target_indices)]
            )
        if next_positions.size:
            conditioning_covariance[np.ix_(next_positions, next_positions)] = (
                next_covariance[np.ix_(next_indices, next_indices)]
            )
            conditioning_to_target[next_positions] = (
                next_covariance[np.ix_(next_indices, target_indices)]
            )
        if current_positions.size and next_positions.size:
            current_next = cross_covariance[
                np.ix_(current_indices, next_indices)
            ]
            conditioning_covariance[np.ix_(current_positions, next_positions)] = (
                current_next
            )
            conditioning_covariance[np.ix_(next_positions, current_positions)] = (
                current_next.T
            )

        conditioning_covariance = 0.5 * (
            conditioning_covariance + conditioning_covariance.T
        )
        target_covariance = next_covariance[
            np.ix_(target_indices, target_indices)
        ]
        target_covariance = 0.5 * (target_covariance + target_covariance.T)
        return (
            conditioning_covariance,
            conditioning_to_target,
            target_covariance,
        )

    def _restricted_covariance_chunk(
        self,
        probabilities: FloatArray,
        modal_probabilities: FloatArray,
        mode_indices: NDArray[np.int64],
        one_step_factors: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        """Return same-time and adjacent-time blocks for a short time chunk."""

        selected_vectors = self.full_basis.eigenvectors[:, mode_indices]
        weighted_gram = np.einsum(
            "xi,tx,xj->tij",
            selected_vectors,
            probabilities,
            selected_vectors,
            optimize=True,
        )
        q = modal_probabilities[:, mode_indices]
        same_time = self.config.n_particles * (
            weighted_gram - q[:, :, None] * q[:, None, :]
        )
        same_time = 0.5 * (same_time + np.swapaxes(same_time, 1, 2))
        cross_time = self.config.n_particles * (
            weighted_gram[:-1] * one_step_factors[mode_indices][None, None, :]
            - q[:-1, :, None] * q[1:, None, :]
        )
        return same_time, cross_time

    def _precompute_local_steps(
        self,
    ) -> tuple[LocalCompletionStep | None, list[LocalPersistentTransitionStep | None]]:
        if self.omitted_modes == 0:
            return None, [None] * max(self.config.n_steps - 1, 0)

        probabilities = multinomial_node_probabilities(self.config)
        vectors = self.full_basis.eigenvectors
        modal_probabilities = probabilities @ vectors
        modal_means = self.config.n_particles * modal_probabilities
        one_step_factors = 1.0 - self.config.dt * self.full_basis.eigenvalues
        start = self.completion_start_step
        r = self.retained_modes
        u = self.omitted_modes
        target_modes = np.arange(r, self.config.n_nodes, dtype=np.int64)

        initial_columns = self._initial_columns()
        initial_active, initial_compact = self._active_columns(initial_columns)
        initial_conditioning_covariance = _modal_multinomial_covariance_block(
            probabilities[start],
            vectors,
            self.config.n_particles,
            initial_active,
            initial_active,
            modal_probability=modal_probabilities[start],
        )
        initial_conditioning_covariance = 0.5 * (
            initial_conditioning_covariance
            + initial_conditioning_covariance.T
        )
        initial_conditioning_to_target = _modal_multinomial_covariance_block(
            probabilities[start],
            vectors,
            self.config.n_particles,
            initial_active,
            target_modes,
            modal_probability=modal_probabilities[start],
        )
        initial_target_covariance = _modal_multinomial_covariance_block(
            probabilities[start],
            vectors,
            self.config.n_particles,
            target_modes,
            target_modes,
            modal_probability=modal_probabilities[start],
        )
        initial_target_covariance = 0.5 * (
            initial_target_covariance + initial_target_covariance.T
        )
        initial_gain, initial_correction = self._solve_restricted_gain(
            initial_conditioning_to_target,
            initial_conditioning_covariance,
            initial_compact,
            initial_active,
            n_columns=r,
        )
        initial_covariance = initial_target_covariance - initial_correction
        initial_factor, initial_diagnostics = (
            self._low_rank_psd_factor_with_diagnostics(initial_covariance)
        )
        initial = LocalCompletionStep(
            retained_mean=modal_means[start, :r],
            omitted_mean=modal_means[start, r:],
            conditional_gain=initial_gain,
            conditional_factor=initial_factor,
            factor_diagnostics=initial_diagnostics,
        )

        transitions: list[LocalPersistentTransitionStep | None] = [
            None
        ] * (self.config.n_steps - 1)
        local_columns = self._transition_columns()
        active_columns, compact_columns = self._active_columns(local_columns)
        compact_topology = self._sparse_topology(
            compact_columns,
            n_columns=active_columns.size,
        )
        stored_topology = self._sparse_topology(
            [active_columns[indices] for indices in compact_columns],
            n_columns=u + 2 * r,
        )
        current_columns = active_columns[active_columns < u + r]
        current_modes = np.where(
            current_columns < u,
            r + current_columns,
            current_columns - u,
        ).astype(np.int64)
        next_modes = (
            active_columns[active_columns >= u + r] - u - r
        ).astype(np.int64)
        covariance_modes = np.unique(
            np.concatenate([target_modes, current_modes, next_modes])
        ).astype(np.int64)
        mode_lookup = np.full(self.config.n_nodes, -1, dtype=np.int64)
        mode_lookup[covariance_modes] = np.arange(covariance_modes.size)
        target_indices = mode_lookup[target_modes]

        # Chunking keeps temporary covariance storage bounded while retaining
        # efficient batched contractions over time.
        chunk_size = 64
        for chunk_start in range(start, self.config.n_steps - 1, chunk_size):
            chunk_stop = min(
                chunk_start + chunk_size,
                self.config.n_steps - 1,
            )
            same_time, cross_time = self._restricted_covariance_chunk(
                probabilities[chunk_start : chunk_stop + 1],
                modal_probabilities[chunk_start : chunk_stop + 1],
                covariance_modes,
                one_step_factors,
            )
            for offset, time_index in enumerate(range(chunk_start, chunk_stop)):
                (
                    conditioning_covariance,
                    conditioning_to_target,
                    target_covariance,
                ) = self._transition_restricted_covariances(
                    same_time[offset],
                    same_time[offset + 1],
                    cross_time[offset],
                    active_columns,
                    mode_lookup,
                    target_indices,
                )
                gain, correction = self._solve_restricted_gain(
                    conditioning_to_target,
                    conditioning_covariance,
                    compact_columns,
                    active_columns,
                    n_columns=u + 2 * r,
                    compact_topology=compact_topology,
                    stored_topology=stored_topology,
                )
                conditional_covariance = target_covariance - correction
                conditioning_mean = np.concatenate(
                    [
                        modal_means[time_index, r:],
                        modal_means[time_index, :r],
                        modal_means[time_index + 1, :r],
                    ]
                )
                factor, diagnostics = self._low_rank_psd_factor_with_diagnostics(
                    conditional_covariance
                )
                transitions[time_index] = LocalPersistentTransitionStep(
                    conditioning_mean=conditioning_mean,
                    target_mean=modal_means[time_index + 1, r:],
                    conditional_gain=gain,
                    conditional_factor=factor,
                    factor_diagnostics=diagnostics,
                )
        return initial, transitions

    def complete(
        self,
        raw_counts: FloatArray,
        rng: np.random.Generator | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        """Return locally conditioned persistent completion."""

        values = self._validate_counts(raw_counts)
        if self.omitted_modes == 0:
            return values.copy(), np.zeros_like(values)
        generator = np.random.default_rng() if rng is None else rng
        completed = values.copy()
        additions = np.zeros_like(values)
        retained_states = values @ self.retained_basis.eigenvectors
        start = self.completion_start_step
        initial = self._initial_step
        if initial is None:
            return completed, additions

        unresolved_state = self._sample_gaussian(
            initial.omitted_mean
            + initial.conditional_gain.matvec(
                retained_states[start] - initial.retained_mean
            ),
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
            unresolved_state = self._sample_gaussian(
                parameters.target_mean
                + parameters.conditional_gain.matvec(
                    conditioning_state - parameters.conditioning_mean
                ),
                parameters.conditional_factor,
                generator,
            )
            addition = self._spatial_addition(unresolved_state)
            additions[time_index + 1] = addition
            completed[time_index + 1] = values[time_index + 1] + addition
        return completed, additions
