"""Innovation-conditioned completion of retained direct-modal trajectories.

For a full modal innovation covariance ``Q_t``, split the innovation into
retained and unresolved coordinates, ``eta = [eta_R, eta_U]``.  This module
uses the exact Gaussian conditional transition

``eta_U | eta_R ~ N(Q_UR Q_RR^+ eta_R,
                     Q_UU - Q_UR Q_RR^+ Q_RU)``.

All covariance blocks come from the same full-rank direct-modal factor.
This is important: singular PSD cleanup performed independently at two ranks
need not produce exactly nested numerical factors.  The constant mass mode is
retained, deterministic, and handled by the pseudoinverse null space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import DiffusionConfig
from .direct_modal import (
    DirectModalGaussianDiffusion,
    _cosine_basis,
    _innovation_factor,
    _modal_covariance,
    _modal_transition_diagonal,
)
from .operators import Eigenbasis
from .references import random_walk_transition_matrix

FloatArray = NDArray[np.float64]


def _completion_ranks(
    n_nodes: int,
    retained_modes: int,
    conditional_rank: int | None,
    pseudoinverse_tolerance: float,
    psd_tolerance: float,
) -> tuple[int, int]:
    if not 1 <= retained_modes <= n_nodes:
        raise ValueError("retained_modes must lie in [1, n_nodes]")
    if conditional_rank is not None and conditional_rank <= 0:
        raise ValueError("conditional_rank must be positive when provided")
    if pseudoinverse_tolerance < 0.0:
        raise ValueError("pseudoinverse_tolerance must be nonnegative")
    if psd_tolerance <= 0.0:
        raise ValueError("psd_tolerance must be positive")
    retained = int(retained_modes)
    unresolved = n_nodes - retained
    rank = unresolved if conditional_rank is None else min(int(conditional_rank), unresolved)
    return retained, rank


def _split_innovation_factor(
    joint_factor: FloatArray, retained: int, pseudoinverse_tolerance: float
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Split Q = F F.T into retained factor, conditional gain and residual factor.

    For retained rows A and unresolved rows B, the SVD A = U S V.T
    separates observed and unobserved latent normals. G = B V_r S_r^-1 U_r.T
    is the conditional gain, and B V_null factors the Schur complement
    without subtracting nearly equal covariances at rank-deficient steps.
    """

    retained_rows = joint_factor[:retained]
    unresolved_rows = joint_factor[retained:]
    left, singular_values, right_transpose = np.linalg.svd(retained_rows, full_matrices=True)
    maximum = float(singular_values[0])
    active = (
        singular_values > np.sqrt(pseudoinverse_tolerance) * maximum
        if maximum > 0.0
        else np.zeros_like(singular_values, dtype=bool)
    )
    active_count = int(np.count_nonzero(active))
    retained_factor = np.zeros((retained, retained), dtype=float)
    if active_count:
        retained_factor[:, :active_count] = left[:, active] * singular_values[active][None, :]
        observed_right = right_transpose[:retained][active].T
        gain = (
            unresolved_rows @ observed_right / singular_values[active][None, :]
        ) @ left[:, active].T
    else:
        gain = np.zeros((len(unresolved_rows), retained), dtype=float)

    right_active = np.zeros(joint_factor.shape[1], dtype=bool)
    right_active[:retained] = active
    null_right = right_transpose[~right_active].T
    return retained_factor, gain, unresolved_rows @ null_right


def _truncate_residual_factor(
    residual_factor: FloatArray, selected: int
) -> tuple[FloatArray, FloatArray]:
    vectors, singular_values, _ = np.linalg.svd(residual_factor, full_matrices=False)
    available = min(selected, singular_values.size)
    conditional_factor = np.zeros((len(residual_factor), selected), dtype=float)
    conditional_factor[:, :available] = vectors[:, :available] * singular_values[:available]
    return conditional_factor, singular_values


def _populate_joint_factor(
    factor: FloatArray,
    retained_factor: FloatArray,
    gain: FloatArray,
    conditional_factor: FloatArray,
) -> None:
    """Fill a zero-initialized retained-plus-conditional factor in draw order."""

    retained = len(retained_factor)
    factor[:retained, :retained] = retained_factor
    factor[retained:, :retained] = gain @ retained_factor
    factor[retained:, retained:] = conditional_factor


def _pack_conditioned_innovation(
    packed: FloatArray,
    full_factor: FloatArray,
    retained: int,
    rank: int,
    pseudoinverse_tolerance: float,
) -> None:
    retained_factor, gain, residual = _split_innovation_factor(
        full_factor, retained, pseudoinverse_tolerance
    )
    conditional_factor, _ = _truncate_residual_factor(residual, rank)
    _populate_joint_factor(packed, retained_factor, gain, conditional_factor)


@dataclass(frozen=True, slots=True)
class InnovationConditioningStep:
    """Cached Gaussian conditioning data for one saved-step transition."""

    retained_covariance: FloatArray
    retained_factor: FloatArray
    conditional_gain: FloatArray
    conditional_covariance: FloatArray
    conditional_factor: FloatArray
    selected_rank: int
    captured_variance_fraction: float

    @property
    def sampled_conditional_covariance(self) -> FloatArray:
        """Covariance represented by the possibly rank-capped factor."""

        return self.conditional_factor @ self.conditional_factor.T


@dataclass(frozen=True, slots=True)
class PreparedInnovationSampler:
    """Joint innovation factors and draw layout for all saved-step transitions."""

    joint_factors: FloatArray
    selected_ranks: NDArray[np.intp]
    latent_count: int
    _active_columns: NDArray[np.bool_] | None

    @property
    def nbytes(self) -> int:
        """Additional array storage used by the prepared sampler."""

        mask_bytes = 0 if self._active_columns is None else self._active_columns.nbytes
        return self.joint_factors.nbytes + self.selected_ranks.nbytes + mask_bytes

    def draw_innovations(self, rng: np.random.Generator) -> FloatArray:
        """Draw all joint innovations in stepwise retained-then-residual order."""

        transitions, _, width = self.joint_factors.shape
        draws = rng.standard_normal(self.latent_count)
        if self._active_columns is None:
            latent = draws.reshape(transitions, width)
        else:
            latent = np.zeros((transitions, width), dtype=float)
            latent[self._active_columns] = draws
        return np.einsum("tij,tj->ti", self.joint_factors, latent, optimize=False)

    def _draw_modal_path(
        self,
        initial_modal: FloatArray,
        one_step_factors: FloatArray,
        rng: np.random.Generator | None,
    ) -> FloatArray:
        generator = np.random.default_rng() if rng is None else rng
        innovations = self.draw_innovations(generator)
        modal = np.empty((len(innovations) + 1, initial_modal.size), dtype=float)
        modal[0] = initial_modal
        for time_index, innovation in enumerate(innovations):
            modal[time_index + 1] = one_step_factors * modal[time_index] + innovation
            modal[time_index + 1, 0] = initial_modal[0]
        return modal


@dataclass(frozen=True, slots=True)
class InnovationConditionedCompletionResult:
    """A completed full-modal and full-spatial Gaussian trajectory."""

    modal_amplitudes: FloatArray
    spatial_counts: FloatArray
    basis: Eigenbasis
    retained_modes: int
    model_name: str = "innovation_conditioned_modal_completion"

    @property
    def retained_modal_amplitudes(self) -> FloatArray:
        """Retained coordinates supplied to, or generated by, the completer."""

        return self.modal_amplitudes[:, : self.retained_modes]

    @property
    def unresolved_modal_amplitudes(self) -> FloatArray:
        """Conditionally generated unresolved coordinates."""

        return self.modal_amplitudes[:, self.retained_modes :]

    @classmethod
    def _from_modal(
        cls, modal: FloatArray, basis: Eigenbasis, retained_modes: int
    ) -> InnovationConditionedCompletionResult:
        return cls(
            modal_amplitudes=modal,
            spatial_counts=modal @ basis.eigenvectors.T,
            basis=basis,
            retained_modes=retained_modes,
        )


@dataclass(frozen=True, slots=True)
class PreparedInnovationCompletion:
    """Runtime state for drawing full trajectories after conditioning setup."""

    initial_modal: FloatArray
    one_step_factors: FloatArray
    basis: Eigenbasis
    retained_modes: int
    sampler: PreparedInnovationSampler

    @staticmethod
    def from_config(
        config: DiffusionConfig,
        *,
        retained_modes: int,
        conditional_rank: int | None = None,
        pseudoinverse_tolerance: float = 1.0e-12,
        psd_tolerance: float = 1.0e-10,
    ) -> PreparedInnovationCompletion:
        """Pack one transition at a time with O(n_nodes**2) setup scratch."""

        retained, rank = _completion_ranks(
            config.n_nodes, retained_modes, conditional_rank, pseudoinverse_tolerance, psd_tolerance
        )
        pseudoinverse_tolerance = float(pseudoinverse_tolerance)
        psd_tolerance = float(psd_tolerance)
        basis = _cosine_basis(config, config.n_nodes)
        transition = random_walk_transition_matrix(config)
        diagonal = _modal_transition_diagonal(config, basis)
        particle_count = float(config.n_particles)
        raw_probability = np.zeros(config.n_nodes, dtype=float)
        raw_probability[config.impulse_index] = 1.0
        initial_modal = particle_count * basis.eigenvectors[config.impulse_index]
        previous = np.zeros((config.n_nodes, config.n_nodes), dtype=float)
        width = retained + rank
        joint = np.zeros((config.n_steps - 1, config.n_nodes, width), dtype=float)
        for time_index, packed in enumerate(joint):
            # The reference advances the raw walk for all times before it clips
            # and normalizes rows. Keep the normalized row out of the recurrence.
            raw_probability = raw_probability @ transition
            probability = np.clip(raw_probability, 0.0, 1.0)
            probability /= probability.sum()
            following = _modal_covariance(probability, basis.eigenvectors, particle_count)
            _pack_conditioned_innovation(
                packed,
                _innovation_factor(previous, following, diagonal, psd_tolerance, time_index),
                retained,
                rank,
                pseudoinverse_tolerance,
            )
            previous = following

        ranks = np.full(len(joint), rank, dtype=np.intp)
        for array in (joint, ranks, initial_modal, diagonal, basis.eigenvectors, basis.eigenvalues):
            array.setflags(write=False)
        return PreparedInnovationCompletion(
            initial_modal=initial_modal,
            one_step_factors=diagonal,
            basis=basis,
            retained_modes=retained,
            sampler=PreparedInnovationSampler(
                joint_factors=joint,
                selected_ranks=ranks,
                latent_count=len(joint) * width,
                _active_columns=None,
            ),
        )

    @staticmethod
    def from_full_history_config(
        config: DiffusionConfig,
        *,
        retained_modes: int,
        conditional_rank: int | None = None,
        pseudoinverse_tolerance: float = 1.0e-12,
        psd_tolerance: float = 1.0e-10,
    ) -> PreparedInnovationCompletion:
        """Prepare a reference runtime using the full conditioning histories."""

        builder = InnovationConditionedModalCompleter(
            config,
            retained_modes=retained_modes,
            conditional_rank=conditional_rank,
            pseudoinverse_tolerance=pseudoinverse_tolerance,
            psd_tolerance=psd_tolerance,
        )
        return builder.prepare_runtime()

    @property
    def nbytes(self) -> int:
        """Resident NumPy storage, including factors and their draw layout."""

        return (
            self.initial_modal.nbytes
            + self.one_step_factors.nbytes
            + self.basis.eigenvalues.nbytes
            + self.basis.eigenvectors.nbytes
            + self.sampler.nbytes
        )

    def run(
        self, rng: np.random.Generator | None = None
    ) -> InnovationConditionedCompletionResult:
        """Draw a trajectory with the same latent order as the batched completer."""

        modal = self.sampler._draw_modal_path(self.initial_modal, self.one_step_factors, rng)
        return InnovationConditionedCompletionResult._from_modal(
            modal, self.basis, self.retained_modes
        )


class InnovationConditionedModalCompleter:
    """Complete a retained direct-modal path one innovation at a time.

    Parameters
    ----------
    config:
        Configuration of the deterministic-impulse reflecting walk.
    retained_modes:
        Number of leading cosine modes supplied by the reduced process.  The
        constant mass mode is always included.
    conditional_rank:
        Optional cap on the residual unresolved conditional-noise rank.  The
        conditional mean is never truncated.  ``None`` uses all unresolved
        directions.
    pseudoinverse_tolerance:
        Relative eigenvalue threshold for ``Q_RR^+``.
    psd_tolerance:
        Relative tolerance for rejecting materially indefinite covariance
        matrices before clipping round-off-sized negative eigenvalues.
    """

    def __init__(
        self,
        config: DiffusionConfig,
        *,
        retained_modes: int,
        conditional_rank: int | None = None,
        pseudoinverse_tolerance: float = 1.0e-12,
        psd_tolerance: float = 1.0e-10,
    ) -> None:
        self.config = config
        self.retained_modes, self.conditional_rank = _completion_ranks(
            config.n_nodes, retained_modes, conditional_rank, pseudoinverse_tolerance, psd_tolerance
        )
        self.unresolved_modes = config.n_nodes - self.retained_modes
        self.pseudoinverse_tolerance = float(pseudoinverse_tolerance)
        self.psd_tolerance = float(psd_tolerance)

        self.full_reference = DirectModalGaussianDiffusion(
            config,
            n_modes=config.n_nodes,
            psd_tolerance=psd_tolerance,
        )
        self.basis = self.full_reference.basis
        self._one_step_factors = self.full_reference.one_step_factors
        self._initial_modal = self.full_reference.modal_means[0].copy()
        self._target_innovation_covariances = (
            self.full_reference.innovation_covariances
        )
        self._conditioning_steps = tuple(
            self._prepare_step(joint_factor)
            for joint_factor in self.full_reference._innovation_factors
        )
        self._batched_sampler: PreparedInnovationSampler | None = None

    @property
    def conditioning_steps(self) -> tuple[InnovationConditioningStep, ...]:
        """Cached transition conditioning data."""

        return self._conditioning_steps

    @property
    def one_step_factors(self) -> FloatArray:
        """Full modal transition diagonal."""

        return self._one_step_factors.copy()

    @property
    def target_innovation_covariances(self) -> FloatArray:
        """Full-rank reference ``Q_t`` matrices used to form every block."""

        return self._target_innovation_covariances.copy()

    @property
    def retained_innovation_covariances(self) -> FloatArray:
        """Cached ``Q_RR`` matrices."""

        return np.stack(
            [step.retained_covariance for step in self._conditioning_steps]
        )

    @property
    def retained_innovation_factors(self) -> FloatArray:
        """PSD-safe square roots of the retained innovation covariances."""

        return np.stack([step.retained_factor for step in self._conditioning_steps])

    @property
    def conditional_gains(self) -> FloatArray:
        """Cached gains ``Q_UR Q_RR^+``."""

        return np.stack([step.conditional_gain for step in self._conditioning_steps])

    @property
    def conditional_covariances(self) -> FloatArray:
        """Full Schur-complement conditional covariances."""

        return np.stack(
            [step.conditional_covariance for step in self._conditioning_steps]
        )

    @property
    def conditional_factors(self) -> FloatArray:
        """Possibly rank-capped conditional covariance factors."""

        return np.stack([step.conditional_factor for step in self._conditioning_steps])

    @property
    def captured_variance_fractions(self) -> FloatArray:
        """Fraction of conditional variance retained at each transition."""

        return np.asarray(
            [step.captured_variance_fraction for step in self._conditioning_steps],
            dtype=float,
        )

    def effective_innovation_covariances(self) -> FloatArray:
        """Materialize covariances generated by the cached sampling factors.

        With full conditional rank these reproduce
        :attr:`target_innovation_covariances` up to eigensolver round-off.  With
        a rank cap they expose exactly which residual variance is omitted.
        Covariances are formed from the joint factor directly; expanding the
        gain through ``G (R R.T) G.T`` is numerically unstable when ``G`` is
        large and ``R`` is rank deficient.
        """

        result = np.empty_like(self._target_innovation_covariances)
        for time_index, step in enumerate(self._conditioning_steps):
            joint_factor = np.zeros(
                (
                    self.config.n_nodes,
                    self.retained_modes + step.selected_rank,
                ),
                dtype=float,
            )
            self._populate_joint_factor(joint_factor, step)
            result[time_index] = joint_factor @ joint_factor.T
        return result

    def run(
        self,
        rng: np.random.Generator | None = None,
    ) -> InnovationConditionedCompletionResult:
        """Sample a completed trajectory using prepared batched innovations.

        Joint factors are prepared lazily on the first call and reused for
        later trajectories.  :meth:`run_stepwise` remains the scalar-order
        reference implementation.
        """

        return self.run_batched(rng)

    def prepare_batched_sampler(self) -> PreparedInnovationSampler:
        """Cache joint factors for repeated trajectory draws."""

        if self._batched_sampler is None:
            retained = self.retained_modes
            steps = self._conditioning_steps
            ranks = np.array([step.selected_rank for step in steps], dtype=np.intp)
            widths = retained + ranks
            max_width = int(widths.max())
            joint = np.zeros((len(steps), self.config.n_nodes, max_width), dtype=float)
            for factor, step, width in zip(joint, steps, widths):
                self._populate_joint_factor(factor[:, :width], step)

            active_columns = None
            if np.any(widths != max_width):
                active_columns = np.arange(max_width) < widths[:, None]
                active_columns.setflags(write=False)
            joint.setflags(write=False)
            ranks.setflags(write=False)
            self._batched_sampler = PreparedInnovationSampler(
                joint_factors=joint,
                selected_ranks=ranks,
                latent_count=int(widths.sum()),
                _active_columns=active_columns,
            )
        return self._batched_sampler

    def prepare_runtime(self) -> PreparedInnovationCompletion:
        """Export standalone sampling state and share the read-only packed sampler."""

        def frozen_copy(array: FloatArray) -> FloatArray:
            owned = array.copy(order="K")
            owned.setflags(write=False)
            return owned

        return PreparedInnovationCompletion(
            initial_modal=frozen_copy(self._initial_modal),
            one_step_factors=frozen_copy(self._one_step_factors),
            basis=Eigenbasis(
                eigenvalues=frozen_copy(self.basis.eigenvalues),
                eigenvectors=frozen_copy(self.basis.eigenvectors),
            ),
            retained_modes=self.retained_modes,
            sampler=self.prepare_batched_sampler(),
        )

    def run_batched(
        self,
        rng: np.random.Generator | None = None,
    ) -> InnovationConditionedCompletionResult:
        """Draw all innovations together and apply the diagonal state recurrence."""

        modal = self.prepare_batched_sampler()._draw_modal_path(
            self._initial_modal, self._one_step_factors, rng
        )
        return InnovationConditionedCompletionResult._from_modal(
            modal, self.basis, self.retained_modes
        )

    def _populate_joint_factor(
        self,
        factor: FloatArray,
        step: InnovationConditioningStep,
    ) -> None:
        """Fill one retained-plus-conditional innovation factor in place."""

        _populate_joint_factor(
            factor, step.retained_factor, step.conditional_gain, step.conditional_factor
        )

    def run_stepwise(
        self,
        rng: np.random.Generator | None = None,
    ) -> InnovationConditionedCompletionResult:
        """Generate retained innovations and their conditional completion.

        Retained and conditional latent normals are drawn in a fixed order at
        each transition.  Full conditional rank therefore samples the same
        Gaussian process law as the full direct-modal reference, though not the
        same bitwise path because its square-root factorization differs.
        """

        generator = np.random.default_rng() if rng is None else rng
        modal = np.empty(
            (self.config.n_steps, self.config.n_nodes),
            dtype=float,
        )
        modal[0] = self._initial_modal
        retained = self.retained_modes
        retained_factors = self._one_step_factors[:retained]
        unresolved_factors = self._one_step_factors[retained:]
        conserved_mass_mode = float(modal[0, 0])

        for time_index, step in enumerate(self._conditioning_steps):
            retained_noise = step.retained_factor @ generator.standard_normal(
                self.retained_modes
            )
            conditional_noise = step.conditional_factor @ generator.standard_normal(
                step.selected_rank
            )
            unresolved_noise = (
                step.conditional_gain @ retained_noise + conditional_noise
            )
            modal[time_index + 1, :retained] = (
                retained_factors * modal[time_index, :retained]
                + retained_noise
            )
            modal[time_index + 1, retained:] = (
                unresolved_factors * modal[time_index, retained:]
                + unresolved_noise
            )
            modal[time_index + 1, 0] = conserved_mass_mode

        return InnovationConditionedCompletionResult._from_modal(
            modal, self.basis, self.retained_modes
        )

    def complete(
        self,
        retained_modal_amplitudes: FloatArray,
        rng: np.random.Generator | None = None,
    ) -> InnovationConditionedCompletionResult:
        """Conditionally complete a supplied retained modal trajectory."""

        retained_path = np.asarray(retained_modal_amplitudes, dtype=float)
        expected_shape = (self.config.n_steps, self.retained_modes)
        if retained_path.shape != expected_shape:
            raise ValueError(
                "retained_modal_amplitudes must have shape "
                f"{expected_shape}, got {retained_path.shape}"
            )
        if not np.all(np.isfinite(retained_path)):
            raise ValueError("retained_modal_amplitudes must be finite")
        initial_retained = self._initial_modal[: self.retained_modes]
        if not np.allclose(
            retained_path[0],
            initial_retained,
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise ValueError("retained trajectory must begin at the projected impulse")
        if not np.allclose(
            retained_path[:, 0],
            self._initial_modal[0],
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise ValueError("retained trajectory must preserve the constant mass mode")

        generator = np.random.default_rng() if rng is None else rng
        retained = self.retained_modes
        modal = np.empty(
            (self.config.n_steps, self.config.n_nodes),
            dtype=float,
        )
        modal[:, :retained] = retained_path
        modal[0, retained:] = self._initial_modal[retained:]
        retained_factors = self._one_step_factors[:retained]
        unresolved_factors = self._one_step_factors[retained:]

        for time_index, step in enumerate(self._conditioning_steps):
            retained_noise = (
                retained_path[time_index + 1]
                - retained_factors * retained_path[time_index]
            )
            conditional_noise = step.conditional_factor @ generator.standard_normal(
                step.selected_rank
            )
            unresolved_noise = (
                step.conditional_gain @ retained_noise + conditional_noise
            )
            modal[time_index + 1, retained:] = (
                unresolved_factors * modal[time_index, retained:]
                + unresolved_noise
            )

        return InnovationConditionedCompletionResult._from_modal(
            modal, self.basis, self.retained_modes
        )

    def _prepare_step(
        self,
        joint_factor: FloatArray,
    ) -> InnovationConditioningStep:
        retained_factor, gain, residual_factor = _split_innovation_factor(
            joint_factor, self.retained_modes, self.pseudoinverse_tolerance
        )
        selected = self.conditional_rank
        conditional_factor, conditional_singular_values = _truncate_residual_factor(
            residual_factor, selected
        )
        retained_covariance = retained_factor @ retained_factor.T
        conditional_covariance = residual_factor @ residual_factor.T
        conditional_covariance = 0.5 * (conditional_covariance + conditional_covariance.T)
        conditional_variances = conditional_singular_values**2
        total_variance = float(np.sum(conditional_variances))
        captured_variance = float(np.sum(conditional_variances[:selected]))
        captured_fraction = (
            1.0 if total_variance <= 0.0 else captured_variance / total_variance
        )

        # Batched preparation caches products of these arrays.  Make the
        # public conditioning records truly immutable so the reference path
        # and its cached packed representation cannot silently diverge.
        for values in (
            retained_covariance,
            retained_factor,
            gain,
            conditional_covariance,
            conditional_factor,
        ):
            values.setflags(write=False)

        return InnovationConditioningStep(
            retained_covariance=retained_covariance,
            retained_factor=retained_factor,
            conditional_gain=gain,
            conditional_covariance=conditional_covariance,
            conditional_factor=conditional_factor,
            selected_rank=selected,
            captured_variance_fraction=captured_fraction,
        )
