"""Direct moment-matched Gaussian dynamics in a retained diffusion basis.

This dense reference prototype samples retained modal counts with

``a_[t+1] = D a_t + L_t z_t``

where ``L_t L_t.T = C_[t+1] - D C_t D``.  The matrices ``C_t`` are exact
projections of the multinomial count covariances for the reflecting,
independent-particle nearest-neighbour walk.  Because the cosine modes span an
invariant subspace of the transition matrix, the model also has the exact
pairwise covariance ``Cov(a_s, a_t) = C_s D**(t-s)`` for ``s <= t``.

The deterministic impulse and constant mass mode are pinned exactly.  A full
spatial path can be reconstructed on request; with fewer modes than nodes that
path is a signed projection and is not constrained to be nonnegative.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from .config import DiffusionConfig
from .operators import Eigenbasis, impulse_initial_condition
from .references import random_walk_transition_matrix

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class DirectModalGaussianResult:
    """One trajectory from :class:`DirectModalGaussianDiffusion`.

    ``spatial_counts`` is ``None`` when ``run(reconstruct_spatial=False)`` is
    used.  Otherwise it contains the full-grid reconstruction of the retained
    modal trajectory.
    """

    modal_amplitudes: FloatArray
    spatial_counts: FloatArray | None
    basis: Eigenbasis
    model_name: str = "direct_modal_gaussian"

    @property
    def has_spatial_reconstruction(self) -> bool:
        """Return whether full spatial reconstructions were requested."""

        return self.spatial_counts is not None


def _cosine_basis(config: DiffusionConfig, n_modes: int | None) -> Eigenbasis:
    """Return the analytic DCT-II eigenbasis of the reflecting path walk."""

    retained = config.n_nodes if n_modes is None else int(n_modes)
    if not 1 <= retained <= config.n_nodes:
        raise ValueError(f"n_modes must be between 1 and {config.n_nodes}")

    nodes = np.arange(config.n_nodes, dtype=float)[:, None]
    modes = np.arange(retained, dtype=float)[None, :]
    vectors = np.cos(np.pi * (nodes + 0.5) * modes / config.n_nodes)
    vectors[:, 0] /= np.sqrt(float(config.n_nodes))
    if retained > 1:
        vectors[:, 1:] *= np.sqrt(2.0 / config.n_nodes)
    eigenvalues = 2.0 * config.jump_rate * (
        1.0 - np.cos(np.pi * np.arange(retained, dtype=float) / config.n_nodes)
    )
    return Eigenbasis(eigenvalues=eigenvalues, eigenvectors=vectors)


class DirectModalGaussianDiffusion:
    """Exact-moment Gaussian transition model in ``M`` retained modes.

    Dense single-particle probabilities and projected covariances are prepared
    once.  The generally singular innovation factors are also cached, so each
    run only performs diagonal propagation and dense ``M x M`` noise products.

    Parameters
    ----------
    config:
        Reflecting nearest-neighbour walk configuration.  All particles begin
        at ``config.impulse_index``.
    n_modes:
        Number of slow cosine modes.  ``None`` retains all spatial modes.
    psd_tolerance:
        Relative tolerance for distinguishing round-off-sized negative
        eigenvalues from a genuinely indefinite innovation covariance.
    """

    def __init__(
        self,
        config: DiffusionConfig,
        n_modes: int | None = None,
        *,
        psd_tolerance: float = 1.0e-10,
    ) -> None:
        if psd_tolerance <= 0.0:
            raise ValueError("psd_tolerance must be positive")

        self.config = config
        self.basis = _cosine_basis(config, n_modes)
        self._transition = random_walk_transition_matrix(config)
        self._one_step_factors = 1.0 - config.dt * self.basis.eigenvalues
        self._psd_tolerance = float(psd_tolerance)
        if np.min(self._one_step_factors) < -1.0e-12:
            raise ValueError(
                "The timestep is too large for the fastest retained mode: "
                "1 - lambda * dt must be nonnegative."
            )

        self._single_particle_probabilities = self._prepare_probabilities()
        self._modal_means, self._modal_covariances = self._prepare_modal_moments()
        (
            self._innovation_covariances,
            self._innovation_factors,
        ) = self._prepare_innovations()

    @property
    def n_modes(self) -> int:
        """Number of retained modal coordinates."""

        return self.basis.n_modes

    @property
    def transition_matrix(self) -> FloatArray:
        """Dense reflecting single-particle transition matrix."""

        return self._transition.copy()

    @property
    def one_step_factors(self) -> FloatArray:
        """Diagonal of the retained modal transition matrix ``D``."""

        return self._one_step_factors.copy()

    @property
    def single_particle_probabilities(self) -> FloatArray:
        """Exact node probabilities at every saved step."""

        return self._single_particle_probabilities.copy()

    @property
    def modal_means(self) -> FloatArray:
        """Exact projected count means, shape ``(time, mode)``."""

        return self._modal_means.copy()

    @property
    def modal_covariances(self) -> FloatArray:
        """Exact same-time projected covariances, shape ``(time, mode, mode)``."""

        return self._modal_covariances.copy()

    @property
    def innovation_covariances(self) -> FloatArray:
        """Cached ``Q_t`` matrices for transitions ``t -> t + 1``."""

        return self._innovation_covariances.copy()

    @property
    def innovation_factors(self) -> FloatArray:
        """Cached square roots ``L_t`` with zero columns for singular directions."""

        return self._innovation_factors.copy()

    def initial_spatial_counts(self) -> FloatArray:
        """Return the deterministic full spatial impulse."""

        return impulse_initial_condition(
            self.config.n_nodes,
            self.config.n_particles,
            self.config.impulse_index,
        )

    def reconstruct_spatial(self, modal_amplitudes: FloatArray) -> FloatArray:
        """Reconstruct one or more retained-modal states on the full grid."""

        modal = np.asarray(modal_amplitudes, dtype=float)
        if modal.ndim == 0 or modal.shape[-1] != self.n_modes:
            raise ValueError(f"modal_amplitudes must have final dimension {self.n_modes}")
        return modal @ self.basis.eigenvectors.T

    def modal_cross_covariance(self, first_step: int, second_step: int) -> FloatArray:
        """Return exact ``Cov(a[first_step], a[second_step])``.

        Exchanging the arguments returns the transpose.  This method avoids
        allocating the potentially large four-dimensional array of every pair.
        """

        first = self._validate_step(first_step)
        second = self._validate_step(second_step)
        if first <= second:
            propagation = self._one_step_factors ** (second - first)
            return self._modal_covariances[first] * propagation[None, :]
        return self.modal_cross_covariance(second, first).T

    def all_modal_cross_covariances(self) -> FloatArray:
        """Materialize covariances for all ordered pairs of saved steps.

        The shape is ``(time, time, mode, mode)`` and entry ``[s, t]`` is
        ``Cov(a_s, a_t)``.  Construction is explicit because storage grows as
        ``n_steps**2 * n_modes**2``.
        """

        n_steps = self.config.n_steps
        covariance = np.empty(
            (n_steps, n_steps, self.n_modes, self.n_modes),
            dtype=float,
        )
        for first in range(n_steps):
            covariance[first, first] = self._modal_covariances[first]
            propagation = np.ones(self.n_modes, dtype=float)
            for second in range(first + 1, n_steps):
                propagation *= self._one_step_factors
                cross = self._modal_covariances[first] * propagation[None, :]
                covariance[first, second] = cross
                covariance[second, first] = cross.T
        return covariance

    def run(
        self,
        rng: np.random.Generator | None = None,
        *,
        reconstruct_spatial: bool = True,
    ) -> DirectModalGaussianResult:
        """Sample one saved-step modal trajectory.

        Equal generator seeds produce equal trajectories.  Set
        ``reconstruct_spatial=False`` to keep the returned stochastic state
        strictly ``M``-dimensional.
        """

        generator = np.random.default_rng() if rng is None else rng
        modal = np.empty((self.config.n_steps, self.n_modes), dtype=float)
        modal[0] = self._modal_means[0]
        conserved_mass_mode = float(modal[0, 0])

        # Draw and transform the independent innovations in batches.  The
        # state recurrence is still sequential, but avoiding one RNG call and
        # one small BLAS dispatch per saved step is important at reduced rank.
        latent = generator.standard_normal(
            (self.config.n_steps - 1, self.n_modes)
        )
        innovations = np.einsum(
            "tij,tj->ti",
            self._innovation_factors,
            latent,
            optimize=False,
        )

        for time_index in range(self.config.n_steps - 1):
            modal[time_index + 1] = self._one_step_factors * modal[time_index]
            modal[time_index + 1] += innovations[time_index]
            # The exact covariance is singular in the mass direction.  Pinning
            # removes only floating-point leakage from its numerical factor.
            modal[time_index + 1, 0] = conserved_mass_mode

        spatial = self.reconstruct_spatial(modal) if reconstruct_spatial else None
        return DirectModalGaussianResult(
            modal_amplitudes=modal,
            spatial_counts=spatial,
            basis=self.basis,
        )

    def _prepare_probabilities(self) -> FloatArray:
        probabilities = np.empty(
            (self.config.n_steps, self.config.n_nodes),
            dtype=float,
        )
        probabilities[0] = 0.0
        probabilities[0, self.config.impulse_index] = 1.0
        for time_index in range(self.config.n_steps - 1):
            probabilities[time_index + 1] = probabilities[time_index] @ self._transition

        # These operations only remove accumulated floating-point error from a
        # nonnegative stochastic recurrence.
        probabilities = np.clip(probabilities, 0.0, 1.0)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities

    def _prepare_modal_moments(self) -> tuple[FloatArray, FloatArray]:
        vectors = self.basis.eigenvectors
        particle_count = float(self.config.n_particles)
        modal_means = particle_count * self._single_particle_probabilities @ vectors
        covariance = np.empty(
            (self.config.n_steps, self.n_modes, self.n_modes),
            dtype=float,
        )

        for time_index, probability in enumerate(self._single_particle_probabilities):
            single_mean = probability @ vectors
            single_second_moment = vectors.T @ (probability[:, None] * vectors)
            current = particle_count * (
                single_second_moment - np.outer(single_mean, single_mean)
            )
            covariance[time_index] = 0.5 * (current + current.T)

        # The initial impulse and constant mass mode are structurally
        # deterministic.  Do not retain dense-formula cancellation noise.
        covariance[0] = 0.0
        covariance[:, 0, :] = 0.0
        covariance[:, :, 0] = 0.0
        modal_means[:, 0] = modal_means[0, 0]
        return modal_means, covariance

    def _prepare_innovations(self) -> tuple[FloatArray, FloatArray]:
        n_transitions = self.config.n_steps - 1
        covariance = np.empty(
            (n_transitions, self.n_modes, self.n_modes),
            dtype=float,
        )
        factors = np.empty_like(covariance)
        diagonal = self._one_step_factors

        for time_index in range(n_transitions):
            propagated = (
                diagonal[:, None]
                * self._modal_covariances[time_index]
                * diagonal[None, :]
            )
            current = self._modal_covariances[time_index + 1] - propagated
            current = 0.5 * (current + current.T)
            current[0, :] = 0.0
            current[:, 0] = 0.0
            factor, psd_covariance = self._singular_psd_factor(current, time_index)
            covariance[time_index] = psd_covariance
            factors[time_index] = factor

        return covariance, factors

    def _singular_psd_factor(
        self,
        covariance: FloatArray,
        time_index: int,
    ) -> tuple[FloatArray, FloatArray]:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
        tolerance = self._psd_tolerance * scale
        if float(eigenvalues[0]) < -tolerance:
            raise RuntimeError(
                "Innovation covariance is not positive semidefinite at "
                f"transition {time_index}: minimum eigenvalue "
                f"{eigenvalues[0]:.6e}."
            )

        clipped = np.clip(eigenvalues, 0.0, None)
        factor = eigenvectors * np.sqrt(clipped)[None, :]
        factor[0, :] = 0.0
        psd_covariance = factor @ factor.T
        psd_covariance = 0.5 * (psd_covariance + psd_covariance.T)
        psd_covariance[0, :] = 0.0
        psd_covariance[:, 0] = 0.0
        return factor, psd_covariance

    def _validate_step(self, step: int) -> int:
        if not isinstance(step, Integral):
            raise TypeError("saved-step indices must be integers")
        value = int(step)
        if not 0 <= value < self.config.n_steps:
            raise IndexError(
                f"saved-step index must lie in [0, {self.config.n_steps - 1}]"
            )
        return value
