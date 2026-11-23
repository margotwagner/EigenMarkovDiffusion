"""Correlated-noise and banked modal diffusion prototypes.

The original EigenMarkov model treats every retained eigenmode as an
independent two-state Markov process. That construction reproduces the modal
mean but omits the cross-mode correlations created by a shared physical
particle transfer.

This module adds two deliberately experimental alternatives:

``CorrelatedModalDiffusion``
    Uses the same modal mean update, but generates one shared spatial
    Gaussian fluctuation field from the local random-walk transition events
    and projects that field into the retained eigenbasis. The resulting modal
    noise is correlated by construction.

``BankedCorrelatedModalDiffusion``
    Adds a spatial residual ledger (a debt/profit bank) and projects each raw
    reconstruction onto the nonnegative constant-mass simplex. The correction
    is carried forward rather than discarded. This guarantees nonnegative
    mass-conserving outputs, but the bank is an ``n_nodes``-dimensional state,
    so this variant is not a purely reduced-order model.

Both classes are research prototypes. They are intended to test hypotheses,
not to replace the independent modal baseline without validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import DiffusionConfig
from .operators import Eigenbasis, eigendecompose, impulse_initial_condition, neumann_laplacian_1d
from .simplex import _project_to_mass_simplex_prevalidated, project_to_mass_simplex

FloatArray = NDArray[np.float64]


def nearest_neighbour_transition_covariance(
    source_counts: FloatArray,
    jump_probability: float,
) -> FloatArray:
    """Return the exact conditional covariance of one node-transition step.

    The calculation treats ``source_counts[i]`` independent particles at each
    source node and sums their categorical destination covariances. Real-valued
    source masses are allowed because the correlated modal model uses this as a
    moment closure.
    """

    counts = np.asarray(source_counts, dtype=float)
    if counts.ndim != 1 or counts.size < 2:
        raise ValueError("source_counts must be a one-dimensional array of length >= 2")
    if np.min(counts) < -1e-12:
        raise ValueError("source_counts must be nonnegative")
    if not 0.0 <= jump_probability <= 0.5 + 1e-12:
        raise ValueError("jump_probability must lie in [0, 0.5]")

    counts = np.clip(counts, 0.0, None)
    p = float(jump_probability)
    n_nodes = counts.size
    covariance = np.zeros((n_nodes, n_nodes), dtype=float)

    for source_index, mass in enumerate(counts):
        probabilities = np.zeros(n_nodes, dtype=float)
        if source_index == 0:
            probabilities[0] = 1.0 - p
            probabilities[1] = p
        elif source_index == n_nodes - 1:
            probabilities[-2] = p
            probabilities[-1] = 1.0 - p
        else:
            probabilities[source_index - 1] = p
            probabilities[source_index] = 1.0 - 2.0 * p
            probabilities[source_index + 1] = p
        covariance += mass * (
            np.diag(probabilities) - np.outer(probabilities, probabilities)
        )

    return covariance


class _NearestNeighbourGaussianWorkspace:
    """Scratch arrays reused by a prepared nearest-neighbour sampler."""

    __slots__ = (
        "boundary_correction",
        "interior",
        "interior_correction",
        "interior_mass_sqrt",
        "interior_sums",
        "left",
        "noise",
        "right",
    )

    def __init__(self, n_nodes: int) -> None:
        n_interior = max(n_nodes - 2, 0)
        self.noise = np.empty(n_nodes, dtype=float)
        self.left = np.empty(2, dtype=float)
        self.interior = np.empty((n_interior, 3), dtype=float)
        self.right = np.empty(2, dtype=float)
        self.boundary_correction = np.empty(2, dtype=float)
        self.interior_sums = np.empty(n_interior, dtype=float)
        self.interior_correction = np.empty((n_interior, 3), dtype=float)
        self.interior_mass_sqrt = np.empty(n_interior, dtype=float)


class _PreparedNearestNeighbourGaussianSampler:
    """Nearest-neighbour Gaussian sampler with cached transition factors.

    Transition probabilities and their square roots are fixed by the grid and
    timestep, so simulators prepare them once. Scratch storage remains local
    to each run, which permits concurrent calls to one simulator instance.
    """

    __slots__ = (
        "_interior_probabilities",
        "_interior_probability_sqrt",
        "_left_probabilities",
        "_left_probability_sqrt",
        "_n_nodes",
        "_right_probabilities",
        "_right_probability_sqrt",
    )

    def __init__(self, n_nodes: int, jump_probability: float) -> None:
        if n_nodes < 2:
            raise ValueError("n_nodes must be at least 2")
        if not 0.0 <= jump_probability <= 0.5 + 1e-12:
            raise ValueError("jump_probability must lie in [0, 0.5]")

        p = float(jump_probability)
        self._n_nodes = int(n_nodes)
        self._left_probabilities = np.asarray([1.0 - p, p], dtype=float)
        self._interior_probabilities = np.asarray([p, 1.0 - 2.0 * p, p], dtype=float)
        self._right_probabilities = np.asarray([p, 1.0 - p], dtype=float)
        self._left_probability_sqrt = np.sqrt(self._left_probabilities)
        self._interior_probability_sqrt = np.sqrt(self._interior_probabilities)
        self._right_probability_sqrt = np.sqrt(self._right_probabilities)

    def new_workspace(self) -> _NearestNeighbourGaussianWorkspace:
        """Allocate scratch storage for one simulation run."""

        return _NearestNeighbourGaussianWorkspace(self._n_nodes)

    def sample(
        self,
        source_counts: FloatArray,
        rng: np.random.Generator,
        workspace: _NearestNeighbourGaussianWorkspace,
    ) -> FloatArray:
        """Sample into ``workspace`` while preserving the legacy RNG stream."""

        counts = np.asarray(source_counts, dtype=float)
        noise = workspace.noise
        noise.fill(0.0)

        # Preserve the original left/interior/right draw order and shapes.
        left = workspace.left
        rng.standard_normal(out=left)
        np.multiply(self._left_probability_sqrt, left, out=left)
        left_sum = float(left.sum())
        np.multiply(
            self._left_probabilities,
            left_sum,
            out=workspace.boundary_correction,
        )
        np.subtract(left, workspace.boundary_correction, out=left)
        np.multiply(np.sqrt(float(counts[0])), left, out=left)
        noise[0] += left[0]
        noise[1] += left[1]

        if self._n_nodes > 2:
            interior = workspace.interior
            rng.standard_normal(out=interior)
            np.multiply(
                self._interior_probability_sqrt[None, :],
                interior,
                out=interior,
            )
            np.sum(interior, axis=1, out=workspace.interior_sums)
            np.multiply(
                self._interior_probabilities[None, :],
                workspace.interior_sums[:, None],
                out=workspace.interior_correction,
            )
            np.subtract(interior, workspace.interior_correction, out=interior)
            np.sqrt(counts[1:-1], out=workspace.interior_mass_sqrt)
            np.multiply(
                workspace.interior_mass_sqrt[:, None],
                interior,
                out=interior,
            )
            noise[:-2] += interior[:, 0]
            noise[1:-1] += interior[:, 1]
            noise[2:] += interior[:, 2]

        right = workspace.right
        rng.standard_normal(out=right)
        np.multiply(self._right_probability_sqrt, right, out=right)
        right_sum = float(right.sum())
        np.multiply(
            self._right_probabilities,
            right_sum,
            out=workspace.boundary_correction,
        )
        np.subtract(right, workspace.boundary_correction, out=right)
        np.multiply(np.sqrt(float(counts[-1])), right, out=right)
        noise[-2] += right[0]
        noise[-1] += right[1]

        # Every source-level categorical fluctuation is zero sum. Remove only
        # floating-point accumulation error from the final field.
        noise[-1] -= float(noise.sum())
        return noise


def sample_nearest_neighbour_gaussian_noise(
    source_counts: FloatArray,
    jump_probability: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Sample the Gaussian approximation to one random-walk transition step.

    For every source node ``i``, the outgoing left/stay/right counts have a
    multinomial covariance. The local noise samples are generated from

    ``diag(p_i) - p_i p_i.T``

    and accumulated at their destination nodes. One physical source event
    therefore changes many retained modes after projection, which creates the
    cross-mode correlations missing from independent modal sampling.

    Parameters
    ----------
    source_counts:
        Nonnegative particle mass at each source node. Values may be real;
        this is a Gaussian moment approximation rather than an integer draw.
    jump_probability:
        Directed nearest-neighbour probability ``D * dt / dx**2``.
    rng:
        NumPy random-number generator.
    """

    counts = np.asarray(source_counts, dtype=float)
    if counts.ndim != 1 or counts.size < 2:
        raise ValueError("source_counts must be a one-dimensional array of length >= 2")
    if np.min(counts) < -1e-12:
        raise ValueError("source_counts must be nonnegative")
    if not 0.0 <= jump_probability <= 0.5 + 1e-12:
        raise ValueError("jump_probability must lie in [0, 0.5]")

    counts = np.clip(counts, 0.0, None)
    sampler = _PreparedNearestNeighbourGaussianSampler(
        counts.size,
        float(jump_probability),
    )
    return sampler.sample(counts, rng, sampler.new_workspace())


@dataclass(frozen=True, slots=True)
class CorrelatedModalResult:
    """Outputs of one correlated modal simulation."""

    spatial_counts: FloatArray
    modal_amplitudes: FloatArray
    noise_source_counts: FloatArray
    basis: Eigenbasis
    model_name: str = "correlated_modal"

    @property
    def minimum_count(self) -> float:
        return float(np.min(self.spatial_counts))


@dataclass(frozen=True, slots=True)
class BankedCorrelatedModalResult:
    """Outputs of one banked correlated modal simulation."""

    spatial_counts: FloatArray
    modal_amplitudes: FloatArray
    raw_spatial_counts: FloatArray
    adjusted_spatial_counts: FloatArray
    bank_balances: FloatArray
    basis: Eigenbasis
    model_name: str = "banked_correlated_modal"

    @property
    def minimum_count(self) -> float:
        return float(np.min(self.spatial_counts))

    @property
    def maximum_bank_l1_fraction(self) -> float:
        total = float(self.spatial_counts[0].sum())
        return float(np.max(np.abs(self.bank_balances).sum(axis=1) / total))


class _CorrelatedModalBase:
    """Shared operator and noise machinery for correlated modal variants."""

    def __init__(self, config: DiffusionConfig, n_modes: int | None = None) -> None:
        self.config = config
        operator = neumann_laplacian_1d(
            config.n_nodes,
            config.length,
            config.diffusion_coefficient,
        )
        self.basis = eigendecompose(operator, n_modes=n_modes)
        self._one_step_factors = 1.0 - config.dt * self.basis.eigenvalues
        self._noise_sampler = _PreparedNearestNeighbourGaussianSampler(
            config.n_nodes,
            config.jump_probability,
        )
        if np.min(self._one_step_factors) < -1e-12:
            raise ValueError(
                "The timestep is too large for the fastest retained mode: "
                "1 - lambda * dt must be nonnegative."
            )

    @property
    def n_modes(self) -> int:
        return self.basis.n_modes

    @property
    def one_step_factors(self) -> FloatArray:
        return self._one_step_factors.copy()

    def initial_spatial_counts(self) -> FloatArray:
        return impulse_initial_condition(
            self.config.n_nodes,
            self.config.n_particles,
            self.config.impulse_index,
        )

    def _noise_source_proxy(self, raw_spatial_counts: FloatArray) -> FloatArray:
        """Return a physical source state used only to set noise covariance."""

        return project_to_mass_simplex(raw_spatial_counts, self.config.n_particles)

    def _modal_noise(
        self,
        source_counts: FloatArray,
        rng: np.random.Generator,
        workspace: _NearestNeighbourGaussianWorkspace,
    ) -> FloatArray:
        spatial_noise = self._noise_sampler.sample(
            source_counts,
            rng,
            workspace,
        )
        modal_noise = self.basis.to_modal(spatial_noise)
        # The constant mode must remain deterministic because every physical
        # transition is a zero-sum debit/credit event.
        modal_noise[0] = 0.0
        return modal_noise


class CorrelatedModalDiffusion(_CorrelatedModalBase):
    """Modal mean dynamics with shared, physically derived Gaussian noise.

    The spatial state is not projected after each update. Negative values can
    therefore still occur. A simplex-projected proxy is used only to define
    nonnegative source intensities for the next conditional noise draw.
    """

    def run(
        self,
        rng: np.random.Generator | None = None,
    ) -> CorrelatedModalResult:
        generator = np.random.default_rng() if rng is None else rng
        initial = self.initial_spatial_counts()
        modal = np.zeros((self.config.n_steps, self.n_modes), dtype=float)
        spatial = np.zeros((self.config.n_steps, self.config.n_nodes), dtype=float)
        noise_sources = np.zeros_like(spatial)
        noise_workspace = self._noise_sampler.new_workspace()

        modal[0] = self.basis.to_modal(initial)
        spatial[0] = self.basis.to_spatial(modal[0])
        noise_sources[0] = self._noise_source_proxy(spatial[0])
        conserved_zero_mode = float(modal[0, 0])

        for time_index in range(self.config.n_steps - 1):
            source = self._noise_source_proxy(spatial[time_index])
            noise_sources[time_index] = source
            next_modal = self._one_step_factors * modal[time_index]
            next_modal += self._modal_noise(source, generator, noise_workspace)
            next_modal[0] = conserved_zero_mode
            modal[time_index + 1] = next_modal
            spatial[time_index + 1] = self.basis.to_spatial(next_modal)

        noise_sources[-1] = self._noise_source_proxy(spatial[-1])
        return CorrelatedModalResult(
            spatial_counts=spatial,
            modal_amplitudes=modal,
            noise_source_counts=noise_sources,
            basis=self.basis,
        )


class BankedCorrelatedModalDiffusion(_CorrelatedModalBase):
    """Correlated modal diffusion with nonnegative projection and a ledger.

    At every step, the raw modal reconstruction is combined with the previous
    spatial bank balance. The adjusted state is projected onto the nonnegative
    simplex with total mass ``N``. The difference between that adjusted state
    and its retained-mode reconstruction becomes the next bank balance:

    ``bank[t+1] = adjusted[t+1] - V @ (V.T @ physical[t+1])``.

    The bank therefore carries both projection correction and omitted-mode
    residual forward. This guarantees physical outputs, but the full spatial
    ledger means the method is not a purely ``M``-dimensional reduced model.
    """

    def run(
        self,
        rng: np.random.Generator | None = None,
    ) -> BankedCorrelatedModalResult:
        generator = np.random.default_rng() if rng is None else rng
        initial = self.initial_spatial_counts()

        modal = np.zeros((self.config.n_steps, self.n_modes), dtype=float)
        physical = np.zeros((self.config.n_steps, self.config.n_nodes), dtype=float)
        raw = np.zeros_like(physical)
        adjusted = np.zeros_like(physical)
        bank = np.zeros_like(physical)
        noise_workspace = self._noise_sampler.new_workspace()

        physical[0] = initial
        modal[0] = self.basis.to_modal(initial)
        raw[0] = self.basis.to_spatial(modal[0])
        adjusted[0] = initial
        bank[0] = adjusted[0] - raw[0]
        conserved_zero_mode = float(modal[0, 0])

        for time_index in range(self.config.n_steps - 1):
            next_modal_candidate = self._one_step_factors * modal[time_index]
            next_modal_candidate += self._modal_noise(
                physical[time_index],
                generator,
                noise_workspace,
            )
            next_modal_candidate[0] = conserved_zero_mode

            raw_next = self.basis.to_spatial(next_modal_candidate)
            adjusted_next = raw_next + bank[time_index]
            physical_next = project_to_mass_simplex(
                adjusted_next,
                self.config.n_particles,
            )
            projected_modal = self.basis.to_modal(physical_next)
            projected_modal[0] = conserved_zero_mode
            retained_reconstruction = self.basis.to_spatial(projected_modal)

            raw[time_index + 1] = raw_next
            adjusted[time_index + 1] = adjusted_next
            physical[time_index + 1] = physical_next
            modal[time_index + 1] = projected_modal
            bank[time_index + 1] = adjusted_next - retained_reconstruction

        return BankedCorrelatedModalResult(
            spatial_counts=physical,
            modal_amplitudes=modal,
            raw_spatial_counts=raw,
            adjusted_spatial_counts=adjusted,
            bank_balances=bank,
            basis=self.basis,
        )


@dataclass(frozen=True, slots=True)
class HandoffCorrelatedModalResult:
    """Outputs of one full-to-reduced correlated modal simulation."""

    spatial_counts: FloatArray
    modal_amplitudes: FloatArray
    noise_source_counts: FloatArray
    basis: Eigenbasis
    final_basis: Eigenbasis
    handoff_step: int
    handoff_time: float
    initial_n_modes: int
    final_n_modes: int
    handoff_projection_relative_l2: float
    modal_update_fraction_of_full: float
    model_name: str = "handoff_correlated_modal"

    @property
    def minimum_count(self) -> float:
        return float(np.min(self.spatial_counts))


@dataclass(frozen=True, slots=True)
class HandoffCorrelatedModalMinimalResult:
    """Counts plus retained coordinates needed by downstream completion."""

    spatial_counts: FloatArray
    retained_modal_amplitudes: FloatArray


class HandoffCorrelatedModalDiffusion:
    """Run a high-rank transient, then hand off to fewer correlated modes.

    Diffusion begins in an ``initial_n_modes`` basis. At ``handoff_time``, the
    current spatial state is projected onto the nested ``final_n_modes`` basis
    and all subsequent updates use only those final modes. This targets the
    central truncation problem of an impulse: high-frequency modes are retained
    during the early transient, when they matter most, and discarded after
    diffusion has smoothed the spatial profile.

    The projection is intentionally visible in ``spatial_counts`` at the
    handoff step. No residual bank or Delta-Sigma feedback is used to hide
    omitted-mode error.
    """

    def __init__(
        self,
        config: DiffusionConfig,
        *,
        final_n_modes: int,
        handoff_time: float,
        initial_n_modes: int | None = None,
    ) -> None:
        self.config = config
        initial = config.n_nodes if initial_n_modes is None else int(initial_n_modes)
        final = int(final_n_modes)
        if not 1 <= final <= initial <= config.n_nodes:
            raise ValueError(
                "mode counts must satisfy 1 <= final_n_modes <= "
                "initial_n_modes <= n_nodes"
            )
        if not config.times[0] - 1e-12 <= handoff_time <= config.times[-1] + 1e-12:
            raise ValueError("handoff_time must lie within the simulated time range")

        operator = neumann_laplacian_1d(
            config.n_nodes,
            config.length,
            config.diffusion_coefficient,
        )
        initial_basis = eigendecompose(operator, n_modes=initial)
        # Slice the high-rank eigendecomposition so the two bases are exactly
        # nested and use identical eigenvector signs.
        final_basis = Eigenbasis(
            eigenvalues=initial_basis.eigenvalues[:final].copy(),
            eigenvectors=initial_basis.eigenvectors[:, :final].copy(),
        )

        initial_factors = 1.0 - config.dt * initial_basis.eigenvalues
        final_factors = 1.0 - config.dt * final_basis.eigenvalues
        if np.min(initial_factors) < -1e-12 or np.min(final_factors) < -1e-12:
            raise ValueError(
                "The timestep is too large for the fastest retained mode: "
                "1 - lambda * dt must be nonnegative."
            )

        self.initial_basis = initial_basis
        self.final_basis = final_basis
        self.initial_n_modes = initial
        self.final_n_modes = final
        self._initial_factors = initial_factors
        self._final_factors = final_factors
        self._noise_sampler = _PreparedNearestNeighbourGaussianSampler(
            config.n_nodes,
            config.jump_probability,
        )
        self.handoff_step = int(np.argmin(np.abs(config.times - float(handoff_time))))
        self.handoff_time = float(config.times[self.handoff_step])

    @property
    def n_modes(self) -> int:
        """Return the final reduced dimension used after handoff."""

        return self.final_n_modes

    def initial_spatial_counts(self) -> FloatArray:
        return impulse_initial_condition(
            self.config.n_nodes,
            self.config.n_particles,
            self.config.impulse_index,
        )

    def _noise_source_proxy(self, raw_spatial_counts: FloatArray) -> FloatArray:
        return project_to_mass_simplex(raw_spatial_counts, self.config.n_particles)

    def _modal_noise(
        self,
        source_counts: FloatArray,
        basis: Eigenbasis,
        rng: np.random.Generator,
        workspace: _NearestNeighbourGaussianWorkspace,
    ) -> FloatArray:
        spatial_noise = self._noise_sampler.sample(
            source_counts,
            rng,
            workspace,
        )
        modal_noise = basis.to_modal(spatial_noise)
        modal_noise[0] = 0.0
        return modal_noise

    def run(
        self,
        rng: np.random.Generator | None = None,
    ) -> HandoffCorrelatedModalResult:
        generator = np.random.default_rng() if rng is None else rng
        initial = self.initial_spatial_counts()
        n_steps = self.config.n_steps

        spatial = np.zeros((n_steps, self.config.n_nodes), dtype=float)
        noise_sources = np.zeros_like(spatial)
        # Store modal amplitudes in the initial/high-rank coordinate array.
        # Entries above final_n_modes are exactly zero after the handoff.
        modal_padded = np.zeros((n_steps, self.initial_n_modes), dtype=float)
        noise_workspace = self._noise_sampler.new_workspace()

        high_modal = self.initial_basis.to_modal(initial)
        modal_padded[0] = high_modal
        spatial[0] = self.initial_basis.to_spatial(high_modal)
        conserved_zero_mode = float(high_modal[0])

        # High-rank transient: compute the state through the handoff time.
        for time_index in range(self.handoff_step):
            source = self._noise_source_proxy(spatial[time_index])
            noise_sources[time_index] = source
            next_modal = self._initial_factors * modal_padded[time_index]
            next_modal += self._modal_noise(
                source,
                self.initial_basis,
                generator,
                noise_workspace,
            )
            next_modal[0] = conserved_zero_mode
            modal_padded[time_index + 1] = next_modal
            spatial[time_index + 1] = self.initial_basis.to_spatial(next_modal)

        # Project the state at the handoff time into the reduced nested basis.
        pre_handoff = spatial[self.handoff_step].copy()
        low_modal = self.final_basis.to_modal(pre_handoff)
        low_modal[0] = conserved_zero_mode
        low_spatial = self.final_basis.to_spatial(low_modal)
        denominator = np.linalg.norm(pre_handoff)
        if denominator == 0.0:
            projection_error = 0.0
        else:
            projection_error = float(np.linalg.norm(pre_handoff - low_spatial) / denominator)
        modal_padded[self.handoff_step] = 0.0
        modal_padded[self.handoff_step, : self.final_n_modes] = low_modal
        spatial[self.handoff_step] = low_spatial

        # Reduced correlated dynamics after the handoff.
        for time_index in range(self.handoff_step, n_steps - 1):
            source = self._noise_source_proxy(spatial[time_index])
            noise_sources[time_index] = source
            current_low = modal_padded[time_index, : self.final_n_modes]
            next_low = self._final_factors * current_low
            next_low += self._modal_noise(
                source,
                self.final_basis,
                generator,
                noise_workspace,
            )
            next_low[0] = conserved_zero_mode
            modal_padded[time_index + 1, : self.final_n_modes] = next_low
            spatial[time_index + 1] = self.final_basis.to_spatial(next_low)

        noise_sources[-1] = self._noise_source_proxy(spatial[-1])
        transition_count = max(n_steps - 1, 1)
        modal_work = (
            self.handoff_step * self.initial_n_modes
            + (transition_count - self.handoff_step) * self.final_n_modes
        )
        full_work = transition_count * self.config.n_nodes

        return HandoffCorrelatedModalResult(
            spatial_counts=spatial,
            modal_amplitudes=modal_padded,
            noise_source_counts=noise_sources,
            basis=self.initial_basis,
            final_basis=self.final_basis,
            handoff_step=self.handoff_step,
            handoff_time=self.handoff_time,
            initial_n_modes=self.initial_n_modes,
            final_n_modes=self.final_n_modes,
            handoff_projection_relative_l2=projection_error,
            modal_update_fraction_of_full=float(modal_work / full_work),
        )

    def run_minimal(
        self,
        rng: np.random.Generator | None = None,
    ) -> HandoffCorrelatedModalMinimalResult:
        """Return the full spatial field and only the retained modal history.

        The detailed :meth:`run` method also stores every conditional-noise
        source and a high-rank padded modal trajectory for diagnostics.  This
        production path instead keeps rolling modal state and the retained
        coordinates consumed by persistent completion.  It preserves the
        same transition and random-number order.
        """

        generator = np.random.default_rng() if rng is None else rng
        initial = self.initial_spatial_counts()
        n_steps = self.config.n_steps
        n_nodes = self.config.n_nodes

        spatial = np.empty((n_steps, n_nodes), dtype=float)
        retained = np.empty((n_steps, self.final_n_modes), dtype=float)
        source = np.empty(n_nodes, dtype=float)
        simplex_workspace = np.empty((1, n_nodes), dtype=float)
        noise_workspace = self._noise_sampler.new_workspace()

        high_modal = self.initial_basis.to_modal(initial)
        retained[0] = high_modal[: self.final_n_modes]
        spatial[0] = self.initial_basis.to_spatial(high_modal)
        conserved_zero_mode = float(high_modal[0])

        for time_index in range(self.handoff_step):
            _project_to_mass_simplex_prevalidated(
                spatial[time_index],
                float(self.config.n_particles),
                source,
                simplex_workspace,
            )
            next_modal = self._initial_factors * high_modal
            next_modal += self._modal_noise(
                source,
                self.initial_basis,
                generator,
                noise_workspace,
            )
            next_modal[0] = conserved_zero_mode
            high_modal = next_modal
            retained[time_index + 1] = next_modal[: self.final_n_modes]
            spatial[time_index + 1] = self.initial_basis.to_spatial(next_modal)

        pre_handoff = spatial[self.handoff_step].copy()
        low_modal = self.final_basis.to_modal(pre_handoff)
        low_modal[0] = conserved_zero_mode
        retained[self.handoff_step] = low_modal
        spatial[self.handoff_step] = self.final_basis.to_spatial(low_modal)

        for time_index in range(self.handoff_step, n_steps - 1):
            _project_to_mass_simplex_prevalidated(
                spatial[time_index],
                float(self.config.n_particles),
                source,
                simplex_workspace,
            )
            next_low = self._final_factors * low_modal
            next_low += self._modal_noise(
                source,
                self.final_basis,
                generator,
                noise_workspace,
            )
            next_low[0] = conserved_zero_mode
            low_modal = next_low
            retained[time_index + 1] = next_low
            spatial[time_index + 1] = self.final_basis.to_spatial(next_low)

        return HandoffCorrelatedModalMinimalResult(
            spatial_counts=spatial,
            retained_modal_amplitudes=retained,
        )
