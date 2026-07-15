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

FloatArray = NDArray[np.float64]


def project_to_mass_simplex(values: FloatArray, total_mass: float) -> FloatArray:
    """Euclidean projection onto ``x >= 0`` and ``sum(x) = total_mass``.

    This is the standard sorting-based simplex projection. It preserves the
    requested total mass to floating-point precision and is the projection
    used by the banked model.
    """

    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if total_mass <= 0:
        raise ValueError("total_mass must be positive")
    if not np.all(np.isfinite(vector)):
        raise ValueError("values must be finite")

    sorted_values = np.sort(vector)[::-1]
    cumulative = np.cumsum(sorted_values) - total_mass
    indices = np.arange(1, vector.size + 1, dtype=float)
    positive = sorted_values - cumulative / indices > 0.0
    if not np.any(positive):
        # This branch is mathematically unreachable for positive total_mass,
        # but retaining it gives a clear failure mode for corrupted inputs.
        raise RuntimeError("simplex projection failed to find an active set")
    rho = int(np.nonzero(positive)[0][-1])
    threshold = cumulative[rho] / float(rho + 1)
    projected = np.maximum(vector - threshold, 0.0)

    # Correct the final few ulps on one active coordinate so downstream mass
    # tests can use a tight tolerance without changing the projection.
    residual = total_mass - float(projected.sum())
    if abs(residual) > 0.0:
        active = int(np.argmax(projected))
        projected[active] += residual
    return projected


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
    p = float(jump_probability)
    noise = np.zeros_like(counts)
    n_nodes = counts.size

    def local_noise(mass: FloatArray, probabilities: FloatArray) -> FloatArray:
        """Vectorized categorical covariance sampler for one or many sources."""

        probabilities = np.asarray(probabilities, dtype=float)
        if mass.ndim == 0:
            z = rng.standard_normal(probabilities.size)
            weighted = np.sqrt(probabilities) * z
            return np.sqrt(float(mass)) * (
                weighted - probabilities * float(weighted.sum())
            )

        z = rng.standard_normal((mass.size, probabilities.size))
        weighted = np.sqrt(probabilities)[None, :] * z
        return np.sqrt(mass)[:, None] * (
            weighted - probabilities[None, :] * weighted.sum(axis=1, keepdims=True)
        )

    # Left reflecting boundary: stay or move right.
    left = local_noise(np.asarray(counts[0]), np.asarray([1.0 - p, p]))
    noise[0] += left[0]
    noise[1] += left[1]

    # Interior nodes: move left, stay, or move right.
    if n_nodes > 2:
        interior = local_noise(
            counts[1:-1],
            np.asarray([p, 1.0 - 2.0 * p, p]),
        )
        noise[:-2] += interior[:, 0]
        noise[1:-1] += interior[:, 1]
        noise[2:] += interior[:, 2]

    # Right reflecting boundary: move left or stay.
    right = local_noise(np.asarray(counts[-1]), np.asarray([p, 1.0 - p]))
    noise[-2] += right[0]
    noise[-1] += right[1]

    # Every source-level categorical fluctuation is zero sum. Remove only
    # floating-point accumulation error from the final field.
    noise[-1] -= float(noise.sum())
    return noise


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
    ) -> FloatArray:
        spatial_noise = sample_nearest_neighbour_gaussian_noise(
            source_counts,
            self.config.jump_probability,
            rng,
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

        modal[0] = self.basis.to_modal(initial)
        spatial[0] = self.basis.to_spatial(modal[0])
        noise_sources[0] = self._noise_source_proxy(spatial[0])
        conserved_zero_mode = float(modal[0, 0])

        for time_index in range(self.config.n_steps - 1):
            source = self._noise_source_proxy(spatial[time_index])
            noise_sources[time_index] = source
            next_modal = self._one_step_factors * modal[time_index]
            next_modal += self._modal_noise(source, generator)
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
