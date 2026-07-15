"""Stochastic eigenmode Markov diffusion.

This is a cleaned and testable implementation of the central model from the
original ``diffusion-model`` repository. Plotting, file I/O, and experimental
clipping rules are deliberately kept outside the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import DiffusionConfig
from .operators import Eigenbasis, eigendecompose, impulse_initial_condition, neumann_laplacian_1d

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class EigenMarkovResult:
    """Outputs of one EigenMarkov simulation."""

    spatial_counts: FloatArray
    modal_amplitudes: FloatArray
    positive_states: IntArray
    negative_states: IntArray
    basis: Eigenbasis
    initial_modal_amplitudes: FloatArray
    modal_particle_weight: float

    @property
    def normalized_counts(self) -> FloatArray:
        total = self.spatial_counts[0].sum()
        return self.spatial_counts / total

    @property
    def minimum_count(self) -> float:
        return float(np.min(self.spatial_counts))


class EigenMarkovDiffusion:
    """Simulate diffusion as independent two-state Markov eigenmodes.

    Each retained nonconstant mode ``k`` is represented by positive and
    negative integer state populations. During one timestep, every state unit
    switches sign with probability ``lambda_k * dt / 2``. Therefore,

        E[m_k(t + dt)] = (1 - lambda_k dt) E[m_k(t)],

    which is the first-order diffusion update in modal coordinates.

    The constant zero mode is stored exactly rather than rounded because it
    does not transition and is responsible for conservation of total mass.

    Notes
    -----
    Reconstructing signed stochastic eigenmodes can produce negative spatial
    values. The class exposes that behavior instead of silently clipping it;
    resolving nonnegativity while preserving the desired statistics remains a
    scientific question for this project.
    """

    def __init__(
        self,
        config: DiffusionConfig,
        n_modes: int | None = None,
        modal_particle_weight: float = 1.0,
        initialization: str = "nearest",
    ) -> None:
        if modal_particle_weight <= 0:
            raise ValueError("modal_particle_weight must be positive")
        if initialization not in {"nearest", "stochastic"}:
            raise ValueError("initialization must be 'nearest' or 'stochastic'")

        self.config = config
        self.modal_particle_weight = float(modal_particle_weight)
        self.initialization = initialization

        operator = neumann_laplacian_1d(
            config.n_nodes,
            config.length,
            config.diffusion_coefficient,
        )
        self.basis = eigendecompose(operator, n_modes=n_modes)

        probabilities = self.transition_probabilities
        if np.max(probabilities) > 1.0 + 1e-12:
            raise ValueError(
                "The timestep is too large for the fastest retained mode: "
                "max(lambda * dt / 2) must be <= 1."
            )

    @property
    def n_modes(self) -> int:
        return self.basis.n_modes

    @property
    def transition_probabilities(self) -> FloatArray:
        return self.basis.eigenvalues * self.config.dt / 2.0

    def initial_modal_amplitudes(self) -> FloatArray:
        initial = impulse_initial_condition(
            self.config.n_nodes,
            self.config.n_particles,
            self.config.impulse_index,
        )
        return self.basis.to_modal(initial)

    def _integerize_units(
        self,
        units: FloatArray,
        rng: np.random.Generator,
    ) -> IntArray:
        if self.initialization == "nearest":
            return np.rint(units).astype(np.int64)

        lower = np.floor(units)
        fractional = units - lower
        return (lower + (rng.random(units.shape) < fractional)).astype(np.int64)

    def run(
        self,
        rng: np.random.Generator | None = None,
    ) -> EigenMarkovResult:
        generator = np.random.default_rng() if rng is None else rng
        modal_initial = self.initial_modal_amplitudes()

        positive = np.zeros((self.config.n_steps, self.n_modes), dtype=np.int64)
        negative = np.zeros_like(positive)

        # The zero mode is handled exactly below. Nonzero modes are represented
        # by integer state units with a configurable physical weight.
        if self.n_modes > 1:
            nonzero_initial = modal_initial[1:]
            units = np.abs(nonzero_initial) / self.modal_particle_weight
            integer_units = self._integerize_units(units, generator)
            positive[0, 1:] = np.where(nonzero_initial >= 0.0, integer_units, 0)
            negative[0, 1:] = np.where(nonzero_initial < 0.0, integer_units, 0)

        probabilities = self.transition_probabilities
        for time_index in range(self.config.n_steps - 1):
            positive[time_index + 1] = positive[time_index]
            negative[time_index + 1] = negative[time_index]

            if self.n_modes <= 1:
                continue

            p = probabilities[1:]
            leave_positive = generator.binomial(positive[time_index, 1:], p)
            leave_negative = generator.binomial(negative[time_index, 1:], p)

            positive[time_index + 1, 1:] = (
                positive[time_index, 1:] - leave_positive + leave_negative
            )
            negative[time_index + 1, 1:] = (
                negative[time_index, 1:] - leave_negative + leave_positive
            )

        modal_amplitudes = self.modal_particle_weight * (
            positive - negative
        ).astype(float)
        modal_amplitudes[:, 0] = modal_initial[0]
        spatial_counts = modal_amplitudes @ self.basis.eigenvectors.T

        return EigenMarkovResult(
            spatial_counts=spatial_counts,
            modal_amplitudes=modal_amplitudes,
            positive_states=positive,
            negative_states=negative,
            basis=self.basis,
            initial_modal_amplitudes=modal_initial,
            modal_particle_weight=self.modal_particle_weight,
        )


class IndependentModalDiffusion(EigenMarkovDiffusion):
    """Explicit name for the original independent two-state modal model."""
