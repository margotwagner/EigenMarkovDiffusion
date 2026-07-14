"""Configuration objects for one-dimensional diffusion simulations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DiffusionConfig:
    """Physical and numerical parameters shared by all simulators.

    Units follow the original project:

    - length: micrometres (µm)
    - time: microseconds (µs)
    - diffusion coefficient: µm²/µs
    - counts: molecules or abstract particles
    """

    n_particles: int = 5_275
    n_nodes: int = 101
    n_steps: int = 101
    impulse_index: int | None = None
    dt: float = 1.0
    length: float = 4.0
    diffusion_coefficient: float = 2.20e-4

    def __post_init__(self) -> None:
        if self.n_particles <= 0:
            raise ValueError("n_particles must be positive")
        if self.n_nodes < 2:
            raise ValueError("n_nodes must be at least 2")
        if self.n_steps < 2:
            raise ValueError("n_steps must be at least 2")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.length <= 0:
            raise ValueError("length must be positive")
        if self.diffusion_coefficient <= 0:
            raise ValueError("diffusion_coefficient must be positive")

        index = self.n_nodes // 2 if self.impulse_index is None else self.impulse_index
        if not 0 <= index < self.n_nodes:
            raise ValueError("impulse_index must be between 0 and n_nodes - 1")
        object.__setattr__(self, "impulse_index", index)

        # A nearest-neighbour random walk has left/right probability p each in
        # the interior, so 2p must not exceed one.
        if 2.0 * self.jump_probability > 1.0 + 1e-12:
            raise ValueError(
                "Unstable timestep: 2 * D * dt / dx^2 must be <= 1. "
                "Decrease dt or use fewer spatial nodes."
            )

    @property
    def dx(self) -> float:
        """Spacing between nodes on a grid that includes both endpoints."""

        return self.length / (self.n_nodes - 1)

    @property
    def jump_rate(self) -> float:
        """Nearest-neighbour diffusion rate, k = D / dx²."""

        return self.diffusion_coefficient / self.dx**2

    @property
    def jump_probability(self) -> float:
        """Probability of one directed nearest-neighbour jump in one step."""

        return self.jump_rate * self.dt

    @property
    def times(self):
        """Simulation times as a NumPy array."""

        import numpy as np

        return np.arange(self.n_steps, dtype=float) * self.dt

    @property
    def positions(self):
        """Spatial grid as a NumPy array."""

        import numpy as np

        return np.linspace(0.0, self.length, self.n_nodes)
