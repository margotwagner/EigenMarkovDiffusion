"""Physical readout layers for real-valued modal diffusion trajectories.

The modal dynamics and the readout are deliberately separated. A modal model
may evolve real-valued spatial reconstructions, while a readout converts those
reconstructions into constrained particle-count observations without feeding
back into the modal state.

Available readouts
------------------
``raw``
    Return the modal reconstruction unchanged.
``simplex_bank``
    Continuous nonnegative, mass-conserving projection with first-order
    temporal residual feedback. This is an output-only analogue of a debt /
    profit bank.
``delta_sigma_temporal``
    First-order temporal delta-sigma readout using an integer simplex
    quantizer. Outputs are integer, nonnegative, bounded by ``N``, and sum to
    ``N`` at every time step.
``delta_sigma_neighbor``
    A neighbor-coupled first-order delta-sigma readout. A configurable fraction
    of each local quantization error is passed to the next spatial node during
    the same time step; the remainder is carried forward in time. Scan
    direction alternates across time to reduce directional bias.
``unresolved_gaussian_completion``
    An analytic same-time moment completion that samples omitted eigenmodes
    conditionally on the retained state after a configurable start time.
``persistent_unresolved_completion``
    A temporally persistent analytic completion that propagates an unresolved
    latent state conditioned on consecutive retained states.

These readouts enforce output constraints and shape quantization error. They do
not, by themselves, repair incorrect stochastic dynamics. In particular, the
recommended pairing is ``correlated_modal`` dynamics with one of the
Delta-Sigma readouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .completion import (
    PersistentUnresolvedGaussianCompleter,
    UnresolvedGaussianCompleter,
)
from .config import DiffusionConfig
from .correlated_modal import project_to_mass_simplex

FloatArray = NDArray[np.float64]
ReadoutName = Literal[
    "raw",
    "simplex_bank",
    "delta_sigma_temporal",
    "delta_sigma_neighbor",
    "unresolved_gaussian_completion",
    "persistent_unresolved_completion",
]
READOUT_NAMES: tuple[ReadoutName, ...] = (
    "raw",
    "simplex_bank",
    "delta_sigma_temporal",
    "delta_sigma_neighbor",
    "unresolved_gaussian_completion",
    "persistent_unresolved_completion",
)


@dataclass(frozen=True, slots=True)
class ReadoutResult:
    """Result of applying one readout to a time-by-space trajectory."""

    counts: FloatArray
    residuals: FloatArray
    adjusted_inputs: FloatArray
    readout_name: str

    @property
    def residual_l1(self) -> FloatArray:
        return np.abs(self.residuals).sum(axis=-1)


def integer_simplex_quantize(values: FloatArray, total_mass: int) -> FloatArray:
    """Quantize onto nonnegative integer counts summing exactly to ``total_mass``.

    The input is first projected onto the continuous mass simplex. The floor of
    each projected value is taken, and the remaining particles are assigned to
    the largest fractional remainders. This is deterministic largest-remainder
    rounding and is permutation-equivariant up to ties.
    """

    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if total_mass <= 0:
        raise ValueError("total_mass must be positive")
    if not np.all(np.isfinite(vector)):
        raise ValueError("values must be finite")

    projected = project_to_mass_simplex(vector, float(total_mass))
    floored = np.floor(projected).astype(np.int64)
    remainder = int(total_mass - int(floored.sum()))
    if remainder < 0 or remainder > vector.size:
        raise RuntimeError("integer simplex rounding produced an invalid remainder")

    if remainder:
        fractions = projected - floored
        # Stable sorting makes exact ties deterministic.
        order = np.argsort(-fractions, kind="stable")
        floored[order[:remainder]] += 1

    result = floored.astype(float)
    if int(result.sum()) != int(total_mass):
        raise RuntimeError("integer simplex quantizer failed to conserve mass")
    return result


def raw_readout(raw_counts: FloatArray, total_mass: int) -> ReadoutResult:
    """Return an unchanged copy of the real-valued modal reconstruction."""

    values = _validate_trajectory(raw_counts, total_mass)
    zeros = np.zeros_like(values)
    return ReadoutResult(
        counts=values.copy(),
        residuals=zeros,
        adjusted_inputs=values.copy(),
        readout_name="raw",
    )


def simplex_bank_readout(raw_counts: FloatArray, total_mass: int) -> ReadoutResult:
    """Apply continuous simplex projection with temporal residual feedback.

    For each time point,

    ``adjusted[t] = raw[t] + residual[t-1]``
    ``output[t] = project_to_mass_simplex(adjusted[t], N)``
    ``residual[t] = adjusted[t] - output[t]``

    The residual is the debt/profit balance carried to the next readout time.
    This readout does not modify the underlying modal trajectory.
    """

    values = _validate_trajectory(raw_counts, total_mass)
    outputs = np.zeros_like(values)
    residuals = np.zeros_like(values)
    adjusted = np.zeros_like(values)
    previous_residual = np.zeros(values.shape[1], dtype=float)

    for time_index, row in enumerate(values):
        current = row + previous_residual
        output = project_to_mass_simplex(current, float(total_mass))
        residual = current - output
        # Remove only floating-point zero-sum drift.
        residual[-1] -= float(residual.sum())

        adjusted[time_index] = current
        outputs[time_index] = output
        residuals[time_index] = residual
        previous_residual = residual

    return ReadoutResult(
        counts=outputs,
        residuals=residuals,
        adjusted_inputs=adjusted,
        readout_name="simplex_bank",
    )


def temporal_delta_sigma_readout(
    raw_counts: FloatArray,
    total_mass: int,
) -> ReadoutResult:
    """First-order temporal Delta-Sigma readout with integer simplex output.

    The quantization residual is carried entirely forward in time. Outputs are
    integer, nonnegative, bounded by ``total_mass``, and conserve total mass at
    each saved time.
    """

    values = _validate_trajectory(raw_counts, total_mass)
    outputs = np.zeros_like(values)
    residuals = np.zeros_like(values)
    adjusted = np.zeros_like(values)
    previous_residual = np.zeros(values.shape[1], dtype=float)

    for time_index, row in enumerate(values):
        current = row + previous_residual
        output = integer_simplex_quantize(current, total_mass)
        residual = current - output
        residual[-1] -= float(residual.sum())

        adjusted[time_index] = current
        outputs[time_index] = output
        residuals[time_index] = residual
        previous_residual = residual

    return ReadoutResult(
        counts=outputs,
        residuals=residuals,
        adjusted_inputs=adjusted,
        readout_name="delta_sigma_temporal",
    )


def neighbor_delta_sigma_readout(
    raw_counts: FloatArray,
    total_mass: int,
    *,
    spatial_error_fraction: float = 0.5,
    alternate_scan: bool = True,
) -> ReadoutResult:
    """Neighbor-coupled first-order Delta-Sigma readout.

    A fraction ``spatial_error_fraction`` of the local quantization error is
    passed to the next spatial location in the current scan. The remaining
    fraction is stored as temporal residual debt. The final node absorbs the
    exact remaining particle count, and its residual is carried in time.

    Scan direction alternates by time step by default, reducing the left/right
    bias of a causal spatial error-diffusion pass.
    """

    values = _validate_trajectory(raw_counts, total_mass)
    alpha = float(spatial_error_fraction)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("spatial_error_fraction must lie in [0, 1]")

    n_steps, n_nodes = values.shape
    outputs = np.zeros_like(values)
    residuals = np.zeros_like(values)
    adjusted = np.zeros_like(values)
    previous_temporal = np.zeros(n_nodes, dtype=float)

    for time_index in range(n_steps):
        current = values[time_index] + previous_temporal
        adjusted[time_index] = current
        work = current.copy()
        output = np.zeros(n_nodes, dtype=float)
        next_temporal = np.zeros(n_nodes, dtype=float)

        reverse = alternate_scan and bool(time_index % 2)
        order = list(range(n_nodes - 1, -1, -1) if reverse else range(n_nodes))
        remaining = int(total_mass)

        for scan_position, node_index in enumerate(order[:-1]):
            target = float(work[node_index])
            quantized = int(np.rint(target))
            quantized = min(max(quantized, 0), remaining)
            output[node_index] = float(quantized)
            remaining -= quantized

            error = target - quantized
            next_temporal[node_index] += (1.0 - alpha) * error
            neighbour_index = order[scan_position + 1]
            work[neighbour_index] += alpha * error

        final_index = order[-1]
        output[final_index] = float(remaining)
        final_error = float(work[final_index] - remaining)
        next_temporal[final_index] += final_error

        # The feedback ledger should be globally zero-sum. Correct only
        # floating-point accumulation drift on the terminal coordinate.
        next_temporal[final_index] -= float(next_temporal.sum())

        outputs[time_index] = output
        residuals[time_index] = next_temporal
        previous_temporal = next_temporal

    return ReadoutResult(
        counts=outputs,
        residuals=residuals,
        adjusted_inputs=adjusted,
        readout_name="delta_sigma_neighbor",
    )



def unresolved_gaussian_completion_readout(
    raw_counts: FloatArray,
    total_mass: int,
    *,
    config: DiffusionConfig,
    retained_modes: int,
    completion_start_time: float = 0.0,
    completion_rank: int | None = None,
    completion_ridge: float = 1.0e-2,
    rng: np.random.Generator | None = None,
) -> ReadoutResult:
    """Restore omitted-mode same-time moments with analytic Gaussian sampling.

    The retained spatial trajectory is left unchanged before
    ``completion_start_time``. At and after that time, the omitted diffusion
    modes are sampled from a ridge-regularized Gaussian conditional model
    derived from the exact finite-step multinomial mean and covariance.

    This is an output-only, analytic proof-of-principle closure. It preserves
    the retained modal coordinates and total mass, but it does not reproduce
    the exact temporal correlation of the omitted modes.
    """

    values = _validate_trajectory(raw_counts, total_mass)
    if config.n_particles != total_mass:
        raise ValueError("config.n_particles must match total_mass")
    completer = UnresolvedGaussianCompleter(
        config,
        retained_modes=retained_modes,
        completion_start_time=completion_start_time,
        completion_rank=completion_rank,
        ridge=completion_ridge,
    )
    completed, additions = completer.complete(values, rng=rng)
    return ReadoutResult(
        counts=completed,
        residuals=additions,
        adjusted_inputs=values.copy(),
        readout_name="unresolved_gaussian_completion",
    )


def persistent_unresolved_completion_readout(
    raw_counts: FloatArray,
    total_mass: int,
    *,
    config: DiffusionConfig,
    retained_modes: int,
    completion_start_time: float = 0.0,
    completion_rank: int | None = None,
    completion_ridge: float = 1.0e-2,
    rng: np.random.Generator | None = None,
) -> ReadoutResult:
    """Restore omitted modes with a persistent analytic Gaussian state.

    The unresolved state is initialized from the exact same-time conditional
    distribution, then propagated from one saved time to the next conditioned
    on its previous value and the retained states at both times. This preserves
    substantially more short-lag temporal covariance than independently
    redrawing the unresolved state at every output time.
    """

    values = _validate_trajectory(raw_counts, total_mass)
    if config.n_particles != total_mass:
        raise ValueError("config.n_particles must match total_mass")
    completer = PersistentUnresolvedGaussianCompleter(
        config,
        retained_modes=retained_modes,
        completion_start_time=completion_start_time,
        completion_rank=completion_rank,
        ridge=completion_ridge,
    )
    completed, additions = completer.complete(values, rng=rng)
    return ReadoutResult(
        counts=completed,
        residuals=additions,
        adjusted_inputs=values.copy(),
        readout_name="persistent_unresolved_completion",
    )

def apply_readout(
    raw_counts: FloatArray,
    total_mass: int,
    *,
    readout: ReadoutName = "raw",
    spatial_error_fraction: float = 0.5,
    config: DiffusionConfig | None = None,
    retained_modes: int | None = None,
    completion_start_time: float = 0.0,
    completion_rank: int | None = None,
    completion_ridge: float = 1.0e-2,
    rng: np.random.Generator | None = None,
) -> ReadoutResult:
    """Apply one named output readout to a modal spatial trajectory."""

    if readout == "raw":
        return raw_readout(raw_counts, total_mass)
    if readout == "simplex_bank":
        return simplex_bank_readout(raw_counts, total_mass)
    if readout == "delta_sigma_temporal":
        return temporal_delta_sigma_readout(raw_counts, total_mass)
    if readout == "delta_sigma_neighbor":
        return neighbor_delta_sigma_readout(
            raw_counts,
            total_mass,
            spatial_error_fraction=spatial_error_fraction,
        )
    if readout == "unresolved_gaussian_completion":
        if config is None or retained_modes is None:
            raise ValueError(
                "unresolved_gaussian_completion requires config and retained_modes"
            )
        return unresolved_gaussian_completion_readout(
            raw_counts,
            total_mass,
            config=config,
            retained_modes=retained_modes,
            completion_start_time=completion_start_time,
            completion_rank=completion_rank,
            completion_ridge=completion_ridge,
            rng=rng,
        )
    if readout == "persistent_unresolved_completion":
        if config is None or retained_modes is None:
            raise ValueError(
                "persistent_unresolved_completion requires config and retained_modes"
            )
        return persistent_unresolved_completion_readout(
            raw_counts,
            total_mass,
            config=config,
            retained_modes=retained_modes,
            completion_start_time=completion_start_time,
            completion_rank=completion_rank,
            completion_ridge=completion_ridge,
            rng=rng,
        )
    allowed = ", ".join(READOUT_NAMES)
    raise ValueError(f"unknown readout {readout!r}; choose one of: {allowed}")


def readout_constraint_diagnostics(
    counts: FloatArray,
    total_mass: int,
    *,
    tolerance: float = 1e-9,
) -> dict[str, float]:
    """Return integer, bound, and mass diagnostics for readout samples."""

    values = np.asarray(counts, dtype=float)
    if values.ndim not in (2, 3):
        raise ValueError("counts must have shape (time, space) or (runs, time, space)")
    noninteger = np.abs(values - np.rint(values)) > tolerance
    masses = values.sum(axis=-1)
    return {
        "noninteger_entry_fraction": float(np.mean(noninteger)),
        "lower_bound_violation_fraction": float(np.mean(values < -tolerance)),
        "upper_bound_violation_fraction": float(
            np.mean(values > float(total_mass) + tolerance)
        ),
        "maximum_instantaneous_mass_error": float(
            np.max(np.abs(masses - float(total_mass)))
        ),
    }


def _validate_trajectory(raw_counts: FloatArray, total_mass: int) -> FloatArray:
    values = np.asarray(raw_counts, dtype=float)
    if values.ndim != 2:
        raise ValueError("raw_counts must have shape (time, space)")
    if values.shape[1] < 2:
        raise ValueError("raw_counts must contain at least two spatial nodes")
    if total_mass <= 0:
        raise ValueError("total_mass must be positive")
    if not np.all(np.isfinite(values)):
        raise ValueError("raw_counts must be finite")
    return values
