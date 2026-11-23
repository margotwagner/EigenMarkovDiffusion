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
    LocalPersistentUnresolvedGaussianCompleter,
    PersistentUnresolvedGaussianCompleter,
    UnresolvedGaussianCompleter,
)
from .config import DiffusionConfig
from .simplex import _project_to_mass_simplex_prevalidated

FloatArray = NDArray[np.float64]
ReadoutName = Literal[
    "raw",
    "simplex_bank",
    "delta_sigma_temporal",
    "delta_sigma_neighbor",
    "unresolved_gaussian_completion",
    "persistent_unresolved_completion",
    "unresolved_gaussian_completion_simplex_bank",
    "persistent_unresolved_completion_simplex_bank",
    "local_persistent_unresolved_completion",
    "local_persistent_unresolved_completion_simplex_bank",
]
READOUT_NAMES: tuple[ReadoutName, ...] = (
    "raw",
    "simplex_bank",
    "delta_sigma_temporal",
    "delta_sigma_neighbor",
    "unresolved_gaussian_completion",
    "persistent_unresolved_completion",
    "unresolved_gaussian_completion_simplex_bank",
    "persistent_unresolved_completion_simplex_bank",
    "local_persistent_unresolved_completion",
    "local_persistent_unresolved_completion_simplex_bank",
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

    projected = np.empty_like(vector)
    floored = np.empty(vector.size, dtype=np.int64)
    workspace = np.empty((1, vector.size), dtype=float)
    return _integer_simplex_quantize_prevalidated(
        vector,
        total_mass,
        projected,
        projected,
        floored,
        workspace,
    )


def _integer_simplex_quantize_prevalidated(
    values: FloatArray,
    total_mass: int,
    out: FloatArray,
    projected: FloatArray,
    floored: NDArray[np.int64],
    workspace: FloatArray,
) -> FloatArray:
    """Quantize one validated row using caller-owned reusable buffers."""

    _project_to_mass_simplex_prevalidated(
        values,
        float(total_mass),
        projected,
        workspace,
    )
    np.floor(projected, out=workspace[0])
    np.copyto(floored, workspace[0], casting="unsafe")
    remainder = int(total_mass - int(floored.sum()))
    if remainder < 0 or remainder > values.size:
        raise RuntimeError("integer simplex rounding produced an invalid remainder")

    if remainder:
        np.subtract(projected, floored, out=workspace[0])
        # Sorting the negated fractions with a stable algorithm retains the
        # historical first-coordinate-wins rule for exact ties.
        np.negative(workspace[0], out=workspace[0])
        order = np.argsort(workspace[0], kind="stable")
        floored[order[:remainder]] += 1

    np.copyto(out, floored, casting="unsafe")
    if int(out.sum()) != int(total_mass):
        raise RuntimeError("integer simplex quantizer failed to conserve mass")
    return out


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
    outputs = np.empty_like(values)
    residuals = np.empty_like(values)
    adjusted = np.empty_like(values)
    previous_residual = np.zeros(values.shape[1], dtype=float)
    workspace = np.empty((1, values.shape[1]), dtype=float)

    for time_index, row in enumerate(values):
        current = adjusted[time_index]
        output = outputs[time_index]
        residual = residuals[time_index]
        np.add(row, previous_residual, out=current)
        _project_to_mass_simplex_prevalidated(
            current,
            float(total_mass),
            output,
            workspace,
        )
        np.subtract(current, output, out=residual)
        # Remove only floating-point zero-sum drift.
        residual[-1] -= float(residual.sum())

        previous_residual = residual

    return ReadoutResult(
        counts=outputs,
        residuals=residuals,
        adjusted_inputs=adjusted,
        readout_name="simplex_bank",
    )


def simplex_bank_counts(raw_counts: FloatArray, total_mass: int) -> FloatArray:
    """Return only simplex-bank counts using rolling row buffers.

    This is equivalent to ``simplex_bank_readout(...).counts`` but does not
    retain the adjusted-input and temporal-residual histories.  The residual
    ledger is still propagated exactly from one row to the next.
    """

    values = _validate_trajectory(raw_counts, total_mass)
    outputs = np.empty_like(values)
    residual = np.zeros(values.shape[1], dtype=float)
    current = np.empty(values.shape[1], dtype=float)
    workspace = np.empty((1, values.shape[1]), dtype=float)

    for row, output in zip(values, outputs, strict=True):
        np.add(row, residual, out=current)
        _project_to_mass_simplex_prevalidated(
            current,
            float(total_mass),
            output,
            workspace,
        )
        np.subtract(current, output, out=residual)
        residual[-1] -= float(residual.sum())

    return outputs


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
    outputs = np.empty_like(values)
    residuals = np.empty_like(values)
    adjusted = np.empty_like(values)
    previous_residual = np.zeros(values.shape[1], dtype=float)
    projected = np.empty(values.shape[1], dtype=float)
    floored = np.empty(values.shape[1], dtype=np.int64)
    workspace = np.empty((1, values.shape[1]), dtype=float)

    for time_index, row in enumerate(values):
        current = adjusted[time_index]
        output = outputs[time_index]
        residual = residuals[time_index]
        np.add(row, previous_residual, out=current)
        _integer_simplex_quantize_prevalidated(
            current,
            total_mass,
            output,
            projected,
            floored,
            workspace,
        )
        np.subtract(current, output, out=residual)
        residual[-1] -= float(residual.sum())

        previous_residual = residual

    return ReadoutResult(
        counts=outputs,
        residuals=residuals,
        adjusted_inputs=adjusted,
        readout_name="delta_sigma_temporal",
    )


def temporal_delta_sigma_counts(
    raw_counts: FloatArray,
    total_mass: int,
) -> FloatArray:
    """Return only temporal Delta-Sigma counts using rolling row buffers.

    This is equivalent to ``temporal_delta_sigma_readout(...).counts`` while
    retaining only the current adjusted row, quantizer work arrays, and the
    temporal residual ledger.
    """

    values = _validate_trajectory(raw_counts, total_mass)
    n_nodes = values.shape[1]
    outputs = np.empty_like(values)
    residual = np.zeros(n_nodes, dtype=float)
    current = np.empty(n_nodes, dtype=float)
    projected = np.empty(n_nodes, dtype=float)
    floored = np.empty(n_nodes, dtype=np.int64)
    workspace = np.empty((1, n_nodes), dtype=float)

    for row, output in zip(values, outputs, strict=True):
        np.add(row, residual, out=current)
        _integer_simplex_quantize_prevalidated(
            current,
            total_mass,
            output,
            projected,
            floored,
            workspace,
        )
        np.subtract(current, output, out=residual)
        residual[-1] -= float(residual.sum())

    return outputs


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
    completion_residual_variance_tolerance: float | None = None,
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
        residual_variance_tolerance=completion_residual_variance_tolerance,
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
    completion_residual_variance_tolerance: float | None = None,
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
        residual_variance_tolerance=completion_residual_variance_tolerance,
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
    completion_bandwidth: int = 12,
    completion_residual_variance_tolerance: float | None = None,
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
            completion_residual_variance_tolerance=(
                completion_residual_variance_tolerance
            ),
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
            completion_residual_variance_tolerance=(
                completion_residual_variance_tolerance
            ),
            rng=rng,
        )
    if readout in {
        "local_persistent_unresolved_completion",
        "local_persistent_unresolved_completion_simplex_bank",
    }:
        if config is None or retained_modes is None:
            raise ValueError(f"{readout} requires config and retained_modes")
        values = _validate_trajectory(raw_counts, total_mass)
        completer = LocalPersistentUnresolvedGaussianCompleter(
            config,
            retained_modes=retained_modes,
            completion_start_time=completion_start_time,
            completion_rank=completion_rank,
            modal_bandwidth=completion_bandwidth,
            ridge=completion_ridge,
            residual_variance_tolerance=(
                completion_residual_variance_tolerance
            ),
        )
        completed, additions = completer.complete(values, rng=rng)
        if readout.endswith("_simplex_bank"):
            constrained = simplex_bank_readout(completed, total_mass)
            return ReadoutResult(
                counts=constrained.counts,
                residuals=constrained.residuals,
                adjusted_inputs=completed,
                readout_name=readout,
            )
        return ReadoutResult(
            counts=completed,
            residuals=additions,
            adjusted_inputs=values.copy(),
            readout_name=readout,
        )
    if readout in {
        "unresolved_gaussian_completion_simplex_bank",
        "persistent_unresolved_completion_simplex_bank",
    }:
        if config is None or retained_modes is None:
            raise ValueError(f"{readout} requires config and retained_modes")
        completion = (
            unresolved_gaussian_completion_readout
            if readout.startswith("unresolved_")
            else persistent_unresolved_completion_readout
        )
        completed = completion(
            raw_counts,
            total_mass,
            config=config,
            retained_modes=retained_modes,
            completion_start_time=completion_start_time,
            completion_rank=completion_rank,
            completion_ridge=completion_ridge,
            completion_residual_variance_tolerance=(
                completion_residual_variance_tolerance
            ),
            rng=rng,
        )
        constrained = simplex_bank_readout(completed.counts, total_mass)
        return ReadoutResult(
            counts=constrained.counts,
            residuals=constrained.residuals,
            adjusted_inputs=completed.counts,
            readout_name=readout,
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
