"""Optional compiled execution of the exact residual-feedback simplex bank."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .readouts import _validate_trajectory, simplex_bank_counts, simplex_bank_readout
from .simplex import _MAX_SAFE_WARM_OFFSET_MULTIPLIER, _MAX_WARM_SUPPORT_UPDATES

FloatArray = NDArray[np.float64]
# Keep separately ordered compiled and NumPy sums comparable at count scale.
_MAX_ADJUSTED_L1 = 1.0e-10 / (16.0 * np.finfo(np.float64).eps)


@dataclass(frozen=True, slots=True)
class CompiledSimplexResult:
    """Counts and ledger histories from the compiled exact simplex bank."""

    counts: FloatArray
    residuals: FloatArray
    adjusted_inputs: FloatArray


@dataclass(frozen=True, slots=True)
class CompiledSimplexDiagnostics:
    """The triggering row and amount of fallback work for one trajectory."""

    total_rows: int
    fallback_row: int | None

    @property
    def used_fallback(self) -> bool:
        return self.fallback_row is not None

    @property
    def fallback_rows(self) -> int:
        """Number of rows replayed through the NumPy sorting bank."""

        return self.total_rows if self.used_fallback else 0


def _warm_simplex_kernel(
    values: FloatArray,
    mass: float,
    outputs: FloatArray,
    residuals: FloatArray,
    adjusted_inputs: FloatArray,
) -> int:
    """Fill trajectory buffers and return -1 or the row requiring replay."""

    n_rows, n_nodes = values.shape
    current = np.empty(n_nodes, dtype=np.float64)
    residual = np.zeros(n_nodes, dtype=np.float64)
    offset_limit = _MAX_SAFE_WARM_OFFSET_MULTIPLIER * max(mass, 1.0)
    adjusted_l1_limit = max(_MAX_ADJUSTED_L1, 2.0 * mass)
    collect_history = residuals.shape[0] > 0
    threshold = 0.0

    for time_index in range(n_rows):
        adjusted_l1 = 0.0
        adjusted_l1_compensation = 0.0
        for node in range(n_nodes):
            value = values[time_index, node] + residual[node]
            increment = abs(value) - adjusted_l1_compensation
            updated = adjusted_l1 + increment
            adjusted_l1_compensation = (updated - adjusted_l1) - increment
            adjusted_l1 = updated
            current[node] = value
        # Cancellation on large signed rows can amplify summation-order
        # differences in the threshold and residual drift correction.
        if not adjusted_l1 <= adjusted_l1_limit:
            return time_index

        if time_index == 0:
            # Seed the threshold with the historical descending-sort rule.
            ordered = np.sort(current)[::-1]
            prefix = 0.0
            prefix_compensation = 0.0
            threshold = np.nan
            for node in range(n_nodes):
                increment = ordered[node] - prefix_compensation
                updated = prefix + increment
                prefix_compensation = (updated - prefix) - increment
                prefix = updated
                candidate = (prefix - mass) / (node + 1)
                if ordered[node] > candidate:
                    threshold = candidate
            if not np.isfinite(threshold):
                return time_index
        else:
            certified = False
            for _ in range(_MAX_WARM_SUPPORT_UPDATES + 1):
                active_sum = 0.0
                active_sum_compensation = 0.0
                active_size = 0
                min_active = np.inf
                max_inactive = -np.inf
                for value in current:
                    if value > threshold:
                        increment = value - active_sum_compensation
                        updated = active_sum + increment
                        active_sum_compensation = (updated - active_sum) - increment
                        active_sum = updated
                        active_size += 1
                        min_active = min(min_active, value)
                    else:
                        max_inactive = max(max_inactive, value)
                if active_size == 0:
                    break
                threshold = (active_sum - mass) / active_size
                if not (-offset_limit <= threshold <= offset_limit):
                    break
                # The candidate solves the mass equation on this support.
                # These bounds certify that its strict KKT support agrees.
                if max_inactive <= threshold < min_active:
                    certified = True
                    break
            if not certified:
                return time_index

        output = outputs[time_index]
        output_sum = 0.0
        output_sum_compensation = 0.0
        largest_index = 0
        for node in range(n_nodes):
            value = max(current[node] - threshold, 0.0)
            output[node] = value
            increment = value - output_sum_compensation
            updated = output_sum + increment
            output_sum_compensation = (updated - output_sum) - increment
            output_sum = updated
            if value > output[largest_index]:
                largest_index = node
        corrected = output[largest_index] + (mass - output_sum)
        if not np.isfinite(corrected) or corrected < 0.0:
            return time_index
        output[largest_index] = corrected

        residual_sum = 0.0
        residual_sum_compensation = 0.0
        for node in range(n_nodes):
            residual[node] = current[node] - output[node]
            increment = residual[node] - residual_sum_compensation
            updated = residual_sum + increment
            residual_sum_compensation = (updated - residual_sum) - increment
            residual_sum = updated
        residual[-1] -= residual_sum
        if not np.isfinite(residual[-1]):
            return time_index
        if collect_history:
            adjusted_inputs[time_index] = current
            residuals[time_index] = residual
    return -1


@lru_cache(maxsize=1)
def _compiled_kernel():
    try:
        from numba import config, njit
    except Exception as exc:
        raise RuntimeError(
            "Compiled warm simplex readout requires a working optional Numba installation."
        ) from exc
    if config.DISABLE_JIT:
        raise RuntimeError("Compiled warm simplex readout requires Numba JIT to be enabled.")

    # Persist the machine-code specialization so a later process pays cache
    # loading rather than the full LLVM compilation cost.
    return njit(cache=compiled_warm_simplex_disk_cache_enabled())(
        _warm_simplex_kernel
    )


def compiled_warm_simplex_disk_cache_enabled() -> bool:
    """Return whether this installation has a requested writable cache."""

    configured = os.environ.get("NUMBA_CACHE_DIR")
    if configured:
        cache_path = Path(configured).expanduser()
        return cache_path.is_dir() and os.access(cache_path, os.W_OK)
    return os.access(Path(__file__).resolve().parent, os.W_OK)


def compiled_warm_simplex_available() -> bool:
    """Report whether the optional compiler can be imported without compiling."""

    try:
        _compiled_kernel()
    except RuntimeError:
        return False
    return True


def compile_warm_simplex_kernel() -> None:
    """Compile the trajectory kernel so callers can measure cold start separately."""

    values = np.zeros((1, 2), dtype=float)
    outputs = np.empty_like(values)
    empty_history = np.empty((0, 2), dtype=float)
    _compiled_kernel()(values, 1.0, outputs, empty_history, empty_history)


def _run_compiled_bank(
    raw_counts: FloatArray,
    total_mass: int,
    *,
    collect_history: bool,
) -> tuple[FloatArray, FloatArray, FloatArray, CompiledSimplexDiagnostics]:
    values = _validate_trajectory(raw_counts, total_mass)
    mass = float(total_mass)
    if not np.isfinite(mass):
        raise ValueError("total_mass must be finite")
    # Use one compiled signature even for strided or read-only inputs.
    values = np.require(values, requirements=("C", "A", "W"))
    outputs = np.empty_like(values)
    history_shape = values.shape if collect_history else (0, values.shape[1])
    residuals = np.empty(history_shape, dtype=float)
    adjusted_inputs = np.empty(history_shape, dtype=float)
    failed_row = _compiled_kernel()(values, mass, outputs, residuals, adjusted_inputs)
    diagnostics = CompiledSimplexDiagnostics(
        total_rows=values.shape[0],
        fallback_row=None if failed_row == -1 else failed_row,
    )
    if diagnostics.used_fallback:
        # Start the exact NumPy bank with a clean ledger on numerical failure.
        if collect_history:
            result = simplex_bank_readout(values, total_mass)
            outputs, residuals, adjusted_inputs = (
                result.counts,
                result.residuals,
                result.adjusted_inputs,
            )
        else:
            outputs = simplex_bank_counts(values, total_mass)
        if (
            not np.all(np.isfinite(outputs))
            or np.any(outputs < -1.0e-12)
            or not np.allclose(outputs.sum(axis=1), mass, rtol=0.0, atol=1.0e-10)
        ):
            raise RuntimeError(
                "NumPy sorting fallback did not produce a feasible simplex trajectory"
            )
    return outputs, residuals, adjusted_inputs, diagnostics


def compiled_warm_simplex_bank_counts(raw_counts: FloatArray, total_mass: int) -> FloatArray:
    """Return exact simplex-bank counts with a rolling residual ledger."""

    counts, _, _, _ = _run_compiled_bank(raw_counts, total_mass, collect_history=False)
    return counts


def compiled_warm_simplex_bank_readout(
    raw_counts: FloatArray,
    total_mass: int,
) -> CompiledSimplexResult:
    """Return exact simplex-bank counts together with the full ledger."""

    counts, residuals, adjusted_inputs, _ = _run_compiled_bank(
        raw_counts, total_mass, collect_history=True
    )
    return CompiledSimplexResult(counts, residuals, adjusted_inputs)


def compiled_warm_simplex_bank_counts_with_diagnostics(
    raw_counts: FloatArray,
    total_mass: int,
) -> tuple[FloatArray, CompiledSimplexDiagnostics]:
    """Return counts and the fallback status of this trajectory."""

    counts, _, _, diagnostics = _run_compiled_bank(raw_counts, total_mass, collect_history=False)
    return counts, diagnostics
