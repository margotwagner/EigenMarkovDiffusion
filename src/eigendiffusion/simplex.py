"""Fast projection onto a nonnegative fixed-mass simplex.

The common case for the modal diffusion models is an input whose Euclidean
simplex projection keeps every coordinate active.  In that case the
projection is only a uniform shift and does not require sorting.  Rows that
would acquire a zero coordinate fall back to the standard sorting-based
algorithm.

The public function performs the same validation as the historical projector
in :mod:`eigendiffusion.correlated_modal`.  The private prevalidated function
also accepts caller-owned output and fallback work buffers so trajectory and
step-wise kernels can avoid repeated allocations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

_MAX_WARM_SUPPORT_UPDATES = 8
_MAX_SAFE_WARM_OFFSET_MULTIPLIER = 1.0e6
_WARM_ZERO_UPDATE = 0
_WARM_CORRECTED = 1
_WARM_FALLBACK = 2


@dataclass(slots=True)
class _WarmSimplexState:
    """Trajectory-local state and scratch masks for warm projection."""

    threshold: float
    support: BoolArray
    candidate: BoolArray
    scratch: BoolArray
    initialized: bool = False

    @classmethod
    def create(cls, size: int) -> _WarmSimplexState:
        return cls(
            threshold=float("nan"),
            support=np.zeros(size, dtype=bool),
            candidate=np.empty(size, dtype=bool),
            scratch=np.empty(size, dtype=bool),
        )


def project_to_mass_simplex(values: FloatArray, total_mass: float) -> FloatArray:
    """Return the Euclidean projection onto ``x >= 0, sum(x) = total_mass``.

    An all-active projection is evaluated in linear time.  Otherwise this
    uses the standard sorting-based active-set projection.  The final
    floating-point mass residual is placed on the first largest projected
    coordinate, matching the historical implementation.
    """

    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if total_mass <= 0:
        raise ValueError("total_mass must be positive")
    if not np.all(np.isfinite(vector)):
        raise ValueError("values must be finite")

    output = np.empty_like(vector)
    return _project_to_mass_simplex_prevalidated(
        vector,
        float(total_mass),
        output,
    )


def _project_to_mass_simplex_prevalidated(
    values: FloatArray,
    total_mass: float,
    out: FloatArray,
    workspace: FloatArray | None = None,
) -> FloatArray:
    """Project one validated row into ``out`` and return ``out``.

    Parameters are assumed to have already passed the public validation.
    ``out`` must have the same one-dimensional shape as ``values`` and may
    alias it. A supplied ``workspace`` is validated for callers that share it
    with other row kernels. Those sequential kernels skip the all-active precheck
    because completed modal fields almost always require the sorting fallback.
    """

    n_values = values.size
    if workspace is not None and (
        workspace.ndim != 2
        or workspace.shape[0] < 1
        or workspace.shape[1] != n_values
    ):
        raise ValueError("workspace must have shape (k, values.size), k >= 1")
    if workspace is None and n_values:
        # A one-off all-active projection is only a uniform shift.
        threshold = (float(np.sum(values)) - total_mass) / float(n_values)
        if float(np.min(values)) > threshold:
            np.subtract(values, threshold, out=out)
            _correct_projected_mass(out, total_mass)
            return out

    # NumPy's allocation-backed sort/cumsum is faster than the caller-owned
    # equivalents for these small rows, and is the historical exact fallback.
    sorted_values = np.sort(values)[::-1]
    cumulative = np.cumsum(sorted_values) - total_mass

    indices = np.arange(1, n_values + 1, dtype=float)
    positive = sorted_values - cumulative / indices > 0.0
    if not np.any(positive):
        # This is mathematically unreachable for a finite vector and positive
        # total mass.  Keep the historical explicit failure for empty or
        # otherwise corrupted prevalidated inputs.
        raise RuntimeError("simplex projection failed to find an active set")
    rho = int(np.nonzero(positive)[0][-1])
    threshold = cumulative[rho] / float(rho + 1)
    np.subtract(values, threshold, out=out)
    np.maximum(out, 0.0, out=out)
    _correct_projected_mass(out, total_mass)
    return out


def _correct_projected_mass(projected: FloatArray, total_mass: float) -> bool:
    """Put the mass residual on one coordinate and report feasibility."""

    residual = total_mass - float(projected.sum())
    if abs(residual) > 0.0:
        active = int(np.argmax(projected))
        corrected = float(projected[active]) + residual
        projected[active] = corrected
        if not (corrected >= 0.0):
            return False
    return True


def _support_threshold(
    values: FloatArray,
    total_mass: float,
    support: BoolArray,
    active_size: int,
    anchor: float,
) -> float:
    """Compute a threshold, or ``nan`` when summation is too ill-conditioned."""

    # The direct sum is fastest for count-scale inputs. A numerically risky
    # common offset returns ``nan`` so the caller uses the historical sort
    # rather than accepting a warm result whose final correction could cross
    # zero. This branch is absent from the physical benchmark workloads.
    offset_limit = (
        _MAX_SAFE_WARM_OFFSET_MULTIPLIER * max(abs(total_mass), 1.0)
    )
    if abs(anchor) > offset_limit:
        return float("nan")
    threshold = (
        float(np.sum(values, where=support)) - total_mass
    ) / float(active_size)
    if not np.isfinite(threshold) or abs(threshold) > offset_limit:
        return float("nan")
    return threshold


def _sorting_projection_and_state(
    values: FloatArray,
    total_mass: float,
    out: FloatArray,
    workspace: FloatArray,
    state: _WarmSimplexState,
    collect_metadata: bool,
) -> tuple[int, int, int]:
    """Run the historical sort and refresh warm state from its result."""

    _project_to_mass_simplex_prevalidated(values, total_mass, out, workspace)
    np.greater(out, 0.0, out=state.candidate)
    active_size = int(np.count_nonzero(state.candidate))
    support_changes = (
        int(np.count_nonzero(state.candidate != state.support))
        if collect_metadata and state.initialized
        else 0
    )
    if active_size:
        first_active = int(np.flatnonzero(state.candidate)[0])
        state.threshold = _support_threshold(
            values,
            total_mass,
            state.candidate,
            active_size,
            float(values[first_active]),
        )
    else:
        state.threshold = float("nan")
    if collect_metadata:
        np.copyto(state.support, state.candidate)
    state.initialized = active_size > 0 and np.isfinite(state.threshold)
    return _WARM_FALLBACK, active_size, support_changes


def _project_to_mass_simplex_warm_prevalidated(
    values: FloatArray,
    total_mass: float,
    out: FloatArray,
    workspace: FloatArray,
    state: _WarmSimplexState,
    *,
    collect_metadata: bool = False,
) -> tuple[int, int, int, int]:
    """Warm-project one row and return path, updates, size, and changes.

    The first row and every failed warm attempt use the unchanged historical
    sorting projector. Later rows seed support with ``values >`` the previous
    threshold and perform at most eight strict active-set support updates.
    """

    if not state.initialized:
        path, active_size, changes = _sorting_projection_and_state(
            values, total_mass, out, workspace, state, collect_metadata
        )
        return path, 0, active_size, changes

    np.greater(values, state.threshold, out=state.candidate)
    candidate = state.candidate
    scratch = state.scratch
    active_size = int(np.count_nonzero(candidate))
    if active_size == 0:
        path, active_size, changes = _sorting_projection_and_state(
            values, total_mass, out, workspace, state, collect_metadata
        )
        return path, 0, active_size, changes

    anchor = float(state.threshold)
    updates = 0
    while True:
        threshold = _support_threshold(
            values,
            total_mass,
            candidate,
            active_size,
            anchor,
        )
        if not np.isfinite(threshold):
            break
        np.greater(values, threshold, out=scratch)
        if np.array_equal(candidate, scratch):
            # Fixed-point equality is the strict active/inactive KKT test.
            np.subtract(values, threshold, out=out)
            np.maximum(out, 0.0, out=out)
            if not _correct_projected_mass(out, total_mass):
                break
            support_changes = (
                int(np.count_nonzero(candidate != state.support))
                if collect_metadata
                else 0
            )
            if collect_metadata:
                np.copyto(state.support, candidate)
            state.threshold = threshold
            path = _WARM_ZERO_UPDATE if updates == 0 else _WARM_CORRECTED
            return path, updates, active_size, support_changes

        if updates >= _MAX_WARM_SUPPORT_UPDATES:
            break
        updates += 1
        candidate, scratch = scratch, candidate
        active_size = int(np.count_nonzero(candidate))
        if active_size == 0:
            break
        anchor = threshold

    path, active_size, changes = _sorting_projection_and_state(
        values, total_mass, out, workspace, state, collect_metadata
    )
    return path, updates, active_size, changes
