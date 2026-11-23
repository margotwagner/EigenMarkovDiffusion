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

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


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
    ``out`` must have the same one-dimensional shape as ``values``.  If
    ``out`` must have the same shape as ``values`` and may alias it.  A
    supplied ``workspace`` is validated for callers that share it with other
    row kernels.  Those sequential kernels skip the all-active precheck
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


def _correct_projected_mass(projected: FloatArray, total_mass: float) -> None:
    """Put the final floating-point mass residual on one active coordinate."""

    residual = total_mass - float(projected.sum())
    if abs(residual) > 0.0:
        active = int(np.argmax(projected))
        projected[active] += residual
