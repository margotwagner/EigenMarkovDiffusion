import numpy as np
import pytest

import eigendiffusion.simplex as simplex_module
from eigendiffusion.readouts import (
    integer_simplex_quantize,
    simplex_bank_counts,
    simplex_bank_readout,
    temporal_delta_sigma_counts,
    temporal_delta_sigma_readout,
)
from eigendiffusion.simplex import (
    _project_to_mass_simplex_prevalidated,
    project_to_mass_simplex,
)


def _sorting_projection_reference(values: np.ndarray, total_mass: float) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    sorted_values = np.sort(vector)[::-1]
    cumulative = np.cumsum(sorted_values) - total_mass
    indices = np.arange(1, vector.size + 1, dtype=float)
    positive = sorted_values - cumulative / indices > 0.0
    rho = int(np.nonzero(positive)[0][-1])
    threshold = cumulative[rho] / float(rho + 1)
    projected = np.maximum(vector - threshold, 0.0)
    residual = total_mass - float(projected.sum())
    if abs(residual) > 0.0:
        projected[int(np.argmax(projected))] += residual
    return projected


@pytest.mark.parametrize(
    ("values", "total_mass"),
    [
        (np.asarray([2.0, 3.0, 5.0]), 13.0),
        (np.asarray([-100.0, 1.0, 2.0]), 4.0),
        (np.asarray([0.0, 1.0]), 1.0),
        (np.asarray([3.0, -1.0, -1.0, 3.0]), 2.0),
        (np.asarray([1.0e12, 1.0e12 + 0.25, -1.0e12]), 7.0),
    ],
)
def test_fast_simplex_matches_sorting_reference(
    values: np.ndarray,
    total_mass: float,
) -> None:
    actual = project_to_mass_simplex(values, total_mass)
    expected = _sorting_projection_reference(values, total_mass)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=5.0e-13)
    assert np.min(actual) >= 0.0
    np.testing.assert_allclose(actual.sum(), total_mass, rtol=0.0, atol=1.0e-12)


def test_all_active_simplex_path_does_not_sort(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_sort(*args: object, **kwargs: object) -> None:
        raise AssertionError("all-active projection unexpectedly sorted")

    monkeypatch.setattr(simplex_module.np, "sort", fail_sort)
    projected = project_to_mass_simplex(np.asarray([2.0, 3.0, 5.0]), 13.0)
    np.testing.assert_array_equal(projected, np.asarray([3.0, 4.0, 6.0]))


def test_prevalidated_simplex_supports_workspace_and_aliased_output() -> None:
    values = np.asarray([-4.0, 1.25, 2.75, 10.0])
    expected = _sorting_projection_reference(values, 9.0)
    aliased = values.copy()
    workspace = np.empty((2, values.size), dtype=float)

    returned = _project_to_mass_simplex_prevalidated(
        aliased,
        9.0,
        aliased,
        workspace,
    )

    assert returned is aliased
    np.testing.assert_array_equal(aliased, expected)


def test_integer_simplex_exact_ties_keep_first_coordinate_order() -> None:
    np.testing.assert_array_equal(
        integer_simplex_quantize(np.full(4, 0.5), 2),
        np.asarray([1.0, 1.0, 0.0, 0.0]),
    )
    np.testing.assert_array_equal(
        integer_simplex_quantize(np.asarray([-2.0, 0.5, 0.5, 0.5, 0.5]), 2),
        np.asarray([0.0, 1.0, 1.0, 0.0, 0.0]),
    )


def _adversarial_mass_conserving_trajectory() -> np.ndarray:
    return np.asarray(
        [
            [2.5, 2.5, 2.5, 2.5],
            [-2.0, 4.0, 4.0, 4.0],
            [0.25, 3.25, 3.25, 3.25],
            [10.0, 0.0, 0.0, 0.0],
            [1.0e-12, 5.0 - 1.0e-12, 2.0, 3.0],
        ]
    )


def test_counts_only_readouts_match_detailed_results_and_keep_ledgers() -> None:
    raw = _adversarial_mass_conserving_trajectory()

    simplex = simplex_bank_readout(raw, 10)
    np.testing.assert_array_equal(simplex_bank_counts(raw, 10), simplex.counts)
    np.testing.assert_allclose(simplex.counts.sum(axis=1), 10.0, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(simplex.residuals.sum(axis=1), 0.0, rtol=0.0, atol=1.0e-12)
    np.testing.assert_array_equal(simplex.adjusted_inputs[0], raw[0])
    np.testing.assert_array_equal(
        simplex.adjusted_inputs[1:],
        raw[1:] + simplex.residuals[:-1],
    )

    temporal = temporal_delta_sigma_readout(raw, 10)
    np.testing.assert_array_equal(temporal_delta_sigma_counts(raw, 10), temporal.counts)
    np.testing.assert_array_equal(temporal.counts, np.rint(temporal.counts))
    np.testing.assert_array_equal(temporal.counts.sum(axis=1), np.full(raw.shape[0], 10.0))
    np.testing.assert_allclose(temporal.residuals.sum(axis=1), 0.0, rtol=0.0, atol=1.0e-12)
    np.testing.assert_array_equal(temporal.adjusted_inputs[0], raw[0])
    np.testing.assert_array_equal(
        temporal.adjusted_inputs[1:],
        raw[1:] + temporal.residuals[:-1],
    )
    np.testing.assert_allclose(
        np.cumsum(temporal.counts - raw, axis=0),
        -temporal.residuals,
        rtol=0.0,
        atol=1.0e-12,
    )


@pytest.mark.parametrize(
    ("values", "total_mass", "message"),
    [
        (np.ones((2, 2)), 2.0, "one-dimensional"),
        (np.asarray([1.0, 2.0]), 0.0, "positive"),
        (np.asarray([1.0, np.nan]), 2.0, "finite"),
    ],
)
def test_fast_simplex_preserves_public_validation(
    values: np.ndarray,
    total_mass: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        project_to_mass_simplex(values, total_mass)


def test_empty_simplex_preserves_explicit_active_set_failure() -> None:
    with pytest.raises(RuntimeError, match="active set"):
        project_to_mass_simplex(np.asarray([], dtype=float), 1.0)
