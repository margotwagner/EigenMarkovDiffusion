import numpy as np

import eigendiffusion.simplex as simplex_module
from eigendiffusion.readouts import (
    simplex_bank_counts,
    warm_simplex_bank_counts,
    warm_simplex_bank_counts_with_diagnostics,
    warm_simplex_bank_readout,
)


def test_warm_simplex_matches_sorting_path_on_random_and_adversarial_rows():
    rng = np.random.default_rng(31)
    cases = [
        (rng.normal(0.0, 30.0, size=(80, 7)), 53),
        (rng.normal(10.0, 100.0, size=(50, 101)), 5275),
        (
            np.asarray(
                [
                    [2.5, 2.5, 2.5, 2.5],
                    [-2.0, 4.0, 4.0, 4.0],
                    [1.0e12, 1.0e12 + 0.25, -1.0e12, 0.0],
                    [10.0, 0.0, 0.0, 0.0],
                    [1.0e-12, 5.0 - 1.0e-12, 2.0, 3.0],
                ]
            ),
            10,
        ),
    ]

    for raw, total_mass in cases:
        expected = simplex_bank_counts(raw, total_mass)
        actual = warm_simplex_bank_counts(raw, total_mass)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-10)


def test_warm_simplex_forced_fallback_matches_sorting_projection():
    # The first row has threshold 99. The bank makes the next adjusted row
    # identically zero, so values > previous_threshold is empty and must fall
    # back to the historical sort.
    raw = np.asarray([[100.0, 0.0, 0.0], [-99.0, 0.0, 99.0]])
    expected = simplex_bank_counts(raw, 1)
    actual, diagnostics = warm_simplex_bank_counts_with_diagnostics(raw, 1)

    np.testing.assert_array_equal(actual, expected)
    assert diagnostics.fallback_rows == 1
    assert diagnostics.fallback_rate == 1.0
    assert diagnostics.support_updates[1] == -1


def test_warm_simplex_large_offset_remains_feasible_and_matches_sorting():
    raw = np.full((2, 10_000), 1.0e8)
    expected = simplex_bank_counts(raw, 1)
    actual = warm_simplex_bank_counts(raw, 1)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-10)
    assert np.min(actual) >= 0.0
    np.testing.assert_allclose(actual.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-12)


def test_warm_counts_detailed_result_constraints_and_diagnostics_agree():
    rng = np.random.default_rng(9)
    raw = rng.normal(2.0, 15.0, size=(40, 13))
    counts, diagnostics = warm_simplex_bank_counts_with_diagnostics(raw, 200)
    detailed = warm_simplex_bank_readout(raw, 200)

    np.testing.assert_array_equal(counts, detailed.counts)
    np.testing.assert_allclose(counts.sum(axis=1), 200.0, rtol=0.0, atol=1.0e-12)
    assert np.min(counts) >= 0.0
    np.testing.assert_allclose(
        detailed.residuals.sum(axis=1),
        0.0,
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_array_equal(detailed.adjusted_inputs[0], raw[0])
    np.testing.assert_allclose(
        detailed.adjusted_inputs[1:],
        raw[1:] + detailed.residuals[:-1],
        rtol=0.0,
        atol=1.0e-12,
    )

    assert diagnostics.total_rows == raw.shape[0]
    assert (
        diagnostics.zero_update_rows
        + diagnostics.corrected_rows
        + diagnostics.fallback_rows
        == diagnostics.warm_rows
    )
    assert np.isclose(
        diagnostics.zero_update_rate
        + diagnostics.corrected_rate
        + diagnostics.fallback_rate,
        1.0,
    )
    assert diagnostics.active_sizes.shape == (raw.shape[0],)
    assert diagnostics.support_changes.shape == (raw.shape[0],)
    assert np.all(
        (diagnostics.support_updates == -1)
        | (diagnostics.support_updates <= 8)
    )
    assert sum(diagnostics.updates_distribution.values()) == (
        diagnostics.zero_update_rows + diagnostics.corrected_rows
    )


def test_first_row_sorts_and_stable_later_rows_avoid_sort(monkeypatch):
    original_sort = simplex_module.np.sort
    sort_calls = 0

    def counted_sort(*args, **kwargs):
        nonlocal sort_calls
        sort_calls += 1
        return original_sort(*args, **kwargs)

    monkeypatch.setattr(simplex_module.np, "sort", counted_sort)
    raw = np.repeat(np.asarray([[8.0, -2.0, 4.0, 0.0]]), 12, axis=0)
    counts, diagnostics = warm_simplex_bank_counts_with_diagnostics(raw, 10)

    assert sort_calls == 1
    assert diagnostics.zero_update_rows == diagnostics.warm_rows
    assert diagnostics.corrected_rows == 0
    assert diagnostics.fallback_rows == 0
    np.testing.assert_allclose(counts.sum(axis=1), 10.0)
