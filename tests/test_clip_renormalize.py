import numpy as np
import pytest

import eigendiffusion
import eigendiffusion.readouts as readouts_module
from eigendiffusion.readouts import (
    ClipRenormalizeDiagnostics,
    clip_renormalize_bank_counts,
    clip_renormalize_bank_counts_with_diagnostics,
    clip_renormalize_bank_readout,
    warm_simplex_bank_counts,
    warm_simplex_bank_readout,
)


def _manual_clip_renormalize(
    raw: np.ndarray,
    total_mass: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outputs = np.empty_like(raw)
    residuals = np.empty_like(raw)
    adjusted = np.empty_like(raw)
    previous = np.zeros(raw.shape[1], dtype=float)

    for time_index, row in enumerate(raw):
        current = row + previous
        clipped = np.maximum(current, 0.0)
        output = clipped * (total_mass / float(clipped.sum()))
        output[int(np.argmax(output))] += total_mass - float(output.sum())
        residual = current - output
        residual[-1] -= float(residual.sum())
        adjusted[time_index] = current
        outputs[time_index] = output
        residuals[time_index] = residual
        previous = residual
    return outputs, residuals, adjusted


def _ordinary_trajectory() -> np.ndarray:
    return np.asarray(
        [
            [-0.25, 4.0, 6.25],
            [1.0, 5.0, 4.0],
            [9.0, -0.2, 1.2],
            [4.0, 3.0, 3.0],
        ]
    )


def test_clip_renormalize_matches_row_semantics_and_reports_diagnostics() -> None:
    raw = _ordinary_trajectory()
    expected_counts, expected_residuals, expected_adjusted = (
        _manual_clip_renormalize(raw, 10.0)
    )

    detailed = clip_renormalize_bank_readout(raw, 10)
    counts, diagnostics = clip_renormalize_bank_counts_with_diagnostics(raw, 10)

    np.testing.assert_array_equal(detailed.counts, expected_counts)
    np.testing.assert_array_equal(detailed.residuals, expected_residuals)
    np.testing.assert_array_equal(detailed.adjusted_inputs, expected_adjusted)
    np.testing.assert_array_equal(counts, detailed.counts)
    np.testing.assert_allclose(counts.sum(axis=1), 10.0, rtol=0.0, atol=1.0e-12)
    assert np.min(counts) >= 0.0
    np.testing.assert_allclose(
        detailed.residuals.sum(axis=1),
        0.0,
        rtol=0.0,
        atol=1.0e-12,
    )
    assert detailed.readout_name == "clip_renormalize_bank"

    clipped = np.maximum(expected_adjusted, 0.0)
    expected_positive_masses = clipped.sum(axis=1)
    expected_negative_counts = (expected_adjusted < 0.0).sum(axis=1)
    expected_negative_masses = np.maximum(-expected_adjusted, 0.0).sum(axis=1)
    np.testing.assert_array_equal(diagnostics.positive_masses, expected_positive_masses)
    np.testing.assert_array_equal(
        diagnostics.scale_factors,
        10.0 / expected_positive_masses,
    )
    np.testing.assert_array_equal(
        diagnostics.negative_entry_counts,
        expected_negative_counts,
    )
    np.testing.assert_array_equal(diagnostics.negative_masses, expected_negative_masses)
    assert diagnostics.attempted_rows == raw.shape[0]
    assert diagnostics.failure_row is None
    assert diagnostics.failure_reason is None
    assert not diagnostics.used_fallback
    assert diagnostics.fallback_count == 0
    assert diagnostics.fallback_rate == 0.0
    assert diagnostics.negative_entries == int(expected_negative_counts.sum())
    assert diagnostics.total_negative_mass == float(expected_negative_masses.sum())
    assert not np.any(diagnostics.guard_failure_flags)
    assert not np.any(diagnostics.fallback_flags)


def test_guard_failure_transactionally_reruns_the_whole_warm_simplex_path() -> None:
    # Row 0 has different clip-renormalize and Euclidean-simplex outputs. Its
    # clip residual makes row 1 entirely nonpositive, triggering the guard.
    raw = np.asarray(
        [
            [-0.25, 4.0, 6.25],
            [0.2, -0.1, -0.2],
            [2.0, 3.0, 5.0],
        ]
    )
    exact = warm_simplex_bank_readout(raw, 10)
    detailed = clip_renormalize_bank_readout(raw, 10)
    counts, diagnostics = clip_renormalize_bank_counts_with_diagnostics(raw, 10)

    np.testing.assert_array_equal(detailed.counts, exact.counts)
    np.testing.assert_array_equal(detailed.residuals, exact.residuals)
    np.testing.assert_array_equal(detailed.adjusted_inputs, exact.adjusted_inputs)
    np.testing.assert_array_equal(counts, exact.counts)
    assert not np.array_equal(
        exact.counts[0],
        _manual_clip_renormalize(raw[:1], 10.0)[0][0],
    )

    assert detailed.readout_name == "clip_renormalize_bank"
    assert diagnostics.used_fallback
    assert diagnostics.fallback_rows == raw.shape[0]
    assert diagnostics.fallback_rate == 1.0
    assert diagnostics.attempted_rows == 2
    assert diagnostics.failure_row == 1
    assert diagnostics.failure_reason == "nonpositive_clipped_mass"
    np.testing.assert_array_equal(
        diagnostics.guard_failure_flags,
        np.asarray([False, True, False]),
    )
    np.testing.assert_array_equal(
        diagnostics.fallback_flags,
        np.ones(raw.shape[0], dtype=bool),
    )
    assert np.isnan(diagnostics.positive_masses[2])
    assert np.isnan(diagnostics.scale_factors[2])
    assert diagnostics.negative_entry_counts[2] == -1
    assert np.isnan(diagnostics.negative_masses[2])


def test_unsafe_normalization_scale_uses_exact_trajectory_fallback() -> None:
    raw = np.asarray([[1.0e-20, 2.0e-20], [0.5, 0.5]])
    counts, diagnostics = clip_renormalize_bank_counts_with_diagnostics(raw, 1)

    np.testing.assert_array_equal(counts, warm_simplex_bank_counts(raw, 1))
    np.testing.assert_array_equal(clip_renormalize_bank_counts(raw, 1), counts)
    assert diagnostics.used_fallback
    assert diagnostics.failure_row == 0
    assert diagnostics.failure_reason == "unsafe_scale_factor"
    assert diagnostics.scale_factors[0] > 1.0e19


def test_clipped_mass_above_five_percent_uses_exact_trajectory_fallback() -> None:
    raw = np.asarray([[-0.6, 4.0, 6.6], [2.0, 3.0, 5.0]])
    counts, diagnostics = clip_renormalize_bank_counts_with_diagnostics(raw, 10)

    np.testing.assert_array_equal(counts, warm_simplex_bank_counts(raw, 10))
    np.testing.assert_array_equal(clip_renormalize_bank_counts(raw, 10), counts)
    assert diagnostics.used_fallback
    assert diagnostics.failure_row == 0
    assert diagnostics.failure_reason == "excess_clipped_mass"
    np.testing.assert_allclose(diagnostics.positive_masses[0], 10.6)
    assert diagnostics.scale_factors[0] < 1.0 / 1.05


def test_counts_only_path_does_not_construct_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_diagnostics(*args: object, **kwargs: object) -> None:
        raise AssertionError("production counts path constructed diagnostics")

    monkeypatch.setattr(
        readouts_module,
        "ClipRenormalizeDiagnostics",
        fail_diagnostics,
    )
    raw = _ordinary_trajectory()
    np.testing.assert_array_equal(
        clip_renormalize_bank_counts(raw, 10),
        clip_renormalize_bank_readout(raw, 10).counts,
    )


def test_clip_renormalize_public_apis_are_exported() -> None:
    assert eigendiffusion.ClipRenormalizeDiagnostics is ClipRenormalizeDiagnostics
    assert eigendiffusion.clip_renormalize_bank_counts is clip_renormalize_bank_counts
    assert (
        eigendiffusion.clip_renormalize_bank_counts_with_diagnostics
        is clip_renormalize_bank_counts_with_diagnostics
    )
    assert eigendiffusion.clip_renormalize_bank_readout is clip_renormalize_bank_readout
