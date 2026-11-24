import builtins

import numpy as np
import pytest

import eigendiffusion
import eigendiffusion.compiled_simplex as compiled_simplex
from eigendiffusion.compiled_simplex import (
    compile_warm_simplex_kernel,
    compiled_warm_simplex_available,
    compiled_warm_simplex_bank_counts,
    compiled_warm_simplex_bank_counts_with_diagnostics,
    compiled_warm_simplex_bank_readout,
    compiled_warm_simplex_disk_cache_enabled,
)
from eigendiffusion.readouts import simplex_bank_readout, warm_simplex_bank_readout


@pytest.fixture(scope="module")
def compiled_kernel():
    if not compiled_warm_simplex_available():
        pytest.skip("Numba is unavailable")
    compile_warm_simplex_kernel()


@pytest.mark.parametrize(
    "raw, mass, message",
    [
        (np.zeros(2), 1, "shape"),
        (np.zeros((2, 1)), 1, "at least two"),
        (np.zeros((1, 2)), 0, "positive"),
        (np.zeros((1, 2)), np.inf, "total_mass must be finite"),
        (np.asarray([[0.0, np.nan]]), 1, "raw_counts must be finite"),
        (np.asarray([[0.0, np.inf]]), 1, "raw_counts must be finite"),
    ],
)
def test_compiled_counts_validates_inputs(raw, mass, message):
    with pytest.raises(ValueError, match=message):
        compiled_warm_simplex_bank_counts(raw, mass)


def test_compiled_exact_bank_matches_warm_and_sorting(compiled_kernel):
    rng = np.random.default_rng(31)
    mass = 300
    raw = rng.normal(10.0, 30.0, size=(80, 30))[:, ::2]
    raw += (mass - raw.sum(axis=1, keepdims=True)) / raw.shape[1]
    raw.flags.writeable = False

    counts, diagnostics = compiled_warm_simplex_bank_counts_with_diagnostics(raw, mass)
    detailed = compiled_warm_simplex_bank_readout(raw, mass)

    assert not diagnostics.used_fallback
    assert diagnostics.total_rows == raw.shape[0]
    assert diagnostics.fallback_rows == 0
    np.testing.assert_array_equal(counts, detailed.counts)
    np.testing.assert_array_equal(counts, compiled_warm_simplex_bank_counts(raw, mass))
    for reference in (warm_simplex_bank_readout(raw, mass), simplex_bank_readout(raw, mass)):
        np.testing.assert_allclose(counts, reference.counts, rtol=0.0, atol=1e-10)
        np.testing.assert_array_equal(counts > 0.0, reference.counts > 0.0)
        np.testing.assert_allclose(detailed.residuals, reference.residuals, rtol=0.0, atol=1e-10)
        np.testing.assert_allclose(
            detailed.adjusted_inputs, reference.adjusted_inputs, rtol=0.0, atol=1e-10
        )
    assert np.min(counts) >= 0.0
    np.testing.assert_allclose(counts.sum(axis=1), mass, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(detailed.residuals.sum(axis=1), 0.0, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(
        np.cumsum(raw - counts, axis=0), detailed.residuals, rtol=0.0, atol=1e-9
    )


def test_compiled_handles_tied_values_and_zero_support_boundaries(compiled_kernel):
    raw = np.asarray(
        [[2.5, 2.5, 2.5, 2.5], [-2.0, 4.0, 4.0, 4.0], [2.0, 1.0, 3.0, 4.0]]
    )
    counts, diagnostics = compiled_warm_simplex_bank_counts_with_diagnostics(raw, 10)
    assert not diagnostics.used_fallback
    np.testing.assert_array_equal(counts, simplex_bank_readout(raw, 10).counts)


@pytest.mark.parametrize(
    "raw, mass, triggering_row",
    [
        (np.asarray([[100.0, 0.0, 0.0], [-99.0, 0.0, 99.0], [0.0, 0.0, 1.0]]), 1, 1),
        (np.full((2, 10_000), 1e8), 1, 0),
        (np.repeat(np.tile([9e5, -9e5], 100)[None, :], 2, axis=0), 5275, 0),
    ],
)
def test_compiled_replays_entire_trajectory_on_guard_failure(
    compiled_kernel, raw, mass, triggering_row
):
    counts, diagnostics = compiled_warm_simplex_bank_counts_with_diagnostics(raw, mass)
    detailed = compiled_warm_simplex_bank_readout(raw, mass)
    reference = simplex_bank_readout(raw, mass)

    assert diagnostics.used_fallback
    assert diagnostics.fallback_row == triggering_row
    assert diagnostics.fallback_rows == raw.shape[0]
    np.testing.assert_array_equal(counts, reference.counts)
    np.testing.assert_array_equal(detailed.counts, reference.counts)
    np.testing.assert_array_equal(detailed.residuals, reference.residuals)
    np.testing.assert_array_equal(detailed.adjusted_inputs, reference.adjusted_inputs)


def test_compiled_preserves_empty_trajectories(compiled_kernel):
    raw = np.empty((0, 2))
    counts, diagnostics = compiled_warm_simplex_bank_counts_with_diagnostics(raw, 10)
    detailed = compiled_warm_simplex_bank_readout(raw, 10)
    assert counts.shape == detailed.counts.shape == raw.shape
    assert detailed.residuals.shape == detailed.adjusted_inputs.shape == raw.shape
    assert diagnostics.total_rows == 0
    assert not diagnostics.used_fallback


def test_compiled_refuses_numerically_infeasible_numpy_fallback(compiled_kernel):
    raw = 1e14 + np.random.default_rng(50).normal(0.0, 0.1, size=(1, 101))
    with pytest.raises(RuntimeError, match="NumPy sorting fallback did not produce"):
        compiled_warm_simplex_bank_counts(raw, 1)


def test_compensated_sums_preserve_large_well_conditioned_row(compiled_kernel):
    raw = np.full((1, 10_000), 2.7)
    counts, diagnostics = compiled_warm_simplex_bank_counts_with_diagnostics(
        raw,
        27_000,
    )

    assert not diagnostics.used_fallback
    np.testing.assert_allclose(counts, 2.7, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        counts.sum(axis=1),
        27_000.0,
        rtol=0.0,
        atol=1.0e-10,
    )


def test_disk_cache_is_best_effort(monkeypatch, tmp_path):
    missing = tmp_path / "missing"
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(missing))
    assert not compiled_warm_simplex_disk_cache_enabled()

    missing.mkdir()
    assert compiled_warm_simplex_disk_cache_enabled()


def test_optional_compiler_unavailable_has_explicit_error(monkeypatch):
    original_import = builtins.__import__

    def import_without_numba(name, *args, **kwargs):
        if name == "numba":
            raise ImportError("test without Numba")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_numba)
    compiled_simplex._compiled_kernel.cache_clear()
    try:
        assert not compiled_warm_simplex_available()
        with pytest.raises(RuntimeError, match="optional Numba installation"):
            compile_warm_simplex_kernel()
        with pytest.raises(RuntimeError, match="optional Numba installation"):
            compiled_warm_simplex_bank_counts(np.zeros((1, 2)), 10)
    finally:
        compiled_simplex._compiled_kernel.cache_clear()


def test_disabled_jit_is_not_reported_as_compiled(monkeypatch):
    if not compiled_warm_simplex_available():
        pytest.skip("Numba JIT is unavailable")
    from numba import config

    monkeypatch.setattr(config, "DISABLE_JIT", True)
    compiled_simplex._compiled_kernel.cache_clear()
    try:
        assert not compiled_warm_simplex_available()
        with pytest.raises(RuntimeError, match="requires Numba JIT to be enabled"):
            compiled_warm_simplex_bank_readout(np.zeros((1, 2)), 10)
    finally:
        compiled_simplex._compiled_kernel.cache_clear()


def test_compiled_simplex_public_apis_are_exported():
    assert eigendiffusion.CompiledSimplexDiagnostics is compiled_simplex.CompiledSimplexDiagnostics
    assert eigendiffusion.CompiledSimplexResult is compiled_simplex.CompiledSimplexResult
    assert eigendiffusion.compile_warm_simplex_kernel is compile_warm_simplex_kernel
    assert eigendiffusion.compiled_warm_simplex_available is compiled_warm_simplex_available
    assert (
        eigendiffusion.compiled_warm_simplex_bank_counts
        is compiled_warm_simplex_bank_counts
    )
    assert (
        eigendiffusion.compiled_warm_simplex_bank_counts_with_diagnostics
        is compiled_warm_simplex_bank_counts_with_diagnostics
    )
    assert (
        eigendiffusion.compiled_warm_simplex_bank_readout
        is compiled_warm_simplex_bank_readout
    )
    assert (
        eigendiffusion.compiled_warm_simplex_disk_cache_enabled
        is compiled_warm_simplex_disk_cache_enabled
    )
