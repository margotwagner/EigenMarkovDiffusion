import numpy as np

from eigendiffusion.config import DiffusionConfig
from eigendiffusion.ensemble import EnsembleResult, apply_readout_ensemble
from eigendiffusion.readouts import (
    apply_readout,
    integer_simplex_quantize,
    neighbor_delta_sigma_readout,
    simplex_bank_readout,
    temporal_delta_sigma_readout,
)


def _toy_trajectory() -> np.ndarray:
    return np.asarray(
        [
            [10.0, 0.0, 0.0],
            [7.2, 2.6, 0.2],
            [4.7, 3.4, 1.9],
            [2.2, 4.1, 3.7],
        ]
    )


def _assert_integer_physical(values: np.ndarray, total: int) -> None:
    np.testing.assert_allclose(values, np.rint(values))
    assert np.min(values) >= 0.0
    assert np.max(values) <= total
    np.testing.assert_allclose(values.sum(axis=-1), total)


def test_integer_simplex_quantizer_is_integer_bounded_and_mass_conserving() -> None:
    output = integer_simplex_quantize(np.asarray([-1.0, 3.2, 8.7]), 10)
    _assert_integer_physical(output[None, :], 10)


def test_temporal_delta_sigma_constraints_and_telescoping_error() -> None:
    raw = _toy_trajectory()
    result = temporal_delta_sigma_readout(raw, 10)
    _assert_integer_physical(result.counts, 10)

    cumulative_error = np.cumsum(result.counts - raw, axis=0)
    np.testing.assert_allclose(cumulative_error, -result.residuals, atol=1e-12)


def test_neighbor_delta_sigma_constraints() -> None:
    result = neighbor_delta_sigma_readout(
        _toy_trajectory(),
        10,
        spatial_error_fraction=0.5,
    )
    _assert_integer_physical(result.counts, 10)
    np.testing.assert_allclose(result.residuals.sum(axis=-1), 0.0, atol=1e-12)


def test_simplex_bank_is_nonnegative_and_mass_conserving() -> None:
    raw = _toy_trajectory().copy()
    raw[2, 0] = -1.0
    raw[2, 1] += 5.7
    result = simplex_bank_readout(raw, 10)
    assert np.min(result.counts) >= 0.0
    np.testing.assert_allclose(result.counts.sum(axis=-1), 10.0)


def test_raw_readout_is_unchanged() -> None:
    raw = _toy_trajectory()
    result = apply_readout(raw, 10, readout="raw")
    np.testing.assert_array_equal(result.counts, raw)


def test_apply_readout_ensemble_keeps_raw_ensemble_available() -> None:
    config = DiffusionConfig(n_particles=10, n_nodes=3, n_steps=4, impulse_index=0)
    raw = _toy_trajectory()
    ensemble = EnsembleResult(runs=np.stack([raw, raw]), model_name="correlated_modal")
    processed = apply_readout_ensemble(
        ensemble,
        config,
        readout="delta_sigma_temporal",
    )
    _assert_integer_physical(processed.runs.reshape(-1, 3), 10)
    assert processed.model_name == "correlated_modal+delta_sigma_temporal"
    assert processed.auxiliary is not None
    assert "readout_residual_l1_fraction" in processed.auxiliary
