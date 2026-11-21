from __future__ import annotations

from eigendiffusion.cli import (
    STRUCTURED_CLOSURE_ACCEPTANCE_METRICS,
    _structured_closure_noninferiority_pass,
    _structured_heldout_bandwidth_passes,
)


def _comparison_row(value: float) -> dict[str, object]:
    row: dict[str, object] = {
        metric: value for metric in STRUCTURED_CLOSURE_ACCEPTANCE_METRICS
    }
    row["mean_negative_mass_fraction"] = 0.0
    row["maximum_mass_error"] = 0.0
    return row


def test_structured_gate_uses_paired_interval_upper_endpoint() -> None:
    baseline = _comparison_row(0.10)
    floor = _comparison_row(0.02)
    candidate = _comparison_row(0.10)
    candidate["mean_relative_l2_error_difference_ci_upper"] = 0.021
    assert not _structured_closure_noninferiority_pass(
        candidate,
        baseline,
        floor,
    )
    candidate["mean_relative_l2_error_difference_ci_upper"] = 0.019
    assert _structured_closure_noninferiority_pass(candidate, baseline, floor)


def test_structured_gate_uses_stricter_temporal_margin() -> None:
    baseline = _comparison_row(0.10)
    floor = _comparison_row(0.04)
    candidate = _comparison_row(0.10)
    candidate["cross_time_covariance_relative_frobenius_error"] = 0.111
    assert not _structured_closure_noninferiority_pass(
        candidate,
        baseline,
        floor,
    )
    candidate["cross_time_covariance_relative_frobenius_error"] = 0.109
    assert _structured_closure_noninferiority_pass(candidate, baseline, floor)


def test_structured_selector_can_fall_through_to_wider_heldout_candidate() -> None:
    rows: list[dict[str, object]] = []
    for case in ("base", "long_domain"):
        floor = {"case": case, "closure": "sampling_floor", **_comparison_row(0.04)}
        dense = {"case": case, "closure": "dense", **_comparison_row(0.10)}
        local_12 = {"case": case, "closure": "local_b12", **_comparison_row(0.10)}
        local_20 = {"case": case, "closure": "local_b20", **_comparison_row(0.10)}
        if case == "long_domain":
            local_12["cross_time_covariance_relative_frobenius_error"] = 0.12
            local_20["cross_time_covariance_relative_frobenius_error"] = 0.109
        rows.extend((floor, dense, local_12, local_20))

    assert not _structured_heldout_bandwidth_passes(rows, 12)
    assert _structured_heldout_bandwidth_passes(rows, 20)

