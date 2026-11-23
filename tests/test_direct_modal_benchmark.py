from __future__ import annotations

import csv
import json

import numpy as np

from eigendiffusion.config import DiffusionConfig
from eigendiffusion.direct_modal import DirectModalGaussianDiffusion
from eigendiffusion.direct_modal_benchmark import (
    DirectModalStudySpec,
    analytic_direct_modal_diagnostics,
    build_parser,
    run_direct_modal_study,
)


def test_analytic_diagnostics_separate_retained_algebra_and_truncation() -> None:
    config = DiffusionConfig(
        n_particles=101,
        n_nodes=7,
        n_steps=8,
        impulse_index=2,
        dt=0.25,
    )
    reduced = analytic_direct_modal_diagnostics(
        DirectModalGaussianDiffusion(config, n_modes=3),
        regime="fixed_steps",
    )
    full = analytic_direct_modal_diagnostics(
        DirectModalGaussianDiffusion(config, n_modes=config.n_nodes),
        regime="fixed_steps",
    )

    assert reduced["retained_algebra_pass"] is True
    assert full["retained_algebra_pass"] is True
    assert float(reduced["raw_covariance_relative_frobenius_truncation_error"]) > 0.0
    assert float(full["raw_mean_relative_l2_truncation_error"]) < 1.0e-12
    assert float(full["raw_variance_relative_l2_truncation_error"]) < 1.0e-12
    assert float(full["raw_covariance_relative_frobenius_truncation_error"]) < 1.0e-7
    assert np.isfinite(float(reduced["raw_variance_relative_l2_truncation_error"]))


def test_tiny_study_writes_reproducible_artifact_schema(tmp_path) -> None:
    spec = DirectModalStudySpec(
        node_counts=(7,),
        mode_counts=(3,),
        regimes=("fixed_steps", "fixed_duration"),
        n_particles=50,
        fixed_steps=5,
        total_time=1.0,
        requested_dt=0.25,
        setup_repeats=1,
        trajectory_repeats=1,
        warmup_runs=0,
        seed=9,
    )
    report = run_direct_modal_study(spec, tmp_path)

    assert report["schema_version"] == 1
    assert len(report["analytic_accuracy"]) == 2
    assert all(row["retained_algebra_pass"] for row in report["analytic_accuracy"])
    assert len(report["logical_memory"]) == 8
    assert {row["method"] for row in report["logical_memory"]} == {
        "direct_modal",
        "direct_raw_spatial",
        "direct_simplex_bank",
        "exact_vectorized_multinomial",
    }

    for name in (
        "timing_samples.csv",
        "analytic_accuracy.csv",
        "logical_memory.csv",
        "summary.json",
    ):
        assert (tmp_path / name).is_file()

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["study_contract"]["modal_model"] == "DirectModalGaussianDiffusion"
    with (tmp_path / "timing_samples.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    # Per configuration: one setup plus four trajectory methods.
    assert len(rows) == 2 * 5
    assert {row["component"] for row in rows} == {"setup", "trajectory_total"}


def test_study_parser_defaults_and_small_override() -> None:
    defaults = build_parser().parse_args([])
    assert defaults.node_counts == [101, 201]
    assert defaults.modes == [20, 35, 50, 75]

    custom = build_parser().parse_args(
        [
            "--node-counts",
            "11",
            "--modes",
            "4",
            "--regimes",
            "fixed_steps",
            "--repeats",
            "2",
        ]
    )
    assert custom.node_counts == [11]
    assert custom.modes == [4]
    assert custom.regimes == ["fixed_steps"]
    assert custom.repeats == 2
