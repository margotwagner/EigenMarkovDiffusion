from __future__ import annotations

import csv
import json

from eigendiffusion.config import DiffusionConfig
from eigendiffusion.innovation_completion import InnovationConditionedModalCompleter
from eigendiffusion.innovation_completion_benchmark import (
    InnovationCompletionScreenSpec,
    analytic_completion_diagnostics,
    build_parser,
    run_innovation_completion_screen,
)


def test_full_rank_analytic_completion_is_a_numerical_oracle() -> None:
    config = DiffusionConfig(
        n_particles=71,
        n_nodes=7,
        n_steps=6,
        impulse_index=2,
        dt=0.25,
    )
    diagnostics = analytic_completion_diagnostics(
        InnovationConditionedModalCompleter(config, retained_modes=3)
    )

    assert diagnostics["full_rank_oracle"] is True
    assert float(diagnostics["mean_relative_l2_error"]) < 1.0e-12
    assert float(diagnostics["full_covariance_relative_frobenius_error"]) < 1.0e-10
    assert (
        float(diagnostics["lag_1_time_covariance_relative_frobenius_error"])
        < 1.0e-10
    )


def test_tiny_completion_screen_writes_artifacts(tmp_path) -> None:
    report = run_innovation_completion_screen(
        InnovationCompletionScreenSpec(
            node_counts=(7,),
            mode_counts=(3,),
            conditional_ranks=(1,),
            n_particles=71,
            total_time=1.0,
            requested_dt=0.25,
            setup_repeats=1,
            trajectory_repeats=1,
            warmup_runs=0,
            seed=4,
        ),
        tmp_path,
    )

    assert report["schema_version"] == 1
    assert len(report["analytic_accuracy"]) == 2
    assert {row["rank_label"] for row in report["analytic_accuracy"]} == {
        "1",
        "full",
    }
    for filename in (
        "timing_samples.csv",
        "analytic_accuracy.csv",
        "logical_memory.csv",
        "summary.json",
    ):
        assert (tmp_path / filename).is_file()
    with (tmp_path / "timing_samples.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 * 4  # one setup plus three trajectory methods per rank
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["study_contract"]["regime"] == "fixed_duration"


def test_completion_screen_parser_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.node_counts == [101, 201]
    assert args.modes == [20, 35, 50, 75]
    assert args.ranks == [5, 10, 20]
    assert args.include_full_rank is True
