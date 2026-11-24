from __future__ import annotations

import csv
import json

import pytest

from eigendiffusion.clip_renormalize_benchmark import (
    ClipRenormalizeBenchmarkCase,
    ClipRenormalizeBenchmarkSpec,
    build_parser,
    run_clip_renormalize_benchmark,
)


def test_tiny_clip_renormalize_benchmark_writes_artifacts(tmp_path) -> None:
    report = run_clip_renormalize_benchmark(
        ClipRenormalizeBenchmarkSpec(
            cases=(ClipRenormalizeBenchmarkCase(7, 3, 1),),
            n_particles=71,
            total_time=1.0,
            requested_dt=0.25,
            cached_trajectories=3,
            readout_warmups=1,
            readout_repeats=2,
            end_to_end_warmups=0,
            end_to_end_repeats=6,
            bootstrap_resamples=100,
            accuracy_runs=6,
            covariance_times=(0.25, 0.5, 1.0),
            distribution_times=(0.5, 1.0),
            temporal_lags=(0.25,),
            temporal_origins=2,
            distribution_projections=4,
            histogram_bins=5,
            seed=7,
        ),
        tmp_path,
    )

    assert report["schema_version"] == 1
    assert report["study_contract"]["paired_completed_fields"] is True
    assert len(report["diagnostics"]) == 1
    assert len(report["accuracy"]) == 3
    assert len(report["gates"]) == 1
    for filename in (
        "timing_samples.csv",
        "diagnostics.csv",
        "accuracy.csv",
        "accuracy_comparisons.csv",
        "summary.json",
    ):
        assert (tmp_path / filename).is_file()
    with (tmp_path / "timing_samples.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 * 2 + 3 * 6
    summary = json.loads(
        (tmp_path / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["study_contract"]["position_balanced_timing"] is True


def test_clip_renormalize_benchmark_parser_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.node_counts == [101, 201]
    assert args.modes == 75
    assert args.completion_rank == 25
    assert args.accuracy_runs == 500
    assert args.accuracy_bootstrap_resamples == 200
    assert args.end_to_end_repeats == 24


def test_clip_renormalize_benchmark_rejects_unbalanced_orders() -> None:
    with pytest.raises(ValueError, match="balance"):
        ClipRenormalizeBenchmarkSpec(end_to_end_repeats=20)
