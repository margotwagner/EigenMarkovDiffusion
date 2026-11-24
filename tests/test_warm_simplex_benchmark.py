from __future__ import annotations

import csv
import json

import pytest

from eigendiffusion.warm_simplex_benchmark import (
    WarmSimplexBenchmarkCase,
    WarmSimplexBenchmarkSpec,
    build_parser,
    run_warm_simplex_benchmark,
)


def test_tiny_warm_simplex_benchmark_writes_paired_artifacts(tmp_path) -> None:
    report = run_warm_simplex_benchmark(
        WarmSimplexBenchmarkSpec(
            cases=(WarmSimplexBenchmarkCase(7, 3, 1),),
            n_particles=71,
            total_time=1.0,
            requested_dt=0.25,
            cached_trajectories=3,
            readout_warmups=1,
            readout_repeats=2,
            end_to_end_warmups=0,
            end_to_end_repeats=2,
            bootstrap_resamples=100,
            seed=7,
        ),
        tmp_path,
    )

    assert report["schema_version"] == 1
    assert len(report["diagnostics"]) == 1
    assert report["diagnostics"][0]["correctness_pass"] is True
    assert len(report["gates"]) == 1
    for filename in ("timing_samples.csv", "diagnostics.csv", "summary.json"):
        assert (tmp_path / filename).is_file()
    with (tmp_path / "timing_samples.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 * 2 + 3 * 2
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["study_contract"]["paired_cached_fields"] is True


def test_warm_simplex_benchmark_parser_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.node_counts == [101, 201]
    assert args.modes == 75
    assert args.completion_rank == 25
    assert args.readout_repeats == 30
    assert args.end_to_end_repeats == 24


def test_warm_simplex_benchmark_rejects_duplicate_node_cases() -> None:
    with pytest.raises(ValueError, match="unique node counts"):
        WarmSimplexBenchmarkSpec(
            cases=(
                WarmSimplexBenchmarkCase(7, 3, 1),
                WarmSimplexBenchmarkCase(7, 4, 1),
            )
        )
