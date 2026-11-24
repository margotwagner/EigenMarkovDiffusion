from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from eigendiffusion.compiled_simplex import compiled_warm_simplex_available
from eigendiffusion.streaming_constructor_benchmark import (
    EXACT_METHOD,
    FULL_HISTORY_METHOD,
    STREAMING_METHOD,
    StreamingConstructorBenchmarkCase,
    StreamingConstructorBenchmarkSpec,
    build_parser,
    run_streaming_constructor_benchmark,
)


def test_streaming_constructor_benchmark_writes_balanced_artifacts(tmp_path) -> None:
    if not compiled_warm_simplex_available():
        pytest.skip("Numba is unavailable")
    spec = StreamingConstructorBenchmarkSpec(
        cases=(StreamingConstructorBenchmarkCase(7, 3, 1),),
        n_particles=71,
        total_time=1.0,
        requested_dt=0.25,
        completion_warmups=1,
        completion_repeats=2,
        end_to_end_warmups=1,
        end_to_end_repeats=6,
        diagnostic_trajectories=3,
        bootstrap_resamples=100,
        seed=7,
    )
    report = run_streaming_constructor_benchmark(spec, tmp_path)

    contract = report["study_contract"]
    assert contract["workload"] == "full_field_every_timestep"
    assert contract["setup_and_compiler_startup_in_trajectory_timing"] is False
    assert contract["peak_rss_gate_nodes"] == 201
    diagnostic = report["diagnostics"][0]
    assert diagnostic["correctness_pass"] is True
    assert diagnostic["joint_factors_maximum_absolute_difference"] == 0.0
    assert diagnostic["runtime_state_matches"] is True
    assert diagnostic["matching_rng_states"] == 3
    assert diagnostic["modal_exact_match_trajectories"] == 3
    assert diagnostic["spatial_exact_match_trajectories"] == 3
    assert diagnostic["physical_count_exact_match_trajectories"] == 3
    assert diagnostic["strict_support_mismatch_entries"] == 0

    resource = report["resources"][0]
    retained_bytes = resource["full_history_retained_array_bytes"]
    assert retained_bytes > 0
    assert resource["streaming_retained_array_bytes"] == retained_bytes
    assert resource["full_history_reported_array_bytes"] == retained_bytes
    assert resource["streaming_reported_array_bytes"] == retained_bytes
    assert report["gates"][0]["retained_storage_pass"] is True
    assert "peak_rss_pass" not in report["gates"][0]
    for method in ("full_history", "streaming"):
        assert resource[f"{method}_cold_prepared_setup_seconds"] > 0
        if Path("/proc/self/status").is_file():
            baseline = resource[f"{method}_rss_baseline_bytes"]
            assert baseline > 0
            for phase in ("after_setup", "peak_through_setup"):
                absolute = resource[f"{method}_rss_{phase}_bytes"]
                assert absolute > 0
                assert resource[f"{method}_rss_{phase}_increase_bytes"] == absolute - baseline
    assert len(report["paired_comparisons"]) == 5
    assert len(report["timing_summaries"]) == 7

    for artifact, component, methods, repetitions in (
        ("setup_samples", "cold_prepared_setup", (FULL_HISTORY_METHOD, STREAMING_METHOD), 2),
        ("timing_samples", "completion_only", (FULL_HISTORY_METHOD, STREAMING_METHOD), 1),
        ("timing_samples", "end_to_end", (FULL_HISTORY_METHOD, STREAMING_METHOD, EXACT_METHOD), 2),
    ):
        with (tmp_path / f"{artifact}.csv").open(newline="", encoding="utf-8") as handle:
            pairs = Counter(
                (row["method"], int(row["execution_position"]))
                for row in csv.DictReader(handle)
                if row["component"] == component
            )
        assert pairs == {
            (method, position): repetitions
            for method in methods
            for position in range(len(methods))
        }
    assert (tmp_path / "diagnostics.csv").is_file()
    assert (tmp_path / "resources.csv").is_file()
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8")) == json.loads(
        json.dumps(report)
    )


def test_streaming_constructor_benchmark_defaults_match_canonical_workload() -> None:
    args = build_parser().parse_args([])
    assert args.node_counts == [101, 201]
    assert args.modes == 75
    assert args.completion_rank == 25
    assert args.particles == 5_275
    assert args.total_time == 100.0
    assert args.setup_repeats == 4
    assert args.completion_repeats == args.end_to_end_repeats == 30
    assert args.diagnostic_trajectories == 32
    assert args.bootstrap_resamples == 10_000
    assert args.output_dir == "outputs/streaming_constructor_benchmark"
