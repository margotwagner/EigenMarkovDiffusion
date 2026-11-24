from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from eigendiffusion.compact_runtime_benchmark import (
    BATCHED_METHOD,
    COMPACT_METHOD,
    EXACT_METHOD,
    CompactRuntimeBenchmarkCase,
    CompactRuntimeBenchmarkSpec,
    build_parser,
    run_compact_runtime_benchmark,
)
from eigendiffusion.compiled_simplex import compiled_warm_simplex_available


def test_compact_runtime_benchmark_writes_balanced_artifacts(tmp_path) -> None:
    if not compiled_warm_simplex_available():
        pytest.skip("Numba is unavailable")
    report = run_compact_runtime_benchmark(
        CompactRuntimeBenchmarkSpec(
            cases=(CompactRuntimeBenchmarkCase(7, 3, 1),),
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
        ),
        tmp_path,
    )

    assert report["study_contract"]["workload"] == "full_field_every_timestep"
    assert report["study_contract"]["setup_and_compiler_startup_in_trajectory_timing"] is False
    diagnostic = report["diagnostics"][0]
    assert diagnostic["correctness_pass"] is True
    assert diagnostic["matching_rng_states"] == 3
    assert diagnostic["modal_exact_match_trajectories"] == 3
    assert diagnostic["spatial_exact_match_trajectories"] == 3
    assert diagnostic["physical_count_exact_match_trajectories"] == 3
    assert diagnostic["strict_support_mismatch_entries"] == 0
    for release_check in (
        "builder_released",
        "full_reference_released",
        "full_reference_histories_released",
        "target_innovation_history_released",
        "initial_modal_backing_released",
        "conditioning_history_released",
        "builder_release_pass",
    ):
        assert diagnostic[release_check] is True
    assert len(report["paired_comparisons"]) == 4

    resource = report["resources"][0]
    assert resource["builder_setup_seconds"] > 0
    assert resource["batch_preparation_seconds"] > 0
    assert resource["compact_conversion_seconds"] > 0
    assert resource["compact_cold_prepared_setup_seconds"] > 0
    assert resource["batched_cold_prepared_setup_seconds"] > resource["builder_setup_seconds"]
    assert resource["compact_retained_array_bytes"] == resource["compact_reported_array_bytes"]
    assert (
        resource["converted_retained_array_bytes_after_builder_release"]
        == resource["compact_retained_array_bytes"]
    )
    assert (
        resource["packed_sampler_array_bytes"]
        < resource["compact_retained_array_bytes"]
        < resource["batched_retained_array_bytes"]
    )
    if Path("/proc/self/status").is_file():
        for method in ("batched", "compact"):
            baseline = resource[f"{method}_rss_baseline_bytes"]
            for phase in ("baseline", "after_setup", "peak_through_setup"):
                assert resource[f"{method}_rss_{phase}_bytes"] > 0
            for phase in ("after_setup", "peak_through_setup"):
                assert (
                    resource[f"{method}_rss_{phase}_increase_bytes"]
                    == resource[f"{method}_rss_{phase}_bytes"] - baseline
                )
    for comparison in ("compact_vs_exact", "batched_vs_exact"):
        value = resource[f"{comparison}_break_even_trajectories"]
        assert value is None or value >= 1
    assert all(
        isinstance(value, bool)
        for name, value in report["gates"][0].items()
        if name != "n_nodes"
    )

    with (tmp_path / "setup_samples.csv").open(newline="", encoding="utf-8") as handle:
        setup_rows = list(csv.DictReader(handle))
    assert len(setup_rows) == 2 * 4
    assert Counter((row["method"], int(row["execution_position"])) for row in setup_rows) == {
        (method, position): 2
        for method in (BATCHED_METHOD, COMPACT_METHOD)
        for position in range(2)
    }
    with (tmp_path / "timing_samples.csv").open(newline="", encoding="utf-8") as handle:
        timing_rows = list(csv.DictReader(handle))
    assert len(timing_rows) == 2 * 2 + 3 * 6
    for component, methods, repetitions in (
        ("completion_only", (BATCHED_METHOD, COMPACT_METHOD), 1),
        ("end_to_end", (BATCHED_METHOD, COMPACT_METHOD, EXACT_METHOD), 2),
    ):
        observed = Counter(
            (row["method"], int(row["execution_position"]))
            for row in timing_rows
            if row["component"] == component
        )
        assert observed == {
            (method, position): repetitions
            for method in methods
            for position in range(len(methods))
        }
    assert (tmp_path / "diagnostics.csv").is_file()
    assert (tmp_path / "resources.csv").is_file()
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8")) == json.loads(
        json.dumps(report)
    )


def test_compact_runtime_benchmark_defaults_match_canonical_workload() -> None:
    args = build_parser().parse_args([])
    assert args.node_counts == [101, 201]
    assert args.modes == 75
    assert args.completion_rank == 25
    assert args.setup_repeats == 4
    assert args.completion_repeats == args.end_to_end_repeats == 30
    assert args.diagnostic_trajectories == 32
    assert args.bootstrap_resamples == 10_000
    assert args.output_dir == "outputs/compact_runtime_benchmark"


@pytest.mark.parametrize(
    "options",
    [{"setup_repeats": 1}, {"completion_repeats": 29}, {"end_to_end_repeats": 29}],
)
def test_compact_runtime_benchmark_requires_balanced_orders(options) -> None:
    with pytest.raises(ValueError, match="repeats must balance"):
        CompactRuntimeBenchmarkSpec(**options)
