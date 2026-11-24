from __future__ import annotations

import csv
import json
from collections import Counter

import pytest

from eigendiffusion.compiled_simplex import compiled_warm_simplex_available
from eigendiffusion.compiled_simplex_benchmark import (
    COMPILED_METHOD,
    EXACT_METHOD,
    WARM_METHOD,
    CompiledSimplexBenchmarkCase,
    CompiledSimplexBenchmarkSpec,
    build_parser,
    run_compiled_simplex_benchmark,
)


def test_tiny_compiled_simplex_benchmark_writes_balanced_artifacts(tmp_path) -> None:
    if not compiled_warm_simplex_available():
        pytest.skip("Numba is unavailable")
    report = run_compiled_simplex_benchmark(
        CompiledSimplexBenchmarkSpec(
            cases=(CompiledSimplexBenchmarkCase(7, 3, 1),),
            n_particles=71,
            total_time=1.0,
            requested_dt=0.25,
            cached_trajectories=3,
            readout_warmups=1,
            readout_repeats=2,
            end_to_end_warmups=1,
            end_to_end_repeats=6,
            bootstrap_resamples=100,
            seed=7,
        ),
        tmp_path,
    )

    assert report["schema_version"] == 1
    assert report["compiler_startup"]["import_and_jit_seconds"] >= 0.0
    assert report["resources"][0]["setup_seconds"] > 0.0
    assert report["study_contract"]["paired_cached_fields"] is True
    assert report["study_contract"]["workload"] == "full_field_every_timestep"
    assert report["study_contract"]["compiler_startup_in_timing_samples"] is False
    diagnostic = report["diagnostics"][0]
    assert diagnostic["cached_trajectories"] == 3
    assert diagnostic["support_identical_above_tolerance"] is True
    assert diagnostic["maximum_adjusted_absolute_difference"] <= 1.0e-10
    assert diagnostic["maximum_residual_absolute_difference"] <= 1.0e-10
    assert diagnostic["maximum_compiled_residual_sum_error"] <= 1.0e-10
    assert diagnostic["maximum_warm_residual_sum_error"] <= 1.0e-10
    assert diagnostic["correctness_pass"] is True
    assert report["gates"][0]["correctness_pass"] is True
    assert len(report["paired_comparisons"]) == 4

    with (tmp_path / "timing_samples.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 * 2 + 3 * 6
    readout_positions = Counter(
        (row["method"], int(row["execution_position"]))
        for row in rows
        if row["component"] == "readout_only"
    )
    assert readout_positions == {
        (method, position): 1 for method in (WARM_METHOD, COMPILED_METHOD) for position in range(2)
    }
    end_to_end_positions = Counter(
        (row["method"], int(row["execution_position"]))
        for row in rows
        if row["component"] == "end_to_end"
    )
    assert end_to_end_positions == {
        (method, position): 2
        for method in (WARM_METHOD, COMPILED_METHOD, EXACT_METHOD)
        for position in range(3)
    }
    assert (tmp_path / "diagnostics.csv").is_file()
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8")) == json.loads(
        json.dumps(report)
    )


def test_compiled_simplex_benchmark_parser_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.node_counts == [101, 201]
    assert args.modes == 75
    assert args.completion_rank == 25
    assert args.cached_trajectories == 32
    assert args.readout_warmups == 2
    assert args.readout_repeats == 30
    assert args.end_to_end_warmups == 1
    assert args.end_to_end_repeats == 24
    assert args.bootstrap_resamples == 10_000
    assert args.output_dir == "outputs/compiled_simplex_benchmark"


@pytest.mark.parametrize("options", [{"readout_repeats": 29}, {"end_to_end_repeats": 23}])
def test_compiled_simplex_benchmark_requires_balanced_orders(options) -> None:
    with pytest.raises(ValueError, match="repeats must balance"):
        CompiledSimplexBenchmarkSpec(**options)
