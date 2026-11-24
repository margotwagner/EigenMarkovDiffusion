from __future__ import annotations

import csv
import json
from collections import Counter

import pytest

from eigendiffusion.compiled_simplex import compiled_warm_simplex_available
from eigendiffusion.joint_innovation_benchmark import (
    BATCHED_METHOD,
    EXACT_METHOD,
    STEPWISE_METHOD,
    JointInnovationBenchmarkCase,
    JointInnovationBenchmarkSpec,
    build_parser,
    run_joint_innovation_benchmark,
)


def test_joint_innovation_benchmark_writes_balanced_artifacts(tmp_path) -> None:
    if not compiled_warm_simplex_available():
        pytest.skip("Numba is unavailable")
    report = run_joint_innovation_benchmark(
        JointInnovationBenchmarkSpec(
            cases=(JointInnovationBenchmarkCase(7, 3, 1),),
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

    assert report["schema_version"] == 1
    assert report["study_contract"]["workload"] == "full_field_every_timestep"
    assert report["study_contract"]["setup_and_compiler_startup_in_timing"] is False
    diagnostic = report["diagnostics"][0]
    assert diagnostic["matching_rng_states"] == 3
    assert diagnostic["support_mismatch_entries_above_tolerance"] == 0
    assert diagnostic["correctness_pass"] is True
    gates = report["gates"][0]
    assert gates["n_nodes"] == 7
    assert gates["correctness_pass"] is True
    assert isinstance(gates["completion_speed_pass"], bool)
    assert isinstance(gates["end_to_end_speed_pass"], bool)
    assert isinstance(gates["exact_crossover_pass"], bool)
    resource = report["resources"][0]
    assert resource["batch_preparation_seconds"] > 0.0
    assert resource["packed_factor_array_bytes"] == 4 * 7 * (3 + 1) * 8
    assert (
        resource["batched_retained_array_bytes"] - resource["stepwise_retained_array_bytes"]
        == resource["packed_sampler_array_bytes"]
    )
    assert len(report["paired_comparisons"]) == 4

    with (tmp_path / "timing_samples.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 * 2 + 3 * 6
    completion_positions = Counter(
        (row["method"], int(row["execution_position"]))
        for row in rows
        if row["component"] == "completion_only"
    )
    assert completion_positions == {
        (method, position): 1
        for method in (STEPWISE_METHOD, BATCHED_METHOD)
        for position in range(2)
    }
    end_to_end_positions = Counter(
        (row["method"], int(row["execution_position"]))
        for row in rows
        if row["component"] == "end_to_end"
    )
    assert end_to_end_positions == {
        (method, position): 2
        for method in (STEPWISE_METHOD, BATCHED_METHOD, EXACT_METHOD)
        for position in range(3)
    }
    assert (tmp_path / "diagnostics.csv").is_file()
    assert (tmp_path / "resources.csv").is_file()
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8")) == json.loads(
        json.dumps(report)
    )


def test_joint_innovation_benchmark_parser_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.node_counts == [101, 201]
    assert args.modes == 75
    assert args.completion_rank == 25
    assert args.completion_warmups == 2
    assert args.completion_repeats == 30
    assert args.end_to_end_warmups == 1
    assert args.end_to_end_repeats == 30
    assert args.diagnostic_trajectories == 32
    assert args.bootstrap_resamples == 10_000
    assert args.output_dir == "outputs/joint_innovation_benchmark"


@pytest.mark.parametrize("options", [{"completion_repeats": 29}, {"end_to_end_repeats": 29}])
def test_joint_innovation_benchmark_requires_balanced_orders(options) -> None:
    with pytest.raises(ValueError, match="repeats must balance"):
        JointInnovationBenchmarkSpec(**options)
