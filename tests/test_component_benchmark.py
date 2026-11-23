import csv
import json

import numpy as np
import pytest

from eigendiffusion.cli import build_parser
from eigendiffusion.component_benchmark import (
    PRIMARY_STANDARD_COMPARATOR_METHOD,
    STANDARD_COMPARATOR_METHOD,
    STANDARD_COMPARATOR_METHODS,
    VECTORIZED_STANDARD_COMPARATOR_METHOD,
    benchmark_component_scaling,
    benchmark_full_field_components,
    build_full_field_component_pipeline,
    fixed_step_component_config,
    measure_component_memory,
    measure_standard_comparator_memory,
    run_full_field_component_pipeline,
    validate_full_field_output,
    write_component_benchmark_csv,
    write_component_benchmark_summary,
    write_component_memory_csv,
    write_component_scaling_csv,
    write_component_scaling_summary,
)
from eigendiffusion.config import DiffusionConfig


def _tiny_config() -> DiffusionConfig:
    return DiffusionConfig(
        n_particles=40,
        n_nodes=7,
        n_steps=5,
        impulse_index=3,
        dt=0.2,
    )


def test_full_field_component_pipeline_is_seed_reproducible() -> None:
    config = _tiny_config()
    pipeline = build_full_field_component_pipeline(
        config,
        retained_modes=4,
        handoff_time=0.4,
        completion_rank=1,
    )

    first = run_full_field_component_pipeline(
        pipeline,
        readout="delta_sigma_temporal",
        seed=12,
    )
    second = run_full_field_component_pipeline(
        pipeline,
        readout="delta_sigma_temporal",
        seed=12,
    )

    assert np.array_equal(first.counts, second.counts)
    assert first.counts.shape == (config.n_steps, config.n_nodes)
    assert np.all(first.counts == np.rint(first.counts))
    assert np.all(first.counts >= 0.0)
    assert np.all(first.counts.sum(axis=1) == config.n_particles)


def test_component_benchmark_records_all_stages_and_writes_artifacts(tmp_path) -> None:
    benchmark = benchmark_full_field_components(
        _tiny_config(),
        retained_modes=4,
        handoff_time=0.4,
        completion_rank=1,
        setup_repeats=1,
        trajectory_repeats=2,
        warmup_runs=0,
        seed=4,
    )

    assert len(benchmark.samples) == 15
    assert all(
        np.isfinite(sample.seconds) and sample.seconds >= 0.0
        for sample in benchmark.samples
    )
    assert benchmark.setup_array_bytes > 0
    assert benchmark.resident_array_bytes >= benchmark.setup_array_bytes
    assert benchmark.output_array_bytes == 5 * 7 * np.dtype(float).itemsize

    summary = benchmark.summary()
    components = summary["components"]
    assert "simulator_setup" in components
    assert "completion_setup" in components
    assert "raw_dynamics" in components
    assert "completion_application" in components
    assert "physical_readout:simplex_bank" in components
    assert "physical_readout:delta_sigma_temporal" in components
    assert "trajectory_total:simplex_bank" in components
    assert "trajectory_total:delta_sigma_temporal" in components
    assert (
        summary["readouts"]["delta_sigma_temporal"]["constraints"][
            "noninteger_entry_fraction"
        ]
        == 0.0
    )

    csv_path = write_component_benchmark_csv(
        benchmark,
        tmp_path / "samples.csv",
    )
    summary_path = write_component_benchmark_summary(
        benchmark,
        tmp_path / "summary.json",
    )
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(benchmark.samples)
    assert rows[0]["regime"] == "fixed_steps"
    assert rows[0]["output_workload"] == "full_field_every_timestep"
    assert json.loads(summary_path.read_text(encoding="utf-8"))["configuration"][
        "n_nodes"
    ] == 7


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (np.zeros((4, 7)), "unexpected shape"),
        (np.full((5, 7), np.nan), "nonfinite"),
        (np.zeros((5, 7)), "mass conservation"),
    ],
)
def test_full_field_validator_rejects_invalid_outputs(
    values: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        validate_full_field_output(values, _tiny_config())


def test_component_benchmark_parser_defaults() -> None:
    args = build_parser().parse_args(["benchmark-components"])
    assert args.nodes == 101
    assert args.steps == 101
    assert args.modes == 75
    assert args.handoff_time == 30.0
    assert args.completion_rank == 5
    assert args.readouts == ["simplex_bank", "delta_sigma_temporal"]
    assert args.setup_repeats == 5
    assert args.repeats == 10


def test_fixed_step_component_config_uses_stable_dt_and_fixed_frame_count() -> None:
    config = fixed_step_component_config(
        n_particles=40,
        n_nodes=201,
        n_steps=11,
        requested_dt=1.0,
        length=4.0,
        diffusion_coefficient=2.2e-4,
        impulse_fraction=0.59,
        stability_safety=0.95,
    )

    dx = 4.0 / 200.0
    expected_dt = 0.95 * dx**2 / (4.0 * 2.2e-4)
    assert config.n_steps == 11
    assert config.dt == pytest.approx(expected_dt)
    assert config.times[-1] == pytest.approx(10.0 * expected_dt)
    assert config.impulse_index == 118
    assert 1.0 - 4.0 * config.jump_rate * config.dt >= -1.0e-12


def test_component_scaling_runs_two_node_counts_and_reports_slopes(tmp_path) -> None:
    scaling = benchmark_component_scaling(
        node_counts=(5, 7),
        regimes=("fixed_steps", "fixed_duration"),
        operating_point="accuracy_first",
        n_particles=40,
        fixed_steps=5,
        total_time=0.8,
        requested_dt=0.2,
        length=4.0,
        diffusion_coefficient=2.2e-4,
        impulse_fraction=0.5,
        readouts=("simplex_bank",),
        setup_repeats=1,
        trajectory_repeats=1,
        warmup_runs=0,
        seed=7,
        isolate_memory=False,
    )

    assert len(scaling.benchmarks) == 4
    assert scaling.memory_records == ()
    assert {
        (benchmark.regime, benchmark.config.n_nodes)
        for benchmark in scaling.benchmarks
    } == {
        ("fixed_steps", 5),
        ("fixed_steps", 7),
        ("fixed_duration", 5),
        ("fixed_duration", 7),
    }
    assert all(benchmark.config.n_steps == 5 for benchmark in scaling.benchmarks)
    assert all(benchmark.readouts == ("simplex_bank",) for benchmark in scaling.benchmarks)

    summary = scaling.summary()
    assert summary["operating_point"] == "accuracy_first"
    assert summary["schema_version"] == 3
    assert len(summary["configurations"]) == 4
    assert summary["memory_records"] == []
    assert len(summary["standard_comparators"]) == 8
    assert summary["standard_memory_records"] == []
    assert len(summary["comparisons"]) == 4
    assert set(summary["scaling_slopes"]) == {"fixed_steps", "fixed_duration"}
    assert set(summary["standard_scaling_slopes"]) == set(
        STANDARD_COMPARATOR_METHODS
    )
    for method in STANDARD_COMPARATOR_METHODS:
        assert set(summary["standard_scaling_slopes"][method]) == {
            "fixed_steps",
            "fixed_duration",
        }
    for regime in ("fixed_steps", "fixed_duration"):
        slopes = summary["scaling_slopes"][regime]
        assert "setup_total" in slopes
        assert "trajectory_total:simplex_bank" in slopes
        assert np.isfinite(slopes["setup_array_bytes"])
        assert slopes["output_array_bytes"] == pytest.approx(1.0)

    for standard in summary["standard_comparators"]:
        assert standard["configuration"]["setup_applicable"] is False
        assert standard["memory"]["output_dtype"] == "int64"
        assert standard["constraints"]["noninteger_entry_fraction"] == 0.0
        assert standard["constraints"]["maximum_instantaneous_mass_error"] == 0.0
    for comparison in summary["comparisons"]:
        assert comparison["primary_standard_method"] == (
            PRIMARY_STANDARD_COMPARATOR_METHOD
        )
        assert set(comparison["standards"]) == set(STANDARD_COMPARATOR_METHODS)
        for readout in ("simplex_bank",):
            readout_comparison = comparison["readouts"][readout]
            ratio = readout_comparison[
                "steady_runtime_ratio_modal_over_standard"
            ]
            assert ratio > 0.0
            configuration = next(
                item
                for item in summary["configurations"]
                if item["configuration"]["regime"] == comparison["regime"]
                and item["configuration"]["n_nodes"] == comparison["n_nodes"]
            )
            modal_seconds = configuration["readouts"][readout][
                "measured_end_to_end_seconds"
            ]
            setup_seconds = configuration["components"]["setup_total"][
                "median_seconds"
            ]
            expected = (modal_seconds + setup_seconds / 10) / comparison[
                "standard_median_seconds"
            ]
            assert readout_comparison[
                "amortized_runtime_ratio_modal_over_standard"
            ]["10"] == pytest.approx(expected)

    samples_path = write_component_scaling_csv(
        scaling,
        tmp_path / "scaling_samples.csv",
    )
    memory_path = write_component_memory_csv(
        scaling,
        tmp_path / "scaling_memory.csv",
    )
    with samples_path.open(newline="", encoding="utf-8") as handle:
        sample_rows = list(csv.DictReader(handle))
    assert {row["pipeline"] for row in sample_rows} == {
        "completed_modal",
        "aggregated_multinomial",
        "aggregated_multinomial_vectorized",
    }
    assert memory_path.read_text(encoding="utf-8") == ""

    summary_path = write_component_scaling_summary(
        scaling,
        tmp_path / "scaling_summary.json",
    )
    written = json.loads(summary_path.read_text(encoding="utf-8"))
    assert written["operating_point"] == "accuracy_first"
    assert set(written["scaling_slopes"]) == {"fixed_steps", "fixed_duration"}


def test_component_scaling_parser_defaults_and_memory_toggle() -> None:
    defaults = build_parser().parse_args(["benchmark-component-scaling"])
    assert defaults.node_counts == [51, 101, 151, 201]
    assert defaults.regimes == ["fixed_steps", "fixed_duration"]
    assert defaults.operating_point == "accuracy_first"
    assert defaults.steps == 101
    assert defaults.isolate_memory is True
    assert defaults.include_standard_comparator is True

    custom = build_parser().parse_args(
        [
            "benchmark-component-scaling",
            "--node-counts",
            "7",
            "9",
            "--regimes",
            "fixed_steps",
            "--operating-point",
            "balanced",
            "--no-isolate-memory",
            "--no-include-standard-comparator",
        ]
    )
    assert custom.node_counts == [7, 9]
    assert custom.regimes == ["fixed_steps"]
    assert custom.operating_point == "balanced"
    assert custom.isolate_memory is False
    assert custom.include_standard_comparator is False


def test_component_scaling_can_disable_standard_comparator() -> None:
    scaling = benchmark_component_scaling(
        node_counts=(7,),
        regimes=("fixed_steps",),
        n_particles=40,
        fixed_steps=5,
        requested_dt=0.2,
        readouts=("simplex_bank",),
        setup_repeats=1,
        trajectory_repeats=1,
        warmup_runs=0,
        isolate_memory=False,
        include_standard_comparator=False,
    )

    assert scaling.benchmarks[0].standard_benchmark is None
    summary = scaling.summary()
    assert summary["standard_comparators"] == []
    assert summary["standard_memory_records"] == []
    assert summary["comparisons"] == []
    assert summary["standard_scaling_slopes"] == {}


def test_isolated_memory_worker_reports_phase_and_logical_memory() -> None:
    config = _tiny_config()
    record = measure_component_memory(
        config,
        regime="fixed_steps",
        operating_point="accuracy_first",
        retained_modes=4,
        handoff_time=0.4,
        completion_rank=1,
        completion_ridge=0.01,
        readout="simplex_bank",
        seed=18,
    )

    assert record.readout == "simplex_bank"
    assert record.baseline_rss_bytes > 0
    assert record.setup_peak_rss_bytes > 0
    assert record.process_peak_after_setup_rss_bytes >= record.baseline_peak_rss_bytes
    assert (
        record.process_peak_after_trajectory_rss_bytes
        >= record.process_peak_after_setup_rss_bytes
    )
    assert record.setup_array_bytes > 0
    assert record.resident_array_bytes >= record.setup_array_bytes
    assert record.output_array_bytes == 5 * 7 * np.dtype(float).itemsize
    assert record.returned_result_array_bytes >= record.output_array_bytes
    assert record.setup_peak_incremental_bytes == max(
        0,
        record.setup_peak_rss_bytes - record.baseline_rss_bytes,
    )


def test_standard_comparator_is_interleaved_and_satisfies_native_contract() -> None:
    benchmark = benchmark_full_field_components(
        _tiny_config(),
        retained_modes=4,
        handoff_time=0.4,
        completion_rank=1,
        readouts=("simplex_bank", "delta_sigma_temporal"),
        setup_repeats=1,
        trajectory_repeats=10,
        warmup_runs=1,
        seed=21,
        include_standard_comparator=True,
    )

    assert {item.method for item in benchmark.standard_benchmarks} == set(
        STANDARD_COMPARATOR_METHODS
    )
    for standard in benchmark.standard_benchmarks:
        assert len(standard.samples) == 10
        assert {sample.execution_position for sample in standard.samples} == {
            0,
            1,
            2,
            3,
        }
        assert standard.output_array_bytes == 5 * 7 * np.dtype(np.int64).itemsize
        assert standard.output_dtype == "int64"
        assert standard.constraint_diagnostics["noninteger_entry_fraction"] == 0.0
        assert (
            standard.constraint_diagnostics["maximum_instantaneous_mass_error"]
            == 0.0
        )

    modal_positions = {
        sample.repeat: sample.execution_position
        for sample in benchmark.samples
        if sample.component == "trajectory_total"
        and sample.readout == "simplex_bank"
    }
    assert set(modal_positions) == set(range(10))
    assert set(modal_positions.values()) == {0, 1, 2, 3}

    orders: dict[int, list[tuple[int, str]]] = {
        repeat: [] for repeat in range(10)
    }
    for standard in benchmark.standard_benchmarks:
        for sample in standard.samples:
            orders[sample.repeat].append(
                (sample.execution_position, standard.method)
            )
    for sample in benchmark.samples:
        if sample.component == "trajectory_total":
            orders[sample.repeat].append((sample.execution_position, sample.readout))
    ordered_names = {
        repeat: [name for _, name in sorted(values)]
        for repeat, values in orders.items()
    }
    task_names = (
        "simplex_bank",
        "delta_sigma_temporal",
        *STANDARD_COMPARATOR_METHODS,
    )
    for first in task_names:
        for second in task_names:
            if first == second:
                continue
            first_precedes = sum(
                names.index(first) < names.index(second)
                for names in ordered_names.values()
            )
            assert first_precedes == 5


@pytest.mark.parametrize(
    ("method", "workspace_vectors"),
    [
        (STANDARD_COMPARATOR_METHOD, 1),
        (VECTORIZED_STANDARD_COMPARATOR_METHOD, 4),
    ],
)
def test_isolated_standard_comparator_memory_worker(
    method: str,
    workspace_vectors: int,
) -> None:
    config = _tiny_config()
    record = measure_standard_comparator_memory(
        config,
        regime="fixed_steps",
        operating_point="accuracy_first",
        seed=23,
        method=method,
    )

    expected_bytes = 5 * 7 * np.dtype(np.int64).itemsize
    assert record.method == method
    assert record.variant == "native_integer_counts"
    assert record.setup_applicable is False
    assert record.setup_array_bytes == 0
    assert record.resident_array_bytes == expected_bytes
    assert record.output_array_bytes == expected_bytes
    assert record.returned_result_array_bytes == expected_bytes
    assert record.transient_core_array_bytes == (
        expected_bytes + workspace_vectors * 7 * 8
    )
    assert record.output_dtype == "int64"
    assert record.process_peak_after_setup_rss_bytes >= record.baseline_peak_rss_bytes
    assert (
        record.process_peak_after_trajectory_rss_bytes
        >= record.process_peak_after_setup_rss_bytes
    )
