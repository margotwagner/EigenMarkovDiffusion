from __future__ import annotations

import numpy as np

from eigendiffusion.benchmarking import (
    BENCHMARK_METHOD_NAMES,
    SELECTED_OPERATING_POINTS,
    benchmark_operating_points,
    benchmark_diffusion_methods,
    deep_numpy_nbytes,
    stable_diffusion_config,
)
from eigendiffusion.cli import build_parser


def test_stable_diffusion_config_preserves_total_time_and_stability() -> None:
    config = stable_diffusion_config(
        n_particles=100,
        n_nodes=51,
        total_time=10.0,
        requested_dt=1.0,
        length=4.0,
        diffusion_coefficient=2.2e-4,
        impulse_fraction=0.59,
    )
    assert np.isclose(config.times[-1], 10.0)
    assert 2.0 * config.jump_probability <= 0.5 + 1e-12

    # The stricter benchmark bound also guarantees nonnegative modal
    # one-step factors for the full Neumann eigenbasis.
    modal_upper_bound = 4.0 * config.jump_rate
    assert 1.0 - modal_upper_bound * config.dt >= -1e-12
    assert config.impulse_index == round(0.59 * 50)


def test_deep_numpy_nbytes_deduplicates_shared_arrays() -> None:
    array = np.zeros((3, 4), dtype=np.float64)
    assert deep_numpy_nbytes((array, array)) == array.nbytes


def test_benchmark_diffusion_methods_smoke() -> None:
    records = benchmark_diffusion_methods(
        node_counts=[11],
        particle_counts=[100],
        base_nodes=11,
        base_particles=100,
        total_time=2.0,
        requested_dt=0.2,
        length=4.0,
        diffusion_coefficient=2.2e-4,
        methods=BENCHMARK_METHOD_NAMES,
        retained_fraction=0.5,
        handoff_time=0.5,
        completion_rank=2,
        repeats=1,
        amortization_runs=10,
        seed=2,
    )
    assert len(records) == 2 * len(BENCHMARK_METHOD_NAMES)
    assert {record.sweep_axis for record in records} == {"nodes", "particles"}
    assert all(record.median_run_seconds >= 0.0 for record in records)
    assert all(record.resident_array_bytes > 0 for record in records)
    assert all(record.n_steps >= 2 for record in records)


def test_benchmark_models_cli_is_registered() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "benchmark-models",
            "--node-counts",
            "31",
            "--particle-counts",
            "1000",
            "--completion-rank",
            "5",
        ]
    )
    assert args.command == "benchmark-models"
    assert args.node_counts == [31]
    assert args.completion_rank == 5


def test_selected_operating_points_scale_and_benchmark() -> None:
    balanced = SELECTED_OPERATING_POINTS[1]
    assert balanced.scaled(101, 100.0) == (50, 10.0, 10)
    records = benchmark_operating_points(
        node_counts=[9],
        n_particles=50,
        total_time=1.0,
        requested_dt=0.2,
        repeats=1,
        amortization_runs=5,
        isolate_peak_rss=False,
        seed=8,
    )
    assert len(records) == 3
    assert {record.operating_point for record in records} == {
        "compute_first",
        "balanced",
        "accuracy_first",
    }
    assert all(record.peak_rss_bytes > 0 for record in records)
    assert all(record.closure == "dense_persistent" for record in records)


def test_operating_point_study_cli_is_registered() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "study-operating-points",
            "--node-counts",
            "11",
            "--accuracy-nodes",
            "11",
            "--accuracy-runs",
            "20",
        ]
    )
    assert args.command == "study-operating-points"
    assert args.accuracy_nodes == 11
    assert args.accuracy_runs == 20
    assert args.closure == "dense_persistent"


def test_structured_closure_study_cli_is_registered() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "study-structured-closure",
            "--bandwidths",
            "2",
            "4",
            "--heldout-runs",
            "25",
        ]
    )
    assert args.command == "study-structured-closure"
    assert args.bandwidths == [2, 4]
    assert args.heldout_runs == 25


def test_fitted_closure_study_cli_is_registered() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "study-fitted-closure",
            "--training-runs",
            "20",
            "--frequency-bins",
            "4",
            "--anchor-conditioned",
            "--physics-conditioned",
            "--innovation-rank",
            "6",
            "--persistent-innovations",
            "--run-ablation",
            "--use-bridge-anchors",
            "--gate-before-lockbox",
        ]
    )
    assert args.command == "study-fitted-closure"
    assert args.training_runs == 20
    assert args.frequency_bins == 4
    assert args.anchor_conditioned
    assert args.physics_conditioned
    assert args.innovation_rank == 6
    assert args.persistent_innovations
    assert args.run_ablation
    assert args.ablation_ranks == [4, 8, 12, 20]
    assert args.exact_anchor_runs == 100
    assert args.conditioning_degrees == [1]
    assert args.conditioning_folds == 3
    assert args.recommended_anchor_count == 4
    assert args.cv_mean_threshold == 0.87
    assert args.cv_max_threshold == 1.05
    assert args.use_bridge_anchors
    assert args.gate_before_lockbox
