import csv

import numpy as np

from eigendiffusion.rank_scaling import (
    BASELINE_POLICY,
    FIXED_DENSE_POLICY,
    FIXED_LOCAL_B12_POLICY,
    FIXED_LOCAL_B20_POLICY,
    CrossCaseRankScalingSpec,
    RankScalingPhysicalCase,
    evaluate_timing_cost_gates,
    intersect_case_eligibility,
    policies_for_case,
    run_cross_case_rank_scaling,
)


def test_case_policies_scale_local_bandwidth_and_positions():
    spec = CrossCaseRankScalingSpec()
    base, long = spec.cases
    base_by_name = {policy.name: policy for policy in policies_for_case(spec, base)}
    long_by_name = {policy.name: policy for policy in policies_for_case(spec, long)}

    assert base.diagnostic_positions == (1.6, 2.0, 2.4)
    assert np.allclose(long.diagnostic_positions, (2.4, 3.0, 3.6))
    assert base_by_name[FIXED_LOCAL_B12_POLICY].modal_bandwidth == 12
    assert base_by_name[FIXED_LOCAL_B20_POLICY].modal_bandwidth == 20
    assert long_by_name[FIXED_LOCAL_B12_POLICY].modal_bandwidth == 18
    assert long_by_name[FIXED_LOCAL_B20_POLICY].modal_bandwidth == 30
    assert base_by_name[FIXED_DENSE_POLICY].completion_closure == "dense_persistent"


def test_cross_case_eligibility_requires_every_case():
    eligible = intersect_case_eligibility(
        {
            "base": [BASELINE_POLICY, FIXED_DENSE_POLICY, FIXED_LOCAL_B12_POLICY],
            "long": [BASELINE_POLICY, FIXED_DENSE_POLICY],
        }
    )
    assert set(eligible) == {BASELINE_POLICY, FIXED_DENSE_POLICY}


def test_timing_cost_gate_uses_largest_physical_grid():
    rows = []
    for policy, setup, resident, runtime in (
        (BASELINE_POLICY, 2.0, 200.0, 1.0),
        (FIXED_DENSE_POLICY, 1.5, 150.0, 1.05),
        (FIXED_LOCAL_B12_POLICY, 1.0, 100.0, 1.051),
    ):
        rows.append(
            {
                "policy": policy,
                "regime": "physical_refinement",
                "n_nodes": 201,
                "setup_seconds": setup,
                "resident_array_bytes": resident,
                "median_run_seconds": runtime,
            }
        )
    gates = evaluate_timing_cost_gates(rows)
    assert gates[FIXED_DENSE_POLICY]["cost_pass"] is True
    assert gates[FIXED_LOCAL_B12_POLICY]["trajectory_runtime_pass"] is False
    assert gates[FIXED_LOCAL_B12_POLICY]["cost_pass"] is False


def test_tiny_cross_case_selection_records_disjoint_seeds(tmp_path):
    cases = (
        RankScalingPhysicalCase("base", 4.0, (9,)),
        RankScalingPhysicalCase("long", 6.0, (9,)),
    )
    spec = CrossCaseRankScalingSpec(
        cases=cases,
        n_particles=40,
        total_time=2.0,
        requested_dt=0.5,
        selection_runs=3,
        confirmation_runs=4,
        selection_bootstrap_resamples=4,
        confirmation_bootstrap_resamples=4,
        bootstrap_blocks=3,
        projections=2,
        histogram_bins=3,
        covariance_times=(0.5, 1.0, 2.0),
        distribution_times=(0.5, 1.0, 2.0),
        temporal_lag=0.5,
        fixed_step_count=5,
        timing_repeats=1,
        selection_seed=101,
        confirmation_seed=202,
        timing_seed=303,
        case_seed_stride=200_000,
    )
    summary = run_cross_case_rank_scaling(spec, tmp_path, stage="selection")
    assert summary["case_seeds"] == {"base": 101, "long": 200_101}
    assert summary["seed_provenance"]["confirmation_seed"] == 202
    assert set(summary["eligible_by_case"]) == {"base", "long"}
    for case in cases:
        timing_path = tmp_path / "selection" / case.name / "screening_accuracy.csv"
        with timing_path.open(newline="", encoding="utf-8") as handle:
            assert len(list(csv.DictReader(handle))) >= 2

    confirmation = run_cross_case_rank_scaling(
        spec,
        tmp_path,
        stage="confirmation",
    )
    assert confirmation["master_seed"] == 202
    assert confirmation["case_seeds"] == {"base": 202, "long": 200_202}
    assert confirmation["inherited_selection_source"].endswith(
        "selection/summary.json"
    )
    assert BASELINE_POLICY in confirmation["selected_policies"]
