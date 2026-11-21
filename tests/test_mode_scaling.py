import csv
import json

import numpy as np

from eigendiffusion.mode_scaling import (
    MODE_SCALING_ACCEPTANCE_METRICS,
    ModeSelectionPolicy,
    ResolutionModeStudySpec,
    _fixed_step_config,
    _physical_config,
    build_analytic_mode_map,
    discrete_mean_cutoff_mode_count,
    mode_policy_noninferiority_pass,
    physical_position_indices,
    run_resolution_mode_study,
    spectral_decay_mode_count,
)
from eigendiffusion.operators import eigendecompose, neumann_laplacian_1d
from eigendiffusion.references import discrete_expected_diffusion


def _comparison_row(value: float) -> dict[str, object]:
    row: dict[str, object] = {
        metric: value for metric in MODE_SCALING_ACCEPTANCE_METRICS
    }
    row["mean_negative_mass_fraction"] = 0.0
    row["maximum_mass_error"] = 0.0
    return row


def test_mode_policy_gate_uses_paired_intervals_and_strict_temporal_margin():
    baseline = _comparison_row(0.10)
    floor = _comparison_row(0.04)
    candidate = _comparison_row(0.10)
    candidate["mean_relative_l2_error_difference_ci_upper"] = 0.039
    candidate["variance_relative_l2_error_difference_ci_upper"] = 0.0
    candidate["covariance_relative_frobenius_error_difference_ci_upper"] = 0.0
    candidate["cross_time_covariance_relative_frobenius_error"] = 0.109
    assert mode_policy_noninferiority_pass(candidate, baseline, floor)

    candidate["mean_relative_l2_error_difference_ci_upper"] = 0.041
    assert not mode_policy_noninferiority_pass(candidate, baseline, floor)
    candidate["mean_relative_l2_error_difference_ci_upper"] = 0.039
    candidate["cross_time_covariance_relative_frobenius_error"] = 0.111
    assert not mode_policy_noninferiority_pass(candidate, baseline, floor)


def test_continuum_spectral_cutoff_is_grid_independent():
    modes = spectral_decay_mode_count(
        length=4.0,
        diffusion_coefficient=2.2e-4,
        handoff_time=30.0,
        tolerance=1.0e-3,
    )
    assert modes == 42
    assert spectral_decay_mode_count(
        length=4.0,
        diffusion_coefficient=2.2e-4,
        handoff_time=30.0,
        tolerance=1.0e-3,
        maximum_modes=30,
    ) == 30


def test_discrete_mean_cutoff_plateaus_under_refinement():
    selected = []
    for n_nodes in (51, 101, 151, 201):
        spec = ResolutionModeStudySpec(node_counts=(n_nodes,))
        config = _physical_config(spec, n_nodes)
        selected.append(
            discrete_mean_cutoff_mode_count(
                config,
                handoff_time=0.3 * config.times[-1],
                post_handoff_relative_l2_tolerance=5.0e-3,
            )
        )
    assert selected == [50, 35, 34, 33]


def test_discrete_mean_cutoff_enforces_worst_post_handoff_time():
    tolerance = 5.0e-3
    spec = ResolutionModeStudySpec(node_counts=(101,))
    config = _physical_config(spec, 101)
    handoff_time = 0.3 * config.times[-1]
    selected = discrete_mean_cutoff_mode_count(
        config,
        handoff_time=handoff_time,
        post_handoff_relative_l2_tolerance=tolerance,
    )
    vectors = eigendecompose(
        neumann_laplacian_1d(
            config.n_nodes,
            config.length,
            config.diffusion_coefficient,
        )
    ).eigenvectors
    modal_mean = discrete_expected_diffusion(config) @ vectors
    handoff_step = int(np.argmin(np.abs(config.times - handoff_time)))
    post = modal_mean[handoff_step:]
    denominator = np.linalg.norm(post, axis=1)

    selected_error = np.linalg.norm(post[:, selected:], axis=1) / denominator
    previous_error = np.linalg.norm(post[:, selected - 1 :], axis=1) / denominator
    assert np.max(selected_error) <= tolerance
    assert np.max(previous_error) > tolerance


def test_analytic_map_keeps_mean_and_full_moment_policies_distinct():
    spec = ResolutionModeStudySpec(node_counts=(51, 101))
    records = build_analytic_mode_map(spec)
    by_key = {(row.n_nodes, row.policy): row for row in records}

    assert by_key[(101, "discrete_mean_cutoff_0p005")].final_modes == 35
    assert by_key[(101, "proportional_75_of_101")].final_modes == 75
    assert (
        by_key[(101, "full_moment_predictor")].final_modes
        >= by_key[(101, "discrete_mean_cutoff_0p005")].final_modes
    )
    assert by_key[(51, "spectral_decay_1e-3")].final_modes == 42


def test_diagnostics_use_physical_positions_and_timing_regimes_are_separate():
    spec = ResolutionModeStudySpec(node_counts=(51, 101), fixed_step_count=17)
    coarse = _physical_config(spec, 51)
    fine = _physical_config(spec, 101)
    positions = (1.6, 2.0, 2.4)
    for config in (coarse, fine):
        indices = physical_position_indices(config, positions)
        recovered = config.positions[indices]
        dx = config.length / (config.n_nodes - 1)
        assert np.all(np.abs(recovered - np.asarray(positions)) <= dx / 2.0)

    physical = _physical_config(spec, 101)
    fixed = _fixed_step_config(spec, 101)
    assert np.isclose(physical.times[-1], spec.total_time)
    assert fixed.n_steps == 17


def test_screening_study_writes_case_specific_floor(tmp_path):
    policies = (
        ModeSelectionPolicy(
            name="proportional",
            kind="proportional",
            reference_modes=4,
            reference_nodes=11,
        ),
        ModeSelectionPolicy(
            name="mean_cutoff",
            kind="discrete_mean_cutoff",
            predictor_post_handoff_mean_tolerance=0.5,
            completion_rank=1,
            modal_bandwidth=1,
        ),
    )
    spec = ResolutionModeStudySpec(
        node_counts=(11,),
        n_particles=80,
        total_time=2.0,
        requested_dt=0.5,
        impulse_fraction=0.5,
        handoff_fraction=0.5,
        policies=policies,
        predictor_candidate_modes=(2, 4, 6, 8, 10),
        covariance_times=(0.5, 1.0, 2.0),
        distribution_times=(0.5, 1.0, 2.0),
        diagnostic_positions=(1.6, 2.0, 2.4),
        temporal_lag=0.5,
        completion_rank=2,
        modal_bandwidth=2,
        screening_runs=3,
        final_runs=4,
        projections=4,
        histogram_bins=4,
        comparison_bootstrap_resamples=10,
        comparison_bootstrap_blocks=3,
        fixed_step_count=5,
        timing_repeats=1,
        seed=17,
    )
    summary = run_resolution_mode_study(spec, tmp_path, stage="screening")

    assert summary["case_specific_sampling_floors"] is True
    assert summary["paired_moment_noninferiority_intervals"] is True
    assert summary["comparison_baselines_by_node"] == {"11": "proportional"}
    assert "proportional" in summary["accuracy_eligible_policies"]
    assert summary["accuracy_rows"] == 3
    with (tmp_path / "screening_accuracy.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["policy"] == "sampling_floor"
    assert {row["policy"] for row in rows[1:]} == {"proportional", "mean_cutoff"}
    assert all("noninferiority_pass" in row for row in rows[1:])
    mean_row = next(row for row in rows if row["policy"] == "mean_cutoff")
    assert mean_row["completion_rank"] == "1"
    assert mean_row["modal_bandwidth"] == "1"
    saved_summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert saved_summary == summary

