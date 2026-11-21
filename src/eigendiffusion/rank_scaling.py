"""Cross-physics selection and timing gates for adaptive completion rank."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence

from .mode_scaling import (
    ModeSelectionPolicy,
    ResolutionModeStudySpec,
    run_resolution_mode_study,
)

RankScalingStage = Literal["selection", "confirmation", "timing"]

BASELINE_POLICY = "proportional_dense_rank5"
FIXED_DENSE_POLICY = "fixed75_dense_energy90"
FIXED_LOCAL_B12_POLICY = "fixed75_local_energy90_b12_scaled"
FIXED_LOCAL_B20_POLICY = "fixed75_local_energy90_b20_scaled"


@dataclass(frozen=True, slots=True)
class RankScalingPhysicalCase:
    """One physical domain used by the cross-case promotion gate."""

    name: str
    length: float
    node_counts: tuple[int, ...] = (151, 201)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("case name must be nonempty")
        if self.length <= 0.0:
            raise ValueError("case length must be positive")
        if not self.node_counts or any(nodes < 3 for nodes in self.node_counts):
            raise ValueError("case node counts must be at least three")
        if len(set(self.node_counts)) != len(self.node_counts):
            raise ValueError("case node counts must be unique")

    @property
    def diagnostic_positions(self) -> tuple[float, float, float]:
        """Return fixed domain-relative diagnostic positions."""

        return (0.4 * self.length, 0.5 * self.length, 0.6 * self.length)


DEFAULT_RANK_SCALING_CASES = (
    RankScalingPhysicalCase("base_length4", 4.0),
    RankScalingPhysicalCase("long_length6", 6.0),
)


@dataclass(frozen=True, slots=True)
class CrossCaseRankScalingSpec:
    """Configuration for selection, independent confirmation, and timing."""

    cases: tuple[RankScalingPhysicalCase, ...] = DEFAULT_RANK_SCALING_CASES
    n_particles: int = 5_275
    total_time: float = 100.0
    requested_dt: float = 1.0
    diffusion_coefficient: float = 2.20e-4
    impulse_fraction: float = 0.59
    stability_safety: float = 0.95
    handoff_fraction: float = 0.30
    fixed_modes: int = 75
    residual_variance_tolerance: float = 0.10
    completion_ridge: float = 1.0e-2
    selection_runs: int = 100
    confirmation_runs: int = 250
    selection_bootstrap_resamples: int = 100
    confirmation_bootstrap_resamples: int = 200
    bootstrap_blocks: int = 20
    confidence_level: float = 0.95
    projections: int = 64
    histogram_bins: int = 30
    covariance_times: tuple[float, ...] = (1.0, 5.0, 20.0, 50.0, 100.0)
    distribution_times: tuple[float, ...] = (20.0, 50.0, 100.0)
    temporal_lag: float = 1.0
    fixed_step_count: int = 101
    timing_repeats: int = 7
    setup_repeats: int = 3
    selection_seed: int = 20_261_130
    confirmation_seed: int = 20_261_201
    timing_seed: int = 20_261_202
    case_seed_stride: int = 1_000_000
    trajectory_runtime_ratio_limit: float = 1.05

    def __post_init__(self) -> None:
        if not self.cases or len({case.name for case in self.cases}) != len(self.cases):
            raise ValueError("physical cases must have unique names")
        if self.n_particles <= 0 or self.fixed_modes <= 0:
            raise ValueError("particle and fixed-mode counts must be positive")
        if self.total_time <= 0.0 or self.requested_dt <= 0.0:
            raise ValueError("time parameters must be positive")
        if self.diffusion_coefficient <= 0.0:
            raise ValueError("diffusion coefficient must be positive")
        if not 0.0 <= self.impulse_fraction <= 1.0:
            raise ValueError("impulse fraction must lie in [0, 1]")
        if not 0.0 < self.handoff_fraction < 1.0:
            raise ValueError("handoff fraction must lie in (0, 1)")
        if not 0.0 < self.residual_variance_tolerance < 1.0:
            raise ValueError("residual variance tolerance must lie in (0, 1)")
        if self.selection_runs <= 1 or self.confirmation_runs <= 1:
            raise ValueError("selection and confirmation runs must exceed one")
        if self.selection_bootstrap_resamples <= 0 or self.confirmation_bootstrap_resamples <= 0:
            raise ValueError("bootstrap resamples must be positive")
        if self.bootstrap_blocks <= 1:
            raise ValueError("bootstrap blocks must exceed one")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence level must lie in (0, 1)")
        if self.timing_repeats <= 0 or self.setup_repeats <= 0 or self.fixed_step_count < 2:
            raise ValueError("timing settings must be positive")
        if self.case_seed_stride <= 100_000:
            raise ValueError("case_seed_stride must separate internal seed ranges")
        if self.trajectory_runtime_ratio_limit < 1.0:
            raise ValueError("runtime ratio limit must be at least one")
        seeds = {self.selection_seed, self.confirmation_seed, self.timing_seed}
        if len(seeds) != 3:
            raise ValueError("selection, confirmation, and timing seeds must differ")


def policies_for_case(
    spec: CrossCaseRankScalingSpec,
    case: RankScalingPhysicalCase,
) -> tuple[ModeSelectionPolicy, ...]:
    """Build case-specific policies with physically scaled local bandwidths."""

    bandwidth_scale = case.length / 4.0
    bandwidth_12 = max(1, int(round(12 * bandwidth_scale)))
    bandwidth_20 = max(1, int(round(20 * bandwidth_scale)))
    adaptive = {
        "fixed_modes": spec.fixed_modes,
        "completion_residual_variance_tolerance": (
            spec.residual_variance_tolerance
        ),
    }
    return (
        ModeSelectionPolicy(
            name=BASELINE_POLICY,
            kind="proportional",
            completion_rank=5,
            completion_closure="dense_persistent",
        ),
        ModeSelectionPolicy(
            name=FIXED_DENSE_POLICY,
            kind="fixed",
            completion_closure="dense_persistent",
            **adaptive,
        ),
        ModeSelectionPolicy(
            name=FIXED_LOCAL_B12_POLICY,
            kind="fixed",
            modal_bandwidth=bandwidth_12,
            completion_closure="local_persistent",
            **adaptive,
        ),
        ModeSelectionPolicy(
            name=FIXED_LOCAL_B20_POLICY,
            kind="fixed",
            modal_bandwidth=bandwidth_20,
            completion_closure="local_persistent",
            **adaptive,
        ),
    )


def intersect_case_eligibility(
    eligible_by_case: dict[str, Sequence[str]],
) -> list[str]:
    """Return policies that pass every physical case."""

    if not eligible_by_case:
        return []
    eligible_sets = [set(values) for values in eligible_by_case.values()]
    return sorted(set.intersection(*eligible_sets))


def _read_timing_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def evaluate_timing_cost_gates(
    rows: Sequence[dict[str, object]],
    *,
    baseline_policy: str = BASELINE_POLICY,
    runtime_ratio_limit: float = 1.05,
) -> dict[str, dict[str, float | int | bool]]:
    """Apply setup, resident-storage, and trajectory-runtime cost gates."""

    physical_rows = [
        row for row in rows if str(row.get("regime")) == "physical_refinement"
    ]
    if not physical_rows:
        return {}
    largest_node = max(int(row["n_nodes"]) for row in physical_rows)
    largest = [row for row in physical_rows if int(row["n_nodes"]) == largest_node]
    baseline = next(
        (row for row in largest if str(row["policy"]) == baseline_policy),
        None,
    )
    if baseline is None:
        raise ValueError("timing rows do not contain the baseline policy")

    baseline_setup = float(baseline["setup_seconds"])
    baseline_resident = float(baseline["resident_array_bytes"])
    baseline_run = float(baseline["median_run_seconds"])
    results: dict[str, dict[str, float | int | bool]] = {}
    for row in largest:
        policy = str(row["policy"])
        if policy == baseline_policy:
            continue
        setup_ratio = float(row["setup_seconds"]) / baseline_setup
        resident_ratio = float(row["resident_array_bytes"]) / baseline_resident
        runtime_ratio = float(row["median_run_seconds"]) / baseline_run
        setup_pass = setup_ratio < 1.0
        resident_pass = resident_ratio < 1.0
        runtime_pass = runtime_ratio <= runtime_ratio_limit
        results[policy] = {
            "n_nodes": largest_node,
            "setup_ratio": setup_ratio,
            "resident_ratio": resident_ratio,
            "trajectory_runtime_ratio": runtime_ratio,
            "setup_pass": setup_pass,
            "resident_pass": resident_pass,
            "trajectory_runtime_pass": runtime_pass,
            "cost_pass": setup_pass and resident_pass and runtime_pass,
        }
    return results


def _prior_selected_policies(
    output_root: Path,
    stage: RankScalingStage,
) -> tuple[list[str] | None, str | None]:
    source_stage = "selection" if stage == "confirmation" else "confirmation"
    if stage == "selection":
        return None, None
    source = output_root / source_stage / "summary.json"
    if not source.exists():
        return None, None
    summary = json.loads(source.read_text(encoding="utf-8"))
    candidates = list(summary.get("eligible_candidate_policies", []))
    return [BASELINE_POLICY, *candidates], str(source)


def _case_resolution_spec(
    spec: CrossCaseRankScalingSpec,
    case: RankScalingPhysicalCase,
    *,
    seed: int,
    runs: int,
    bootstrap_resamples: int,
) -> ResolutionModeStudySpec:
    return ResolutionModeStudySpec(
        node_counts=case.node_counts,
        n_particles=spec.n_particles,
        total_time=spec.total_time,
        requested_dt=spec.requested_dt,
        length=case.length,
        diffusion_coefficient=spec.diffusion_coefficient,
        impulse_fraction=spec.impulse_fraction,
        stability_safety=spec.stability_safety,
        handoff_fraction=spec.handoff_fraction,
        policies=policies_for_case(spec, case),
        covariance_times=spec.covariance_times,
        distribution_times=spec.distribution_times,
        diagnostic_positions=case.diagnostic_positions,
        temporal_lag=spec.temporal_lag,
        completion_rank=5,
        completion_ridge=spec.completion_ridge,
        screening_runs=runs,
        final_runs=runs,
        projections=spec.projections,
        histogram_bins=spec.histogram_bins,
        comparison_bootstrap_resamples=bootstrap_resamples,
        comparison_bootstrap_blocks=spec.bootstrap_blocks,
        comparison_confidence_level=spec.confidence_level,
        fixed_step_count=spec.fixed_step_count,
        timing_repeats=spec.timing_repeats,
        setup_repeats=spec.setup_repeats,
        seed=seed,
    )


def run_cross_case_rank_scaling(
    spec: CrossCaseRankScalingSpec,
    output_directory: str | Path,
    *,
    stage: RankScalingStage,
    selected_policy_names: Sequence[str] | None = None,
) -> dict[str, object]:
    """Run and aggregate one stage of the adaptive-rank promotion workflow."""

    if stage not in {"selection", "confirmation", "timing"}:
        raise ValueError("stage must be selection, confirmation, or timing")
    output_root = Path(output_directory)
    stage_directory = output_root / stage
    stage_directory.mkdir(parents=True, exist_ok=True)

    inherited_selection_source: str | None = None
    selected = None if selected_policy_names is None else list(selected_policy_names)
    if selected is None:
        selected, inherited_selection_source = _prior_selected_policies(
            output_root,
            stage,
        )
    if selected is not None and BASELINE_POLICY not in selected:
        selected = [BASELINE_POLICY, *selected]

    if stage == "selection":
        master_seed = spec.selection_seed
        runs = spec.selection_runs
        bootstrap_resamples = spec.selection_bootstrap_resamples
    elif stage == "confirmation":
        master_seed = spec.confirmation_seed
        runs = spec.confirmation_runs
        bootstrap_resamples = spec.confirmation_bootstrap_resamples
    else:
        master_seed = spec.timing_seed
        runs = spec.selection_runs
        bootstrap_resamples = spec.selection_bootstrap_resamples

    case_summaries: dict[str, dict[str, object]] = {}
    eligible_by_case: dict[str, list[str]] = {}
    timing_gates_by_case: dict[str, dict[str, dict[str, float | int | bool]]] = {}
    case_seeds: dict[str, int] = {}
    for case_index, case in enumerate(spec.cases):
        case_seed = master_seed + case_index * spec.case_seed_stride
        case_seeds[case.name] = case_seed
        resolution_spec = _case_resolution_spec(
            spec,
            case,
            seed=case_seed,
            runs=runs,
            bootstrap_resamples=bootstrap_resamples,
        )
        case_directory = stage_directory / case.name
        case_summary = run_resolution_mode_study(
            resolution_spec,
            case_directory,
            stage="analytic" if stage == "timing" else "screening",
            selected_policy_names=selected,
            include_timing=stage == "timing",
        )
        case_summaries[case.name] = {
            "length": case.length,
            "node_counts": list(case.node_counts),
            "diagnostic_positions": list(case.diagnostic_positions),
            "seed": case_seed,
            "output_directory": str(case_directory),
            "summary": case_summary,
            "policies": [asdict(policy) for policy in policies_for_case(spec, case)],
        }
        if stage != "timing":
            eligible_by_case[case.name] = list(
                case_summary["accuracy_eligible_policies"]
            )
        else:
            timing_rows = _read_timing_rows(case_directory / "timing_scaling.csv")
            timing_gates_by_case[case.name] = evaluate_timing_cost_gates(
                timing_rows,
                runtime_ratio_limit=spec.trajectory_runtime_ratio_limit,
            )

    if stage == "timing":
        confirmation_path = output_root / "confirmation" / "summary.json"
        confirmed: list[str] = []
        if confirmation_path.exists():
            confirmed = list(
                json.loads(confirmation_path.read_text(encoding="utf-8")).get(
                    "eligible_candidate_policies",
                    [],
                )
            )
        cost_pass_by_case = {
            case_name: sorted(
                policy
                for policy, gate in gates.items()
                if bool(gate["cost_pass"])
            )
            for case_name, gates in timing_gates_by_case.items()
        }
        cost_eligible = intersect_case_eligibility(cost_pass_by_case)
        promotion_eligible = sorted(set(confirmed) & set(cost_eligible))
        accuracy_eligible: list[str] = [BASELINE_POLICY, *confirmed]
        eligible_candidates = confirmed
    else:
        accuracy_eligible = intersect_case_eligibility(eligible_by_case)
        eligible_candidates = [
            policy for policy in accuracy_eligible if policy != BASELINE_POLICY
        ]
        cost_pass_by_case = {}
        cost_eligible = []
        promotion_eligible = []

    combined: dict[str, object] = {
        "stage": stage,
        "master_seed": master_seed,
        "case_seeds": case_seeds,
        "seed_provenance": {
            "selection_seed": spec.selection_seed,
            "confirmation_seed": spec.confirmation_seed,
            "timing_seed": spec.timing_seed,
            "case_seed_stride": spec.case_seed_stride,
        },
        "runs": 0 if stage == "timing" else runs,
        "selected_policies": selected,
        "inherited_selection_source": inherited_selection_source,
        "eligible_by_case": eligible_by_case,
        "accuracy_eligible_policies": accuracy_eligible,
        "eligible_candidate_policies": eligible_candidates,
        "timing_gates_by_case": timing_gates_by_case,
        "cost_pass_by_case": cost_pass_by_case,
        "cost_eligible_policies": cost_eligible,
        "promotion_eligible_policies": promotion_eligible,
        "case_summaries": case_summaries,
    }
    (stage_directory / "summary.json").write_text(
        json.dumps(combined, indent=2) + "\n",
        encoding="utf-8",
    )
    return combined
