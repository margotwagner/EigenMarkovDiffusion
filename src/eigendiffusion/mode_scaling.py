"""Resolution-aware scaling studies for the physics-based modal model.

The original performance sweep scales the retained mode count and completion
rank in proportion to the number of spatial nodes.  That is a useful baseline,
but it is not a physical selection law.  For diffusion on a fixed domain the
continuous modal decay is approximately

``exp(-D (k pi / L)**2 t)``,

so a fixed decay tolerance can select an absolute spectral cutoff that
approaches a grid-independent value under refinement.  This module compares
that rule with proportional, fixed-count, and analytic-error-predictor rules.

Two timing regimes are deliberately kept separate:

``physical_refinement``
    Fix ``L``, ``D``, and the final physical time, and reduce ``dt`` to remain
    stable.  This is the honest end-to-end refinement cost, including the
    resulting growth in the number of time steps.

``fixed_steps``
    Fix the number of stable updates and report time per update.  The final
    physical time is allowed to change so that spatial algebra scaling is not
    confounded with the ``dt ~ dx**2`` refinement requirement.
"""

from __future__ import annotations

import csv
import gc
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Literal, Sequence

import numpy as np

from .benchmarking import deep_numpy_nbytes, stable_diffusion_config
from .completion import LocalPersistentUnresolvedGaussianCompleter
from .config import DiffusionConfig
from .correlated_modal import HandoffCorrelatedModalDiffusion
from .distribution_metrics import summarize_distribution_ensemble
from .ensemble import apply_readout_ensemble, run_modal_ensemble, run_random_walk_ensemble
from .handoff import HandoffPrediction, predict_handoff_grid
from .metrics import (
    paired_bootstrap_error_difference_intervals,
    relative_frobenius_error,
    summarize_ensemble,
)
from .operators import eigendecompose, neumann_laplacian_1d
from .readouts import simplex_bank_readout
from .references import (
    discrete_expected_diffusion,
    multinomial_marginal_covariance,
    multinomial_marginal_variance,
)
from .temporal import (
    empirical_mean_cross_time_covariance,
    multinomial_mean_cross_time_covariance,
    valid_time_origins,
)

ModePolicyKind = Literal[
    "proportional",
    "fixed",
    "spectral_decay",
    "discrete_mean_cutoff",
    "full_moment_predictor",
]
StudyStage = Literal["analytic", "screening", "final"]
TimingRegime = Literal["physical_refinement", "fixed_steps"]

MODE_SCALING_ACCEPTANCE_METRICS = (
    "mean_relative_l2_error",
    "mean_max_time_relative_l2_error",
    "variance_relative_l2_error",
    "variance_max_time_relative_l2_error",
    "covariance_relative_frobenius_error",
    "cross_time_covariance_relative_frobenius_error",
    "marginal_normalized_wasserstein",
    "sliced_normalized_wasserstein",
)
MODE_SCALING_BOOTSTRAP_METRICS = frozenset(
    {
        "mean_relative_l2_error",
        "variance_relative_l2_error",
        "covariance_relative_frobenius_error",
    }
)


def mode_policy_noninferiority_pass(
    candidate: dict[str, object],
    baseline: dict[str, object],
    sampling_floor: dict[str, object],
) -> bool:
    """Return whether a policy is noninferior to the proportional baseline.

    Moment comparisons use paired candidate-minus-baseline bootstrap interval
    endpoints when available. Other ensemble metrics use the observed paired
    difference. The case-specific reference-reference error supplies the
    Monte Carlo margin; lag covariance deliberately receives one quarter of
    that margin because temporal persistence is a primary closure target.
    """

    for metric in MODE_SCALING_ACCEPTANCE_METRICS:
        margin_multiplier = (
            0.25
            if metric == "cross_time_covariance_relative_frobenius_error"
            else 1.0
        )
        margin = margin_multiplier * float(sampling_floor[metric])
        interval_key = f"{metric}_difference_ci_upper"
        if metric in MODE_SCALING_BOOTSTRAP_METRICS and interval_key in candidate:
            difference_upper = float(candidate[interval_key])
        else:
            difference_upper = float(candidate[metric]) - float(baseline[metric])
        if difference_upper > margin:
            return False

    negative_mass_tolerance = max(
        float(sampling_floor["mean_negative_mass_fraction"]),
        1.0e-12,
    )
    if float(candidate["mean_negative_mass_fraction"]) > (
        float(baseline["mean_negative_mass_fraction"])
        + negative_mass_tolerance
    ):
        return False
    mass_tolerance = max(float(sampling_floor["maximum_mass_error"]), 1.0e-9)
    return float(candidate["maximum_mass_error"]) <= (
        float(baseline["maximum_mass_error"]) + mass_tolerance
    )


@dataclass(frozen=True, slots=True)
class ModeSelectionPolicy:
    """One retained-mode selection policy.

    Unused parameters are retained in the dataclass so policies remain easy to
    serialize.  Validation only applies the parameters relevant to ``kind``.
    """

    name: str
    kind: ModePolicyKind
    reference_modes: int = 75
    reference_nodes: int = 101
    fixed_modes: int = 75
    decay_tolerance: float = 1.0e-3
    amplitude_bound: float = 1.0
    predictor_mean_tolerance: float = 2.0e-2
    predictor_post_handoff_mean_tolerance: float = 5.0e-3
    predictor_variance_tolerance: float = 1.0e-1
    predictor_covariance_tolerance: float = 2.0e-1
    completion_rank: int | None = None
    modal_bandwidth: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("policy name must be nonempty")
        if self.kind not in {
            "proportional",
            "fixed",
            "spectral_decay",
            "discrete_mean_cutoff",
            "full_moment_predictor",
        }:
            raise ValueError(f"unknown mode policy kind: {self.kind}")
        if self.reference_modes <= 0 or self.reference_nodes < 2:
            raise ValueError("reference modes/nodes must be positive")
        if self.fixed_modes <= 0:
            raise ValueError("fixed_modes must be positive")
        if not 0.0 < self.decay_tolerance < self.amplitude_bound:
            raise ValueError("decay_tolerance must lie between zero and amplitude_bound")
        tolerances = (
            self.predictor_mean_tolerance,
            self.predictor_post_handoff_mean_tolerance,
            self.predictor_variance_tolerance,
            self.predictor_covariance_tolerance,
        )
        if any(value < 0.0 for value in tolerances):
            raise ValueError("predictor tolerances must be nonnegative")
        if self.completion_rank is not None and self.completion_rank <= 0:
            raise ValueError("policy completion_rank must be positive")
        if self.modal_bandwidth is not None and self.modal_bandwidth < 0:
            raise ValueError("policy modal_bandwidth must be nonnegative")


DEFAULT_MODE_POLICIES: tuple[ModeSelectionPolicy, ...] = (
    ModeSelectionPolicy(name="proportional_75_of_101", kind="proportional"),
    ModeSelectionPolicy(name="fixed_75", kind="fixed"),
    ModeSelectionPolicy(name="spectral_decay_1e-3", kind="spectral_decay"),
    ModeSelectionPolicy(
        name="discrete_mean_cutoff_0p005",
        kind="discrete_mean_cutoff",
        predictor_post_handoff_mean_tolerance=5.0e-3,
    ),
    ModeSelectionPolicy(
        name="full_moment_predictor",
        kind="full_moment_predictor",
        predictor_mean_tolerance=5.0e-3,
        predictor_variance_tolerance=2.5e-1,
        predictor_covariance_tolerance=3.0e-1,
    ),
)

# A five-mode grid keeps the full-moment policy affordable while avoiding the
# very coarse operating-point-only choices used by the original sweep.  It is
# a diagnostic baseline, not the proposed plateauing rule.
DEFAULT_PREDICTOR_CANDIDATE_MODES: tuple[int, ...] = tuple(range(5, 301, 5))


@dataclass(frozen=True, slots=True)
class ResolutionModeStudySpec:
    """Configuration for an analytic, screening, or final scaling study."""

    node_counts: tuple[int, ...] = (101, 151, 201)
    n_particles: int = 5_275
    total_time: float = 100.0
    requested_dt: float = 1.0
    length: float = 4.0
    diffusion_coefficient: float = 2.20e-4
    impulse_fraction: float = 0.59
    stability_safety: float = 0.95
    handoff_fraction: float = 0.30
    policies: tuple[ModeSelectionPolicy, ...] = DEFAULT_MODE_POLICIES
    predictor_candidate_modes: tuple[int, ...] = DEFAULT_PREDICTOR_CANDIDATE_MODES
    covariance_times: tuple[float, ...] = (1.0, 5.0, 20.0, 50.0, 100.0)
    distribution_times: tuple[float, ...] = (20.0, 50.0, 100.0)
    diagnostic_positions: tuple[float, ...] = (1.6, 2.0, 2.4)
    temporal_lag: float = 1.0
    completion_rank: int = 5
    modal_bandwidth: int = 12
    completion_ridge: float = 1.0e-2
    screening_runs: int = 250
    final_runs: int = 2_000
    projections: int = 64
    histogram_bins: int = 30
    comparison_bootstrap_resamples: int = 200
    comparison_bootstrap_blocks: int = 20
    comparison_confidence_level: float = 0.95
    fixed_step_count: int = 101
    timing_repeats: int = 7
    seed: int = 20261120

    def __post_init__(self) -> None:
        if not self.node_counts or any(nodes < 3 for nodes in self.node_counts):
            raise ValueError("node_counts must contain values of at least three")
        if len(set(self.node_counts)) != len(self.node_counts):
            raise ValueError("node_counts must not contain duplicates")
        if self.n_particles <= 0:
            raise ValueError("n_particles must be positive")
        if self.total_time <= 0.0 or self.requested_dt <= 0.0:
            raise ValueError("time parameters must be positive")
        if self.length <= 0.0 or self.diffusion_coefficient <= 0.0:
            raise ValueError("physical parameters must be positive")
        if not 0.0 <= self.impulse_fraction <= 1.0:
            raise ValueError("impulse_fraction must lie in [0, 1]")
        if not 0.0 < self.handoff_fraction < 1.0:
            raise ValueError("handoff_fraction must lie in (0, 1)")
        if not self.policies or len({policy.name for policy in self.policies}) != len(
            self.policies
        ):
            raise ValueError("policies must have unique names")
        if not self.predictor_candidate_modes or any(
            modes <= 0 for modes in self.predictor_candidate_modes
        ):
            raise ValueError("predictor_candidate_modes must be positive")
        if not self.covariance_times or not self.distribution_times:
            raise ValueError("diagnostic times must be nonempty")
        if not self.diagnostic_positions:
            raise ValueError("diagnostic_positions must be nonempty")
        if min(self.diagnostic_positions) < 0.0 or max(self.diagnostic_positions) > self.length:
            raise ValueError("diagnostic positions must lie in the physical domain")
        if self.temporal_lag <= 0.0:
            raise ValueError("temporal_lag must be positive")
        if self.completion_rank <= 0 or self.modal_bandwidth < 0:
            raise ValueError("completion rank/bandwidth must be valid")
        if self.screening_runs <= 1 or self.final_runs <= 1:
            raise ValueError("ensemble run counts must exceed one")
        if self.projections <= 0 or self.histogram_bins < 2:
            raise ValueError("distribution settings must be positive")
        if self.comparison_bootstrap_resamples <= 0:
            raise ValueError("comparison bootstrap resamples must be positive")
        if self.comparison_bootstrap_blocks <= 1:
            raise ValueError("comparison bootstrap blocks must exceed one")
        if not 0.0 < self.comparison_confidence_level < 1.0:
            raise ValueError("comparison confidence level must lie in (0, 1)")
        if self.fixed_step_count < 2 or self.timing_repeats <= 0:
            raise ValueError("timing settings must be positive")


@dataclass(frozen=True, slots=True)
class AnalyticModeScalingRecord:
    """Analytic error and work prediction for one grid and one policy."""

    policy: str
    policy_kind: str
    n_nodes: int
    n_steps: int
    dt: float
    total_time: float
    handoff_time: float
    final_modes: int
    retained_fraction: float
    first_omitted_continuous_decay: float
    predicted_mean_relative_l2_error: float
    predicted_post_handoff_mean_relative_l2_error: float
    predicted_variance_relative_l2_error: float
    predicted_covariance_relative_frobenius_error: float
    modal_update_fraction_of_full: float

    def as_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ModeTimingRecord:
    """Setup and repeated trajectory timing in one scaling regime."""

    policy: str
    regime: str
    n_nodes: int
    n_steps: int
    dt: float
    total_time: float
    handoff_time: float
    final_modes: int
    completion_rank: int
    modal_bandwidth: int
    repeats: int
    setup_seconds: float
    median_run_seconds: float
    mean_run_seconds: float
    std_run_seconds: float
    median_seconds_per_transition: float
    resident_array_bytes: int

    def as_dict(self) -> dict[str, str | int | float]:
        row = asdict(self)
        row["resident_array_megabytes"] = self.resident_array_bytes / 1024**2
        return row


def spectral_decay_mode_count(
    *,
    length: float,
    diffusion_coefficient: float,
    handoff_time: float,
    tolerance: float,
    amplitude_bound: float = 1.0,
    maximum_modes: int | None = None,
) -> int:
    """Return the smallest cutoff whose first omitted continuum mode decays.

    Retaining ``M`` modes means that the first omitted Neumann wavenumber is
    ``k=M`` when the constant mode is indexed by zero.
    """

    if length <= 0.0 or diffusion_coefficient <= 0.0 or handoff_time <= 0.0:
        raise ValueError("length, diffusion coefficient, and handoff time must be positive")
    if not 0.0 < tolerance < amplitude_bound:
        raise ValueError("tolerance must lie between zero and amplitude_bound")
    cutoff = (
        length
        / (np.pi * np.sqrt(diffusion_coefficient * handoff_time))
        * np.sqrt(np.log(amplitude_bound / tolerance))
    )
    modes = max(1, int(np.ceil(cutoff)))
    if maximum_modes is not None:
        if maximum_modes <= 0:
            raise ValueError("maximum_modes must be positive")
        modes = min(modes, int(maximum_modes))
    return modes


def physical_position_indices(
    config: DiffusionConfig,
    positions: Sequence[float],
) -> list[int]:
    """Map fixed physical positions to the nearest nodes, without duplicates."""

    requested = np.asarray(positions, dtype=float)
    if requested.ndim != 1 or requested.size == 0:
        raise ValueError("positions must be a nonempty vector")
    if not np.all(np.isfinite(requested)):
        raise ValueError("positions must be finite")
    if np.min(requested) < 0.0 or np.max(requested) > config.length:
        raise ValueError("positions must lie in the physical domain")
    indices = [int(np.argmin(np.abs(config.positions - position))) for position in requested]
    return list(dict.fromkeys(indices))


def _nearest_time_indices(config: DiffusionConfig, times: Sequence[float]) -> list[int]:
    requested = np.asarray(times, dtype=float)
    if requested.ndim != 1 or requested.size == 0:
        raise ValueError("times must be a nonempty vector")
    if np.min(requested) < 0.0 or np.max(requested) > config.times[-1] + 1.0e-12:
        raise ValueError("diagnostic times must lie in the simulated interval")
    indices = [int(np.argmin(np.abs(config.times - time))) for time in requested]
    return list(dict.fromkeys(indices))


def _physical_config(spec: ResolutionModeStudySpec, n_nodes: int) -> DiffusionConfig:
    return stable_diffusion_config(
        n_particles=spec.n_particles,
        n_nodes=n_nodes,
        total_time=spec.total_time,
        requested_dt=spec.requested_dt,
        length=spec.length,
        diffusion_coefficient=spec.diffusion_coefficient,
        impulse_fraction=spec.impulse_fraction,
        stability_safety=spec.stability_safety,
    )


def _fixed_step_config(spec: ResolutionModeStudySpec, n_nodes: int) -> DiffusionConfig:
    dx = spec.length / (n_nodes - 1)
    maximum_dt = dx**2 / (4.0 * spec.diffusion_coefficient)
    dt = min(spec.requested_dt, spec.stability_safety * maximum_dt)
    return DiffusionConfig(
        n_particles=spec.n_particles,
        n_nodes=n_nodes,
        n_steps=spec.fixed_step_count,
        impulse_index=int(round(spec.impulse_fraction * (n_nodes - 1))),
        dt=dt,
        length=spec.length,
        diffusion_coefficient=spec.diffusion_coefficient,
    )


def _static_policy_modes(
    policy: ModeSelectionPolicy,
    config: DiffusionConfig,
    handoff_time: float,
) -> int:
    maximum = config.n_nodes - 1
    if policy.kind == "proportional":
        selected = int(round(config.n_nodes * policy.reference_modes / policy.reference_nodes))
    elif policy.kind == "fixed":
        selected = policy.fixed_modes
    elif policy.kind == "spectral_decay":
        selected = spectral_decay_mode_count(
            length=config.length,
            diffusion_coefficient=config.diffusion_coefficient,
            handoff_time=handoff_time,
            tolerance=policy.decay_tolerance,
            amplitude_bound=policy.amplitude_bound,
        )
    else:
        raise ValueError("predictor policies require analytic predictions")
    return min(maximum, max(1, selected))


def discrete_mean_cutoff_mode_count(
    config: DiffusionConfig,
    *,
    handoff_time: float,
    post_handoff_relative_l2_tolerance: float,
) -> int:
    """Select the exact worst-time discrete mean cutoff after handoff.

    This deliberately ignores unresolved covariance.  The completed-model
    experiment tests whether the analytic unresolved closure can restore that
    covariance while the dynamical retained-mode count follows this plateauing
    mean criterion.  The returned count is the smallest ``M`` for which every
    saved time at or after handoff satisfies

    ``||mu_modal[t, M:]||_2 / ||mu_modal[t]||_2 <= tolerance``.
    """

    if handoff_time < config.times[0] or handoff_time > config.times[-1]:
        raise ValueError("handoff_time must lie in the simulated interval")
    if post_handoff_relative_l2_tolerance < 0.0:
        raise ValueError("post-handoff tolerance must be nonnegative")
    operator = neumann_laplacian_1d(
        config.n_nodes,
        config.length,
        config.diffusion_coefficient,
    )
    vectors = eigendecompose(operator).eigenvectors
    modal_mean = discrete_expected_diffusion(config) @ vectors
    handoff_step = int(np.argmin(np.abs(config.times - handoff_time)))
    post_modal_mean = modal_mean[handoff_step:]
    total_energy_by_time = np.sum(post_modal_mean**2, axis=1)
    if not np.any(total_energy_by_time > 0.0):
        return 1
    tail_energy_by_time = np.cumsum(post_modal_mean[:, ::-1] ** 2, axis=1)[:, ::-1]
    for retained_modes in range(1, config.n_nodes):
        relative_error_by_time = np.zeros_like(total_energy_by_time)
        np.divide(
            tail_energy_by_time[:, retained_modes],
            total_energy_by_time,
            out=relative_error_by_time,
            where=total_energy_by_time > 0.0,
        )
        np.sqrt(relative_error_by_time, out=relative_error_by_time)
        if (
            float(np.max(relative_error_by_time))
            <= post_handoff_relative_l2_tolerance
        ):
            return retained_modes
    # Retaining the complete basis has identically zero projection error.
    return config.n_nodes


def _prediction_satisfies(policy: ModeSelectionPolicy, row: HandoffPrediction) -> bool:
    if policy.kind == "discrete_mean_cutoff":
        return (
            row.predicted_post_handoff_mean_relative_l2_error
            <= policy.predictor_post_handoff_mean_tolerance
        )
    return (
        row.predicted_mean_relative_l2_error <= policy.predictor_mean_tolerance
        and row.predicted_variance_relative_l2_error <= policy.predictor_variance_tolerance
        and row.predicted_covariance_relative_frobenius_error
        <= policy.predictor_covariance_tolerance
    )


def build_analytic_mode_map(
    spec: ResolutionModeStudySpec,
) -> list[AnalyticModeScalingRecord]:
    """Evaluate every mode policy against exact projected multinomial moments."""

    records: list[AnalyticModeScalingRecord] = []
    for n_nodes in spec.node_counts:
        config = _physical_config(spec, n_nodes)
        handoff_time = float(config.times[-1] * spec.handoff_fraction)
        static_modes = {
            policy.name: _static_policy_modes(policy, config, handoff_time)
            for policy in spec.policies
            if policy.kind
            not in {"discrete_mean_cutoff", "full_moment_predictor"}
        }
        exact_mean_modes = {
            policy.name: discrete_mean_cutoff_mode_count(
                config,
                handoff_time=handoff_time,
                post_handoff_relative_l2_tolerance=(
                    policy.predictor_post_handoff_mean_tolerance
                ),
            )
            for policy in spec.policies
            if policy.kind == "discrete_mean_cutoff"
        }
        candidate_modes = {
            min(config.n_nodes, int(value))
            for value in spec.predictor_candidate_modes
            if value > 0
        }
        candidate_modes.update(static_modes.values())
        candidate_modes.update(exact_mean_modes.values())
        candidate_modes.add(config.n_nodes)
        covariance_indices = _nearest_time_indices(config, spec.covariance_times)
        predictions = predict_handoff_grid(
            config,
            final_modes=sorted(candidate_modes),
            handoff_times=[handoff_time],
            covariance_time_indices=covariance_indices,
        )
        by_modes = {row.final_n_modes: row for row in predictions}

        selected_modes = {**static_modes, **exact_mean_modes}
        for policy in spec.policies:
            if policy.kind != "full_moment_predictor":
                continue
            eligible = [row for row in predictions if _prediction_satisfies(policy, row)]
            winner = (
                min(eligible, key=lambda row: row.final_n_modes)
                if eligible
                else predictions[-1]
            )
            selected_modes[policy.name] = winner.final_n_modes

        for policy in spec.policies:
            modes = selected_modes[policy.name]
            prediction = by_modes[modes]
            omitted_decay = np.exp(
                -config.diffusion_coefficient
                * (modes * np.pi / config.length) ** 2
                * prediction.handoff_time
            )
            records.append(
                AnalyticModeScalingRecord(
                    policy=policy.name,
                    policy_kind=policy.kind,
                    n_nodes=config.n_nodes,
                    n_steps=config.n_steps,
                    dt=config.dt,
                    total_time=float(config.times[-1]),
                    handoff_time=prediction.handoff_time,
                    final_modes=modes,
                    retained_fraction=modes / config.n_nodes,
                    first_omitted_continuous_decay=float(omitted_decay),
                    predicted_mean_relative_l2_error=(
                        prediction.predicted_mean_relative_l2_error
                    ),
                    predicted_post_handoff_mean_relative_l2_error=(
                        prediction.predicted_post_handoff_mean_relative_l2_error
                    ),
                    predicted_variance_relative_l2_error=(
                        prediction.predicted_variance_relative_l2_error
                    ),
                    predicted_covariance_relative_frobenius_error=(
                        prediction.predicted_covariance_relative_frobenius_error
                    ),
                    modal_update_fraction_of_full=(
                        prediction.modal_update_fraction_of_full
                    ),
                )
            )
    return records


def _evaluate_ensemble(
    config: DiffusionConfig,
    reference_runs: np.ndarray,
    model_runs: np.ndarray,
    spec: ResolutionModeStudySpec,
    *,
    handoff_time: float,
    projection_seed: int,
    exact_mean: np.ndarray | None = None,
    exact_variance: np.ndarray | None = None,
    exact_covariance: np.ndarray | None = None,
    exact_cross_covariance: np.ndarray | None = None,
) -> dict[str, int | float]:
    time_indices = _nearest_time_indices(config, spec.distribution_times)
    node_indices = physical_position_indices(config, spec.diagnostic_positions)
    distribution = summarize_distribution_ensemble(
        reference_runs,
        model_runs,
        time_indices=time_indices,
        node_indices=node_indices,
        n_projections=spec.projections,
        histogram_bins=spec.histogram_bins,
        seed=projection_seed,
    )
    if exact_mean is None:
        exact_mean = discrete_expected_diffusion(config)
    if exact_variance is None:
        exact_variance = multinomial_marginal_variance(config)
    if exact_covariance is None:
        exact_covariance = multinomial_marginal_covariance(config)
    moment = summarize_ensemble(
        exact_mean,
        exact_variance,
        model_runs,
        config.n_particles,
        reference_covariance=exact_covariance,
        covariance_time_indices=time_indices,
    )

    lag_steps = max(1, int(round(spec.temporal_lag / config.dt)))
    lag_steps = min(lag_steps, config.n_steps - 1)
    handoff_step = int(np.argmin(np.abs(config.times - handoff_time)))
    latest_start = config.n_steps - lag_steps - 1
    origins = valid_time_origins(
        config.n_steps,
        lag_steps,
        start_step=min(handoff_step, latest_start),
    )
    exact_cross = exact_cross_covariance
    if exact_cross is None:
        exact_cross = multinomial_mean_cross_time_covariance(
            config,
            lag_steps,
            start_indices=origins,
        )
    empirical_cross = empirical_mean_cross_time_covariance(
        model_runs,
        lag_steps,
        start_indices=origins,
    )
    return {
        **distribution.as_dict(),
        "mean_relative_l2_error": moment.mean_relative_l2_error,
        "mean_max_time_relative_l2_error": moment.mean_max_time_relative_l2_error,
        "mean_final_time_relative_l2_error": moment.mean_final_time_relative_l2_error,
        "mean_error_to_reference_se_ratio": moment.mean_error_to_reference_se_ratio,
        "variance_relative_l2_error": moment.variance_relative_l2_error,
        "variance_max_time_relative_l2_error": moment.variance_max_time_relative_l2_error,
        "variance_final_time_relative_l2_error": moment.variance_final_time_relative_l2_error,
        "covariance_relative_frobenius_error": float(
            moment.covariance_relative_frobenius_error
        ),
        "cross_time_covariance_relative_frobenius_error": relative_frobenius_error(
            exact_cross,
            empirical_cross,
        ),
        "temporal_lag_steps": lag_steps,
        "temporal_lag_time": lag_steps * config.dt,
        "mean_negative_mass_fraction": moment.mean_negative_mass_fraction,
        "maximum_mass_error": moment.maximum_mass_error,
    }


def _policy_completion_settings(
    spec: ResolutionModeStudySpec,
    candidate: AnalyticModeScalingRecord,
) -> tuple[int, int]:
    policy = next(policy for policy in spec.policies if policy.name == candidate.policy)
    requested_rank = (
        spec.completion_rank
        if policy.completion_rank is None
        else policy.completion_rank
    )
    bandwidth = (
        spec.modal_bandwidth
        if policy.modal_bandwidth is None
        else policy.modal_bandwidth
    )
    return (
        min(requested_rank, candidate.n_nodes - candidate.final_modes),
        bandwidth,
    )


def run_accuracy_screen(
    spec: ResolutionModeStudySpec,
    analytic_records: Sequence[AnalyticModeScalingRecord],
    *,
    n_runs: int,
    selected_policy_names: set[str] | None = None,
) -> list[dict[str, object]]:
    """Run completed ensembles and paired noninferiority tests by resolution."""

    if n_runs <= 1:
        raise ValueError("n_runs must exceed one")
    rows: list[dict[str, object]] = []
    for node_index, n_nodes in enumerate(spec.node_counts):
        config = _physical_config(spec, n_nodes)
        candidates = [
            row
            for row in analytic_records
            if row.n_nodes == n_nodes
            and (selected_policy_names is None or row.policy in selected_policy_names)
        ]
        if not candidates:
            continue
        baseline_candidate = next(
            (candidate for candidate in candidates if candidate.policy_kind == "proportional"),
            max(candidates, key=lambda candidate: candidate.final_modes),
        )
        ordered_candidates = [
            baseline_candidate,
            *(candidate for candidate in candidates if candidate is not baseline_candidate),
        ]
        reference_a = run_random_walk_ensemble(
            config,
            n_runs=n_runs,
            seed=spec.seed + 10_000 + node_index,
            method="multinomial",
        )
        reference_b = run_random_walk_ensemble(
            config,
            n_runs=n_runs,
            seed=spec.seed + 20_000 + node_index,
            method="multinomial",
        )
        projection_seed = spec.seed + 30_000 + node_index
        exact_mean = discrete_expected_diffusion(config)
        exact_variance = multinomial_marginal_variance(config)
        exact_covariance = multinomial_marginal_covariance(config)
        lag_steps = max(1, int(round(spec.temporal_lag / config.dt)))
        lag_steps = min(lag_steps, config.n_steps - 1)
        handoff_step = int(
            np.argmin(np.abs(config.times - baseline_candidate.handoff_time))
        )
        origins = valid_time_origins(
            config.n_steps,
            lag_steps,
            start_step=min(handoff_step, config.n_steps - lag_steps - 1),
        )
        exact_cross = multinomial_mean_cross_time_covariance(
            config,
            lag_steps,
            start_indices=origins,
        )
        floor_metrics = _evaluate_ensemble(
            config,
            reference_a.runs,
            reference_b.runs,
            spec,
            handoff_time=candidates[0].handoff_time,
            projection_seed=projection_seed,
            exact_mean=exact_mean,
            exact_variance=exact_variance,
            exact_covariance=exact_covariance,
            exact_cross_covariance=exact_cross,
        )
        rows.append(
            {
                "policy": "sampling_floor",
                "policy_kind": "reference",
                "comparison_baseline_policy": baseline_candidate.policy,
                "noninferiority_pass": None,
                "n_nodes": n_nodes,
                "n_steps": config.n_steps,
                "dt": config.dt,
                "runs": n_runs,
                "handoff_time": candidates[0].handoff_time,
                "final_modes": config.n_nodes,
                "completion_rank": 0,
                "modal_bandwidth": 0,
                **floor_metrics,
            }
        )

        covariance_indices = _nearest_time_indices(config, spec.distribution_times)

        def raw_ensemble(candidate: AnalyticModeScalingRecord):
            return run_modal_ensemble(
                config,
                n_runs=n_runs,
                model="handoff_correlated_modal",
                n_modes=candidate.final_modes,
                initial_n_modes=config.n_nodes,
                handoff_time=candidate.handoff_time,
                seed=spec.seed + 40_000 + node_index,
            )

        def completed_runs_and_metrics(
            candidate: AnalyticModeScalingRecord,
            raw,
        ) -> tuple[np.ndarray, dict[str, int | float]]:
            completion_rank, modal_bandwidth = _policy_completion_settings(
                spec,
                candidate,
            )
            if completion_rank == 0:
                completed = apply_readout_ensemble(
                    raw,
                    config,
                    readout="simplex_bank",
                    seed=spec.seed + 50_000 + node_index,
                )
            else:
                completed = apply_readout_ensemble(
                    raw,
                    config,
                    readout="local_persistent_unresolved_completion_simplex_bank",
                    retained_modes=candidate.final_modes,
                    completion_start_time=candidate.handoff_time,
                    completion_rank=completion_rank,
                    completion_ridge=spec.completion_ridge,
                    completion_bandwidth=modal_bandwidth,
                    seed=spec.seed + 50_000 + node_index,
                )
            metrics = _evaluate_ensemble(
                config,
                reference_a.runs,
                completed.runs,
                spec,
                handoff_time=candidate.handoff_time,
                projection_seed=projection_seed,
                exact_mean=exact_mean,
                exact_variance=exact_variance,
                exact_covariance=exact_covariance,
                exact_cross_covariance=exact_cross,
            )
            completed_runs = completed.runs
            del completed
            gc.collect()
            return completed_runs, metrics

        active_raw_modes = baseline_candidate.final_modes
        active_raw = raw_ensemble(baseline_candidate)
        baseline_runs, baseline_metrics = completed_runs_and_metrics(
            baseline_candidate,
            active_raw,
        )
        baseline_rank, baseline_bandwidth = _policy_completion_settings(
            spec,
            baseline_candidate,
        )
        baseline_configuration = (
            baseline_candidate.final_modes,
            baseline_rank,
            baseline_bandwidth,
        )
        for candidate_index, candidate in enumerate(ordered_candidates):
            completion_rank, modal_bandwidth = _policy_completion_settings(
                spec,
                candidate,
            )
            candidate_configuration = (
                candidate.final_modes,
                completion_rank,
                modal_bandwidth,
            )
            if candidate_configuration == baseline_configuration:
                metrics = dict(baseline_metrics)
                difference_intervals = {
                    metric: (0.0, 0.0)
                    for metric in MODE_SCALING_BOOTSTRAP_METRICS
                }
            else:
                if candidate.final_modes != active_raw_modes:
                    del active_raw
                    gc.collect()
                    active_raw = raw_ensemble(candidate)
                    active_raw_modes = candidate.final_modes
                candidate_runs, metrics = completed_runs_and_metrics(
                    candidate,
                    active_raw,
                )
                difference_intervals = paired_bootstrap_error_difference_intervals(
                    exact_mean,
                    exact_variance,
                    baseline_runs,
                    candidate_runs,
                    reference_covariance=exact_covariance,
                    covariance_time_indices=covariance_indices,
                    n_resamples=spec.comparison_bootstrap_resamples,
                    confidence_level=spec.comparison_confidence_level,
                    n_blocks=spec.comparison_bootstrap_blocks,
                    seed=spec.seed + 60_000 + 1_000 * node_index + candidate_index,
                )
                del candidate_runs
                gc.collect()
            comparison_metrics: dict[str, object] = dict(metrics)
            for metric, (lower, upper) in difference_intervals.items():
                comparison_metrics[f"{metric}_difference_ci_lower"] = lower
                comparison_metrics[f"{metric}_difference_ci_upper"] = upper
            comparison_metrics["noninferiority_pass"] = (
                True
                if candidate_configuration == baseline_configuration
                else mode_policy_noninferiority_pass(
                    comparison_metrics,
                    baseline_metrics,
                    floor_metrics,
                )
            )
            rows.append(
                {
                    "policy": candidate.policy,
                    "policy_kind": candidate.policy_kind,
                    "comparison_baseline_policy": baseline_candidate.policy,
                    "n_nodes": n_nodes,
                    "n_steps": config.n_steps,
                    "dt": config.dt,
                    "runs": n_runs,
                    "handoff_time": candidate.handoff_time,
                    "final_modes": candidate.final_modes,
                    "retained_fraction": candidate.final_modes / config.n_nodes,
                    "completion_rank": completion_rank,
                    "modal_bandwidth": modal_bandwidth,
                    **comparison_metrics,
                }
            )
        del reference_a, reference_b, baseline_runs, active_raw
        gc.collect()
    return rows


def _benchmark_one_policy(
    spec: ResolutionModeStudySpec,
    *,
    policy: str,
    config: DiffusionConfig,
    regime: TimingRegime,
    final_modes: int,
    seed: int,
) -> ModeTimingRecord:
    handoff_time = float(config.times[-1] * spec.handoff_fraction)
    policy_spec = next(item for item in spec.policies if item.name == policy)
    requested_rank = (
        spec.completion_rank
        if policy_spec.completion_rank is None
        else policy_spec.completion_rank
    )
    completion_rank = min(requested_rank, config.n_nodes - final_modes)
    modal_bandwidth = (
        spec.modal_bandwidth
        if policy_spec.modal_bandwidth is None
        else policy_spec.modal_bandwidth
    )
    start = perf_counter()
    simulator = HandoffCorrelatedModalDiffusion(
        config=config,
        initial_n_modes=config.n_nodes,
        final_n_modes=final_modes,
        handoff_time=handoff_time,
    )
    completer = (
        None
        if completion_rank == 0
        else LocalPersistentUnresolvedGaussianCompleter(
            config,
            retained_modes=final_modes,
            completion_start_time=handoff_time,
            completion_rank=completion_rank,
            modal_bandwidth=modal_bandwidth,
            ridge=spec.completion_ridge,
        )
    )
    setup_seconds = perf_counter() - start
    resident_bytes = deep_numpy_nbytes((simulator, completer))

    timings: list[float] = []
    sequences = np.random.SeedSequence(seed).spawn(spec.timing_repeats * 2)
    for repeat in range(spec.timing_repeats):
        start = perf_counter()
        raw = simulator.run(rng=np.random.default_rng(sequences[2 * repeat]))
        if completer is None:
            completed = raw.spatial_counts
        else:
            completed, _ = completer.complete(
                raw.spatial_counts,
                rng=np.random.default_rng(sequences[2 * repeat + 1]),
            )
        constrained = simplex_bank_readout(completed, config.n_particles).counts
        elapsed = perf_counter() - start
        if not np.allclose(constrained.sum(axis=1), config.n_particles, atol=1.0e-7):
            raise RuntimeError("timed completed output failed mass conservation")
        if np.min(constrained) < -1.0e-10:
            raise RuntimeError("timed completed output failed nonnegativity")
        timings.append(elapsed)
    values = np.asarray(timings, dtype=float)
    return ModeTimingRecord(
        policy=policy,
        regime=regime,
        n_nodes=config.n_nodes,
        n_steps=config.n_steps,
        dt=config.dt,
        total_time=float(config.times[-1]),
        handoff_time=simulator.handoff_time,
        final_modes=final_modes,
        completion_rank=completion_rank,
        modal_bandwidth=modal_bandwidth,
        repeats=spec.timing_repeats,
        setup_seconds=setup_seconds,
        median_run_seconds=float(np.median(values)),
        mean_run_seconds=float(np.mean(values)),
        std_run_seconds=float(np.std(values, ddof=0)),
        median_seconds_per_transition=float(np.median(values) / (config.n_steps - 1)),
        resident_array_bytes=resident_bytes,
    )


def benchmark_mode_scaling(
    spec: ResolutionModeStudySpec,
    analytic_records: Sequence[AnalyticModeScalingRecord],
    *,
    selected_policy_names: set[str] | None = None,
) -> list[ModeTimingRecord]:
    """Benchmark policies under physical-refinement and fixed-step regimes."""

    records: list[ModeTimingRecord] = []
    for node_index, n_nodes in enumerate(spec.node_counts):
        candidates = [
            row
            for row in analytic_records
            if row.n_nodes == n_nodes
            and (selected_policy_names is None or row.policy in selected_policy_names)
        ]
        for regime_index, regime in enumerate(("physical_refinement", "fixed_steps")):
            config = (
                _physical_config(spec, n_nodes)
                if regime == "physical_refinement"
                else _fixed_step_config(spec, n_nodes)
            )
            timing_cache: dict[tuple[int, int, int], ModeTimingRecord] = {}
            for policy_index, candidate in enumerate(candidates):
                rank, bandwidth = _policy_completion_settings(spec, candidate)
                configuration = (candidate.final_modes, rank, bandwidth)
                if configuration not in timing_cache:
                    timing_cache[configuration] = _benchmark_one_policy(
                        spec,
                        policy=candidate.policy,
                        config=config,
                        regime=regime,
                        final_modes=candidate.final_modes,
                        seed=(
                            spec.seed
                            + 60_000
                            + 1_000 * node_index
                            + 100 * regime_index
                            + policy_index
                        ),
                    )
                cached = timing_cache[configuration]
                records.append(replace(cached, policy=candidate.policy))
    return records


def _write_csv(rows: Sequence[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_resolution_mode_study(
    spec: ResolutionModeStudySpec,
    output_directory: str | Path,
    *,
    stage: StudyStage = "analytic",
    selected_policy_names: Sequence[str] | None = None,
    include_timing: bool = False,
) -> dict[str, object]:
    """Run one staged study and write reproducible CSV/JSON artifacts."""

    if stage not in {"analytic", "screening", "final"}:
        raise ValueError("stage must be analytic, screening, or final")
    selected = None if selected_policy_names is None else set(selected_policy_names)
    known = {policy.name for policy in spec.policies}
    if selected is not None and not selected <= known:
        raise ValueError(f"unknown selected policies: {sorted(selected - known)}")

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    analytic = build_analytic_mode_map(spec)
    _write_csv([row.as_dict() for row in analytic], directory / "analytic_mode_map.csv")

    accuracy_rows: list[dict[str, object]] = []
    n_runs = 0
    if stage != "analytic":
        n_runs = spec.screening_runs if stage == "screening" else spec.final_runs
        accuracy_rows = run_accuracy_screen(
            spec,
            analytic,
            n_runs=n_runs,
            selected_policy_names=selected,
        )
        _write_csv(accuracy_rows, directory / f"{stage}_accuracy.csv")

    evaluated_nodes = sorted(
        {
            int(row["n_nodes"])
            for row in accuracy_rows
            if row["policy"] != "sampling_floor"
        }
    )
    accuracy_eligible_policies = sorted(
        {
            str(row["policy"])
            for row in accuracy_rows
            if row["policy"] != "sampling_floor"
            and all(
                bool(candidate["noninferiority_pass"])
                for candidate in accuracy_rows
                if candidate["policy"] == row["policy"]
            )
            and {
                int(candidate["n_nodes"])
                for candidate in accuracy_rows
                if candidate["policy"] == row["policy"]
            }
            == set(evaluated_nodes)
        }
    )
    comparison_baselines = {
        str(row["n_nodes"]): str(row["comparison_baseline_policy"])
        for row in accuracy_rows
        if row["policy"] == "sampling_floor"
    }

    timing_records: list[ModeTimingRecord] = []
    if include_timing:
        timing_records = benchmark_mode_scaling(
            spec,
            analytic,
            selected_policy_names=selected,
        )
        _write_csv(
            [row.as_dict() for row in timing_records],
            directory / "timing_scaling.csv",
        )

    summary: dict[str, object] = {
        "stage": stage,
        "node_counts": list(spec.node_counts),
        "policies": [
            policy.name
            for policy in spec.policies
            if selected is None or policy.name in selected
        ],
        "seed": spec.seed,
        "ensemble_runs": n_runs,
        "case_specific_sampling_floors": stage != "analytic",
        "paired_moment_noninferiority_intervals": stage != "analytic",
        "comparison_bootstrap_resamples": spec.comparison_bootstrap_resamples,
        "comparison_bootstrap_blocks": spec.comparison_bootstrap_blocks,
        "comparison_confidence_level": spec.comparison_confidence_level,
        "comparison_baselines_by_node": comparison_baselines,
        "accuracy_eligible_policies": accuracy_eligible_policies,
        "fixed_physical_diagnostic_positions": list(spec.diagnostic_positions),
        "fixed_physical_temporal_lag": spec.temporal_lag,
        "timing_regimes": (
            ["physical_refinement", "fixed_steps"] if include_timing else []
        ),
        "analytic_rows": len(analytic),
        "accuracy_rows": len(accuracy_rows),
        "timing_rows": len(timing_records),
    }
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary

