"""Command-line interface for validation, sweeps, comparisons, and benchmarks."""

from __future__ import annotations

import argparse
import cProfile
import csv
import gc
import io
import json
import pstats
from pathlib import Path

import numpy as np

from .benchmarking import (
    BENCHMARK_METHOD_NAMES,
    DEFAULT_COMPLETION_CLOSURE,
    DEFAULT_LOCAL_MODAL_BANDWIDTH,
    SELECTED_OPERATING_POINTS,
    benchmark_operating_points,
    deep_numpy_nbytes,
    benchmark_diffusion_methods,
    benchmark_random_walks,
    stable_diffusion_config,
    write_benchmark_csv,
    write_diffusion_method_benchmark_csv,
    write_operating_point_benchmark_csv,
)
from .config import DiffusionConfig
from .completion import (
    LocalPersistentUnresolvedGaussianCompleter,
    PersistentUnresolvedGaussianCompleter,
)
from .component_benchmark import (
    COMPONENT_BENCHMARK_REGIMES,
    COMPONENT_READOUT_NAMES,
    benchmark_component_scaling,
    benchmark_full_field_components,
    write_component_benchmark_csv,
    write_component_memory_csv,
    write_component_scaling_csv,
    write_component_scaling_summary,
    write_component_benchmark_summary,
)
from .distribution_metrics import summarize_distribution_ensemble
from .fitted_study import run_fitted_closure_study
from .ensemble import (
    MODAL_MODEL_NAMES,
    EnsembleResult,
    ModalModelName,
    apply_readout_ensemble,
    run_modal_ensemble,
    run_random_walk_ensemble,
)
from .handoff import pareto_handoff_indices, predict_handoff_grid
from .metrics import (
    bootstrap_error_intervals,
    paired_bootstrap_error_difference_intervals,
    relative_frobenius_error,
    relative_l2_error,
    relative_l2_error_from_time,
    summarize_ensemble,
    time_resolved_relative_l2_error,
)
from .plotting import (
    plot_diffusion_method_benchmark,
    plot_handoff_sweep,
    plot_handoff_selection,
    plot_completion_rank_sweep,
    plot_mode_sweep,
    plot_model_comparison,
    plot_operating_point_study,
    plot_structured_closure_study,
    plot_random_walk_benchmark,
    plot_temporal_correlation_comparison,
    plot_trial_convergence,
    plot_validation,
)
from .readouts import READOUT_NAMES, ReadoutName, readout_constraint_diagnostics
from .references import (
    continuous_expected_diffusion,
    discrete_expected_diffusion,
    multinomial_marginal_covariance,
    multinomial_marginal_variance,
)
from .temporal import (
    empirical_mean_cross_time_covariance,
    mean_node_lag_correlation_from_runs,
    multinomial_mean_cross_time_covariance,
    multinomial_mean_node_lag_correlation,
    valid_time_origins,
)


STRUCTURED_CLOSURE_ACCEPTANCE_METRICS = (
    "mean_relative_l2_error",
    "mean_max_time_relative_l2_error",
    "variance_relative_l2_error",
    "variance_max_time_relative_l2_error",
    "covariance_relative_frobenius_error",
    "cross_time_covariance_relative_frobenius_error",
    "marginal_normalized_wasserstein",
    "sliced_normalized_wasserstein",
)
STRUCTURED_CLOSURE_BOOTSTRAP_METRICS = frozenset(
    {
        "mean_relative_l2_error",
        "variance_relative_l2_error",
        "covariance_relative_frobenius_error",
    }
)


def _structured_closure_noninferiority_pass(
    candidate: dict[str, object],
    baseline: dict[str, object],
    sampling_floor: dict[str, object],
) -> bool:
    """Apply predeclared, case-matched closure noninferiority margins."""

    for metric in STRUCTURED_CLOSURE_ACCEPTANCE_METRICS:
        margin_multiplier = (
            0.25
            if metric == "cross_time_covariance_relative_frobenius_error"
            else 1.0
        )
        margin = margin_multiplier * float(sampling_floor[metric])
        interval_key = f"{metric}_difference_ci_upper"
        if metric in STRUCTURED_CLOSURE_BOOTSTRAP_METRICS and interval_key in candidate:
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


def _structured_heldout_bandwidth_passes(
    heldout_rows: list[dict[str, object]],
    modal_bandwidth: int,
) -> bool:
    """Return whether one local bandwidth passes every held-out case."""

    by_case_and_closure = {
        (str(row["case"]), str(row["closure"])): row for row in heldout_rows
    }
    closure = f"local_b{modal_bandwidth}"
    candidates = [
        row for row in heldout_rows if str(row["closure"]) == closure
    ]
    if not candidates:
        return False
    return all(
        _structured_closure_noninferiority_pass(
            candidate,
            by_case_and_closure[(str(candidate["case"]), "dense")],
            by_case_and_closure[(str(candidate["case"]), "sampling_floor")],
        )
        for candidate in candidates
    )


def _add_shared_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_runs: bool = True,
) -> None:
    parser.add_argument("--particles", type=int, default=5_275)
    parser.add_argument("--nodes", type=int, default=101)
    parser.add_argument("--steps", type=int, default=101)
    parser.add_argument("--impulse-index", type=int, default=None)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--length", type=float, default=4.0)
    parser.add_argument("--diffusion", type=float, default=2.20e-4)
    if include_runs:
        parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--modal-particle-weight",
        type=float,
        default=1.0,
        help="Used only by independent_modal.",
    )
    parser.add_argument(
        "--initialization",
        choices=("nearest", "stochastic"),
        default="nearest",
        help="Used only by independent_modal.",
    )


def _add_modal_model_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--modal-model",
        choices=MODAL_MODEL_NAMES,
        default="independent_modal",
        help=(
            "Modal formulation to run. independent_modal is the unchanged original "
            "EigenMarkov model."
        ),
    )


def _add_handoff_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--initial-modes",
        type=int,
        default=None,
        help=(
            "Initial high-rank basis for handoff_correlated_modal. "
            "Defaults to all spatial modes."
        ),
    )
    parser.add_argument(
        "--handoff-time",
        type=float,
        default=10.0,
        help=(
            "Time in microseconds at which handoff_correlated_modal projects "
            "from the initial basis to --modes final modes."
        ),
    )


def _add_completion_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--completion-start-time",
        type=float,
        default=None,
        help=(
            "Start time for unresolved_gaussian_completion. Defaults to the "
            "handoff time for handoff_correlated_modal and 0 otherwise."
        ),
    )
    parser.add_argument(
        "--completion-rank",
        type=int,
        default=None,
        help=(
            "Number of leading unresolved conditional-covariance directions "
            "to sample. Defaults to all omitted modes."
        ),
    )
    parser.add_argument(
        "--completion-ridge",
        type=float,
        default=1.0e-2,
        help="Ridge fraction used to stabilize Gaussian conditional completion.",
    )
    parser.add_argument(
        "--completion-bandwidth",
        type=int,
        default=DEFAULT_LOCAL_MODAL_BANDWIDTH,
        help="Modal-neighborhood radius for local persistent completion.",
    )


def _add_readout_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--readout",
        choices=READOUT_NAMES,
        default="raw",
        help=(
            "Output-only physical readout. Delta-Sigma readouts do not feed "
            "back into the modal dynamics."
        ),
    )
    parser.add_argument(
        "--spatial-error-fraction",
        type=float,
        default=0.5,
        help=(
            "Fraction of Delta-Sigma quantization error borrowed by the next "
            "spatial node; used only by delta_sigma_neighbor."
        ),
    )
    _add_completion_arguments(parser)


def _add_covariance_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--covariance-times",
        type=float,
        nargs="+",
        default=[1.0, 5.0, 20.0, 100.0],
        help="Times used for same-time spatial covariance error.",
    )


def _config_from_args(args: argparse.Namespace) -> DiffusionConfig:
    return DiffusionConfig(
        n_particles=args.particles,
        n_nodes=args.nodes,
        n_steps=args.steps,
        impulse_index=args.impulse_index,
        dt=args.dt,
        length=args.length,
        diffusion_coefficient=args.diffusion,
    )


def _nearest_time_indices(
    times: np.ndarray,
    requested_times: list[float],
    *,
    exclude_zero: bool = False,
) -> tuple[list[int], list[float]]:
    indices: list[int] = []
    selected_times: list[float] = []
    for requested in requested_times:
        if requested < times[0] - 1e-12 or requested > times[-1] + 1e-12:
            continue
        index = int(np.argmin(np.abs(times - requested)))
        if exclude_zero and index == 0:
            continue
        if index not in indices:
            indices.append(index)
            selected_times.append(float(times[index]))
    if not indices:
        fallback = 1 if exclude_zero and times.size > 1 else 0
        indices = [fallback]
        selected_times = [float(times[fallback])]
    return indices, selected_times


def _time_column(start_time: float) -> str:
    text = f"{start_time:g}".replace("-", "m").replace(".", "p")
    return f"mean_relative_l2_t_ge_{text}us"


def _bank_diagnostics(ensemble: EnsembleResult) -> dict[str, float]:
    if not ensemble.auxiliary or "bank_l1_fraction" not in ensemble.auxiliary:
        return {}
    values = ensemble.auxiliary["bank_l1_fraction"]
    return {
        "mean_bank_l1_fraction": float(np.mean(values)),
        "maximum_bank_l1_fraction": float(np.max(values)),
        "final_mean_bank_l1_fraction": float(np.mean(values[:, -1])),
    }


def _handoff_diagnostics(ensemble: EnsembleResult) -> dict[str, float]:
    if not ensemble.auxiliary or "handoff_projection_relative_l2" not in ensemble.auxiliary:
        return {}
    values = ensemble.auxiliary["handoff_projection_relative_l2"]
    diagnostics = {
        "mean_handoff_projection_relative_l2": float(np.mean(values)),
        "maximum_handoff_projection_relative_l2": float(np.max(values)),
    }
    work = ensemble.auxiliary.get("handoff_modal_update_fraction_of_full")
    if work is not None:
        diagnostics["handoff_modal_update_fraction_of_full"] = float(np.mean(work))
    return diagnostics


def _readout_diagnostics(ensemble: EnsembleResult, total_particles: int) -> dict[str, float]:
    values = readout_constraint_diagnostics(ensemble.runs, total_particles)
    if ensemble.auxiliary and "readout_residual_l1_fraction" in ensemble.auxiliary:
        residual = ensemble.auxiliary["readout_residual_l1_fraction"]
        values.update(
            {
                "mean_readout_residual_l1_fraction": float(np.mean(residual)),
                "maximum_readout_residual_l1_fraction": float(np.max(residual)),
                "final_mean_readout_residual_l1_fraction": float(
                    np.mean(residual[:, -1])
                ),
            }
        )
    if ensemble.auxiliary and "completion_l1_fraction" in ensemble.auxiliary:
        completion = ensemble.auxiliary["completion_l1_fraction"]
        values.update(
            {
                "mean_completion_l1_fraction": float(np.mean(completion)),
                "maximum_completion_l1_fraction": float(np.max(completion)),
                "final_mean_completion_l1_fraction": float(
                    np.mean(completion[:, -1])
                ),
            }
        )
    if ensemble.auxiliary and "constraint_residual_l1_fraction" in ensemble.auxiliary:
        constraint = ensemble.auxiliary["constraint_residual_l1_fraction"]
        values.update(
            {
                "mean_constraint_residual_l1_fraction": float(np.mean(constraint)),
                "maximum_constraint_residual_l1_fraction": float(np.max(constraint)),
                "final_mean_constraint_residual_l1_fraction": float(
                    np.mean(constraint[:, -1])
                ),
            }
        )
    return values


def _resolved_completion_start_time(args: argparse.Namespace) -> float:
    if args.completion_start_time is not None:
        return float(args.completion_start_time)
    if getattr(args, "modal_model", None) == "handoff_correlated_modal":
        return float(args.handoff_time)
    return 0.0


def _apply_selected_readout(
    ensemble: EnsembleResult,
    config: DiffusionConfig,
    args: argparse.Namespace,
    *,
    n_modes: int,
    readout: ReadoutName | None = None,
    completion_start_time: float | None = None,
    seed: int | None = None,
) -> EnsembleResult:
    selected = args.readout if readout is None else readout
    start_time = (
        _resolved_completion_start_time(args)
        if completion_start_time is None
        else float(completion_start_time)
    )
    readout_seed = args.seed + 10_000 if seed is None else int(seed)
    return apply_readout_ensemble(
        ensemble,
        config,
        readout=selected,
        spatial_error_fraction=args.spatial_error_fraction,
        retained_modes=n_modes,
        completion_start_time=start_time,
        completion_rank=args.completion_rank,
        completion_ridge=args.completion_ridge,
        completion_bandwidth=args.completion_bandwidth,
        seed=readout_seed,
    )



def _run_selected_modal(
    config: DiffusionConfig,
    args: argparse.Namespace,
    *,
    n_modes: int,
    seed: int,
    model: ModalModelName | None = None,
) -> EnsembleResult:
    selected_model = args.modal_model if model is None else model
    return run_modal_ensemble(
        config,
        n_runs=args.runs,
        model=selected_model,
        n_modes=n_modes,
        modal_particle_weight=args.modal_particle_weight,
        initialization=args.initialization,
        initial_n_modes=args.initial_modes,
        handoff_time=args.handoff_time,
        seed=seed,
    )


def run_validation(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    n_modes = config.n_nodes if args.modes is None else args.modes

    continuous_reference = continuous_expected_diffusion(config)
    discrete_reference = discrete_expected_diffusion(config)
    analytic_variance = multinomial_marginal_variance(config)
    analytic_covariance = multinomial_marginal_covariance(config)

    random_walk = run_random_walk_ensemble(
        config,
        n_runs=args.runs,
        seed=args.seed,
        method=args.random_walk_method,
    )
    raw_modal = _run_selected_modal(
        config,
        args,
        n_modes=n_modes,
        seed=args.seed + 1,
    )
    modal = _apply_selected_readout(raw_modal, config, args, n_modes=n_modes)

    covariance_indices, covariance_times = _nearest_time_indices(
        config.times,
        args.covariance_times,
        exclude_zero=True,
    )
    modal_metrics = summarize_ensemble(
        discrete_reference,
        analytic_variance,
        modal.runs,
        config.n_particles,
        reference_covariance=analytic_covariance,
        covariance_time_indices=covariance_indices,
    )
    rw_metrics = summarize_ensemble(
        discrete_reference,
        analytic_variance,
        random_walk.runs,
        config.n_particles,
        reference_covariance=analytic_covariance,
        covariance_time_indices=covariance_indices,
    )

    output = plot_validation(
        config=config,
        continuous_reference=continuous_reference,
        discrete_reference=discrete_reference,
        analytic_variance=analytic_variance,
        modal_runs=modal.runs,
        random_walk_runs=random_walk.runs,
        output_path=args.output,
        profile_step=args.profile_step,
        modal_label=f"{args.modal_model}+{args.readout}",
    )

    data_path = Path(args.data_output) if args.data_output else output.with_suffix(".npz")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    save_items: dict[str, object] = {
        "continuous_reference": continuous_reference,
        "discrete_reference": discrete_reference,
        "analytic_variance": analytic_variance,
        "analytic_covariance_selected": analytic_covariance[covariance_indices],
        "covariance_time_indices": np.asarray(covariance_indices, dtype=int),
        "covariance_times": np.asarray(covariance_times, dtype=float),
        "raw_modal_runs": raw_modal.runs,
        "modal_runs": modal.runs,
        "modal_model": np.asarray(args.modal_model),
        "readout": np.asarray(args.readout),
        "completion_start_time": np.asarray(_resolved_completion_start_time(args)),
        "completion_rank": np.asarray(-1 if args.completion_rank is None else args.completion_rank),
        "completion_ridge": np.asarray(args.completion_ridge),
        "random_walk_runs": random_walk.runs,
        "random_walk_method": np.asarray(args.random_walk_method),
        "times": config.times,
        "positions": config.positions,
    }
    if modal.auxiliary:
        for key, value in modal.auxiliary.items():
            save_items[f"modal_aux_{key}"] = value
    # Preserve the previous data key for users loading old independent-model outputs.
    if args.modal_model == "independent_modal" and args.readout == "raw":
        save_items["eigenmarkov_runs"] = modal.runs
    np.savez_compressed(data_path, **save_items)

    modal_report = modal_metrics.as_dict()
    modal_report.update(_bank_diagnostics(raw_modal))
    modal_report.update(_handoff_diagnostics(raw_modal))
    modal_report.update(_readout_diagnostics(modal, config.n_particles))
    report = {
        "configuration": {
            "particles": config.n_particles,
            "nodes": config.n_nodes,
            "steps": config.n_steps,
            "impulse_index": config.impulse_index,
            "dt": config.dt,
            "length": config.length,
            "diffusion_coefficient": config.diffusion_coefficient,
            "modes": n_modes,
            "initial_modes": args.initial_modes,
            "handoff_time": args.handoff_time,
            "runs": args.runs,
            "modal_model": args.modal_model,
            "readout": args.readout,
            "spatial_error_fraction": args.spatial_error_fraction,
            "completion_start_time": _resolved_completion_start_time(args),
            "completion_rank": args.completion_rank,
            "completion_ridge": args.completion_ridge,
            "modal_particle_weight": args.modal_particle_weight,
            "initialization": args.initialization,
            "random_walk_method": args.random_walk_method,
            "covariance_times": covariance_times,
        },
        "reference_difference": {
            "continuous_vs_discrete_relative_l2": relative_l2_error(
                discrete_reference,
                continuous_reference,
            )
        },
        f"{args.modal_model}+{args.readout}_vs_discrete": modal_report,
        "random_walk_vs_discrete": rw_metrics.as_dict(),
        "figure": str(output),
        "data": str(data_path),
    }
    print(json.dumps(report, indent=2))
    return 0


def run_sweep(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    if args.modes is None:
        candidates = [2, 5, 10, 20, 50, 75, config.n_nodes]
        requested_modes = sorted({mode for mode in candidates if mode <= config.n_nodes})
    else:
        requested_modes = sorted(set(args.modes))
    for mode_count in requested_modes:
        if not 1 <= mode_count <= config.n_nodes:
            raise ValueError(f"mode count {mode_count} is outside [1, {config.n_nodes}]")

    valid_start_times = sorted(
        {
            float(start)
            for start in args.error_start_times
            if config.times[0] - 1e-12 <= start <= config.times[-1] + 1e-12
        }
    )
    if not valid_start_times:
        valid_start_times = [0.0]

    discrete_reference = discrete_expected_diffusion(config)
    analytic_variance = multinomial_marginal_variance(config)
    rows: list[dict[str, float | int | str]] = []

    for mode_count in requested_modes:
        raw_ensemble = _run_selected_modal(
            config,
            args,
            n_modes=mode_count,
            seed=args.seed + mode_count,
        )
        ensemble = _apply_selected_readout(
            raw_ensemble, config, args, n_modes=mode_count
        )
        diagnostics = summarize_ensemble(
            discrete_reference,
            analytic_variance,
            ensemble.runs,
            config.n_particles,
        )
        diagnostic_values = diagnostics.as_dict()
        diagnostic_values.pop("covariance_relative_frobenius_error", None)
        row: dict[str, float | int | str] = {
            "modal_model": args.modal_model,
            "readout": args.readout,
            "n_modes": mode_count,
            **diagnostic_values,
            **_bank_diagnostics(raw_ensemble),
            **_handoff_diagnostics(raw_ensemble),
            **_readout_diagnostics(ensemble, config.n_particles),
        }
        mean = ensemble.mean
        for start_time in valid_start_times:
            row[_time_column(start_time)] = relative_l2_error_from_time(
                discrete_reference,
                mean,
                config.times,
                start_time,
            )
        rows.append(row)
        print(
            f"model={args.modal_model}+{args.readout:20s}  modes={mode_count:4d}  "
            f"mean_l2={diagnostics.mean_relative_l2_error:.6g}  "
            f"variance_l2={diagnostics.variance_relative_l2_error:.6g}  "
            f"negative_mass={diagnostics.mean_negative_mass_fraction:.6g}"
        )

    csv_path = Path(args.csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    time_window_errors = {
        start_time: [float(row[_time_column(start_time)]) for row in rows]
        for start_time in valid_start_times
    }
    figure = plot_mode_sweep(
        modes=[int(row["n_modes"]) for row in rows],
        time_window_errors=time_window_errors,
        variance_errors=[float(row["variance_relative_l2_error"]) for row in rows],
        negative_mass_fractions=[
            float(row["mean_negative_mass_fraction"]) for row in rows
        ],
        negative_entry_fractions=[float(row["negative_entry_fraction"]) for row in rows],
        output_path=args.output,
        model_label=f"{args.modal_model}+{args.readout}",
    )
    print(f"Saved {csv_path}")
    print(f"Saved {figure}")
    return 0


def run_model_comparison(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    n_modes = config.n_nodes if args.modes is None else args.modes
    discrete_reference = discrete_expected_diffusion(config)
    analytic_variance = multinomial_marginal_variance(config)
    analytic_covariance = multinomial_marginal_covariance(config)
    covariance_indices, covariance_times = _nearest_time_indices(
        config.times,
        args.covariance_times,
        exclude_zero=True,
    )

    random_walk = run_random_walk_ensemble(
        config,
        n_runs=args.runs,
        seed=args.seed,
        method=args.random_walk_method,
    )
    model_runs: dict[str, np.ndarray] = {
        f"random_walk_{args.random_walk_method}": random_walk.runs
    }
    ensembles: dict[str, EnsembleResult] = {}
    for offset, model_name in enumerate(args.models, start=1):
        ensemble = run_modal_ensemble(
            config,
            n_runs=args.runs,
            model=model_name,
            n_modes=n_modes,
            modal_particle_weight=args.modal_particle_weight,
            initialization=args.initialization,
            initial_n_modes=args.initial_modes,
            handoff_time=args.handoff_time,
            seed=args.seed + offset,
        )
        ensembles[model_name] = ensemble
        model_runs[model_name] = ensemble.runs

    diagnostics: dict[str, dict[str, float | None]] = {}
    for name, runs in model_runs.items():
        values = summarize_ensemble(
            discrete_reference,
            analytic_variance,
            runs,
            config.n_particles,
            reference_covariance=analytic_covariance,
            covariance_time_indices=covariance_indices,
        ).as_dict()
        if name in ensembles:
            values.update(_bank_diagnostics(ensembles[name]))
            values.update(_handoff_diagnostics(ensembles[name]))
        diagnostics[name] = values

    figure = plot_model_comparison(
        config=config,
        discrete_reference=discrete_reference,
        analytic_variance=analytic_variance,
        model_runs=model_runs,
        diagnostics=diagnostics,
        output_path=args.output,
        profile_step=args.profile_step,
    )

    csv_path = Path(args.csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"model": name, **values} for name, values in diagnostics.items()]
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    data_path = Path(args.data_output) if args.data_output else figure.with_suffix(".npz")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    save_items: dict[str, object] = {
        "discrete_reference": discrete_reference,
        "analytic_variance": analytic_variance,
        "covariance_time_indices": np.asarray(covariance_indices, dtype=int),
        "covariance_times": np.asarray(covariance_times, dtype=float),
        "times": config.times,
        "positions": config.positions,
    }
    for name, runs in model_runs.items():
        save_items[f"runs_{name}"] = runs
    for name, ensemble in ensembles.items():
        if ensemble.auxiliary:
            for key, value in ensemble.auxiliary.items():
                save_items[f"aux_{name}_{key}"] = value
    np.savez_compressed(data_path, **save_items)

    report = {
        "configuration": {
            "particles": config.n_particles,
            "nodes": config.n_nodes,
            "steps": config.n_steps,
            "modes": n_modes,
            "runs": args.runs,
            "models": list(args.models),
            "random_walk_method": args.random_walk_method,
            "covariance_times": covariance_times,
        },
        "diagnostics": diagnostics,
        "figure": str(figure),
        "csv": str(csv_path),
        "data": str(data_path),
    }
    print(json.dumps(report, indent=2))
    return 0


def run_trial_convergence(args: argparse.Namespace) -> int:
    """Measure ensemble-metric convergence from nested trajectory prefixes."""

    config = _config_from_args(args)
    n_modes = config.n_nodes if args.modes is None else args.modes
    trial_counts = sorted(set(args.trial_counts))
    if not trial_counts or trial_counts[0] < 2:
        raise ValueError("trial counts must contain integers of at least 2")
    maximum_trials = trial_counts[-1]

    discrete_reference = discrete_expected_diffusion(config)
    analytic_variance = multinomial_marginal_variance(config)
    analytic_covariance = multinomial_marginal_covariance(config)
    covariance_indices, covariance_times = _nearest_time_indices(
        config.times,
        args.covariance_times,
        exclude_zero=True,
    )

    csv_path = Path(args.csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str | None]] = []
    full_mean_errors: dict[str, np.ndarray] = {}

    method_names = [f"random_walk_{args.random_walk_method}", *args.models]

    def write_rows() -> None:
        if not rows:
            return
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    for method_index, method_name in enumerate(method_names):
        print(
            f"Simulating {method_name}: {maximum_trials} trials...",
            flush=True,
        )
        if method_name.startswith("random_walk_"):
            ensemble = run_random_walk_ensemble(
                config,
                n_runs=maximum_trials,
                seed=args.seed,
                method=args.random_walk_method,
            )
        else:
            ensemble = run_modal_ensemble(
                config,
                n_runs=maximum_trials,
                model=method_name,
                n_modes=n_modes,
                modal_particle_weight=args.modal_particle_weight,
                initialization=args.initialization,
                initial_n_modes=args.initial_modes,
                handoff_time=args.handoff_time,
                seed=args.seed + method_index,
            )

        full_mean_errors[method_name] = time_resolved_relative_l2_error(
            discrete_reference,
            ensemble.runs.mean(axis=0),
        )
        for count_index, trial_count in enumerate(trial_counts):
            prefix = ensemble.runs[:trial_count]
            diagnostics = summarize_ensemble(
                discrete_reference,
                analytic_variance,
                prefix,
                config.n_particles,
                reference_covariance=analytic_covariance,
                covariance_time_indices=covariance_indices,
            ).as_dict()
            intervals = bootstrap_error_intervals(
                discrete_reference,
                analytic_variance,
                prefix,
                reference_covariance=analytic_covariance,
                covariance_time_indices=covariance_indices,
                n_resamples=args.bootstrap_resamples,
                confidence_level=args.confidence_level,
                n_blocks=args.bootstrap_blocks,
                seed=args.seed + 10_000 * method_index + count_index,
            )
            row: dict[str, float | int | str | None] = {
                "model": method_name,
                "trial_count": trial_count,
                **diagnostics,
            }
            for metric, (lower, upper) in intervals.items():
                row[f"{metric}_ci_lower"] = lower
                row[f"{metric}_ci_upper"] = upper
            rows.append(row)
            print(
                f"  n={trial_count}: "
                f"mean={diagnostics['mean_relative_l2_error']:.6g}, "
                f"variance={diagnostics['variance_relative_l2_error']:.6g}, "
                f"covariance={diagnostics['covariance_relative_frobenius_error']:.6g}",
                flush=True,
            )

        write_rows()
        del ensemble
        gc.collect()

    figure = plot_trial_convergence(
        rows=rows,
        times=config.times,
        mean_error_by_time=full_mean_errors,
        output_path=args.output,
    )
    data_path = Path(args.data_output) if args.data_output else figure.with_suffix(".npz")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    save_items: dict[str, object] = {
        "times": config.times,
        "trial_counts": np.asarray(trial_counts, dtype=int),
        "covariance_time_indices": np.asarray(covariance_indices, dtype=int),
        "covariance_times": np.asarray(covariance_times, dtype=float),
    }
    for name, values in full_mean_errors.items():
        save_items[f"mean_relative_l2_by_time_{name}"] = values
    np.savez_compressed(data_path, **save_items)

    report = {
        "configuration": {
            "particles": config.n_particles,
            "nodes": config.n_nodes,
            "steps": config.n_steps,
            "modes": n_modes,
            "trial_counts": trial_counts,
            "models": method_names,
            "bootstrap_resamples": args.bootstrap_resamples,
            "confidence_level": args.confidence_level,
            "covariance_times": covariance_times,
        },
        "figure": str(figure),
        "csv": str(csv_path),
        "data": str(data_path),
    }
    print(json.dumps(report, indent=2))
    return 0


def run_readout_comparison(args: argparse.Namespace) -> int:
    """Compare output-only readouts on one shared raw modal ensemble."""

    config = _config_from_args(args)
    n_modes = config.n_nodes if args.modes is None else args.modes
    discrete_reference = discrete_expected_diffusion(config)
    analytic_variance = multinomial_marginal_variance(config)
    analytic_covariance = multinomial_marginal_covariance(config)
    covariance_indices, covariance_times = _nearest_time_indices(
        config.times,
        args.covariance_times,
        exclude_zero=True,
    )

    random_walk = run_random_walk_ensemble(
        config,
        n_runs=args.runs,
        seed=args.seed,
        method=args.random_walk_method,
    )
    raw_modal = run_modal_ensemble(
        config,
        n_runs=args.runs,
        model=args.modal_model,
        n_modes=n_modes,
        modal_particle_weight=args.modal_particle_weight,
        initialization=args.initialization,
        initial_n_modes=args.initial_modes,
        handoff_time=args.handoff_time,
        seed=args.seed + 1,
    )

    model_runs: dict[str, np.ndarray] = {
        f"random_walk_{args.random_walk_method}": random_walk.runs
    }
    readout_ensembles: dict[str, EnsembleResult] = {}
    for readout_name in args.readouts:
        ensemble = apply_readout_ensemble(
            raw_modal,
            config,
            readout=readout_name,
            spatial_error_fraction=args.spatial_error_fraction,
            retained_modes=n_modes,
            completion_start_time=_resolved_completion_start_time(args),
            completion_rank=args.completion_rank,
            completion_ridge=args.completion_ridge,
            completion_bandwidth=args.completion_bandwidth,
            seed=args.seed + 10_000,
        )
        label = f"{args.modal_model}+{readout_name}"
        readout_ensembles[label] = ensemble
        model_runs[label] = ensemble.runs

    diagnostics: dict[str, dict[str, float | None]] = {}
    for name, runs in model_runs.items():
        values = summarize_ensemble(
            discrete_reference,
            analytic_variance,
            runs,
            config.n_particles,
            reference_covariance=analytic_covariance,
            covariance_time_indices=covariance_indices,
        ).as_dict()
        if name in readout_ensembles:
            values.update(_readout_diagnostics(readout_ensembles[name], config.n_particles))
        diagnostics[name] = values

    figure = plot_model_comparison(
        config=config,
        discrete_reference=discrete_reference,
        analytic_variance=analytic_variance,
        model_runs=model_runs,
        diagnostics=diagnostics,
        output_path=args.output,
        profile_step=args.profile_step,
    )

    csv_path = Path(args.csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"model": name, **values} for name, values in diagnostics.items()]
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    data_path = Path(args.data_output) if args.data_output else figure.with_suffix(".npz")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    save_items: dict[str, object] = {
        "raw_modal_runs": raw_modal.runs,
        "discrete_reference": discrete_reference,
        "analytic_variance": analytic_variance,
        "times": config.times,
        "positions": config.positions,
        "modal_model": np.asarray(args.modal_model),
        "readouts": np.asarray(args.readouts),
    }
    for name, runs in model_runs.items():
        save_items[f"runs_{name}"] = runs
    for name, ensemble in readout_ensembles.items():
        if ensemble.auxiliary:
            for key, value in ensemble.auxiliary.items():
                save_items[f"aux_{name}_{key}"] = value
    np.savez_compressed(data_path, **save_items)

    report = {
        "configuration": {
            "particles": config.n_particles,
            "nodes": config.n_nodes,
            "steps": config.n_steps,
            "modes": n_modes,
            "runs": args.runs,
            "modal_model": args.modal_model,
            "readouts": list(args.readouts),
            "spatial_error_fraction": args.spatial_error_fraction,
            "completion_start_time": _resolved_completion_start_time(args),
            "completion_rank": args.completion_rank,
            "completion_ridge": args.completion_ridge,
            "covariance_times": covariance_times,
        },
        "diagnostics": diagnostics,
        "figure": str(figure),
        "csv": str(csv_path),
        "data": str(data_path),
    }
    print(json.dumps(report, indent=2))
    return 0


def run_handoff_selection_study(args: argparse.Namespace) -> int:
    """Predict handoff error and optionally validate it with stochastic runs."""

    config = _config_from_args(args)
    initial_modes = config.n_nodes if args.initial_modes is None else args.initial_modes
    final_modes = sorted(set(args.final_modes))
    handoff_times = sorted(set(float(value) for value in args.handoff_times))
    covariance_indices, covariance_times = _nearest_time_indices(
        config.times,
        args.covariance_times,
        exclude_zero=True,
    )
    predictions = predict_handoff_grid(
        config,
        final_modes=final_modes,
        handoff_times=handoff_times,
        covariance_time_indices=covariance_indices,
        initial_n_modes=initial_modes,
    )
    frontier = pareto_handoff_indices(predictions)
    rows: list[dict[str, float | int | str | bool | None]] = []
    for index, prediction in enumerate(predictions):
        values: dict[str, float | int | str | bool | None] = {
            **prediction.as_dict(),
            "pareto_optimal": index in frontier,
            "meets_tolerances": (
                prediction.predicted_mean_relative_l2_error <= args.mean_tolerance
                and prediction.predicted_variance_relative_l2_error
                <= args.variance_tolerance
                and prediction.predicted_covariance_relative_frobenius_error
                <= args.covariance_tolerance
            ),
        }
        rows.append(values)

    feasible = [row for row in rows if bool(row["meets_tolerances"])]
    if feasible:
        compute_choice = min(
            feasible,
            key=lambda row: float(row["modal_update_fraction_of_full"]),
        )
        compute_choice["recommendation"] = "lowest_work_within_tolerances"
    pareto_rows = [row for row in rows if bool(row["pareto_optimal"])]
    if pareto_rows:
        accuracy_choice = min(
            pareto_rows,
            key=lambda row: (
                float(row["predicted_mean_relative_l2_error"])
                + float(row["predicted_variance_relative_l2_error"])
                + float(row["predicted_covariance_relative_frobenius_error"])
            ),
        )
        existing = accuracy_choice.get("recommendation")
        accuracy_choice["recommendation"] = (
            "accuracy_first"
            if existing is None
            else f"{existing};accuracy_first"
        )

    if args.runs > 0:
        discrete_reference = discrete_expected_diffusion(config)
        analytic_variance = multinomial_marginal_variance(config)
        analytic_covariance = multinomial_marginal_covariance(config)
        print(f"Simulating full correlated baseline: {args.runs} trials", flush=True)
        baseline = run_modal_ensemble(
            config,
            n_runs=args.runs,
            model="correlated_modal",
            n_modes=initial_modes,
            seed=args.seed,
        )
        baseline_diagnostics = summarize_ensemble(
            discrete_reference,
            analytic_variance,
            baseline.runs,
            config.n_particles,
            reference_covariance=analytic_covariance,
            covariance_time_indices=covariance_indices,
        )
        baseline_mean = baseline_diagnostics.mean_relative_l2_error
        baseline_variance = baseline_diagnostics.variance_relative_l2_error
        baseline_covariance = float(
            baseline_diagnostics.covariance_relative_frobenius_error
        )
        del baseline
        gc.collect()

        for row_index, (row, prediction) in enumerate(zip(rows, predictions, strict=True)):
            print(
                f"Validating handoff={prediction.handoff_time:g} us, "
                f"modes={prediction.final_n_modes}: {args.runs} trials",
                flush=True,
            )
            ensemble = run_modal_ensemble(
                config,
                n_runs=args.runs,
                model="handoff_correlated_modal",
                n_modes=prediction.final_n_modes,
                initial_n_modes=initial_modes,
                handoff_time=prediction.handoff_time,
                seed=args.seed + 1_000 + row_index,
            )
            diagnostics = summarize_ensemble(
                discrete_reference,
                analytic_variance,
                ensemble.runs,
                config.n_particles,
                reference_covariance=analytic_covariance,
                covariance_time_indices=covariance_indices,
            )
            row.update(
                {
                    "baseline_mean_relative_l2_error": baseline_mean,
                    "baseline_variance_relative_l2_error": baseline_variance,
                    "baseline_covariance_relative_frobenius_error": baseline_covariance,
                    "predicted_mean_with_sampling_error": float(
                        np.hypot(
                            prediction.predicted_mean_relative_l2_error,
                            baseline_mean,
                        )
                    ),
                    "predicted_variance_with_sampling_error": float(
                        np.hypot(
                            prediction.predicted_variance_relative_l2_error,
                            baseline_variance,
                        )
                    ),
                    "predicted_covariance_with_sampling_error": float(
                        np.hypot(
                            prediction.predicted_covariance_relative_frobenius_error,
                            baseline_covariance,
                        )
                    ),
                    "observed_mean_relative_l2_error": diagnostics.mean_relative_l2_error,
                    "observed_variance_relative_l2_error": (
                        diagnostics.variance_relative_l2_error
                    ),
                    "observed_covariance_relative_frobenius_error": (
                        diagnostics.covariance_relative_frobenius_error
                    ),
                    **_handoff_diagnostics(ensemble),
                }
            )
            del ensemble
            gc.collect()

    csv_path = Path(args.csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    figure = plot_handoff_selection(
        rows=rows,
        final_modes=final_modes,
        handoff_times=handoff_times,
        output_path=args.output,
    )
    validation_summary: dict[str, dict[str, float]] = {}
    if args.runs > 0:
        validation_specs = {
            "mean": (
                "predicted_mean_with_sampling_error",
                "observed_mean_relative_l2_error",
            ),
            "variance": (
                "predicted_variance_with_sampling_error",
                "observed_variance_relative_l2_error",
            ),
            "covariance": (
                "predicted_covariance_with_sampling_error",
                "observed_covariance_relative_frobenius_error",
            ),
        }
        for name, (predicted_key, observed_key) in validation_specs.items():
            predicted = np.asarray([float(row[predicted_key]) for row in rows])
            observed = np.asarray([float(row[observed_key]) for row in rows])
            validation_summary[name] = {
                "correlation": float(np.corrcoef(predicted, observed)[0, 1]),
                "relative_rmse": float(
                    np.linalg.norm(predicted - observed) / np.linalg.norm(observed)
                ),
                "median_prediction_to_observation_ratio": float(
                    np.median(predicted / observed)
                ),
            }
    report = {
        "configuration": {
            "particles": config.n_particles,
            "nodes": config.n_nodes,
            "steps": config.n_steps,
            "initial_modes": initial_modes,
            "final_modes": final_modes,
            "handoff_times": handoff_times,
            "validation_runs": args.runs,
            "covariance_times": covariance_times,
            "tolerances": {
                "mean": args.mean_tolerance,
                "variance": args.variance_tolerance,
                "covariance": args.covariance_tolerance,
            },
        },
        "recommendations": [
            row for row in rows if row.get("recommendation") is not None
        ],
        "validation_summary": validation_summary,
        "figure": str(figure),
        "csv": str(csv_path),
    }
    print(json.dumps(report, indent=2))
    return 0


def run_handoff_sweep(args: argparse.Namespace) -> int:
    """Sweep handoff times and final mode counts for adaptive correlated diffusion."""

    config = _config_from_args(args)
    initial_modes = config.n_nodes if args.initial_modes is None else args.initial_modes
    final_modes = sorted(set(args.final_modes))
    handoff_times = sorted(set(float(value) for value in args.handoff_times))
    for mode_count in final_modes:
        if not 1 <= mode_count <= initial_modes:
            raise ValueError(
                f"final mode count {mode_count} must lie in [1, {initial_modes}]"
            )
    for handoff_time in handoff_times:
        if handoff_time < config.times[0] - 1e-12 or handoff_time > config.times[-1] + 1e-12:
            raise ValueError(
                f"handoff time {handoff_time:g} is outside the simulated range"
            )

    discrete_reference = discrete_expected_diffusion(config)
    analytic_variance = multinomial_marginal_variance(config)
    rows: list[dict[str, float | int | str]] = []

    for handoff_time in handoff_times:
        for final_mode_count in final_modes:
            raw = run_modal_ensemble(
                config,
                n_runs=args.runs,
                model="handoff_correlated_modal",
                n_modes=final_mode_count,
                initial_n_modes=initial_modes,
                handoff_time=handoff_time,
                seed=args.seed + int(round(1000 * handoff_time)) + final_mode_count,
            )
            completion_start = (
                handoff_time
                if args.completion_start_time is None
                else float(args.completion_start_time)
            )
            ensemble = apply_readout_ensemble(
                raw,
                config,
                readout=args.readout,
                spatial_error_fraction=args.spatial_error_fraction,
                retained_modes=final_mode_count,
                completion_start_time=completion_start,
                completion_rank=args.completion_rank,
                completion_ridge=args.completion_ridge,
                seed=args.seed + 20_000 + int(round(1000 * handoff_time)) + final_mode_count,
            )
            diagnostics = summarize_ensemble(
                discrete_reference,
                analytic_variance,
                ensemble.runs,
                config.n_particles,
            )
            row: dict[str, float | int | str] = {
                "modal_model": "handoff_correlated_modal",
                "readout": args.readout,
                "initial_n_modes": initial_modes,
                "final_n_modes": final_mode_count,
                "requested_handoff_time": handoff_time,
                **diagnostics.as_dict(),
                **_handoff_diagnostics(raw),
                **_readout_diagnostics(ensemble, config.n_particles),
            }
            for start_time in args.error_start_times:
                if start_time <= config.times[-1] + 1e-12:
                    row[_time_column(float(start_time))] = relative_l2_error_from_time(
                        discrete_reference,
                        ensemble.mean,
                        config.times,
                        float(start_time),
                    )
            rows.append(row)
            print(
                f"handoff={handoff_time:6g} us  final_modes={final_mode_count:4d}  "
                f"mean_l2={diagnostics.mean_relative_l2_error:.6g}  "
                f"variance_l2={diagnostics.variance_relative_l2_error:.6g}  "
                f"negative_mass={diagnostics.mean_negative_mass_fraction:.6g}"
            )

    csv_path = Path(args.csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    figure = plot_handoff_sweep(
        rows=rows,
        final_modes=final_modes,
        handoff_times=handoff_times,
        late_start_time=args.late_start_time,
        output_path=args.output,
        readout_label=args.readout,
    )
    print(f"Saved {csv_path}")
    print(f"Saved {figure}")
    return 0


def run_completion_rank_sweep(args: argparse.Namespace) -> int:
    """Sweep completion rank, persistence, and optional physical constraints."""

    config = _config_from_args(args)
    n_modes = config.n_nodes if args.modes is None else int(args.modes)
    max_rank = config.n_nodes - n_modes
    ranks = sorted(set(int(value) for value in args.completion_ranks))
    if any(rank < 0 or rank > max_rank for rank in ranks):
        raise ValueError(
            f"completion ranks must lie in [0, {max_rank}] for {n_modes} retained modes"
        )

    discrete_reference = discrete_expected_diffusion(config)
    analytic_variance = multinomial_marginal_variance(config)
    analytic_covariance = multinomial_marginal_covariance(config)
    covariance_indices, covariance_times = _nearest_time_indices(
        config.times,
        args.covariance_times,
        exclude_zero=True,
    )

    raw = run_modal_ensemble(
        config,
        n_runs=args.runs,
        model="handoff_correlated_modal",
        n_modes=n_modes,
        initial_n_modes=args.initial_modes,
        handoff_time=args.handoff_time,
        seed=args.seed + 1,
    )

    completion_start = (
        args.handoff_time
        if args.completion_start_time is None
        else args.completion_start_time
    )
    lag_steps = int(round(args.temporal_lag / config.dt))
    if lag_steps <= 0 or not np.isclose(lag_steps * config.dt, args.temporal_lag):
        raise ValueError("temporal_lag must be a positive integer multiple of dt")
    start_step = _nearest_step(config.times, completion_start)
    temporal_origins = valid_time_origins(
        config.n_steps,
        lag_steps,
        start_step=start_step,
    )
    reference_temporal_covariance = multinomial_mean_cross_time_covariance(
        config,
        lag_steps,
        start_indices=temporal_origins,
    )

    rows: list[dict[str, float | int | str | None]] = []
    for rank in ranks:
        if rank == 0:
            candidates = [("raw", raw)]
        else:
            candidates = []
            for readout in args.completion_readouts:
                completion_family = (
                    0 if readout.startswith("unresolved_gaussian") else 1
                )
                ensemble = apply_readout_ensemble(
                    raw,
                    config,
                    readout=readout,
                    retained_modes=n_modes,
                    completion_start_time=completion_start,
                    completion_rank=rank,
                    completion_ridge=args.completion_ridge,
                    seed=args.seed + 10_000 + 100 * rank + completion_family,
                )
                candidates.append((readout, ensemble))

        for label, ensemble in candidates:
            diagnostics = summarize_ensemble(
                discrete_reference,
                analytic_variance,
                ensemble.runs,
                config.n_particles,
                reference_covariance=analytic_covariance,
                covariance_time_indices=covariance_indices,
            )
            observed_temporal_covariance = empirical_mean_cross_time_covariance(
                ensemble.runs,
                lag_steps,
                start_indices=temporal_origins,
            )
            temporal_error = relative_frobenius_error(
                reference_temporal_covariance,
                observed_temporal_covariance,
            )
            completion_l1 = 0.0
            if ensemble.auxiliary and "completion_l1_fraction" in ensemble.auxiliary:
                completion_l1 = float(
                    np.mean(ensemble.auxiliary["completion_l1_fraction"])
                )
            rows.append(
                {
                    "modal_model": "handoff_correlated_modal",
                    "readout": label,
                    "initial_n_modes": (
                        config.n_nodes
                        if args.initial_modes is None
                        else args.initial_modes
                    ),
                    "final_n_modes": n_modes,
                    "handoff_time": args.handoff_time,
                    "completion_rank": rank,
                    "completion_ridge": args.completion_ridge,
                    "temporal_lag": args.temporal_lag,
                    "cross_time_covariance_relative_frobenius_error": temporal_error,
                    "mean_completion_l1_fraction": completion_l1,
                    **diagnostics.as_dict(),
                    **_readout_diagnostics(ensemble, config.n_particles),
                }
            )
            print(
                f"readout={label:48s}  completion_rank={rank:3d}  "
                f"mean={diagnostics.mean_relative_l2_error:.6g}  "
                f"variance={diagnostics.variance_relative_l2_error:.6g}  "
                f"covariance={diagnostics.covariance_relative_frobenius_error:.6g}  "
                f"temporal={temporal_error:.6g}",
                flush=True,
            )

    csv_path = Path(args.csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    figure = plot_completion_rank_sweep(
        rows=rows,
        output_path=args.output,
    )
    print(f"Saved {csv_path}")
    print(f"Saved {figure}")
    print(f"Covariance times: {covariance_times}")
    return 0



def _nearest_step(times: np.ndarray, requested_time: float) -> int:
    if requested_time < times[0] - 1e-12 or requested_time > times[-1] + 1e-12:
        raise ValueError(
            f"requested time {requested_time:g} is outside "
            f"[{times[0]:g}, {times[-1]:g}]"
        )
    return int(np.argmin(np.abs(times - float(requested_time))))


def run_temporal_correlation_comparison(args: argparse.Namespace) -> int:
    """Compare exact and empirical cross-time covariance after modal handoff."""

    config = _config_from_args(args)
    final_modes = int(args.modes)
    initial_modes = config.n_nodes if args.initial_modes is None else int(args.initial_modes)
    if not 1 <= final_modes <= initial_modes <= config.n_nodes:
        raise ValueError("require 1 <= modes <= initial_modes <= nodes")

    lag_steps = sorted(
        {
            max(1, int(np.rint(float(lag_time) / config.dt)))
            for lag_time in args.lags
        }
    )
    if not lag_steps or max(lag_steps) >= config.n_steps:
        raise ValueError("lags must contain positive values shorter than the trajectory")
    lag_times = [float(step * config.dt) for step in lag_steps]

    start_step = _nearest_step(config.times, args.temporal_start_time)
    end_step = (
        config.n_steps - 1
        if args.temporal_end_time is None
        else _nearest_step(config.times, args.temporal_end_time)
    )
    if end_step <= start_step:
        raise ValueError("temporal_end_time must be later than temporal_start_time")

    radius = int(args.node_window_radius)
    if radius < 0:
        raise ValueError("node_window_radius must be nonnegative")
    window_nodes = np.arange(
        max(0, config.impulse_index - radius),
        min(config.n_nodes, config.impulse_index + radius + 1),
        dtype=np.int64,
    )
    impulse_nodes = np.asarray([config.impulse_index], dtype=np.int64)

    random_walk = run_random_walk_ensemble(
        config,
        n_runs=args.runs,
        seed=args.seed,
        method=args.random_walk_method,
    )
    full_correlated = run_modal_ensemble(
        config,
        n_runs=args.runs,
        model="correlated_modal",
        n_modes=config.n_nodes,
        seed=args.seed + 1,
    )
    handoff_raw = run_modal_ensemble(
        config,
        n_runs=args.runs,
        model="handoff_correlated_modal",
        n_modes=final_modes,
        initial_n_modes=initial_modes,
        handoff_time=args.handoff_time,
        seed=args.seed + 2,
    )
    completion_start = (
        args.handoff_time
        if args.completion_start_time is None
        else float(args.completion_start_time)
    )
    handoff_completed = apply_readout_ensemble(
        handoff_raw,
        config,
        readout="unresolved_gaussian_completion",
        retained_modes=final_modes,
        completion_start_time=completion_start,
        completion_rank=args.completion_rank,
        completion_ridge=args.completion_ridge,
        seed=args.seed + 10_000,
    )
    handoff_persistent = apply_readout_ensemble(
        handoff_raw,
        config,
        readout="persistent_unresolved_completion",
        retained_modes=final_modes,
        completion_start_time=completion_start,
        completion_rank=args.completion_rank,
        completion_ridge=args.completion_ridge,
        seed=args.seed + 20_000,
    )

    model_runs: dict[str, np.ndarray] = {
        f"random_walk_{args.random_walk_method}": random_walk.runs,
        "full_correlated_modal": full_correlated.runs,
        "handoff_raw": handoff_raw.runs,
        "handoff_completion": handoff_completed.runs,
        "handoff_persistent_completion": handoff_persistent.runs,
    }

    covariance_errors: dict[str, list[float]] = {name: [] for name in model_runs}
    impulse_correlations: dict[str, list[float]] = {
        "analytic_reference": []
    }
    window_correlations: dict[str, list[float]] = {
        "analytic_reference": []
    }
    for name in model_runs:
        impulse_correlations[name] = []
        window_correlations[name] = []

    rows: list[dict[str, float | int | str]] = []
    profile_step = max(1, int(np.rint(float(args.profile_lag) / config.dt)))
    profile_origins = valid_time_origins(
        config.n_steps,
        profile_step,
        start_step=start_step,
        end_step=end_step,
    )
    if profile_origins.size == 0:
        raise ValueError("profile_lag has no valid time origins in the requested window")
    analytic_profile_covariance = multinomial_mean_cross_time_covariance(
        config,
        profile_step,
        start_indices=profile_origins,
    )
    profile_covariances: dict[str, np.ndarray] = {
        "analytic_reference": np.diag(analytic_profile_covariance)
    }

    for lag_step, lag_time in zip(lag_steps, lag_times, strict=True):
        origins = valid_time_origins(
            config.n_steps,
            lag_step,
            start_step=start_step,
            end_step=end_step,
        )
        if origins.size == 0:
            raise ValueError(
                f"lag {lag_time:g} µs has no valid time origins in the requested window"
            )
        analytic_covariance = multinomial_mean_cross_time_covariance(
            config,
            lag_step,
            start_indices=origins,
        )
        analytic_impulse = multinomial_mean_node_lag_correlation(
            config,
            lag_step,
            impulse_nodes,
            start_indices=origins,
        )
        analytic_window = multinomial_mean_node_lag_correlation(
            config,
            lag_step,
            window_nodes,
            start_indices=origins,
        )
        impulse_correlations["analytic_reference"].append(analytic_impulse)
        window_correlations["analytic_reference"].append(analytic_window)
        rows.append(
            {
                "model": "analytic_reference",
                "lag_steps": lag_step,
                "lag_time_us": lag_time,
                "n_time_origins": int(origins.size),
                "cross_time_covariance_relative_frobenius_error": 0.0,
                "impulse_node_lag_correlation": analytic_impulse,
                "central_window_lag_correlation": analytic_window,
            }
        )

        for name, runs in model_runs.items():
            observed_covariance = empirical_mean_cross_time_covariance(
                runs,
                lag_step,
                start_indices=origins,
            )
            covariance_error = relative_frobenius_error(
                analytic_covariance,
                observed_covariance,
            )
            impulse_correlation = mean_node_lag_correlation_from_runs(
                runs,
                lag_step,
                impulse_nodes,
                start_indices=origins,
            )
            window_correlation = mean_node_lag_correlation_from_runs(
                runs,
                lag_step,
                window_nodes,
                start_indices=origins,
            )
            covariance_errors[name].append(covariance_error)
            impulse_correlations[name].append(impulse_correlation)
            window_correlations[name].append(window_correlation)
            rows.append(
                {
                    "model": name,
                    "lag_steps": lag_step,
                    "lag_time_us": lag_time,
                    "n_time_origins": int(origins.size),
                    "cross_time_covariance_relative_frobenius_error": covariance_error,
                    "impulse_node_lag_correlation": impulse_correlation,
                    "central_window_lag_correlation": window_correlation,
                }
            )

    for name, runs in model_runs.items():
        covariance = empirical_mean_cross_time_covariance(
            runs,
            profile_step,
            start_indices=profile_origins,
        )
        profile_covariances[name] = np.diag(covariance)

    figure = plot_temporal_correlation_comparison(
        lag_times=lag_times,
        covariance_errors=covariance_errors,
        impulse_correlations=impulse_correlations,
        window_correlations=window_correlations,
        profile_positions=config.positions,
        profile_covariances=profile_covariances,
        profile_lag_time=float(profile_step * config.dt),
        output_path=args.output,
    )

    csv_path = Path(args.csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    data_path = Path(args.data_output) if args.data_output else figure.with_suffix(".npz")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    save_items: dict[str, object] = {
        "lag_steps": np.asarray(lag_steps, dtype=int),
        "lag_times": np.asarray(lag_times, dtype=float),
        "positions": config.positions,
        "profile_lag_steps": np.asarray(profile_step),
        "profile_lag_time": np.asarray(profile_step * config.dt),
        "window_nodes": window_nodes,
        "temporal_start_time": np.asarray(config.times[start_step]),
        "temporal_end_time": np.asarray(config.times[end_step]),
    }
    for name, values in covariance_errors.items():
        save_items[f"covariance_error_{name}"] = np.asarray(values, dtype=float)
    for name, values in impulse_correlations.items():
        save_items[f"impulse_correlation_{name}"] = np.asarray(values, dtype=float)
    for name, values in window_correlations.items():
        save_items[f"window_correlation_{name}"] = np.asarray(values, dtype=float)
    for name, values in profile_covariances.items():
        save_items[f"profile_covariance_{name}"] = np.asarray(values, dtype=float)
    np.savez_compressed(data_path, **save_items)

    report = {
        "configuration": {
            "particles": config.n_particles,
            "nodes": config.n_nodes,
            "steps": config.n_steps,
            "runs": args.runs,
            "initial_modes": initial_modes,
            "final_modes": final_modes,
            "handoff_time": args.handoff_time,
            "completion_start_time": completion_start,
            "completion_rank": args.completion_rank,
            "completion_ridge": args.completion_ridge,
            "lags_us": lag_times,
            "temporal_start_time": float(config.times[start_step]),
            "temporal_end_time": float(config.times[end_step]),
            "node_window_radius": radius,
        },
        "figure": str(figure),
        "csv": str(csv_path),
        "data": str(data_path),
    }
    print(json.dumps(report, indent=2))
    return 0


def run_diffusion_method_benchmark(args: argparse.Namespace) -> int:
    methods = tuple(args.methods)
    records = benchmark_diffusion_methods(
        node_counts=sorted(set(args.node_counts)),
        particle_counts=sorted(set(args.particle_counts)),
        base_nodes=args.base_nodes,
        base_particles=args.base_particles,
        total_time=args.total_time,
        requested_dt=args.dt,
        length=args.length,
        diffusion_coefficient=args.diffusion,
        impulse_fraction=args.impulse_fraction,
        methods=methods,
        final_modes=args.final_modes,
        retained_fraction=args.retained_fraction,
        handoff_time=args.handoff_time,
        completion_rank=args.completion_rank,
        completion_ridge=args.completion_ridge,
        repeats=args.repeats,
        amortization_runs=args.amortization_runs,
        stability_safety=args.stability_safety,
        seed=args.seed,
    )
    csv_path = write_diffusion_method_benchmark_csv(records, args.csv_output)
    figure_path = plot_diffusion_method_benchmark(records, args.output)

    print(
        "axis      size   method                              setup_s  "
        "run_s    amortized_s  resident_MiB"
    )
    for record in records:
        size = record.n_nodes if record.sweep_axis == "nodes" else record.n_particles
        print(
            f"{record.sweep_axis:9s} {size:6d}  {record.method:34s} "
            f"{record.setup_seconds:8.4f}  {record.median_run_seconds:7.4f}  "
            f"{record.amortized_seconds_per_run:11.4f}  "
            f"{record.resident_array_megabytes:12.3f}"
        )

    report = {
        "configuration": {
            "methods": list(methods),
            "node_counts": sorted(set(args.node_counts)),
            "particle_counts": sorted(set(args.particle_counts)),
            "base_nodes": args.base_nodes,
            "base_particles": args.base_particles,
            "total_time": args.total_time,
            "requested_dt": args.dt,
            "length": args.length,
            "diffusion_coefficient": args.diffusion,
            "impulse_fraction": args.impulse_fraction,
            "final_modes": args.final_modes,
            "retained_fraction": args.retained_fraction,
            "handoff_time": args.handoff_time,
            "completion_rank": args.completion_rank,
            "completion_ridge": args.completion_ridge,
            "repeats": args.repeats,
            "amortization_runs": args.amortization_runs,
            "stability_safety": args.stability_safety,
        },
        "figure": str(figure_path),
        "csv": str(csv_path),
    }
    print(json.dumps(report, indent=2))
    return 0


def run_component_benchmark(args: argparse.Namespace) -> int:
    """Measure the canonical completed pipeline one stage at a time."""

    if not 0.0 <= args.impulse_fraction <= 1.0:
        raise ValueError("impulse_fraction must lie in [0, 1]")
    impulse_index = int(round(args.impulse_fraction * (args.nodes - 1)))
    config = DiffusionConfig(
        n_particles=args.particles,
        n_nodes=args.nodes,
        n_steps=args.steps,
        impulse_index=impulse_index,
        dt=args.dt,
        length=args.length,
        diffusion_coefficient=args.diffusion,
    )
    benchmark = benchmark_full_field_components(
        config,
        retained_modes=args.modes,
        handoff_time=args.handoff_time,
        completion_rank=args.completion_rank,
        completion_ridge=args.completion_ridge,
        readouts=tuple(args.readouts),
        setup_repeats=args.setup_repeats,
        trajectory_repeats=args.repeats,
        warmup_runs=args.warmups,
        seed=args.seed,
    )
    csv_path = write_component_benchmark_csv(benchmark, args.csv_output)
    summary_path = write_component_benchmark_summary(
        benchmark,
        args.summary_output,
    )
    summary = benchmark.summary()
    summary["artifacts"] = {
        "samples_csv": str(csv_path),
        "summary_json": str(summary_path),
    }
    print(json.dumps(summary, indent=2))
    return 0


def run_component_scaling_benchmark(args: argparse.Namespace) -> int:
    """Run two-regime component timing and isolated-memory scaling."""

    scaling = benchmark_component_scaling(
        node_counts=args.node_counts,
        regimes=tuple(args.regimes),
        operating_point=args.operating_point,
        n_particles=args.particles,
        fixed_steps=args.steps,
        total_time=args.total_time,
        requested_dt=args.dt,
        length=args.length,
        diffusion_coefficient=args.diffusion,
        impulse_fraction=args.impulse_fraction,
        stability_safety=args.stability_safety,
        completion_ridge=args.completion_ridge,
        readouts=tuple(args.readouts),
        setup_repeats=args.setup_repeats,
        trajectory_repeats=args.repeats,
        warmup_runs=args.warmups,
        seed=args.seed,
        isolate_memory=args.isolate_memory,
        memory_sampler_interval_ms=args.memory_sampler_interval_ms,
        include_standard_comparator=args.include_standard_comparator,
    )
    timing_path = write_component_scaling_csv(scaling, args.csv_output)
    memory_path = write_component_memory_csv(scaling, args.memory_output)
    summary_path = write_component_scaling_summary(scaling, args.summary_output)
    summary = scaling.summary()
    report = {
        "configuration": {
            "node_counts": sorted(set(args.node_counts)),
            "regimes": list(dict.fromkeys(args.regimes)),
            "operating_point": args.operating_point,
            "readouts": list(dict.fromkeys(args.readouts)),
            "setup_repeats": args.setup_repeats,
            "trajectory_repeats": args.repeats,
            "isolate_memory": args.isolate_memory,
            "include_standard_comparator": args.include_standard_comparator,
        },
        "scaling_slopes": summary["scaling_slopes"],
        "standard_scaling_slopes": summary["standard_scaling_slopes"],
        "comparisons": summary["comparisons"],
        "artifacts": {
            "samples_csv": str(timing_path),
            "memory_csv": str(memory_path),
            "summary_json": str(summary_path),
        },
    }
    print(json.dumps(report, indent=2))
    return 0


def run_operating_point_study(args: argparse.Namespace) -> int:
    """Benchmark selected pipelines and compare their full distributions."""

    node_counts = sorted(set(args.node_counts))
    records = benchmark_operating_points(
        node_counts=node_counts,
        n_particles=args.particles,
        total_time=args.total_time,
        requested_dt=args.dt,
        length=args.length,
        diffusion_coefficient=args.diffusion,
        impulse_fraction=args.impulse_fraction,
        completion_ridge=args.completion_ridge,
        repeats=args.repeats,
        amortization_runs=args.amortization_runs,
        stability_safety=args.stability_safety,
        seed=args.seed,
        isolate_peak_rss=not args.no_isolate_peak_rss,
        closure=args.closure,
        modal_bandwidth=DEFAULT_LOCAL_MODAL_BANDWIDTH,
    )
    benchmark_csv = write_operating_point_benchmark_csv(records, args.csv_output)

    config = stable_diffusion_config(
        n_particles=args.particles,
        n_nodes=args.accuracy_nodes,
        total_time=args.total_time,
        requested_dt=args.dt,
        length=args.length,
        diffusion_coefficient=args.diffusion,
        impulse_fraction=args.impulse_fraction,
        stability_safety=args.stability_safety,
    )
    time_indices, selected_times = _nearest_time_indices(
        config.times,
        args.distribution_times,
        exclude_zero=True,
    )
    node_indices = sorted(
        {
            min(config.n_nodes - 1, max(0, config.impulse_index + offset))
            for offset in args.distribution_node_offsets
        }
    )
    reference_a = run_random_walk_ensemble(
        config,
        n_runs=args.accuracy_runs,
        seed=args.seed + 100_000,
        method="multinomial",
    )
    reference_b = run_random_walk_ensemble(
        config,
        n_runs=args.accuracy_runs,
        seed=args.seed + 200_000,
        method="multinomial",
    )
    analytic_mean = discrete_expected_diffusion(config)
    analytic_variance = multinomial_marginal_variance(config)
    analytic_covariance = multinomial_marginal_covariance(config)
    accuracy_rows: list[dict[str, float | int | str | None]] = []

    floor_metrics = summarize_distribution_ensemble(
        reference_a.runs,
        reference_b.runs,
        time_indices=time_indices,
        node_indices=node_indices,
        n_projections=args.projections,
        histogram_bins=args.histogram_bins,
        seed=args.seed,
    )
    accuracy_rows.append(
        {
            "operating_point": "sampling_floor",
            "n_nodes": config.n_nodes,
            "final_modes": config.n_nodes,
            "handoff_time": 0.0,
            "completion_rank": 0,
            **floor_metrics.as_dict(),
        }
    )

    for point_index, point in enumerate(SELECTED_OPERATING_POINTS):
        modes, handoff_time, completion_rank = point.scaled(
            config.n_nodes,
            float(config.times[-1]),
        )
        raw = run_modal_ensemble(
            config,
            n_runs=args.accuracy_runs,
            model="handoff_correlated_modal",
            n_modes=modes,
            initial_n_modes=config.n_nodes,
            handoff_time=handoff_time,
            seed=args.seed + 300_000 + point_index,
        )
        completion_readout = (
            "local_persistent_unresolved_completion_simplex_bank"
            if args.closure == "local_persistent"
            else "persistent_unresolved_completion_simplex_bank"
        )
        completed = apply_readout_ensemble(
            raw,
            config,
            readout=completion_readout,
            retained_modes=modes,
            completion_start_time=handoff_time,
            completion_rank=completion_rank,
            completion_ridge=args.completion_ridge,
            completion_bandwidth=DEFAULT_LOCAL_MODAL_BANDWIDTH,
            seed=args.seed + 400_000 + point_index,
        )
        distribution = summarize_distribution_ensemble(
            reference_a.runs,
            completed.runs,
            time_indices=time_indices,
            node_indices=node_indices,
            n_projections=args.projections,
            histogram_bins=args.histogram_bins,
            seed=args.seed,
        )
        moment = summarize_ensemble(
            analytic_mean,
            analytic_variance,
            completed.runs,
            config.n_particles,
            reference_covariance=analytic_covariance,
            covariance_time_indices=time_indices,
        )
        accuracy_rows.append(
            {
                "operating_point": point.name,
                "n_nodes": config.n_nodes,
                "final_modes": modes,
                "handoff_time": handoff_time,
                "completion_rank": completion_rank,
                **distribution.as_dict(),
                "mean_relative_l2_error": moment.mean_relative_l2_error,
                "variance_relative_l2_error": moment.variance_relative_l2_error,
                "covariance_relative_frobenius_error": (
                    moment.covariance_relative_frobenius_error
                ),
                "mean_negative_mass_fraction": moment.mean_negative_mass_fraction,
                "maximum_mass_error": moment.maximum_mass_error,
            }
        )
        del raw, completed
        gc.collect()

    accuracy_csv = Path(args.distribution_csv_output)
    accuracy_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in accuracy_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with accuracy_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(accuracy_rows)

    model_rows = [
        row for row in accuracy_rows if row["operating_point"] != "sampling_floor"
    ]
    floor_row = accuracy_rows[0]

    def distribution_score(row: dict[str, float | int | str | None]) -> float:
        return 0.5 * (
            float(row["marginal_normalized_wasserstein"])
            + float(row["sliced_normalized_wasserstein"])
        )

    best_accuracy = min(distribution_score(row) for row in model_rows)
    sampling_floor = distribution_score(floor_row)
    accuracy_threshold = best_accuracy + sampling_floor
    eligible = [
        row for row in model_rows if distribution_score(row) <= accuracy_threshold
    ]
    base_benchmarks = {
        record.operating_point: record
        for record in records
        if record.n_nodes == args.accuracy_nodes
    }
    if len(base_benchmarks) != len(SELECTED_OPERATING_POINTS):
        raise ValueError("accuracy_nodes must also be present in node_counts")
    recommended = min(
        eligible,
        key=lambda row: base_benchmarks[str(row["operating_point"])].amortized_seconds_per_run,
    )
    recommended_name = str(recommended["operating_point"])
    figure = plot_operating_point_study(records, accuracy_rows, args.output)

    summary = {
        "recommended_default": recommended_name,
        "completion_closure": args.closure,
        "selection_rule": (
            "lowest amortized runtime among points whose mean normalized "
            "distribution score is within one reference sampling-floor score "
            "of the best observed score"
        ),
        "accuracy_threshold": accuracy_threshold,
        "sampling_floor_score": sampling_floor,
        "distribution_times": selected_times,
        "distribution_nodes": node_indices,
        "benchmark_csv": str(benchmark_csv),
        "distribution_csv": str(accuracy_csv),
        "figure": str(figure),
    }
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(
        "nodes  point            modes  handoff  rank  setup_s  run_s  "
        "amortized_s  peak_RSS_MiB"
    )
    for record in records:
        print(
            f"{record.n_nodes:5d}  {record.operating_point:16s} "
            f"{record.final_modes:5d}  {record.handoff_time:7.2f}  "
            f"{record.completion_rank:4d}  {record.setup_seconds:7.3f}  "
            f"{record.median_run_seconds:6.3f}  "
            f"{record.amortized_seconds_per_run:11.3f}  "
            f"{record.peak_rss_megabytes:12.1f}"
        )
    print("\nDistribution accuracy at the reference grid:")
    for row in accuracy_rows:
        print(
            f"{str(row['operating_point']):16s}  "
            f"marginal_W1={float(row['marginal_normalized_wasserstein']):.4f}  "
            f"JS={float(row['marginal_jensen_shannon']):.4f}  "
            f"sliced_W1={float(row['sliced_normalized_wasserstein']):.4f}"
        )
    print(f"\nRecommended default: {recommended_name}")
    print(json.dumps(summary, indent=2))
    return 0


def run_structured_closure_study(args: argparse.Namespace) -> int:
    """Select and validate a sparse local persistent completion closure."""

    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    point = SELECTED_OPERATING_POINTS[2]

    def make_config(
        *,
        nodes: int,
        particles: int,
        impulse_fraction: float,
        diffusion: float,
        length: float = 4.0,
    ) -> DiffusionConfig:
        return stable_diffusion_config(
            n_particles=particles,
            n_nodes=nodes,
            total_time=args.total_time,
            requested_dt=args.dt,
            length=length,
            diffusion_coefficient=diffusion,
            impulse_fraction=impulse_fraction,
            stability_safety=args.stability_safety,
        )

    def write_rows(rows: list[dict[str, object]], path: Path) -> None:
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def evaluate_runs(
        config: DiffusionConfig,
        reference_runs: np.ndarray,
        model_runs: np.ndarray,
        *,
        time_indices: list[int],
        node_indices: list[int],
        start_step: int,
        seed: int,
    ) -> dict[str, float | int]:
        distribution = summarize_distribution_ensemble(
            reference_runs,
            model_runs,
            time_indices=time_indices,
            node_indices=node_indices,
            n_projections=args.projections,
            histogram_bins=args.histogram_bins,
            seed=seed,
        )
        analytic_mean = discrete_expected_diffusion(config)
        analytic_variance = multinomial_marginal_variance(config)
        analytic_covariance = multinomial_marginal_covariance(config)
        moment = summarize_ensemble(
            analytic_mean,
            analytic_variance,
            model_runs,
            config.n_particles,
            reference_covariance=analytic_covariance,
            covariance_time_indices=time_indices,
        )
        origins = valid_time_origins(
            config.n_steps,
            1,
            start_step=min(start_step, config.n_steps - 2),
        )
        exact_cross = multinomial_mean_cross_time_covariance(
            config,
            1,
            start_indices=origins,
        )
        empirical_cross = empirical_mean_cross_time_covariance(
            model_runs,
            1,
            start_indices=origins,
        )
        return {
            **distribution.as_dict(),
            "mean_relative_l2_error": moment.mean_relative_l2_error,
            "mean_max_time_relative_l2_error": (
                moment.mean_max_time_relative_l2_error
            ),
            "variance_relative_l2_error": moment.variance_relative_l2_error,
            "variance_max_time_relative_l2_error": (
                moment.variance_max_time_relative_l2_error
            ),
            "covariance_relative_frobenius_error": float(
                moment.covariance_relative_frobenius_error
            ),
            "cross_time_covariance_relative_frobenius_error": (
                relative_frobenius_error(exact_cross, empirical_cross)
            ),
            "mean_negative_mass_fraction": moment.mean_negative_mass_fraction,
            "maximum_mass_error": moment.maximum_mass_error,
        }

    base_config = make_config(
        nodes=args.accuracy_nodes,
        particles=args.particles,
        impulse_fraction=args.impulse_fraction,
        diffusion=args.diffusion,
    )
    modes, handoff_time, completion_rank = point.scaled(
        base_config.n_nodes,
        float(base_config.times[-1]),
    )
    time_indices, selected_times = _nearest_time_indices(
        base_config.times,
        args.distribution_times,
        exclude_zero=True,
    )
    node_indices = sorted(
        {
            min(base_config.n_nodes - 1, max(0, base_config.impulse_index + offset))
            for offset in args.distribution_node_offsets
        }
    )
    reference_a = run_random_walk_ensemble(
        base_config,
        n_runs=args.accuracy_runs,
        seed=args.seed + 10_000,
        method="multinomial",
    )
    reference_b = run_random_walk_ensemble(
        base_config,
        n_runs=args.accuracy_runs,
        seed=args.seed + 20_000,
        method="multinomial",
    )
    raw = run_modal_ensemble(
        base_config,
        n_runs=args.accuracy_runs,
        model="handoff_correlated_modal",
        n_modes=modes,
        initial_n_modes=base_config.n_nodes,
        handoff_time=handoff_time,
        seed=args.seed + 30_000,
    )
    accuracy_rows: list[dict[str, float | int | str | None]] = []
    floor = evaluate_runs(
        base_config,
        reference_a.runs,
        reference_b.runs,
        time_indices=time_indices,
        node_indices=node_indices,
        start_step=int(np.searchsorted(base_config.times, handoff_time)),
        seed=args.seed,
    )
    accuracy_rows.append(
        {"closure": "sampling_floor", "modal_bandwidth": 0, **floor}
    )

    dense = apply_readout_ensemble(
        raw,
        base_config,
        readout="persistent_unresolved_completion_simplex_bank",
        retained_modes=modes,
        completion_start_time=handoff_time,
        completion_rank=completion_rank,
        completion_ridge=args.completion_ridge,
        seed=args.seed + 40_000,
    )
    dense_metrics = evaluate_runs(
        base_config,
        reference_a.runs,
        dense.runs,
        time_indices=time_indices,
        node_indices=node_indices,
        start_step=int(np.searchsorted(base_config.times, handoff_time)),
        seed=args.seed,
    )
    accuracy_rows.append(
        {"closure": "dense", "modal_bandwidth": 0, **dense_metrics}
    )
    dense_runs = dense.runs
    del dense

    for bandwidth_index, bandwidth in enumerate(sorted(set(args.bandwidths))):
        local = apply_readout_ensemble(
            raw,
            base_config,
            readout="local_persistent_unresolved_completion_simplex_bank",
            retained_modes=modes,
            completion_start_time=handoff_time,
            completion_rank=completion_rank,
            completion_ridge=args.completion_ridge,
            completion_bandwidth=bandwidth,
            seed=args.seed + 40_000,
        )
        local_metrics = evaluate_runs(
            base_config,
            reference_a.runs,
            local.runs,
            time_indices=time_indices,
            node_indices=node_indices,
            start_step=int(np.searchsorted(base_config.times, handoff_time)),
            seed=args.seed,
        )
        difference_intervals = paired_bootstrap_error_difference_intervals(
            discrete_expected_diffusion(base_config),
            multinomial_marginal_variance(base_config),
            dense_runs,
            local.runs,
            reference_covariance=multinomial_marginal_covariance(base_config),
            covariance_time_indices=time_indices,
            n_resamples=args.comparison_bootstrap_resamples,
            confidence_level=args.comparison_confidence_level,
            n_blocks=args.comparison_bootstrap_blocks,
            seed=args.seed + 50_000 + bandwidth_index,
        )
        for metric, (lower, upper) in difference_intervals.items():
            local_metrics[f"{metric}_difference_ci_lower"] = lower
            local_metrics[f"{metric}_difference_ci_upper"] = upper
        accuracy_rows.append(
            {
                "closure": f"local_b{bandwidth}",
                "modal_bandwidth": bandwidth,
                **local_metrics,
            }
        )
        del local
        gc.collect()

    del dense_runs
    gc.collect()

    eligible: list[dict[str, float | int | str | None]] = []
    for row in accuracy_rows[2:]:
        if _structured_closure_noninferiority_pass(row, dense_metrics, floor):
            eligible.append(row)
    if eligible:
        selected_row = min(eligible, key=lambda row: int(row["modal_bandwidth"]))
    else:
        selected_row = min(
            accuracy_rows[2:],
            key=lambda row: np.mean(
                [
                    float(row[metric])
                    / max(float(dense_metrics[metric]), 1.0e-12)
                    for metric in STRUCTURED_CLOSURE_ACCEPTANCE_METRICS
                ]
            ),
        )
    base_candidate_bandwidths = sorted(
        int(row["modal_bandwidth"])
        for row in (eligible if eligible else [selected_row])
    )
    selected_bandwidth = int(selected_row["modal_bandwidth"])
    write_rows(
        [dict(row) for row in accuracy_rows],
        output_directory / "base_accuracy.csv",
    )

    heldout_cases = (
        ("shifted_impulse", 101, args.particles, 0.35, args.diffusion, 4.0),
        ("low_particles", 101, 2_000, args.impulse_fraction, args.diffusion, 4.0),
        (
            "high_diffusion",
            101,
            args.particles,
            args.impulse_fraction,
            2.0 * args.diffusion,
            4.0,
        ),
        ("fine_grid", 151, args.particles, args.impulse_fraction, args.diffusion, 4.0),
        ("long_domain", 101, args.particles, args.impulse_fraction, args.diffusion, 6.0),
    )
    heldout_rows: list[dict[str, float | int | str | None]] = []
    for case_index, (
        case_name,
        nodes,
        particles,
        impulse_fraction,
        diffusion,
        length,
    ) in enumerate(heldout_cases):
        config = make_config(
            nodes=nodes,
            particles=particles,
            impulse_fraction=impulse_fraction,
            diffusion=diffusion,
            length=length,
        )
        case_modes, case_handoff, case_rank = point.scaled(
            config.n_nodes,
            float(config.times[-1]),
        )
        case_times, _ = _nearest_time_indices(
            config.times,
            args.distribution_times,
            exclude_zero=True,
        )
        case_nodes = sorted(
            {
                min(config.n_nodes - 1, max(0, config.impulse_index + offset))
                for offset in args.distribution_node_offsets
            }
        )
        reference = run_random_walk_ensemble(
            config,
            n_runs=args.heldout_runs,
            seed=args.seed + 100_000 + case_index,
            method="multinomial",
        )
        reference_floor = run_random_walk_ensemble(
            config,
            n_runs=args.heldout_runs,
            seed=args.seed + 105_000 + case_index,
            method="multinomial",
        )
        case_raw = run_modal_ensemble(
            config,
            n_runs=args.heldout_runs,
            model="handoff_correlated_modal",
            n_modes=case_modes,
            initial_n_modes=config.n_nodes,
            handoff_time=case_handoff,
            seed=args.seed + 110_000 + case_index,
        )
        comparison_seed = args.seed + case_index
        case_floor = evaluate_runs(
            config,
            reference.runs,
            reference_floor.runs,
            time_indices=case_times,
            node_indices=case_nodes,
            start_step=int(np.searchsorted(config.times, case_handoff)),
            seed=comparison_seed,
        )
        heldout_rows.append(
            {
                "case": case_name,
                "closure": "sampling_floor",
                "nodes": nodes,
                "particles": particles,
                "impulse_fraction": impulse_fraction,
                "diffusion": diffusion,
                "length": length,
                "runs": args.heldout_runs,
                **case_floor,
            }
        )
        dense_case_runs: np.ndarray | None = None
        closure_candidates = [
            (
                "dense",
                "persistent_unresolved_completion_simplex_bank",
                0,
            ),
            *(
                (
                    f"local_b{bandwidth}",
                    "local_persistent_unresolved_completion_simplex_bank",
                    bandwidth,
                )
                for bandwidth in base_candidate_bandwidths
            ),
        ]
        for closure_name, readout, closure_bandwidth in closure_candidates:
            completed = apply_readout_ensemble(
                case_raw,
                config,
                readout=readout,
                retained_modes=case_modes,
                completion_start_time=case_handoff,
                completion_rank=case_rank,
                completion_ridge=args.completion_ridge,
                completion_bandwidth=(
                    selected_bandwidth
                    if closure_name == "dense"
                    else closure_bandwidth
                ),
                seed=args.seed + 120_000 + case_index,
            )
            metrics = evaluate_runs(
                config,
                reference.runs,
                completed.runs,
                time_indices=case_times,
                node_indices=case_nodes,
                start_step=int(np.searchsorted(config.times, case_handoff)),
                seed=comparison_seed,
            )
            if closure_name == "dense":
                dense_case_runs = completed.runs
            else:
                if dense_case_runs is None:
                    raise RuntimeError("dense closure must be evaluated first")
                difference_intervals = paired_bootstrap_error_difference_intervals(
                    discrete_expected_diffusion(config),
                    multinomial_marginal_variance(config),
                    dense_case_runs,
                    completed.runs,
                    reference_covariance=multinomial_marginal_covariance(config),
                    covariance_time_indices=case_times,
                    n_resamples=args.comparison_bootstrap_resamples,
                    confidence_level=args.comparison_confidence_level,
                    n_blocks=args.comparison_bootstrap_blocks,
                    seed=(
                        args.seed
                        + 130_000
                        + 1_000 * case_index
                        + closure_bandwidth
                    ),
                )
                for metric, (lower, upper) in difference_intervals.items():
                    metrics[f"{metric}_difference_ci_lower"] = lower
                    metrics[f"{metric}_difference_ci_upper"] = upper
            heldout_rows.append(
                {
                    "case": case_name,
                    "closure": closure_name,
                    "nodes": nodes,
                    "particles": particles,
                    "impulse_fraction": impulse_fraction,
                    "diffusion": diffusion,
                    "length": length,
                    "runs": args.heldout_runs,
                    **metrics,
                }
            )
            del completed
            gc.collect()
        del reference, reference_floor, case_raw, dense_case_runs
        gc.collect()
    heldout_pass_by_bandwidth = {
        bandwidth: _structured_heldout_bandwidth_passes(
            [dict(row) for row in heldout_rows],
            bandwidth,
        )
        for bandwidth in base_candidate_bandwidths
    }
    heldout_eligible_bandwidths = [
        bandwidth
        for bandwidth in base_candidate_bandwidths
        if heldout_pass_by_bandwidth[bandwidth]
    ]
    heldout_pass = bool(eligible) and bool(heldout_eligible_bandwidths)
    if heldout_eligible_bandwidths:
        selected_bandwidth = min(heldout_eligible_bandwidths)
    write_rows(
        [dict(row) for row in heldout_rows],
        output_directory / "heldout_accuracy.csv",
    )

    profile_rows: list[dict[str, object]] = []
    for closure_name, completer_class, kwargs in (
        ("dense", PersistentUnresolvedGaussianCompleter, {}),
        (
            f"local_b{selected_bandwidth}",
            LocalPersistentUnresolvedGaussianCompleter,
            {"modal_bandwidth": selected_bandwidth},
        ),
    ):
        profiler = cProfile.Profile()
        profiler.enable()
        completer = completer_class(
            base_config,
            retained_modes=modes,
            completion_start_time=handoff_time,
            completion_rank=completion_rank,
            ridge=args.completion_ridge,
            **kwargs,
        )
        profiler.disable()
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(
            "cumulative"
        ).print_stats(30)
        (output_directory / f"{closure_name}_setup_profile.txt").write_text(
            stream.getvalue(),
            encoding="utf-8",
        )
        transitions = [
            step for step in completer._transition_steps if step is not None
        ]
        profile_rows.append(
            {
                "closure": closure_name,
                "total_array_bytes": deep_numpy_nbytes(completer),
                "gain_array_bytes": deep_numpy_nbytes(
                    [step.conditional_gain for step in transitions]
                ),
                "factor_array_bytes": deep_numpy_nbytes(
                    [step.conditional_factor for step in transitions]
                ),
                "mean_array_bytes": deep_numpy_nbytes(
                    [
                        (step.conditioning_mean, step.target_mean)
                        for step in transitions
                    ]
                ),
            }
        )
        del completer
        gc.collect()
    write_rows(profile_rows, output_directory / "storage_profile.csv")

    dense_benchmark = benchmark_operating_points(
        node_counts=sorted(set(args.node_counts)),
        n_particles=args.particles,
        total_time=args.total_time,
        requested_dt=args.dt,
        length=4.0,
        diffusion_coefficient=args.diffusion,
        impulse_fraction=args.impulse_fraction,
        operating_points=(point,),
        completion_ridge=args.completion_ridge,
        repeats=args.repeats,
        amortization_runs=args.amortization_runs,
        stability_safety=args.stability_safety,
        seed=args.seed,
        closure="dense_persistent",
    )
    local_benchmark = benchmark_operating_points(
        node_counts=sorted(set(args.node_counts)),
        n_particles=args.particles,
        total_time=args.total_time,
        requested_dt=args.dt,
        length=4.0,
        diffusion_coefficient=args.diffusion,
        impulse_fraction=args.impulse_fraction,
        operating_points=(point,),
        completion_ridge=args.completion_ridge,
        repeats=args.repeats,
        amortization_runs=args.amortization_runs,
        stability_safety=args.stability_safety,
        seed=args.seed,
        closure="local_persistent",
        modal_bandwidth=selected_bandwidth,
    )
    benchmark_records = [*dense_benchmark, *local_benchmark]
    write_operating_point_benchmark_csv(
        benchmark_records,
        output_directory / "scaling.csv",
    )

    largest_node = max(args.node_counts)
    dense_largest = next(
        row for row in dense_benchmark if row.n_nodes == largest_node
    )
    local_largest = next(
        row for row in local_benchmark if row.n_nodes == largest_node
    )
    trajectory_runtime_pass = (
        local_largest.median_run_seconds
        <= 1.05 * dense_largest.median_run_seconds
    )
    scaling_pass = (
        local_largest.setup_seconds < dense_largest.setup_seconds
        and local_largest.resident_array_bytes < dense_largest.resident_array_bytes
        and local_largest.peak_rss_bytes <= 1.05 * dense_largest.peak_rss_bytes
        and trajectory_runtime_pass
    )
    accepted = bool(eligible) and heldout_pass and scaling_pass
    plotted_heldout_rows = [
        row
        for row in heldout_rows
        if str(row["closure"])
        in {"sampling_floor", "dense", f"local_b{selected_bandwidth}"}
    ]
    figure = plot_structured_closure_study(
        accuracy_rows,
        plotted_heldout_rows,
        benchmark_records,
        output_directory / "structured_closure.png",
    )
    summary = {
        "selected_bandwidth": (
            selected_bandwidth if heldout_pass else None
        ),
        "profiled_local_bandwidth": selected_bandwidth,
        "base_accuracy_eligible": bool(eligible),
        "heldout_pass": heldout_pass,
        "heldout_pass_by_bandwidth": {
            str(bandwidth): passed
            for bandwidth, passed in heldout_pass_by_bandwidth.items()
        },
        "scaling_pass": scaling_pass,
        "trajectory_runtime_pass": trajectory_runtime_pass,
        "accepted_as_default": accepted,
        "production_default": (
            f"local_persistent_bandwidth_{selected_bandwidth}"
            if accepted
            else "dense_persistent"
        ),
        "accuracy_runs": args.accuracy_runs,
        "heldout_runs": args.heldout_runs,
        "case_specific_sampling_floors": True,
        "comparison_bootstrap_resamples": args.comparison_bootstrap_resamples,
        "comparison_bootstrap_blocks": args.comparison_bootstrap_blocks,
        "comparison_confidence_level": args.comparison_confidence_level,
        "distribution_times": selected_times,
        "largest_grid": {
            "nodes": largest_node,
            "dense_setup_seconds": dense_largest.setup_seconds,
            "local_setup_seconds": local_largest.setup_seconds,
            "dense_median_run_seconds": dense_largest.median_run_seconds,
            "local_median_run_seconds": local_largest.median_run_seconds,
            "dense_resident_mib": dense_largest.resident_array_megabytes,
            "local_resident_mib": local_largest.resident_array_megabytes,
            "dense_peak_rss_mib": dense_largest.peak_rss_megabytes,
            "local_peak_rss_mib": local_largest.peak_rss_megabytes,
        },
        "figure": str(figure),
    }
    (output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0

def run_random_walk_benchmark(args: argparse.Namespace) -> int:
    base_config = DiffusionConfig(
        n_particles=args.particle_counts[0],
        n_nodes=args.nodes,
        n_steps=args.steps,
        impulse_index=args.impulse_index,
        dt=args.dt,
        length=args.length,
        diffusion_coefficient=args.diffusion,
    )
    particle_counts = sorted(set(args.particle_counts))
    records = benchmark_random_walks(
        base_config=base_config,
        particle_counts=particle_counts,
        repeats=args.repeats,
        seed=args.seed,
    )
    csv_path = write_benchmark_csv(records, args.csv_output)
    figure_path = plot_random_walk_benchmark(records, args.output)

    by_particles: dict[int, dict[str, object]] = {}
    for record in records:
        by_particles.setdefault(record.n_particles, {})[record.method] = record

    print("particles  naive_s  multinomial_s  speedup  naive_MiB  multinomial_MiB")
    for n_particles in particle_counts:
        naive = by_particles[n_particles]["naive"]
        multinomial = by_particles[n_particles]["multinomial"]
        speedup = naive.median_seconds / multinomial.median_seconds
        print(
            f"{n_particles:9d}  {naive.median_seconds:7.4f}  "
            f"{multinomial.median_seconds:13.4f}  {speedup:7.2f}x  "
            f"{naive.core_array_megabytes:9.3f}  "
            f"{multinomial.core_array_megabytes:15.3f}"
        )

    print(f"Saved {csv_path}")
    print(f"Saved {figure_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eigendiffusion",
        description="Stochastic diffusion using independent and correlated modal models.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate",
        help="Validate one selected modal model against analytic and random-walk references.",
    )
    _add_shared_arguments(validate)
    _add_modal_model_argument(validate)
    _add_handoff_arguments(validate)
    _add_readout_argument(validate)
    _add_covariance_arguments(validate)
    validate.add_argument("--modes", type=int, default=None)
    validate.add_argument(
        "--random-walk-method",
        choices=("naive", "multinomial"),
        default="multinomial",
        help="Random-walk implementation used for validation.",
    )
    validate.add_argument("--profile-step", type=int, default=None)
    validate.add_argument("--output", default="outputs/readouts/validation.png")
    validate.add_argument("--data-output", default=None)
    validate.set_defaults(handler=run_validation)

    sweep = subparsers.add_parser(
        "sweep-modes",
        help="Sweep retained modes for one selected modal formulation.",
    )
    _add_shared_arguments(sweep)
    _add_modal_model_argument(sweep)
    _add_handoff_arguments(sweep)
    _add_readout_argument(sweep)
    sweep.add_argument("--modes", type=int, nargs="+", default=None)
    sweep.add_argument(
        "--error-start-times",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 5.0, 10.0, 20.0],
        help="Report mean error using only times at or after each listed time.",
    )
    sweep.add_argument("--output", default="outputs/readouts/mode_sweep.png")
    sweep.add_argument("--csv-output", default="outputs/readouts/mode_sweep.csv")
    sweep.set_defaults(handler=run_sweep)

    compare = subparsers.add_parser(
        "compare-modal-models",
        help="Compare independent, correlated, and banked correlated modal formulations.",
    )
    _add_shared_arguments(compare)
    _add_handoff_arguments(compare)
    _add_covariance_arguments(compare)
    compare.add_argument(
        "--models",
        nargs="+",
        choices=MODAL_MODEL_NAMES,
        default=[
            "independent_modal",
            "correlated_modal",
            "banked_correlated_modal",
        ],
    )
    compare.add_argument("--modes", type=int, default=None)
    compare.add_argument(
        "--random-walk-method",
        choices=("naive", "multinomial"),
        default="multinomial",
    )
    compare.add_argument("--profile-step", type=int, default=None)
    compare.add_argument("--output", default="outputs/modal_model_comparison.png")
    compare.add_argument("--csv-output", default="outputs/modal_model_comparison.csv")
    compare.add_argument("--data-output", default=None)
    compare.set_defaults(handler=run_model_comparison)

    convergence = subparsers.add_parser(
        "study-trial-convergence",
        help=(
            "Measure how mean, variance, and covariance errors converge as the "
            "number of stochastic trials increases."
        ),
    )
    _add_shared_arguments(convergence, include_runs=False)
    _add_handoff_arguments(convergence)
    _add_covariance_arguments(convergence)
    convergence.add_argument(
        "--models",
        nargs="+",
        choices=MODAL_MODEL_NAMES,
        default=[
            "independent_modal",
            "correlated_modal",
            "banked_correlated_modal",
        ],
    )
    convergence.add_argument("--modes", type=int, default=None)
    convergence.add_argument(
        "--random-walk-method",
        choices=("naive", "multinomial"),
        default="multinomial",
    )
    convergence.add_argument(
        "--trial-counts",
        type=int,
        nargs="+",
        default=[100, 250, 500, 1_000, 2_000, 5_000],
        help=(
            "Nested ensemble sizes. Each method is simulated only at the maximum "
            "count and smaller counts use deterministic prefixes."
        ),
    )
    convergence.add_argument("--bootstrap-resamples", type=int, default=200)
    convergence.add_argument("--bootstrap-blocks", type=int, default=20)
    convergence.add_argument("--confidence-level", type=float, default=0.95)
    convergence.add_argument(
        "--output",
        default="outputs/convergence/trial_convergence.png",
    )
    convergence.add_argument(
        "--csv-output",
        default="outputs/convergence/trial_convergence.csv",
    )
    convergence.add_argument("--data-output", default=None)
    convergence.set_defaults(handler=run_trial_convergence)

    compare_readouts = subparsers.add_parser(
        "compare-readouts",
        help="Compare raw, banked, and Delta-Sigma readouts on one modal ensemble.",
    )
    _add_shared_arguments(compare_readouts)
    _add_modal_model_argument(compare_readouts)
    _add_handoff_arguments(compare_readouts)
    _add_covariance_arguments(compare_readouts)
    _add_completion_arguments(compare_readouts)
    compare_readouts.add_argument(
        "--readouts",
        nargs="+",
        choices=READOUT_NAMES,
        default=[
            "raw",
            "simplex_bank",
            "delta_sigma_temporal",
            "delta_sigma_neighbor",
        ],
    )
    compare_readouts.add_argument(
        "--spatial-error-fraction",
        type=float,
        default=0.5,
    )
    compare_readouts.add_argument("--modes", type=int, default=None)
    compare_readouts.add_argument(
        "--random-walk-method",
        choices=("naive", "multinomial"),
        default="multinomial",
    )
    compare_readouts.add_argument("--profile-step", type=int, default=None)
    compare_readouts.add_argument(
        "--output",
        default="outputs/readouts/comparisons/readout_comparison.png",
    )
    compare_readouts.add_argument(
        "--csv-output",
        default="outputs/readouts/comparisons/readout_comparison.csv",
    )
    compare_readouts.add_argument("--data-output", default=None)
    compare_readouts.set_defaults(handler=run_readout_comparison)

    handoff_selection = subparsers.add_parser(
        "study-handoff-selection",
        help=(
            "Predict mean/variance/covariance truncation error for handoff choices "
            "and optionally validate them with stochastic ensembles."
        ),
    )
    _add_shared_arguments(handoff_selection)
    _add_covariance_arguments(handoff_selection)
    handoff_selection.add_argument("--initial-modes", type=int, default=None)
    handoff_selection.add_argument(
        "--final-modes",
        type=int,
        nargs="+",
        default=[20, 35, 50, 75],
    )
    handoff_selection.add_argument(
        "--handoff-times",
        type=float,
        nargs="+",
        default=[1.0, 5.0, 10.0, 20.0, 50.0],
    )
    handoff_selection.add_argument("--mean-tolerance", type=float, default=0.005)
    handoff_selection.add_argument("--variance-tolerance", type=float, default=0.25)
    handoff_selection.add_argument("--covariance-tolerance", type=float, default=0.30)
    handoff_selection.add_argument(
        "--output",
        default="outputs/handoff_selection/prediction_validation.png",
    )
    handoff_selection.add_argument(
        "--csv-output",
        default="outputs/handoff_selection/prediction_validation.csv",
    )
    handoff_selection.set_defaults(handler=run_handoff_selection_study)

    handoff_sweep = subparsers.add_parser(
        "sweep-handoff",
        help="Sweep full-to-reduced handoff times and final correlated mode counts.",
    )
    _add_shared_arguments(handoff_sweep)
    handoff_sweep.add_argument(
        "--initial-modes",
        type=int,
        default=None,
        help="Initial high-rank basis; defaults to all spatial modes.",
    )
    _add_readout_argument(handoff_sweep)
    handoff_sweep.add_argument(
        "--final-modes",
        type=int,
        nargs="+",
        default=[10, 20, 30, 50],
    )
    handoff_sweep.add_argument(
        "--handoff-times",
        type=float,
        nargs="+",
        default=[1.0, 5.0, 10.0, 20.0],
    )
    handoff_sweep.add_argument(
        "--error-start-times",
        type=float,
        nargs="+",
        default=[0.0, 5.0, 10.0, 20.0],
    )
    handoff_sweep.add_argument(
        "--late-start-time",
        type=float,
        default=20.0,
        help="Evaluation window shown as the late-time mean-error panel.",
    )
    handoff_sweep.add_argument(
        "--output",
        default="outputs/adaptive_handoff/handoff_sweep.png",
    )
    handoff_sweep.add_argument(
        "--csv-output",
        default="outputs/adaptive_handoff/handoff_sweep.csv",
    )
    handoff_sweep.set_defaults(handler=run_handoff_sweep)

    completion_sweep = subparsers.add_parser(
        "sweep-completion-rank",
        help="Sweep unresolved Gaussian completion rank on a shared handoff ensemble.",
    )
    _add_shared_arguments(completion_sweep)
    _add_covariance_arguments(completion_sweep)
    completion_sweep.add_argument("--modes", type=int, default=50)
    completion_sweep.add_argument("--initial-modes", type=int, default=None)
    completion_sweep.add_argument("--handoff-time", type=float, default=10.0)
    completion_sweep.add_argument("--completion-start-time", type=float, default=None)
    completion_sweep.add_argument(
        "--completion-ranks",
        type=int,
        nargs="+",
        default=[0, 5, 10, 20, 51],
        help="Use rank 0 for the raw handoff baseline.",
    )
    completion_sweep.add_argument(
        "--completion-readouts",
        nargs="+",
        choices=(
            "unresolved_gaussian_completion",
            "persistent_unresolved_completion",
            "unresolved_gaussian_completion_simplex_bank",
            "persistent_unresolved_completion_simplex_bank",
        ),
        default=[
            "unresolved_gaussian_completion",
            "persistent_unresolved_completion",
            "unresolved_gaussian_completion_simplex_bank",
            "persistent_unresolved_completion_simplex_bank",
        ],
    )
    completion_sweep.add_argument("--completion-ridge", type=float, default=1.0e-2)
    completion_sweep.add_argument(
        "--temporal-lag",
        type=float,
        default=1.0,
        help="Lag used for the cross-time covariance error.",
    )
    completion_sweep.add_argument(
        "--output",
        default="outputs/unresolved_completion/completion_rank_sweep.png",
    )
    completion_sweep.add_argument(
        "--csv-output",
        default="outputs/unresolved_completion/completion_rank_sweep.csv",
    )
    completion_sweep.set_defaults(handler=run_completion_rank_sweep)

    temporal = subparsers.add_parser(
        "compare-temporal-correlation",
        help=(
            "Compare exact and empirical cross-time covariance for random walk, "
            "full correlated, raw handoff, independent completion, and "
            "persistent completion models."
        ),
    )
    _add_shared_arguments(temporal)
    _add_handoff_arguments(temporal)
    _add_completion_arguments(temporal)
    temporal.add_argument("--modes", type=int, default=50)
    temporal.add_argument(
        "--lags",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 5.0, 10.0, 20.0],
        help="Time lags in microseconds.",
    )
    temporal.add_argument(
        "--temporal-start-time",
        type=float,
        default=10.0,
        help="Earliest time origin included in the temporal statistics.",
    )
    temporal.add_argument(
        "--temporal-end-time",
        type=float,
        default=None,
        help="Latest destination time; defaults to the final saved time.",
    )
    temporal.add_argument(
        "--node-window-radius",
        type=int,
        default=10,
        help="Radius around the impulse node used for the central-window correlation.",
    )
    temporal.add_argument(
        "--profile-lag",
        type=float,
        default=5.0,
        help="Lag used for the spatial persistence profile panel.",
    )
    temporal.add_argument(
        "--random-walk-method",
        choices=("naive", "multinomial"),
        default="multinomial",
    )
    temporal.add_argument(
        "--output",
        default="outputs/temporal_correlation/temporal_correlation_comparison.png",
    )
    temporal.add_argument(
        "--csv-output",
        default="outputs/temporal_correlation/temporal_correlation_comparison.csv",
    )
    temporal.add_argument("--data-output", default=None)
    temporal.set_defaults(handler=run_temporal_correlation_comparison)


    method_benchmark = subparsers.add_parser(
        "benchmark-models",
        help=(
            "Benchmark multinomial random walk, full correlated modal, raw handoff, "
            "and independent/persistent completion pipelines."
        ),
    )
    method_benchmark.add_argument(
        "--methods",
        nargs="+",
        choices=BENCHMARK_METHOD_NAMES,
        default=list(BENCHMARK_METHOD_NAMES),
    )
    method_benchmark.add_argument(
        "--node-counts",
        type=int,
        nargs="+",
        default=[31, 51, 101, 201],
        help=(
            "Spatial resolutions for the node-scaling sweep. The timestep is "
            "automatically reduced when needed to keep the same physical duration."
        ),
    )
    method_benchmark.add_argument(
        "--particle-counts",
        type=int,
        nargs="+",
        default=[1_000, 5_275, 10_000],
    )
    method_benchmark.add_argument("--base-nodes", type=int, default=101)
    method_benchmark.add_argument("--base-particles", type=int, default=5_275)
    method_benchmark.add_argument("--total-time", type=float, default=100.0)
    method_benchmark.add_argument("--dt", type=float, default=1.0)
    method_benchmark.add_argument("--length", type=float, default=4.0)
    method_benchmark.add_argument("--diffusion", type=float, default=2.20e-4)
    method_benchmark.add_argument(
        "--impulse-fraction",
        type=float,
        default=0.59,
        help="Impulse location as a fraction of the spatial domain.",
    )
    method_benchmark.add_argument(
        "--final-modes",
        type=int,
        default=None,
        help=(
            "Fixed final mode count. When omitted, each grid retains "
            "--retained-fraction of its modes."
        ),
    )
    method_benchmark.add_argument("--retained-fraction", type=float, default=0.5)
    method_benchmark.add_argument("--handoff-time", type=float, default=10.0)
    method_benchmark.add_argument("--completion-rank", type=int, default=10)
    method_benchmark.add_argument("--completion-ridge", type=float, default=0.01)
    method_benchmark.add_argument("--repeats", type=int, default=3)
    method_benchmark.add_argument(
        "--amortization-runs",
        type=int,
        default=100,
        help="Number of trajectories over which one-time setup is amortized.",
    )
    method_benchmark.add_argument("--stability-safety", type=float, default=0.95)
    method_benchmark.add_argument("--seed", type=int, default=0)
    method_benchmark.add_argument(
        "--output",
        default="outputs/performance/model_scaling.png",
    )
    method_benchmark.add_argument(
        "--csv-output",
        default="outputs/performance/model_scaling.csv",
    )
    method_benchmark.set_defaults(handler=run_diffusion_method_benchmark)

    component_benchmark = subparsers.add_parser(
        "benchmark-components",
        help=(
            "Measure setup, dynamics, completion, and physical-readout costs "
            "for one full-field completed modal pipeline."
        ),
    )
    component_benchmark.add_argument("--particles", type=int, default=5_275)
    component_benchmark.add_argument("--nodes", type=int, default=101)
    component_benchmark.add_argument("--steps", type=int, default=101)
    component_benchmark.add_argument("--dt", type=float, default=1.0)
    component_benchmark.add_argument("--length", type=float, default=4.0)
    component_benchmark.add_argument("--diffusion", type=float, default=2.20e-4)
    component_benchmark.add_argument(
        "--impulse-fraction",
        type=float,
        default=0.59,
        help="Impulse location as a fraction of the spatial domain.",
    )
    component_benchmark.add_argument("--modes", type=int, default=75)
    component_benchmark.add_argument("--handoff-time", type=float, default=30.0)
    component_benchmark.add_argument("--completion-rank", type=int, default=5)
    component_benchmark.add_argument("--completion-ridge", type=float, default=0.01)
    component_benchmark.add_argument(
        "--readouts",
        nargs="+",
        choices=COMPONENT_READOUT_NAMES,
        default=list(COMPONENT_READOUT_NAMES),
    )
    component_benchmark.add_argument("--setup-repeats", type=int, default=5)
    component_benchmark.add_argument("--repeats", type=int, default=10)
    component_benchmark.add_argument("--warmups", type=int, default=1)
    component_benchmark.add_argument("--seed", type=int, default=0)
    component_benchmark.add_argument(
        "--csv-output",
        default="outputs/component_benchmark/canonical_samples.csv",
    )
    component_benchmark.add_argument(
        "--summary-output",
        default="outputs/component_benchmark/canonical_summary.json",
    )
    component_benchmark.set_defaults(handler=run_component_benchmark)

    component_scaling = subparsers.add_parser(
        "benchmark-component-scaling",
        help=(
            "Scale full-field component timings under fixed-step and fixed-duration "
            "regimes, with optional fresh-process memory measurements."
        ),
    )
    component_scaling.add_argument(
        "--node-counts",
        type=int,
        nargs="+",
        default=[51, 101, 151, 201],
    )
    component_scaling.add_argument(
        "--regimes",
        nargs="+",
        choices=COMPONENT_BENCHMARK_REGIMES,
        default=list(COMPONENT_BENCHMARK_REGIMES),
    )
    component_scaling.add_argument(
        "--operating-point",
        choices=[point.name for point in SELECTED_OPERATING_POINTS],
        default="accuracy_first",
    )
    component_scaling.add_argument("--particles", type=int, default=5_275)
    component_scaling.add_argument("--steps", type=int, default=101)
    component_scaling.add_argument("--total-time", type=float, default=100.0)
    component_scaling.add_argument("--dt", type=float, default=1.0)
    component_scaling.add_argument("--length", type=float, default=4.0)
    component_scaling.add_argument("--diffusion", type=float, default=2.20e-4)
    component_scaling.add_argument("--impulse-fraction", type=float, default=0.59)
    component_scaling.add_argument("--stability-safety", type=float, default=0.95)
    component_scaling.add_argument("--completion-ridge", type=float, default=0.01)
    component_scaling.add_argument(
        "--readouts",
        nargs="+",
        choices=COMPONENT_READOUT_NAMES,
        default=list(COMPONENT_READOUT_NAMES),
    )
    component_scaling.add_argument("--setup-repeats", type=int, default=5)
    component_scaling.add_argument("--repeats", type=int, default=10)
    component_scaling.add_argument("--warmups", type=int, default=1)
    component_scaling.add_argument("--seed", type=int, default=0)
    component_scaling.add_argument(
        "--isolate-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    component_scaling.add_argument(
        "--include-standard-comparator",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include both the loop and vectorized exact aggregated multinomial "
            "random walks on the same time grids and full-field output "
            "contract; the vectorized implementation is primary."
        ),
    )
    component_scaling.add_argument(
        "--memory-sampler-interval-ms",
        type=float,
        default=1.0,
    )
    component_scaling.add_argument(
        "--csv-output",
        default="outputs/component_benchmark/scaling_samples.csv",
    )
    component_scaling.add_argument(
        "--memory-output",
        default="outputs/component_benchmark/scaling_memory.csv",
    )
    component_scaling.add_argument(
        "--summary-output",
        default="outputs/component_benchmark/scaling_summary.json",
    )
    component_scaling.set_defaults(handler=run_component_scaling_benchmark)

    operating_points = subparsers.add_parser(
        "study-operating-points",
        help=(
            "Benchmark the compute-first, balanced, and accuracy-first completed "
            "pipelines and measure distribution-level error."
        ),
    )
    operating_points.add_argument(
        "--node-counts",
        type=int,
        nargs="+",
        default=[51, 101, 151, 201],
    )
    operating_points.add_argument("--accuracy-nodes", type=int, default=101)
    operating_points.add_argument("--particles", type=int, default=5_275)
    operating_points.add_argument("--total-time", type=float, default=100.0)
    operating_points.add_argument("--dt", type=float, default=1.0)
    operating_points.add_argument("--length", type=float, default=4.0)
    operating_points.add_argument("--diffusion", type=float, default=2.20e-4)
    operating_points.add_argument("--impulse-fraction", type=float, default=0.59)
    operating_points.add_argument("--completion-ridge", type=float, default=0.01)
    operating_points.add_argument(
        "--closure",
        choices=("dense_persistent", "local_persistent"),
        default=DEFAULT_COMPLETION_CLOSURE,
        help="Analytic completion closure used consistently for accuracy and timing.",
    )
    operating_points.add_argument("--repeats", type=int, default=3)
    operating_points.add_argument("--amortization-runs", type=int, default=100)
    operating_points.add_argument("--stability-safety", type=float, default=0.95)
    operating_points.add_argument("--accuracy-runs", type=int, default=500)
    operating_points.add_argument(
        "--distribution-times",
        type=float,
        nargs="+",
        default=[20.0, 50.0, 100.0],
    )
    operating_points.add_argument(
        "--distribution-node-offsets",
        type=int,
        nargs="+",
        default=[-10, 0, 10],
        help="Node offsets from the impulse used for marginal distances.",
    )
    operating_points.add_argument("--projections", type=int, default=64)
    operating_points.add_argument("--histogram-bins", type=int, default=30)
    operating_points.add_argument("--seed", type=int, default=5)
    operating_points.add_argument(
        "--no-isolate-peak-rss",
        action="store_true",
        help="Run in-process; intended only for fast tests, not headline peak RSS.",
    )
    operating_points.add_argument(
        "--output",
        default="outputs/operating_points/benchmark_and_distribution.png",
    )
    operating_points.add_argument(
        "--csv-output",
        default="outputs/operating_points/scaling.csv",
    )
    operating_points.add_argument(
        "--distribution-csv-output",
        default="outputs/operating_points/distribution_accuracy.csv",
    )
    operating_points.add_argument(
        "--summary-output",
        default="outputs/operating_points/summary.json",
    )
    operating_points.set_defaults(handler=run_operating_point_study)

    structured = subparsers.add_parser(
        "study-structured-closure",
        help=(
            "Profile dense completion, select a local modal bandwidth, validate "
            "held-out physics, and benchmark scaling."
        ),
    )
    structured.add_argument("--node-counts", type=int, nargs="+", default=[51, 101, 151, 201])
    structured.add_argument("--accuracy-nodes", type=int, default=101)
    structured.add_argument("--particles", type=int, default=5_275)
    structured.add_argument("--total-time", type=float, default=100.0)
    structured.add_argument("--dt", type=float, default=1.0)
    structured.add_argument("--diffusion", type=float, default=2.20e-4)
    structured.add_argument("--impulse-fraction", type=float, default=0.59)
    structured.add_argument("--completion-ridge", type=float, default=0.01)
    structured.add_argument("--bandwidths", type=int, nargs="+", default=[4, 8, 12, 20])
    structured.add_argument("--accuracy-runs", type=int, default=500)
    structured.add_argument("--heldout-runs", type=int, default=250)
    structured.add_argument("--repeats", type=int, default=3)
    structured.add_argument("--amortization-runs", type=int, default=100)
    structured.add_argument("--stability-safety", type=float, default=0.95)
    structured.add_argument(
        "--distribution-times",
        type=float,
        nargs="+",
        default=[20.0, 50.0, 100.0],
    )
    structured.add_argument(
        "--distribution-node-offsets",
        type=int,
        nargs="+",
        default=[-10, 0, 10],
    )
    structured.add_argument("--projections", type=int, default=64)
    structured.add_argument("--histogram-bins", type=int, default=30)
    structured.add_argument(
        "--comparison-bootstrap-resamples",
        type=int,
        default=200,
        help="Paired block-bootstrap resamples for closure-minus-dense moment errors.",
    )
    structured.add_argument(
        "--comparison-bootstrap-blocks",
        type=int,
        default=20,
        help="Independent trajectory blocks used by paired comparison intervals.",
    )
    structured.add_argument(
        "--comparison-confidence-level",
        type=float,
        default=0.95,
        help="Confidence level for paired closure noninferiority intervals.",
    )
    structured.add_argument("--seed", type=int, default=6)
    structured.add_argument(
        "--output-directory",
        default="outputs/structured_closure",
    )
    structured.set_defaults(handler=run_structured_closure_study)

    fitted = subparsers.add_parser(
        "study-fitted-closure",
        help=(
            "Train a sparse closure from particle trajectories, validate frozen "
            "coefficients on unseen physics, and benchmark inference scaling."
        ),
    )
    fitted.add_argument("--node-counts", type=int, nargs="+", default=[51, 101, 151, 201])
    fitted.add_argument("--particles", type=int, default=5_275)
    fitted.add_argument("--total-time", type=float, default=100.0)
    fitted.add_argument("--dt", type=float, default=1.0)
    fitted.add_argument("--stability-safety", type=float, default=0.95)
    fitted.add_argument("--bandwidth", type=int, default=12)
    fitted.add_argument("--frequency-bins", type=int, default=8)
    fitted.add_argument("--time-bins", type=int, default=5)
    fitted.add_argument("--residual-grid-size", type=int, default=32)
    fitted.add_argument(
        "--anchor-conditioned",
        action="store_true",
        help=(
            "fit separate physical anchors, interpolate nearby models, and use "
            "low-rank Gaussian innovations"
        ),
    )
    fitted.add_argument(
        "--physics-conditioned",
        action="store_true",
        help="fit one shared regression conditioned on dimensionless physics",
    )
    fitted.add_argument("--conditioning-degree", type=int, choices=[1, 2], default=1)
    fitted.add_argument("--conditioning-ridge", type=float, default=1.0e-2)
    fitted.add_argument(
        "--run-conditioning-cv",
        action="store_true",
        help="select conditioning complexity with configuration-level folds",
    )
    fitted.add_argument(
        "--conditioning-degrees",
        type=int,
        nargs="+",
        choices=[1, 2],
        default=[1],
    )
    fitted.add_argument("--conditioning-folds", type=int, default=3)
    fitted.add_argument("--recommended-anchor-count", type=int, default=4)
    fitted.add_argument("--use-bridge-anchors", action="store_true")
    fitted.add_argument("--gate-before-lockbox", action="store_true")
    fitted.add_argument("--run-lockbox", action="store_true")
    fitted.add_argument("--cv-mean-threshold", type=float, default=0.87)
    fitted.add_argument("--cv-max-threshold", type=float, default=1.05)
    fitted.add_argument(
        "--lockbox-manifest",
        default="configs/fitted_closure_lockbox.csv",
    )
    fitted.add_argument("--innovation-rank", type=int, default=12)
    fitted.add_argument("--nearest-anchors", type=int, default=3)
    fitted.add_argument(
        "--persistent-innovations",
        action="store_true",
        help="fit a stable conditional Gaussian transition for ordered residuals",
    )
    fitted.add_argument(
        "--run-ablation",
        action="store_true",
        help="measure exact-anchor rank, persistence, and interpolation errors",
    )
    fitted.add_argument(
        "--ablation-ranks",
        type=int,
        nargs="+",
        default=[4, 8, 12, 20],
    )
    fitted.add_argument("--exact-anchor-runs", type=int, default=100)
    fitted.add_argument("--fit-ridge", type=float, default=1.0e-3)
    fitted.add_argument("--max-residual-samples", type=int, default=4_000)
    fitted.add_argument("--regression-samples-per-target", type=int, default=2_000)
    fitted.add_argument("--training-runs", type=int, default=100)
    fitted.add_argument("--validation-runs", type=int, default=500)
    fitted.add_argument("--heldout-runs", type=int, default=250)
    fitted.add_argument("--repeats", type=int, default=3)
    fitted.add_argument("--amortization-runs", type=int, default=100)
    fitted.add_argument(
        "--distribution-times",
        type=float,
        nargs="+",
        default=[20.0, 50.0, 100.0],
    )
    fitted.add_argument(
        "--distribution-node-offsets",
        type=int,
        nargs="+",
        default=[-10, 0, 10],
    )
    fitted.add_argument("--projections", type=int, default=64)
    fitted.add_argument("--histogram-bins", type=int, default=30)
    fitted.add_argument(
        "--random-walk-particle-counts",
        type=int,
        nargs="+",
        default=[1_000, 5_275],
    )
    fitted.add_argument("--seed", type=int, default=7)
    fitted.add_argument(
        "--output-directory",
        default="outputs/fitted_closure",
    )
    fitted.set_defaults(handler=run_fitted_closure_study)

    benchmark = subparsers.add_parser(
        "benchmark-random-walk",
        help="Benchmark naive particle trajectories against multinomial node transitions.",
    )
    benchmark.add_argument(
        "--particle-counts",
        type=int,
        nargs="+",
        default=[100, 500, 1_000, 5_275],
    )
    benchmark.add_argument("--nodes", type=int, default=101)
    benchmark.add_argument("--steps", type=int, default=101)
    benchmark.add_argument("--impulse-index", type=int, default=None)
    benchmark.add_argument("--dt", type=float, default=1.0)
    benchmark.add_argument("--length", type=float, default=4.0)
    benchmark.add_argument("--diffusion", type=float, default=2.20e-4)
    benchmark.add_argument("--repeats", type=int, default=3)
    benchmark.add_argument("--seed", type=int, default=0)
    benchmark.add_argument(
        "--output",
        default="outputs/random_walk_benchmark.png",
    )
    benchmark.add_argument(
        "--csv-output",
        default="outputs/random_walk_benchmark.csv",
    )
    benchmark.set_defaults(handler=run_random_walk_benchmark)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
