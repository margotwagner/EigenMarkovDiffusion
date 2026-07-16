"""Command-line interface for validation, sweeps, comparisons, and benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .benchmarking import benchmark_random_walks, write_benchmark_csv
from .config import DiffusionConfig
from .ensemble import (
    MODAL_MODEL_NAMES,
    EnsembleResult,
    ModalModelName,
    apply_readout_ensemble,
    run_modal_ensemble,
    run_random_walk_ensemble,
)
from .metrics import (
    relative_frobenius_error,
    relative_l2_error,
    relative_l2_error_from_time,
    summarize_ensemble,
)
from .plotting import (
    plot_handoff_sweep,
    plot_completion_rank_sweep,
    plot_mode_sweep,
    plot_model_comparison,
    plot_random_walk_benchmark,
    plot_temporal_correlation_comparison,
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


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--particles", type=int, default=5_275)
    parser.add_argument("--nodes", type=int, default=101)
    parser.add_argument("--steps", type=int, default=101)
    parser.add_argument("--impulse-index", type=int, default=None)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--length", type=float, default=4.0)
    parser.add_argument("--diffusion", type=float, default=2.20e-4)
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
    """Sweep unresolved Gaussian completion rank on one shared handoff ensemble."""

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

    rows: list[dict[str, float | int | str | None]] = []
    for rank in ranks:
        if rank == 0:
            ensemble = raw
            label = "raw"
        else:
            ensemble = apply_readout_ensemble(
                raw,
                config,
                readout="unresolved_gaussian_completion",
                retained_modes=n_modes,
                completion_start_time=(
                    args.handoff_time
                    if args.completion_start_time is None
                    else args.completion_start_time
                ),
                completion_rank=rank,
                completion_ridge=args.completion_ridge,
                seed=args.seed + 10_000 + rank,
            )
            label = "unresolved_gaussian_completion"

        diagnostics = summarize_ensemble(
            discrete_reference,
            analytic_variance,
            ensemble.runs,
            config.n_particles,
            reference_covariance=analytic_covariance,
            covariance_time_indices=covariance_indices,
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
                    config.n_nodes if args.initial_modes is None else args.initial_modes
                ),
                "final_n_modes": n_modes,
                "handoff_time": args.handoff_time,
                "completion_rank": rank,
                "completion_ridge": args.completion_ridge,
                "mean_completion_l1_fraction": completion_l1,
                **diagnostics.as_dict(),
            }
        )
        print(
            f"completion_rank={rank:3d}  "
            f"mean={diagnostics.mean_relative_l2_error:.6g}  "
            f"variance={diagnostics.variance_relative_l2_error:.6g}  "
            f"covariance={diagnostics.covariance_relative_frobenius_error:.6g}"
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
        ranks=[int(row["completion_rank"]) for row in rows],
        mean_errors=[float(row["mean_relative_l2_error"]) for row in rows],
        variance_errors=[float(row["variance_relative_l2_error"]) for row in rows],
        covariance_errors=[
            float(row["covariance_relative_frobenius_error"]) for row in rows
        ],
        negative_mass_fractions=[
            float(row["mean_negative_mass_fraction"]) for row in rows
        ],
        completion_l1_fractions=[
            float(row["mean_completion_l1_fraction"]) for row in rows
        ],
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
    completion_sweep.add_argument("--completion-ridge", type=float, default=1.0e-2)
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
