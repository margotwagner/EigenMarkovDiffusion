"""Command-line interface for validation, sweeps, and benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .benchmarking import benchmark_random_walks, write_benchmark_csv
from .config import DiffusionConfig
from .ensemble import run_eigenmarkov_ensemble, run_random_walk_ensemble
from .metrics import (
    relative_l2_error,
    relative_l2_error_from_time,
    summarize_ensemble,
)
from .plotting import (
    plot_mode_sweep,
    plot_random_walk_benchmark,
    plot_validation,
)
from .references import (
    continuous_expected_diffusion,
    discrete_expected_diffusion,
    multinomial_marginal_covariance,
    multinomial_marginal_variance,
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
    parser.add_argument("--modal-particle-weight", type=float, default=1.0)
    parser.add_argument(
        "--initialization",
        choices=("nearest", "stochastic"),
        default="nearest",
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
    eigenmarkov = run_eigenmarkov_ensemble(
        config,
        n_runs=args.runs,
        n_modes=n_modes,
        modal_particle_weight=args.modal_particle_weight,
        initialization=args.initialization,
        seed=args.seed + 1,
    )

    covariance_indices, covariance_times = _nearest_time_indices(
        config.times,
        args.covariance_times,
        exclude_zero=True,
    )
    em_metrics = summarize_ensemble(
        discrete_reference,
        analytic_variance,
        eigenmarkov.runs,
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
        eigenmarkov_runs=eigenmarkov.runs,
        random_walk_runs=random_walk.runs,
        output_path=args.output,
        profile_step=args.profile_step,
    )

    data_path = Path(args.data_output) if args.data_output else output.with_suffix(".npz")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        data_path,
        continuous_reference=continuous_reference,
        discrete_reference=discrete_reference,
        analytic_variance=analytic_variance,
        analytic_covariance_selected=analytic_covariance[covariance_indices],
        covariance_time_indices=np.asarray(covariance_indices, dtype=int),
        covariance_times=np.asarray(covariance_times, dtype=float),
        eigenmarkov_runs=eigenmarkov.runs,
        random_walk_runs=random_walk.runs,
        random_walk_method=args.random_walk_method,
        times=config.times,
        positions=config.positions,
    )

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
            "runs": args.runs,
            "modal_particle_weight": args.modal_particle_weight,
            "random_walk_method": args.random_walk_method,
            "covariance_times": covariance_times,
        },
        "reference_difference": {
            "continuous_vs_discrete_relative_l2": relative_l2_error(
                discrete_reference,
                continuous_reference,
            )
        },
        "eigenmarkov_vs_discrete": em_metrics.as_dict(),
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
    rows: list[dict[str, float | int]] = []

    for mode_count in requested_modes:
        ensemble = run_eigenmarkov_ensemble(
            config,
            n_runs=args.runs,
            n_modes=mode_count,
            modal_particle_weight=args.modal_particle_weight,
            initialization=args.initialization,
            seed=args.seed + mode_count,
        )
        diagnostics = summarize_ensemble(
            discrete_reference,
            analytic_variance,
            ensemble.runs,
            config.n_particles,
        )
        diagnostic_values = diagnostics.as_dict()
        diagnostic_values.pop("covariance_relative_frobenius_error", None)
        row: dict[str, float | int] = {
            "n_modes": mode_count,
            **diagnostic_values,
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
            f"modes={mode_count:4d}  "
            f"mean_l2={diagnostics.mean_relative_l2_error:.6g}  "
            f"variance_l2={diagnostics.variance_relative_l2_error:.6g}  "
            f"negative_mass={diagnostics.mean_negative_mass_fraction:.6g}"
        )

    csv_path = Path(args.csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
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
    )
    print(f"Saved {csv_path}")
    print(f"Saved {figure}")
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
        description="Stochastic diffusion using Markov eigenmodes.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate",
        help="Compare EigenMarkov with exact discrete, continuous, and random-walk references.",
    )
    _add_shared_arguments(validate)
    validate.add_argument("--modes", type=int, default=None)
    validate.add_argument(
        "--random-walk-method",
        choices=("naive", "multinomial"),
        default="multinomial",
        help="Random-walk implementation used for validation.",
    )
    validate.add_argument(
        "--covariance-times",
        type=float,
        nargs="+",
        default=[1.0, 5.0, 20.0, 100.0],
        help="Times used for same-time spatial covariance error.",
    )
    validate.add_argument("--profile-step", type=int, default=None)
    validate.add_argument("--output", default="outputs/validation.png")
    validate.add_argument("--data-output", default=None)
    validate.set_defaults(handler=run_validation)

    sweep = subparsers.add_parser(
        "sweep-modes",
        help="Measure time-resolved mean, variance, and nonnegativity versus retained modes.",
    )
    _add_shared_arguments(sweep)
    sweep.add_argument("--modes", type=int, nargs="+", default=None)
    sweep.add_argument(
        "--error-start-times",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 5.0, 10.0, 20.0],
        help="Report mean error using only times at or after each listed time.",
    )
    sweep.add_argument("--output", default="outputs/mode_sweep.png")
    sweep.add_argument("--csv-output", default="outputs/mode_sweep.csv")
    sweep.set_defaults(handler=run_sweep)

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
