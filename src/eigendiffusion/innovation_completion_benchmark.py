"""Focused accuracy/runtime screen for direct innovation completion.

The retained transition is screened separately by :mod:`direct_modal_benchmark`.
This module asks the next decision question: how much full-field covariance is
recovered by low-rank conditional innovations, and does the completed pipeline
remain cheaper than the exact vectorized multinomial walk?
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np

from .baselines import vectorized_multinomial_random_walk_diffusion
from .benchmarking import deep_numpy_nbytes, stable_diffusion_config
from .config import DiffusionConfig
from .innovation_completion import InnovationConditionedModalCompleter
from .readouts import simplex_bank_counts

COMPLETED_RAW_METHOD = "innovation_completed_raw"
COMPLETED_SIMPLEX_METHOD = "innovation_completed_simplex"
EXACT_METHOD = "exact_vectorized_multinomial"
TIMED_METHODS = (COMPLETED_RAW_METHOD, COMPLETED_SIMPLEX_METHOD, EXACT_METHOD)


@dataclass(frozen=True, slots=True)
class InnovationCompletionScreenSpec:
    """Configuration for the fixed-duration completion screen."""

    node_counts: tuple[int, ...] = (101, 201)
    mode_counts: tuple[int, ...] = (20, 35, 50, 75)
    conditional_ranks: tuple[int, ...] = (5, 10, 20)
    include_full_rank: bool = True
    n_particles: int = 5_275
    total_time: float = 100.0
    requested_dt: float = 1.0
    length: float = 4.0
    diffusion_coefficient: float = 2.20e-4
    impulse_fraction: float = 0.59
    stability_safety: float = 0.95
    setup_repeats: int = 1
    trajectory_repeats: int = 15
    warmup_runs: int = 1
    seed: int = 20_261_123

    def __post_init__(self) -> None:
        if not self.node_counts or any(value < 2 for value in self.node_counts):
            raise ValueError("node_counts must contain integers >= 2")
        if not self.mode_counts or any(value <= 0 for value in self.mode_counts):
            raise ValueError("mode_counts must contain positive integers")
        if any(value <= 0 for value in self.conditional_ranks):
            raise ValueError("conditional_ranks must contain positive integers")
        if self.setup_repeats <= 0 or self.trajectory_repeats <= 0:
            raise ValueError("setup_repeats and trajectory_repeats must be positive")
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs must be nonnegative")


def _configuration(
    spec: InnovationCompletionScreenSpec,
    n_nodes: int,
) -> DiffusionConfig:
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


def _relative_from_squares(error_squared: float, reference_squared: float) -> float:
    if reference_squared == 0.0:
        return 0.0 if error_squared == 0.0 else float("inf")
    return float(np.sqrt(max(error_squared, 0.0) / reference_squared))


def analytic_completion_diagnostics(
    completer: InnovationConditionedModalCompleter,
) -> dict[str, int | float | str | bool]:
    """Propagate the effective low-rank covariance and compare exact moments."""

    config = completer.config
    reference = completer.full_reference
    exact_covariances = reference._modal_covariances
    target_innovations = completer._target_innovation_covariances
    effective_innovations = completer.effective_innovation_covariances()
    diagonal = completer._one_step_factors
    vectors = completer.basis.eigenvectors

    covariance = np.zeros((config.n_nodes, config.n_nodes), dtype=float)
    modal_mean = reference._modal_means[0].copy()
    covariance_error_squared = 0.0
    covariance_reference_squared = 0.0
    variance_error_squared = 0.0
    variance_reference_squared = 0.0
    lag_durations = (1.0, 5.0, 20.0)
    lag_steps = {
        duration: max(1, int(round(duration / config.dt)))
        for duration in lag_durations
    }
    lag_error_squared = {duration: 0.0 for duration in lag_durations}
    lag_reference_squared = {duration: 0.0 for duration in lag_durations}
    innovation_error_squared = 0.0
    innovation_reference_squared = 0.0
    mean_error_squared = 0.0
    mean_reference_squared = 0.0
    maximum_covariance_error = 0.0
    final_covariance_error = 0.0

    for time_index, effective in enumerate(effective_innovations):
        innovation_difference = effective - target_innovations[time_index]
        innovation_error_squared += float(np.sum(innovation_difference**2))
        innovation_reference_squared += float(
            np.sum(target_innovations[time_index] ** 2)
        )

        covariance = (
            diagonal[:, None] * covariance * diagonal[None, :] + effective
        )
        exact = exact_covariances[time_index + 1]
        difference = covariance - exact
        difference_squared = float(np.sum(difference**2))
        exact_squared = float(np.sum(exact**2))
        covariance_error_squared += difference_squared
        covariance_reference_squared += exact_squared
        if exact_squared > 0.0:
            current_covariance_error = float(
                np.sqrt(difference_squared / exact_squared)
            )
            maximum_covariance_error = max(
                maximum_covariance_error, current_covariance_error
            )
            final_covariance_error = current_covariance_error

        variance = np.einsum(
            "ni,ij,nj->n", vectors, covariance, vectors, optimize=True
        )
        exact_variance = np.einsum(
            "ni,ij,nj->n", vectors, exact, vectors, optimize=True
        )
        variance_error_squared += float(np.sum((variance - exact_variance) ** 2))
        variance_reference_squared += float(np.sum(exact_variance**2))

        saved_step = time_index + 1
        for duration, step_count in lag_steps.items():
            if saved_step + step_count >= config.n_steps:
                continue
            propagation = diagonal**step_count
            candidate_cross = covariance * propagation[None, :]
            exact_cross = exact * propagation[None, :]
            lag_error_squared[duration] += float(
                np.sum((candidate_cross - exact_cross) ** 2)
            )
            lag_reference_squared[duration] += float(np.sum(exact_cross**2))

        modal_mean *= diagonal
        exact_mean = reference._modal_means[time_index + 1]
        mean_error_squared += float(np.sum((modal_mean - exact_mean) ** 2))
        mean_reference_squared += float(np.sum(exact_mean**2))

    captured = completer.captured_variance_fractions
    full_rank = completer.conditional_rank == completer.unresolved_modes
    result: dict[str, int | float | str | bool] = {
        "n_particles": config.n_particles,
        "n_nodes": config.n_nodes,
        "n_steps": config.n_steps,
        "dt": config.dt,
        "total_time": float(config.times[-1]),
        "retained_modes": completer.retained_modes,
        "conditional_rank": completer.conditional_rank,
        "rank_label": "full" if full_rank else str(completer.conditional_rank),
        "unresolved_modes": completer.unresolved_modes,
        "captured_conditional_variance_minimum": float(np.min(captured)),
        "captured_conditional_variance_median": float(np.median(captured)),
        "captured_conditional_variance_mean": float(np.mean(captured)),
        "innovation_covariance_relative_frobenius_error": (
            _relative_from_squares(
                innovation_error_squared, innovation_reference_squared
            )
        ),
        "full_covariance_relative_frobenius_error": _relative_from_squares(
            covariance_error_squared, covariance_reference_squared
        ),
        "maximum_time_covariance_relative_frobenius_error": (
            maximum_covariance_error
        ),
        "final_time_covariance_relative_frobenius_error": final_covariance_error,
        "marginal_variance_relative_l2_error": _relative_from_squares(
            variance_error_squared, variance_reference_squared
        ),
        "mean_relative_l2_error": _relative_from_squares(
            mean_error_squared, mean_reference_squared
        ),
        "full_rank_oracle": full_rank,
    }
    for duration in lag_durations:
        label = str(int(duration))
        result[f"lag_{label}_time_covariance_relative_frobenius_error"] = (
            _relative_from_squares(
                lag_error_squared[duration],
                lag_reference_squared[duration],
            )
        )
    return result


def _validate_spatial(
    values: np.ndarray,
    config: DiffusionConfig,
    *,
    physical: bool,
) -> None:
    if values.shape != (config.n_steps, config.n_nodes):
        raise RuntimeError("completion screen returned the wrong field shape")
    if not np.all(np.isfinite(values)):
        raise RuntimeError("completion screen returned nonfinite values")
    if not np.allclose(
        values.sum(axis=1), config.n_particles, atol=1.0e-7, rtol=0.0
    ):
        raise RuntimeError("completion screen failed mass conservation")
    if physical and float(np.min(values)) < -1.0e-10:
        raise RuntimeError("physical completion output contains negative counts")


def _timed_action(
    method: str,
    completer: InnovationConditionedModalCompleter,
    config: DiffusionConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if method == EXACT_METHOD:
        return vectorized_multinomial_random_walk_diffusion(config, rng=rng)
    result = completer.run_stepwise(rng=rng)
    if method == COMPLETED_RAW_METHOD:
        return result.spatial_counts
    if method == COMPLETED_SIMPLEX_METHOD:
        return simplex_bank_counts(result.spatial_counts, config.n_particles)
    raise ValueError(f"unknown timed method: {method}")


def _benchmark_case(
    spec: InnovationCompletionScreenSpec,
    config: DiffusionConfig,
    *,
    retained_modes: int,
    conditional_rank: int | None,
    case_index: int,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    setup_samples: list[float] = []
    completer: InnovationConditionedModalCompleter | None = None
    for _ in range(spec.setup_repeats):
        completer = None
        gc.collect()
        start = perf_counter()
        completer = InnovationConditionedModalCompleter(
            config,
            retained_modes=retained_modes,
            conditional_rank=conditional_rank,
        )
        setup_samples.append(perf_counter() - start)
    if completer is None:  # pragma: no cover - positive repeats are validated
        raise RuntimeError("completion setup did not produce a completer")

    rank_label = (
        "full"
        if completer.conditional_rank == completer.unresolved_modes
        else str(completer.conditional_rank)
    )
    columns: dict[str, object] = {
        "n_particles": config.n_particles,
        "n_nodes": config.n_nodes,
        "n_steps": config.n_steps,
        "dt": config.dt,
        "total_time": float(config.times[-1]),
        "retained_modes": retained_modes,
        "conditional_rank": completer.conditional_rank,
        "rank_label": rank_label,
    }
    rows = [
        {
            **columns,
            "method": COMPLETED_RAW_METHOD,
            "component": "setup",
            "repeat": repeat,
            "execution_position": -1,
            "seconds": seconds,
        }
        for repeat, seconds in enumerate(setup_samples)
    ]

    for warmup in range(spec.warmup_runs):
        for channel, method in enumerate(TIMED_METHODS):
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [spec.seed, case_index, 0, warmup, channel]
                )
            )
            output = _timed_action(method, completer, config, rng)
            _validate_spatial(output, config, physical=method != COMPLETED_RAW_METHOD)

    # Rotate and reverse method order so no method owns one timing position.
    orders = (
        TIMED_METHODS,
        TIMED_METHODS[::-1],
        TIMED_METHODS[1:] + TIMED_METHODS[:1],
        (TIMED_METHODS[1:] + TIMED_METHODS[:1])[::-1],
        TIMED_METHODS[2:] + TIMED_METHODS[:2],
        (TIMED_METHODS[2:] + TIMED_METHODS[:2])[::-1],
    )
    was_enabled = gc.isenabled()
    if was_enabled:
        gc.disable()
    try:
        for repeat in range(spec.trajectory_repeats):
            for position, method in enumerate(orders[repeat % len(orders)]):
                channel = TIMED_METHODS.index(method)
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [spec.seed, case_index, 1, repeat, channel]
                    )
                )
                start = perf_counter()
                output = _timed_action(method, completer, config, rng)
                seconds = perf_counter() - start
                _validate_spatial(
                    output,
                    config,
                    physical=method != COMPLETED_RAW_METHOD,
                )
                rows.append(
                    {
                        **columns,
                        "method": method,
                        "component": "trajectory_total",
                        "repeat": repeat,
                        "execution_position": position,
                        "seconds": seconds,
                    }
                )
    finally:
        if was_enabled:
            gc.enable()

    memory = {
        **columns,
        "setup_array_bytes": deep_numpy_nbytes(completer),
        "raw_returned_array_bytes": (
            config.n_steps
            * (config.n_nodes + config.n_nodes)
            * np.dtype(float).itemsize
        ),
        "counts_only_output_bytes": (
            config.n_steps * config.n_nodes * np.dtype(float).itemsize
        ),
        "logical_only": True,
    }
    accuracy = analytic_completion_diagnostics(completer)
    return rows, memory, accuracy


def _write_csv(rows: Sequence[dict[str, object]], path: Path) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _timing_summary(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[float]] = {}
    for row in rows:
        key = (
            row["n_nodes"],
            row["n_steps"],
            row["retained_modes"],
            row["rank_label"],
            row["method"],
            row["component"],
        )
        groups.setdefault(key, []).append(float(row["seconds"]))
    summary: list[dict[str, object]] = []
    for key, samples in groups.items():
        n_nodes, n_steps, modes, rank, method, component = key
        values = np.asarray(samples, dtype=float)
        summary.append(
            {
                "n_nodes": n_nodes,
                "n_steps": n_steps,
                "retained_modes": modes,
                "rank_label": rank,
                "method": method,
                "component": component,
                "samples": values.size,
                "median_seconds": float(np.median(values)),
                "mean_seconds": float(np.mean(values)),
                "std_seconds": float(np.std(values, ddof=0)),
                "min_seconds": float(np.min(values)),
                "max_seconds": float(np.max(values)),
            }
        )
    exact = {
        (row["n_nodes"], row["n_steps"], row["retained_modes"], row["rank_label"]): float(
            row["median_seconds"]
        )
        for row in summary
        if row["method"] == EXACT_METHOD and row["component"] == "trajectory_total"
    }
    setup = {
        (row["n_nodes"], row["n_steps"], row["retained_modes"], row["rank_label"]): float(
            row["median_seconds"]
        )
        for row in summary
        if row["component"] == "setup"
    }
    for row in summary:
        if row["component"] != "trajectory_total":
            row["median_ratio_over_exact"] = None
            row["setup_amortization_break_even_runs"] = None
            continue
        key = (
            row["n_nodes"],
            row["n_steps"],
            row["retained_modes"],
            row["rank_label"],
        )
        exact_seconds = exact[key]
        run_seconds = float(row["median_seconds"])
        row["median_ratio_over_exact"] = run_seconds / exact_seconds
        savings = exact_seconds - run_seconds
        row["setup_amortization_break_even_runs"] = (
            setup[key] / savings
            if row["method"] == COMPLETED_RAW_METHOD and savings > 0.0
            else None
        )
    return sorted(
        summary,
        key=lambda row: tuple(
            str(row[key])
            for key in (
                "n_nodes",
                "retained_modes",
                "rank_label",
                "component",
                "method",
            )
        ),
    )


def run_innovation_completion_screen(
    spec: InnovationCompletionScreenSpec,
    output_dir: str | Path,
) -> dict[str, object]:
    """Run the completion screen and write long-form CSV plus JSON output."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    timing_rows: list[dict[str, object]] = []
    memory_rows: list[dict[str, object]] = []
    accuracy_rows: list[dict[str, object]] = []
    case_index = 0
    for n_nodes in sorted(set(spec.node_counts)):
        config = _configuration(spec, n_nodes)
        for retained_modes in sorted(set(spec.mode_counts)):
            if retained_modes > n_nodes:
                continue
            ranks: list[int | None] = list(dict.fromkeys(spec.conditional_ranks))
            if spec.include_full_rank:
                ranks.append(None)
            for rank in ranks:
                if rank is not None and rank >= config.n_nodes - retained_modes:
                    # The full oracle already represents this case once.
                    if not spec.include_full_rank:
                        rank = None
                    else:
                        continue
                rows, memory, accuracy = _benchmark_case(
                    spec,
                    config,
                    retained_modes=retained_modes,
                    conditional_rank=rank,
                    case_index=case_index,
                )
                timing_rows.extend(rows)
                memory_rows.append(memory)
                accuracy_rows.append(accuracy)
                case_index += 1

    timing_summary = _timing_summary(timing_rows)
    timing_path = destination / "timing_samples.csv"
    memory_path = destination / "logical_memory.csv"
    accuracy_path = destination / "analytic_accuracy.csv"
    summary_path = destination / "summary.json"
    _write_csv(timing_rows, timing_path)
    _write_csv(memory_rows, memory_path)
    _write_csv(accuracy_rows, accuracy_path)
    report: dict[str, object] = {
        "schema_version": 1,
        "study_contract": {
            "regime": "fixed_duration",
            "output_workload": "full_field_every_timestep",
            "accuracy": "analytic Gaussian first/second temporal moments",
            "physical_output": "simplex_bank",
            "exact_comparator": EXACT_METHOD,
            "memory": "logical unique NumPy array bytes, not process RSS",
        },
        "specification": asdict(spec),
        "runtime_environment": {
            name: os.environ.get(name)
            for name in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "timing_summaries": timing_summary,
        "analytic_accuracy": accuracy_rows,
        "logical_memory": memory_rows,
        "artifacts": {
            "timing_samples_csv": str(timing_path),
            "analytic_accuracy_csv": str(accuracy_path),
            "logical_memory_csv": str(memory_path),
            "summary_json": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Screen low-rank innovation completion after direct-modal gating."
    )
    parser.add_argument("--node-counts", nargs="+", type=int, default=[101, 201])
    parser.add_argument("--modes", nargs="+", type=int, default=[20, 35, 50, 75])
    parser.add_argument("--ranks", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument(
        "--no-full-rank", action="store_false", dest="include_full_rank"
    )
    parser.add_argument("--particles", type=int, default=5_275)
    parser.add_argument("--total-time", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--length", type=float, default=4.0)
    parser.add_argument("--diffusion", type=float, default=2.20e-4)
    parser.add_argument("--impulse-fraction", type=float, default=0.59)
    parser.add_argument("--stability-safety", type=float, default=0.95)
    parser.add_argument("--setup-repeats", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20_261_123)
    parser.add_argument(
        "--output-dir", default="outputs/innovation_completion_screen"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = InnovationCompletionScreenSpec(
        node_counts=tuple(args.node_counts),
        mode_counts=tuple(args.modes),
        conditional_ranks=tuple(args.ranks),
        include_full_rank=args.include_full_rank,
        n_particles=args.particles,
        total_time=args.total_time,
        requested_dt=args.dt,
        length=args.length,
        diffusion_coefficient=args.diffusion,
        impulse_fraction=args.impulse_fraction,
        stability_safety=args.stability_safety,
        setup_repeats=args.setup_repeats,
        trajectory_repeats=args.repeats,
        warmup_runs=args.warmups,
        seed=args.seed,
    )
    report = run_innovation_completion_screen(spec, args.output_dir)
    print(
        json.dumps(
            {
                "cases": len(report["analytic_accuracy"]),
                "artifacts": report["artifacts"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
