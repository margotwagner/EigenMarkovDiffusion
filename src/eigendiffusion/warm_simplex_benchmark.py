"""Paired benchmark for warm-started exact simplex-bank projection."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np

from .baselines import vectorized_multinomial_random_walk_diffusion
from .benchmarking import stable_diffusion_config
from .config import DiffusionConfig
from .innovation_completion import InnovationConditionedModalCompleter
from .readouts import (
    simplex_bank_counts,
    simplex_bank_readout,
    warm_simplex_bank_counts,
    warm_simplex_bank_counts_with_diagnostics,
    warm_simplex_bank_readout,
)

SORTING_METHOD = "sorting_simplex_bank"
WARM_METHOD = "warm_simplex_bank"
EXACT_METHOD = "exact_vectorized_multinomial"


@dataclass(frozen=True, slots=True)
class WarmSimplexBenchmarkCase:
    """One completed-Gaussian workload for readout comparison."""

    n_nodes: int
    retained_modes: int = 75
    completion_rank: int = 25


@dataclass(frozen=True, slots=True)
class WarmSimplexBenchmarkSpec:
    """Protocol for paired readout-only and end-to-end timings."""

    cases: tuple[WarmSimplexBenchmarkCase, ...] = (
        WarmSimplexBenchmarkCase(101),
        WarmSimplexBenchmarkCase(201),
    )
    n_particles: int = 5_275
    total_time: float = 100.0
    requested_dt: float = 1.0
    length: float = 4.0
    diffusion_coefficient: float = 2.20e-4
    impulse_fraction: float = 0.59
    stability_safety: float = 0.95
    cached_trajectories: int = 32
    readout_warmups: int = 2
    readout_repeats: int = 30
    end_to_end_warmups: int = 1
    end_to_end_repeats: int = 24
    bootstrap_resamples: int = 10_000
    seed: int = 20_261_123

    def __post_init__(self) -> None:
        if not self.cases:
            raise ValueError("cases must not be empty")
        for case in self.cases:
            if case.n_nodes < 2:
                raise ValueError("case node counts must be at least two")
            if not 1 <= case.retained_modes <= case.n_nodes:
                raise ValueError("retained_modes must lie in [1, n_nodes]")
            if case.completion_rank <= 0:
                raise ValueError("completion_rank must be positive")
        node_counts = [case.n_nodes for case in self.cases]
        if len(set(node_counts)) != len(node_counts):
            raise ValueError("cases must have unique node counts")
        if self.readout_warmups < 0 or self.end_to_end_warmups < 0:
            raise ValueError("warmup counts must be nonnegative")
        if self.readout_repeats <= 0 or self.end_to_end_repeats <= 0:
            raise ValueError("repeat counts must be positive")
        required_cache = self.readout_warmups + self.readout_repeats
        if self.cached_trajectories < required_cache:
            raise ValueError(
                "cached_trajectories must cover readout warmups and repeats"
            )
        if self.bootstrap_resamples <= 0:
            raise ValueError("bootstrap_resamples must be positive")


def _configuration(
    spec: WarmSimplexBenchmarkSpec,
    case: WarmSimplexBenchmarkCase,
) -> DiffusionConfig:
    return stable_diffusion_config(
        n_particles=spec.n_particles,
        n_nodes=case.n_nodes,
        total_time=spec.total_time,
        requested_dt=spec.requested_dt,
        length=spec.length,
        diffusion_coefficient=spec.diffusion_coefficient,
        impulse_fraction=spec.impulse_fraction,
        stability_safety=spec.stability_safety,
    )


def _columns(
    config: DiffusionConfig,
    case: WarmSimplexBenchmarkCase,
) -> dict[str, int | float]:
    return {
        "n_particles": config.n_particles,
        "n_nodes": config.n_nodes,
        "n_steps": config.n_steps,
        "dt": config.dt,
        "total_time": float(config.times[-1]),
        "retained_modes": case.retained_modes,
        "completion_rank": min(
            case.completion_rank,
            config.n_nodes - case.retained_modes,
        ),
    }


def _validate_output(
    values: np.ndarray,
    config: DiffusionConfig,
) -> None:
    if values.shape != (config.n_steps, config.n_nodes):
        raise RuntimeError("readout benchmark returned the wrong output shape")
    if not np.all(np.isfinite(values)):
        raise RuntimeError("readout benchmark returned nonfinite output")
    if float(np.min(values)) < -1.0e-12:
        raise RuntimeError("readout benchmark returned negative output")
    if not np.allclose(
        values.sum(axis=1),
        config.n_particles,
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise RuntimeError("readout benchmark failed mass conservation")


def _paired_ratio_interval(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, float]:
    if baseline.shape != candidate.shape or baseline.ndim != 1:
        raise ValueError("paired timing samples must be equal-length vectors")
    observed = float(np.median(candidate) / np.median(baseline))
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        baseline.size,
        size=(n_resamples, baseline.size),
    )
    ratios = np.median(candidate[indices], axis=1) / np.median(
        baseline[indices], axis=1
    )
    lower, upper = np.quantile(ratios, [0.025, 0.975])
    return {
        "median_ratio": observed,
        "ratio_ci_lower": float(lower),
        "ratio_ci_upper": float(upper),
    }


def _diagnose_cached_fields(
    raw_fields: np.ndarray,
    config: DiffusionConfig,
) -> dict[str, int | float | bool | dict[str, int]]:
    maximum_output_difference = 0.0
    maximum_residual_difference = 0.0
    maximum_adjusted_difference = 0.0
    maximum_mass_error = 0.0
    minimum_output = float("inf")
    bitwise_entries = 0
    bitwise_rows = 0
    support_equal_rows = 0
    total_entries = 0
    total_rows = 0
    zero_update_rows = 0
    corrected_rows = 0
    fallback_rows = 0
    support_updates: list[int] = []
    active_sizes: list[int] = []
    support_changes: list[int] = []

    for raw in raw_fields:
        sorting = simplex_bank_readout(raw, config.n_particles)
        warm = warm_simplex_bank_readout(raw, config.n_particles)
        diagnostic_counts, diagnostics = warm_simplex_bank_counts_with_diagnostics(
            raw,
            config.n_particles,
        )
        if not np.array_equal(diagnostic_counts, warm.counts):
            raise RuntimeError("instrumented and uninstrumented warm paths differ")

        output_difference = np.abs(sorting.counts - warm.counts)
        residual_difference = np.abs(sorting.residuals - warm.residuals)
        adjusted_difference = np.abs(
            sorting.adjusted_inputs - warm.adjusted_inputs
        )
        maximum_output_difference = max(
            maximum_output_difference,
            float(np.max(output_difference)),
        )
        maximum_residual_difference = max(
            maximum_residual_difference,
            float(np.max(residual_difference)),
        )
        maximum_adjusted_difference = max(
            maximum_adjusted_difference,
            float(np.max(adjusted_difference)),
        )
        maximum_mass_error = max(
            maximum_mass_error,
            float(
                np.max(
                    np.abs(warm.counts.sum(axis=1) - config.n_particles)
                )
            ),
        )
        minimum_output = min(minimum_output, float(np.min(warm.counts)))
        equal_entries = sorting.counts == warm.counts
        bitwise_entries += int(np.count_nonzero(equal_entries))
        bitwise_rows += int(np.count_nonzero(np.all(equal_entries, axis=1)))
        support_equal_rows += int(
            np.count_nonzero(
                np.all(
                    (sorting.counts > 0.0) == (warm.counts > 0.0),
                    axis=1,
                )
            )
        )
        total_entries += sorting.counts.size
        total_rows += sorting.counts.shape[0]
        zero_update_rows += diagnostics.zero_update_rows
        corrected_rows += diagnostics.corrected_rows
        fallback_rows += diagnostics.fallback_rows
        successful_updates = diagnostics.support_updates[
            diagnostics.support_updates >= 0
        ]
        support_updates.extend(int(value) for value in successful_updates)
        active_sizes.extend(int(value) for value in diagnostics.active_sizes)
        support_changes.extend(
            int(value) for value in diagnostics.support_changes[1:]
        )

    warm_rows = zero_update_rows + corrected_rows + fallback_rows
    update_values = np.asarray(support_updates, dtype=float)
    changes = np.asarray(support_changes, dtype=float)
    sizes = np.asarray(active_sizes, dtype=float)
    update_histogram = {
        str(value): support_updates.count(value)
        for value in sorted(set(support_updates))
    }
    correctness_pass = bool(
        maximum_output_difference <= 1.0e-10
        and maximum_residual_difference <= 1.0e-10
        and maximum_adjusted_difference <= 1.0e-10
        and maximum_mass_error <= 1.0e-10
        and minimum_output >= -1.0e-12
        and support_equal_rows == total_rows
    )
    return {
        "cached_trajectories": raw_fields.shape[0],
        "total_rows": total_rows,
        "warm_rows": warm_rows,
        "zero_update_rows": zero_update_rows,
        "corrected_rows": corrected_rows,
        "fallback_rows": fallback_rows,
        "zero_update_rate": zero_update_rows / max(warm_rows, 1),
        "corrected_rate": corrected_rows / max(warm_rows, 1),
        "fallback_rate": fallback_rows / max(warm_rows, 1),
        "support_update_mean": float(np.mean(update_values)),
        "support_update_p95": float(np.percentile(update_values, 95.0)),
        "support_update_maximum": int(np.max(update_values)),
        "support_update_histogram": update_histogram,
        "active_size_mean": float(np.mean(sizes)),
        "support_change_mean": float(np.mean(changes)),
        "support_change_p95": float(np.percentile(changes, 95.0)),
        "bitwise_equal_entry_fraction": bitwise_entries / total_entries,
        "bitwise_equal_row_fraction": bitwise_rows / total_rows,
        "support_equal_row_fraction": support_equal_rows / total_rows,
        "maximum_output_absolute_difference": maximum_output_difference,
        "maximum_residual_absolute_difference": maximum_residual_difference,
        "maximum_adjusted_absolute_difference": maximum_adjusted_difference,
        "maximum_mass_error": maximum_mass_error,
        "minimum_output": minimum_output,
        "correctness_pass": correctness_pass,
    }


def _benchmark_case(
    spec: WarmSimplexBenchmarkSpec,
    case: WarmSimplexBenchmarkCase,
    case_index: int,
) -> tuple[
    list[dict[str, object]],
    dict[str, object],
    dict[str, object],
]:
    config = _configuration(spec, case)
    columns = _columns(config, case)
    gc.collect()
    start = perf_counter()
    completer = InnovationConditionedModalCompleter(
        config,
        retained_modes=case.retained_modes,
        conditional_rank=case.completion_rank,
    )
    setup_seconds = perf_counter() - start

    sequences = np.random.SeedSequence(
        [spec.seed, case_index, 0]
    ).spawn(spec.cached_trajectories)
    start = perf_counter()
    raw_fields = np.stack(
        [
            completer.run(np.random.default_rng(sequence)).spatial_counts
            for sequence in sequences
        ]
    )
    cache_generation_seconds = perf_counter() - start
    timing_rows: list[dict[str, object]] = []

    for index in range(spec.readout_warmups):
        raw = raw_fields[index]
        simplex_bank_counts(raw, config.n_particles)
        warm_simplex_bank_counts(raw, config.n_particles)

    was_enabled = gc.isenabled()
    if was_enabled:
        gc.disable()
    try:
        for repeat in range(spec.readout_repeats):
            raw = raw_fields[spec.readout_warmups + repeat]
            order = (
                (SORTING_METHOD, WARM_METHOD)
                if repeat % 2 == 0
                else (WARM_METHOD, SORTING_METHOD)
            )
            for position, method in enumerate(order):
                start = perf_counter()
                output = (
                    simplex_bank_counts(raw, config.n_particles)
                    if method == SORTING_METHOD
                    else warm_simplex_bank_counts(raw, config.n_particles)
                )
                seconds = perf_counter() - start
                _validate_output(output, config)
                timing_rows.append(
                    {
                        **columns,
                        "component": "readout_only",
                        "method": method,
                        "repeat": repeat,
                        "execution_position": position,
                        "seconds": seconds,
                    }
                )
    finally:
        if was_enabled:
            gc.enable()

    for warmup in range(spec.end_to_end_warmups):
        for method in (SORTING_METHOD, WARM_METHOD, EXACT_METHOD):
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [spec.seed, case_index, 1, warmup, int(method == EXACT_METHOD)]
                )
            )
            if method == EXACT_METHOD:
                output = vectorized_multinomial_random_walk_diffusion(config, rng)
            else:
                raw = completer.run(rng).spatial_counts
                output = (
                    simplex_bank_counts(raw, config.n_particles)
                    if method == SORTING_METHOD
                    else warm_simplex_bank_counts(raw, config.n_particles)
                )
            _validate_output(output, config)

    orders = (
        (SORTING_METHOD, WARM_METHOD, EXACT_METHOD),
        (EXACT_METHOD, WARM_METHOD, SORTING_METHOD),
        (WARM_METHOD, EXACT_METHOD, SORTING_METHOD),
        (SORTING_METHOD, EXACT_METHOD, WARM_METHOD),
        (EXACT_METHOD, SORTING_METHOD, WARM_METHOD),
        (WARM_METHOD, SORTING_METHOD, EXACT_METHOD),
    )
    was_enabled = gc.isenabled()
    if was_enabled:
        gc.disable()
    try:
        for repeat in range(spec.end_to_end_repeats):
            for position, method in enumerate(orders[repeat % len(orders)]):
                channel = 1 if method == EXACT_METHOD else 0
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [spec.seed, case_index, 2, repeat, channel]
                    )
                )
                start = perf_counter()
                if method == EXACT_METHOD:
                    output = vectorized_multinomial_random_walk_diffusion(
                        config, rng
                    )
                else:
                    raw = completer.run(rng).spatial_counts
                    output = (
                        simplex_bank_counts(raw, config.n_particles)
                        if method == SORTING_METHOD
                        else warm_simplex_bank_counts(raw, config.n_particles)
                    )
                seconds = perf_counter() - start
                _validate_output(output, config)
                timing_rows.append(
                    {
                        **columns,
                        "component": "end_to_end",
                        "method": method,
                        "repeat": repeat,
                        "execution_position": position,
                        "seconds": seconds,
                    }
                )
    finally:
        if was_enabled:
            gc.enable()

    diagnostics = {**columns, **_diagnose_cached_fields(raw_fields, config)}
    resources = {
        **columns,
        "setup_seconds": setup_seconds,
        "cache_generation_seconds": cache_generation_seconds,
        "cached_trajectory_bytes": int(raw_fields.nbytes),
        "cached_trajectory_megabytes": raw_fields.nbytes / 1024**2,
    }
    return timing_rows, diagnostics, resources


def _timing_report(
    rows: Sequence[dict[str, object]],
    *,
    bootstrap_resamples: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    groups: dict[tuple[int, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (int(row["n_nodes"]), str(row["component"]), str(row["method"]))
        groups.setdefault(key, []).append(row)
    summaries: list[dict[str, object]] = []
    samples_by_key: dict[tuple[int, str, str], np.ndarray] = {}
    for key, selected in groups.items():
        selected = sorted(selected, key=lambda row: int(row["repeat"]))
        values = np.asarray([float(row["seconds"]) for row in selected])
        samples_by_key[key] = values
        n_nodes, component, method = key
        summaries.append(
            {
                "n_nodes": n_nodes,
                "component": component,
                "method": method,
                "samples": values.size,
                "median_seconds": float(np.median(values)),
                "mean_seconds": float(np.mean(values)),
                "std_seconds": float(np.std(values, ddof=0)),
                "min_seconds": float(np.min(values)),
                "max_seconds": float(np.max(values)),
            }
        )

    comparisons: list[dict[str, object]] = []
    for n_nodes in sorted({int(row["n_nodes"]) for row in rows}):
        comparison_specs = (
            ("readout_only", SORTING_METHOD, WARM_METHOD, "warm_over_sorting"),
            ("end_to_end", SORTING_METHOD, WARM_METHOD, "warm_over_sorting"),
            ("end_to_end", EXACT_METHOD, WARM_METHOD, "warm_over_exact"),
            ("end_to_end", EXACT_METHOD, SORTING_METHOD, "sorting_over_exact"),
        )
        for offset, (component, baseline, candidate, label) in enumerate(
            comparison_specs
        ):
            interval = _paired_ratio_interval(
                samples_by_key[(n_nodes, component, baseline)],
                samples_by_key[(n_nodes, component, candidate)],
                n_resamples=bootstrap_resamples,
                seed=seed + n_nodes * 100 + offset,
            )
            comparisons.append(
                {
                    "n_nodes": n_nodes,
                    "component": component,
                    "comparison": label,
                    "baseline_method": baseline,
                    "candidate_method": candidate,
                    **interval,
                }
            )
    return summaries, comparisons


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


def run_warm_simplex_benchmark(
    spec: WarmSimplexBenchmarkSpec,
    output_dir: str | Path,
) -> dict[str, object]:
    """Run paired warm-versus-sorting readout experiments."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    timing_rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    resources: list[dict[str, object]] = []
    for case_index, case in enumerate(spec.cases):
        rows, case_diagnostics, case_resources = _benchmark_case(
            spec, case, case_index
        )
        timing_rows.extend(rows)
        diagnostics.append(case_diagnostics)
        resources.append(case_resources)

    timing_summaries, comparisons = _timing_report(
        timing_rows,
        bootstrap_resamples=spec.bootstrap_resamples,
        seed=spec.seed,
    )
    readout_comparisons = {
        int(row["n_nodes"]): row
        for row in comparisons
        if row["component"] == "readout_only"
    }
    end_comparisons = {
        int(row["n_nodes"]): row
        for row in comparisons
        if row["component"] == "end_to_end"
        and row["comparison"] == "warm_over_sorting"
    }
    exact_comparisons = {
        int(row["n_nodes"]): row
        for row in comparisons
        if row["component"] == "end_to_end"
        and row["comparison"] == "warm_over_exact"
    }
    gates = []
    for row in diagnostics:
        n_nodes = int(row["n_nodes"])
        readout = readout_comparisons[n_nodes]
        end_to_end = end_comparisons[n_nodes]
        exact = exact_comparisons[n_nodes]
        gates.append(
            {
                "n_nodes": n_nodes,
                "correctness_pass": bool(row["correctness_pass"]),
                "algorithm_behavior_pass": bool(
                    float(row["fallback_rate"]) <= 0.05
                    and float(row["support_update_p95"]) <= 2.0
                ),
                "readout_speed_pass": bool(
                    float(readout["median_ratio"]) <= 0.85
                    and float(readout["ratio_ci_upper"]) < 1.0
                ),
                "end_to_end_improvement_pass": bool(
                    float(end_to_end["median_ratio"]) <= 0.95
                    and float(end_to_end["ratio_ci_upper"]) < 1.0
                ),
                "exact_crossover_pass": bool(
                    float(exact["median_ratio"]) < 1.0
                    and float(exact["ratio_ci_upper"]) < 1.0
                ),
            }
        )

    timing_path = destination / "timing_samples.csv"
    diagnostics_path = destination / "diagnostics.csv"
    summary_path = destination / "summary.json"
    _write_csv(timing_rows, timing_path)
    _write_csv(diagnostics, diagnostics_path)
    report: dict[str, object] = {
        "schema_version": 1,
        "study_contract": {
            "workload": "full_field_every_timestep",
            "regime": "fixed_duration",
            "model_setup_in_end_to_end_timing": False,
            "model_setup_reused_across_trajectories": True,
            "readout_baseline": SORTING_METHOD,
            "candidate": WARM_METHOD,
            "end_to_end_comparator": EXACT_METHOD,
            "end_to_end_order_period": 6,
            "end_to_end_position_balanced": spec.end_to_end_repeats % 6 == 0,
            "paired_cached_fields": True,
            "diagnostics_outside_timing": True,
            "accuracy_basis": (
                "warm and sorting readouts agree numerically; stochastic "
                "completion accuracy was established by the preceding "
                "innovation-completion study"
            ),
        },
        "specification": asdict(spec),
        "runtime_environment": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "thread_limits": {
                name: os.environ.get(name)
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        },
        "timing_summaries": timing_summaries,
        "paired_comparisons": comparisons,
        "diagnostics": diagnostics,
        "resources": resources,
        "gates": gates,
        "artifacts": {
            "timing_samples_csv": str(timing_path),
            "diagnostics_csv": str(diagnostics_path),
            "summary_json": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark warm-started exact simplex-bank projection."
    )
    parser.add_argument("--node-counts", type=int, nargs="+", default=[101, 201])
    parser.add_argument("--modes", type=int, default=75)
    parser.add_argument("--completion-rank", type=int, default=25)
    parser.add_argument("--particles", type=int, default=5_275)
    parser.add_argument("--total-time", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--length", type=float, default=4.0)
    parser.add_argument("--diffusion", type=float, default=2.20e-4)
    parser.add_argument("--impulse-fraction", type=float, default=0.59)
    parser.add_argument("--stability-safety", type=float, default=0.95)
    parser.add_argument("--cached-trajectories", type=int, default=32)
    parser.add_argument("--readout-warmups", type=int, default=2)
    parser.add_argument("--readout-repeats", type=int, default=30)
    parser.add_argument("--end-to-end-warmups", type=int, default=1)
    parser.add_argument("--end-to-end-repeats", type=int, default=24)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_261_123)
    parser.add_argument(
        "--output-dir", default="outputs/warm_simplex_benchmark"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = WarmSimplexBenchmarkSpec(
        cases=tuple(
            WarmSimplexBenchmarkCase(
                n_nodes=n_nodes,
                retained_modes=args.modes,
                completion_rank=args.completion_rank,
            )
            for n_nodes in args.node_counts
        ),
        n_particles=args.particles,
        total_time=args.total_time,
        requested_dt=args.dt,
        length=args.length,
        diffusion_coefficient=args.diffusion,
        impulse_fraction=args.impulse_fraction,
        stability_safety=args.stability_safety,
        cached_trajectories=args.cached_trajectories,
        readout_warmups=args.readout_warmups,
        readout_repeats=args.readout_repeats,
        end_to_end_warmups=args.end_to_end_warmups,
        end_to_end_repeats=args.end_to_end_repeats,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    report = run_warm_simplex_benchmark(spec, args.output_dir)
    print(
        json.dumps(
            {"gates": report["gates"], "artifacts": report["artifacts"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
