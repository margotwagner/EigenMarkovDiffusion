"""Paired benchmark for joint batched innovation sampling."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from itertools import permutations
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np

from .baselines import vectorized_multinomial_random_walk_diffusion
from .benchmarking import deep_numpy_nbytes
from .compiled_simplex import (
    compile_warm_simplex_kernel,
    compiled_warm_simplex_available,
    compiled_warm_simplex_bank_counts,
    compiled_warm_simplex_bank_counts_with_diagnostics,
)
from .config import DiffusionConfig
from .innovation_completion import (
    InnovationConditionedModalCompleter,
    PreparedInnovationSampler,
)
from .innovation_completion_benchmark import _validate_spatial
from .warm_simplex_benchmark import (
    EXACT_METHOD,
    WarmSimplexBenchmarkCase as JointInnovationBenchmarkCase,
    _columns,
    _configuration,
    _paired_ratio_interval,
    _validate_output,
    _write_csv,
)

STEPWISE_METHOD = "stepwise_innovation_completion"
BATCHED_METHOD = "batched_innovation_completion"
_COMPLETION = "completion_only"
_END_TO_END = "end_to_end"
_COMPLETION_ORDERS = tuple(permutations((STEPWISE_METHOD, BATCHED_METHOD)))
_END_TO_END_ORDERS = tuple(permutations((STEPWISE_METHOD, BATCHED_METHOD, EXACT_METHOD)))
_PATH_TOLERANCE = 1.0e-8
_COVARIANCE_TOLERANCE = 1.0e-10
_SUPPORT_TOLERANCE = 1.0e-12
_STEPWISE_TARGET_RATIO = 0.75
_CROSSOVER_TARGET_RATIO = 0.95
_THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)


@dataclass(frozen=True, slots=True)
class JointInnovationBenchmarkSpec:
    """Fixed-duration benchmark with balanced method orders."""

    cases: tuple[JointInnovationBenchmarkCase, ...] = (
        JointInnovationBenchmarkCase(101),
        JointInnovationBenchmarkCase(201),
    )
    n_particles: int = 5_275
    total_time: float = 100.0
    requested_dt: float = 1.0
    length: float = 4.0
    diffusion_coefficient: float = 2.20e-4
    impulse_fraction: float = 0.59
    stability_safety: float = 0.95
    completion_warmups: int = 2
    completion_repeats: int = 30
    end_to_end_warmups: int = 1
    end_to_end_repeats: int = 30
    diagnostic_trajectories: int = 32
    bootstrap_resamples: int = 10_000
    seed: int = 20_261_123

    def __post_init__(self) -> None:
        node_counts = {case.n_nodes for case in self.cases}
        if not self.cases or len(node_counts) != len(self.cases):
            raise ValueError("cases must contain unique node counts")
        for case in self.cases:
            if case.n_nodes < 2 or not 1 <= case.retained_modes <= case.n_nodes:
                raise ValueError("require n_nodes >= 2 and retained_modes in [1, n_nodes]")
            if case.completion_rank <= 0:
                raise ValueError("completion_rank must be positive")
        if min(self.completion_warmups, self.end_to_end_warmups) < 0:
            raise ValueError("warmup counts must be nonnegative")
        if self.completion_repeats <= 0 or self.completion_repeats % len(_COMPLETION_ORDERS):
            raise ValueError("completion repeats must balance both method orders")
        if self.end_to_end_repeats <= 0 or self.end_to_end_repeats % len(_END_TO_END_ORDERS):
            raise ValueError("end-to-end repeats must balance all six method orders")
        if min(self.diagnostic_trajectories, self.bootstrap_resamples) <= 0:
            raise ValueError("diagnostic trajectories and bootstrap resamples must be positive")


def _check_covariance(
    completer: InnovationConditionedModalCompleter,
    sampler: PreparedInnovationSampler,
) -> dict[str, float]:
    retained = completer.retained_modes
    error_squared = 0.0
    reference_squared = 0.0
    maximum_error = 0.0
    maximum_factor_error = 0.0
    for step, joint_factor in zip(
        completer.conditioning_steps, sampler.joint_factors, strict=True
    ):
        # Evaluate the stepwise map on unit latent columns before forming its Gram
        # matrix. Applying the gain to Q_RR first is prone to cancellation.
        stepwise_factor = np.zeros_like(joint_factor)
        stepwise_factor[:retained, :retained] = step.retained_factor
        for column in range(retained):
            stepwise_factor[retained:, column] = (
                step.conditional_gain @ step.retained_factor[:, column]
            )
        stepwise_factor[
            retained:, retained : retained + step.selected_rank
        ] = step.conditional_factor
        maximum_factor_error = max(
            maximum_factor_error, float(np.max(np.abs(joint_factor - stepwise_factor)))
        )
        reference = stepwise_factor @ stepwise_factor.T
        difference = joint_factor @ joint_factor.T - reference
        error_squared += float(np.sum(difference**2))
        reference_squared += float(np.sum(reference**2))
        maximum_error = max(maximum_error, float(np.max(np.abs(difference))))
    scaled_error = error_squared / reference_squared if reference_squared else error_squared
    return {
        "effective_covariance_relative_frobenius_error": float(np.sqrt(scaled_error)),
        "effective_covariance_maximum_absolute_error": maximum_error,
        "packed_factor_maximum_absolute_difference": maximum_factor_error,
    }


def _diagnose_paths(
    spec: JointInnovationBenchmarkSpec,
    config: DiffusionConfig,
    completer: InnovationConditionedModalCompleter,
    sampler: PreparedInnovationSampler,
) -> dict[str, object]:
    covariance = _check_covariance(completer, sampler)
    maximum_modal_error = 0.0
    maximum_spatial_error = 0.0
    maximum_count_error = 0.0
    maximum_raw_mass_error = 0.0
    maximum_physical_mass_error = 0.0
    minimum_physical_count = float("inf")
    strict_support_mismatches = 0
    tolerant_support_mismatches = 0
    stepwise_fallbacks = 0
    batched_fallbacks = 0
    matching_rng_states = 0
    for repeat in range(spec.diagnostic_trajectories):
        sequence = np.random.SeedSequence([spec.seed, config.n_nodes, 4, repeat])
        stepwise_rng = np.random.default_rng(sequence)
        batched_rng = np.random.default_rng(sequence)
        stepwise = completer.run_stepwise(stepwise_rng)
        batched = completer.run_batched(batched_rng)
        matching_rng_states += int(
            stepwise_rng.bit_generator.state == batched_rng.bit_generator.state
        )
        maximum_modal_error = max(
            maximum_modal_error,
            float(np.max(np.abs(stepwise.modal_amplitudes - batched.modal_amplitudes))),
        )
        maximum_spatial_error = max(
            maximum_spatial_error,
            float(np.max(np.abs(stepwise.spatial_counts - batched.spatial_counts))),
        )

        physical_counts = []
        fallback_counts = []
        for result in (stepwise, batched):
            raw = result.spatial_counts
            _validate_spatial(raw, config, physical=False)
            maximum_raw_mass_error = max(
                maximum_raw_mass_error, float(np.max(np.abs(raw.sum(axis=1) - config.n_particles)))
            )
            counts, diagnostics = compiled_warm_simplex_bank_counts_with_diagnostics(
                raw, config.n_particles
            )
            _validate_output(counts, config)
            maximum_physical_mass_error = max(
                maximum_physical_mass_error,
                float(np.max(np.abs(counts.sum(axis=1) - config.n_particles))),
            )
            minimum_physical_count = min(minimum_physical_count, float(np.min(counts)))
            physical_counts.append(counts)
            fallback_counts.append(int(diagnostics.used_fallback))

        stepwise_counts, batched_counts = physical_counts
        stepwise_fallbacks += fallback_counts[0]
        batched_fallbacks += fallback_counts[1]
        maximum_count_error = max(
            maximum_count_error, float(np.max(np.abs(stepwise_counts - batched_counts)))
        )
        support_mismatches = (stepwise_counts > 0) != (batched_counts > 0)
        strict_support_mismatches += int(np.count_nonzero(support_mismatches))
        tolerant_support_mismatches += int(
            np.count_nonzero(
                support_mismatches
                & (np.maximum(stepwise_counts, batched_counts) > _SUPPORT_TOLERANCE)
            )
        )

    return {
        **covariance,
        "diagnostic_trajectories": spec.diagnostic_trajectories,
        "matching_rng_states": matching_rng_states,
        "modal_maximum_absolute_difference": maximum_modal_error,
        "spatial_maximum_absolute_difference": maximum_spatial_error,
        "physical_count_maximum_absolute_difference": maximum_count_error,
        "raw_maximum_mass_error": maximum_raw_mass_error,
        "physical_maximum_mass_error": maximum_physical_mass_error,
        "physical_minimum_count": minimum_physical_count,
        "strict_support_mismatch_entries": strict_support_mismatches,
        "support_mismatch_entries_above_tolerance": tolerant_support_mismatches,
        "stepwise_fallback_trajectories": stepwise_fallbacks,
        "batched_fallback_trajectories": batched_fallbacks,
        "correctness_pass": (
            covariance["effective_covariance_relative_frobenius_error"] <= _COVARIANCE_TOLERANCE
            and max(maximum_modal_error, maximum_spatial_error, maximum_count_error)
            <= _PATH_TOLERANCE
            and matching_rng_states == spec.diagnostic_trajectories
            and tolerant_support_mismatches == 0
            and minimum_physical_count >= 0.0
        ),
    }


def _time_component(
    spec: JointInnovationBenchmarkSpec,
    component: str,
    config: DiffusionConfig,
    completer: InnovationConditionedModalCompleter,
) -> list[dict[str, object]]:
    end_to_end = component == _END_TO_END
    if end_to_end:
        orders = _END_TO_END_ORDERS
        warmups = spec.end_to_end_warmups
        repeats = spec.end_to_end_repeats
    else:
        orders = _COMPLETION_ORDERS
        warmups = spec.completion_warmups
        repeats = spec.completion_repeats

    def run(method: str, rng: np.random.Generator) -> np.ndarray:
        if method == EXACT_METHOD:
            return vectorized_multinomial_random_walk_diffusion(config, rng)
        sample = completer.run_batched if method == BATCHED_METHOD else completer.run_stepwise
        raw = sample(rng).spatial_counts
        if end_to_end:
            return compiled_warm_simplex_bank_counts(raw, config.n_particles)
        return raw

    rows: list[dict[str, object]] = []
    for repeat in range(-warmups, repeats):
        order = orders[repeat % len(orders)]
        for position, method in enumerate(order):
            sequence = np.random.SeedSequence([
                spec.seed,
                config.n_nodes,
                int(end_to_end),
                repeat + warmups,
                int(method == EXACT_METHOD),
            ])
            rng = np.random.default_rng(sequence)
            start = perf_counter()
            output = run(method, rng)
            seconds = perf_counter() - start
            if end_to_end:
                _validate_output(output, config)
                if method == EXACT_METHOD and not np.issubdtype(output.dtype, np.integer):
                    raise RuntimeError("the exact walk must return integer counts")
            else:
                _validate_spatial(output, config, physical=False)
            if repeat >= 0:
                rows.append(
                    {
                        "component": component,
                        "method": method,
                        "repeat": repeat,
                        "execution_position": position,
                        "seconds": seconds,
                    }
                )
    return rows


def _benchmark_case(
    spec: JointInnovationBenchmarkSpec,
    case: JointInnovationBenchmarkCase,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    config = _configuration(spec, case)
    columns = _columns(config, case)
    gc.collect()
    start = perf_counter()
    completer = InnovationConditionedModalCompleter(
        config, retained_modes=case.retained_modes, conditional_rank=case.completion_rank
    )
    setup_seconds = perf_counter() - start
    stepwise_bytes = deep_numpy_nbytes(completer)

    start = perf_counter()
    sampler = completer.prepare_batched_sampler()
    batch_preparation_seconds = perf_counter() - start
    batched_bytes = deep_numpy_nbytes(completer)

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        timing_rows = _time_component(spec, _COMPLETION, config, completer)
        timing_rows.extend(_time_component(spec, _END_TO_END, config, completer))
    finally:
        if gc_was_enabled:
            gc.enable()

    diagnostics = _diagnose_paths(spec, config, completer, sampler)
    resources = {
        "stepwise_setup_seconds": setup_seconds,
        "batch_preparation_seconds": batch_preparation_seconds,
        "stepwise_retained_array_bytes": stepwise_bytes,
        "batched_retained_array_bytes": batched_bytes,
        "packed_factor_array_bytes": int(sampler.joint_factors.nbytes),
        "packed_sampler_array_bytes": int(sampler.nbytes),
    }
    return (
        [{**columns, **row} for row in timing_rows],
        {**columns, **diagnostics},
        {**columns, **resources},
    )


def _timing_report(
    spec: JointInnovationBenchmarkSpec, rows: Sequence[dict[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_group: dict[tuple[int, str, str], list[tuple[int, float]]] = {}
    for row in rows:
        key = int(row["n_nodes"]), str(row["component"]), str(row["method"])
        by_group.setdefault(key, []).append((int(row["repeat"]), float(row["seconds"])))
    samples = {
        key: np.array([seconds for _, seconds in sorted(paired)])
        for key, paired in by_group.items()
    }
    timings = [
        {
            "n_nodes": n_nodes,
            "component": component,
            "method": method,
            "samples": int(values.size),
            "median_seconds": float(np.median(values)),
            "mean_seconds": float(np.mean(values)),
            "std_seconds": float(np.std(values)),
        }
        for (n_nodes, component, method), values in samples.items()
    ]
    comparisons = []
    protocols = (
        (_COMPLETION, STEPWISE_METHOD, BATCHED_METHOD, "batched_over_stepwise"),
        (_END_TO_END, STEPWISE_METHOD, BATCHED_METHOD, "batched_over_stepwise"),
        (_END_TO_END, EXACT_METHOD, BATCHED_METHOD, "batched_over_exact"),
        (_END_TO_END, EXACT_METHOD, STEPWISE_METHOD, "stepwise_over_exact"),
    )
    for case in spec.cases:
        n_nodes = case.n_nodes
        for offset, (component, baseline, candidate, label) in enumerate(protocols):
            interval = _paired_ratio_interval(
                samples[n_nodes, component, baseline],
                samples[n_nodes, component, candidate],
                n_resamples=spec.bootstrap_resamples,
                seed=spec.seed + n_nodes * len(protocols) + offset,
            )
            comparisons.append(
                {
                    "n_nodes": n_nodes,
                    "component": component,
                    "comparison": label,
                    "baseline_method": baseline,
                    "candidate_method": candidate,
                    **interval,
                    "speedup": 1.0 / interval["median_ratio"],
                    "candidate_faster_ci_excludes_one": interval["ratio_ci_upper"] < 1.0,
                }
            )
    return timings, comparisons


def _break_even(
    setup_seconds: float, baseline_seconds: float, candidate_seconds: float
) -> int | None:
    savings = baseline_seconds - candidate_seconds
    return ceil(setup_seconds / savings) if savings > 0.0 else None


def _passes_speed_gate(comparison: dict[str, object], target_ratio: float) -> bool:
    return (
        float(comparison["median_ratio"]) <= target_ratio
        and float(comparison["ratio_ci_upper"]) < 1.0
    )


def run_joint_innovation_benchmark(
    spec: JointInnovationBenchmarkSpec, output_dir: str | Path
) -> dict[str, object]:
    """Benchmark prepared sampling, full-field pipelines, and paired correctness."""

    start = perf_counter()
    if not compiled_warm_simplex_available():
        raise RuntimeError("the joint innovation benchmark requires an available Numba backend")
    compile_warm_simplex_kernel()
    compiler_startup_seconds = perf_counter() - start

    timing_rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    resources: list[dict[str, object]] = []
    for case in spec.cases:
        rows, case_diagnostics, case_resources = _benchmark_case(spec, case)
        timing_rows.extend(rows)
        diagnostics.append(case_diagnostics)
        resources.append(case_resources)

    timings, comparisons = _timing_report(spec, timing_rows)
    comparison_lookup = {
        (row["n_nodes"], row["component"], row["comparison"]): row for row in comparisons
    }
    gates = []
    for diagnostic in diagnostics:
        n_nodes = diagnostic["n_nodes"]
        gates.append(
            {
                "n_nodes": n_nodes,
                "correctness_pass": diagnostic["correctness_pass"],
                "completion_speed_pass": _passes_speed_gate(
                    comparison_lookup[n_nodes, _COMPLETION, "batched_over_stepwise"],
                    _STEPWISE_TARGET_RATIO,
                ),
                "end_to_end_speed_pass": _passes_speed_gate(
                    comparison_lookup[n_nodes, _END_TO_END, "batched_over_stepwise"],
                    _STEPWISE_TARGET_RATIO,
                ),
                "exact_crossover_pass": _passes_speed_gate(
                    comparison_lookup[n_nodes, _END_TO_END, "batched_over_exact"],
                    _CROSSOVER_TARGET_RATIO,
                ),
            }
        )

    medians = {
        (row["n_nodes"], row["component"], row["method"]): float(row["median_seconds"])
        for row in timings
    }
    for row in resources:
        n_nodes = row["n_nodes"]
        pack = float(row["batch_preparation_seconds"])
        setup = float(row["stepwise_setup_seconds"])
        exact = medians[n_nodes, _END_TO_END, EXACT_METHOD]
        stepwise = medians[n_nodes, _END_TO_END, STEPWISE_METHOD]
        batched = medians[n_nodes, _END_TO_END, BATCHED_METHOD]
        row["batched_vs_stepwise_break_even_trajectories"] = _break_even(pack, stepwise, batched)
        row["stepwise_vs_exact_break_even_trajectories"] = _break_even(setup, exact, stepwise)
        row["batched_vs_exact_break_even_trajectories"] = _break_even(setup + pack, exact, batched)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    timing_path = destination / "timing_samples.csv"
    diagnostics_path = destination / "diagnostics.csv"
    resources_path = destination / "resources.csv"
    summary_path = destination / "summary.json"
    _write_csv(timing_rows, timing_path)
    _write_csv(diagnostics, diagnostics_path)
    _write_csv(resources, resources_path)

    thread_limits = {name: os.environ.get(name) for name in _THREAD_VARIABLES}
    report: dict[str, object] = {
        "schema_version": 1,
        "study_contract": {
            "regime": "fixed_duration",
            "workload": "full_field_every_timestep",
            "completion_only_includes_spatial_reconstruction": True,
            "modal_pipeline_readout": "compiled_exact_simplex_bank",
            "exact_comparator": EXACT_METHOD,
            "shared_modal_seeds_within_repeat": True,
            "diagnostics_outside_timing": True,
            "setup_and_compiler_startup_in_timing": False,
            "completion_order_period": len(_COMPLETION_ORDERS),
            "end_to_end_order_period": len(_END_TO_END_ORDERS),
            "paired_bootstrap_confidence_level": 0.95,
            "path_absolute_tolerance": _PATH_TOLERANCE,
            "effective_covariance_relative_tolerance": _COVARIANCE_TOLERANCE,
            "support_zero_tolerance": _SUPPORT_TOLERANCE,
            "batched_over_stepwise_target_median_ratio": _STEPWISE_TARGET_RATIO,
            "batched_over_exact_target_median_ratio": _CROSSOVER_TARGET_RATIO,
            "speed_gate_requires_ratio_ci_upper_below_one": True,
            "memory": "logical array bytes retained by setup, excluding transient arrays",
        },
        "specification": asdict(spec),
        "runtime_environment": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "numba_version": getattr(sys.modules.get("numba"), "__version__", None),
            "platform": platform.platform(),
            "thread_limits": thread_limits,
            "one_thread_protocol": all(value == "1" for value in thread_limits.values()),
        },
        "compiler_startup_seconds": compiler_startup_seconds,
        "timing_summaries": timings,
        "paired_comparisons": comparisons,
        "diagnostics": diagnostics,
        "resources": resources,
        "gates": gates,
        "artifacts": {
            "timing_samples_csv": str(timing_path),
            "diagnostics_csv": str(diagnostics_path),
            "resources_csv": str(resources_path),
            "summary_json": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    defaults = JointInnovationBenchmarkSpec()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-counts", nargs="+", type=int, default=[101, 201])
    parser.add_argument("--modes", type=int, default=75)
    parser.add_argument("--completion-rank", type=int, default=25)
    parser.add_argument("--particles", type=int, default=defaults.n_particles)
    parser.add_argument("--total-time", type=float, default=defaults.total_time)
    parser.add_argument("--dt", type=float, default=defaults.requested_dt)
    parser.add_argument("--length", type=float, default=defaults.length)
    parser.add_argument("--diffusion", type=float, default=defaults.diffusion_coefficient)
    parser.add_argument("--impulse-fraction", type=float, default=defaults.impulse_fraction)
    parser.add_argument("--stability-safety", type=float, default=defaults.stability_safety)
    parser.add_argument("--completion-warmups", type=int, default=defaults.completion_warmups)
    parser.add_argument("--completion-repeats", type=int, default=defaults.completion_repeats)
    parser.add_argument("--end-to-end-warmups", type=int, default=defaults.end_to_end_warmups)
    parser.add_argument("--end-to-end-repeats", type=int, default=defaults.end_to_end_repeats)
    parser.add_argument(
        "--diagnostic-trajectories",
        type=int,
        default=defaults.diagnostic_trajectories,
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=defaults.bootstrap_resamples)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--output-dir", default="outputs/joint_innovation_benchmark")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    options = vars(parser.parse_args(argv))
    output_dir = options.pop("output_dir")
    node_counts = options.pop("node_counts")
    modes = options.pop("modes")
    completion_rank = options.pop("completion_rank")
    options["n_particles"] = options.pop("particles")
    options["requested_dt"] = options.pop("dt")
    options["diffusion_coefficient"] = options.pop("diffusion")
    try:
        spec = JointInnovationBenchmarkSpec(
            cases=tuple(
                JointInnovationBenchmarkCase(n, modes, completion_rank)
                for n in node_counts
            ),
            **options,
        )
        report = run_joint_innovation_benchmark(spec, output_dir)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps({"gates": report["gates"], "artifacts": report["artifacts"]}, indent=2))
    return int(any(not row["correctness_pass"] for row in report["gates"]))


if __name__ == "__main__":
    raise SystemExit(main())
