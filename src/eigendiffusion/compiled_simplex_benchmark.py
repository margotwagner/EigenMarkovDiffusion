"""Paired benchmark for the compiled exact warm-simplex bank."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from itertools import permutations
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np

from .baselines import vectorized_multinomial_random_walk_diffusion
from .compiled_simplex import (
    compile_warm_simplex_kernel,
    compiled_warm_simplex_available,
    compiled_warm_simplex_bank_counts,
    compiled_warm_simplex_bank_counts_with_diagnostics,
    compiled_warm_simplex_bank_readout,
    compiled_warm_simplex_disk_cache_enabled,
)
from .config import DiffusionConfig
from .innovation_completion import InnovationConditionedModalCompleter
from .readouts import (
    warm_simplex_bank_counts,
    warm_simplex_bank_counts_with_diagnostics,
    warm_simplex_bank_readout,
)
from .warm_simplex_benchmark import (
    EXACT_METHOD,
    WARM_METHOD,
    WarmSimplexBenchmarkCase as CompiledSimplexBenchmarkCase,
    WarmSimplexBenchmarkSpec,
    _columns,
    _configuration,
    _paired_ratio_interval,
    _validate_output,
    _write_csv,
)
from .warm_simplex_benchmark import build_parser as _build_warm_parser

COMPILED_METHOD = "compiled_warm_simplex_bank"
_READOUT_ORDERS = ((WARM_METHOD, COMPILED_METHOD), (COMPILED_METHOD, WARM_METHOD))
_END_TO_END_ORDERS = tuple(permutations((WARM_METHOD, COMPILED_METHOD, EXACT_METHOD)))
_READOUTS = {
    WARM_METHOD: warm_simplex_bank_counts,
    COMPILED_METHOD: compiled_warm_simplex_bank_counts,
}
_THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)
_SUPPORT_TOLERANCE = 1.0e-12
_CORRECTNESS_TOLERANCE = 1.0e-10
_READOUT_TARGET_RATIO = 0.25
_CROSSOVER_TARGET_RATIO = 0.95


@dataclass(frozen=True, slots=True)
class CompiledSimplexBenchmarkSpec(WarmSimplexBenchmarkSpec):
    """The warm benchmark workload with strictly balanced execution orders."""

    def __post_init__(self) -> None:
        WarmSimplexBenchmarkSpec.__post_init__(self)
        if self.readout_repeats % len(_READOUT_ORDERS):
            raise ValueError("readout repeats must balance the two method orders")
        if self.end_to_end_repeats % len(_END_TO_END_ORDERS):
            raise ValueError("end-to-end repeats must balance the six method orders")


def _diagnose_cached_fields(
    raw_fields: np.ndarray,
    config: DiffusionConfig,
) -> dict[str, object]:
    maximum_output_difference = 0.0
    maximum_adjusted_difference = 0.0
    maximum_residual_difference = 0.0
    maximum_compiled_residual_sum = 0.0
    maximum_warm_residual_sum = 0.0
    maximum_mass_error = 0.0
    minimum_output = float("inf")
    strict_support_mismatches = 0
    tolerant_support_mismatches = 0
    warm_fallback_rows = 0
    compiled_fallback_rows = 0
    compiled_fallback_trajectories = 0
    first_fallback_rows: list[int] = []

    for raw in raw_fields:
        warm = warm_simplex_bank_readout(raw, config.n_particles)
        compiled = compiled_warm_simplex_bank_readout(raw, config.n_particles)
        warm_counts, warm_diagnostics = warm_simplex_bank_counts_with_diagnostics(
            raw, config.n_particles
        )
        compiled_counts, compiled_diagnostics = compiled_warm_simplex_bank_counts_with_diagnostics(
            raw, config.n_particles
        )
        if not np.array_equal(
            compiled_counts, compiled_warm_simplex_bank_counts(raw, config.n_particles)
        ):
            raise RuntimeError("compiled diagnostic and counts-only paths differ")
        if not np.array_equal(warm_counts, warm_simplex_bank_counts(raw, config.n_particles)):
            raise RuntimeError("warm diagnostic and counts-only paths differ")
        if not np.array_equal(warm.counts, warm_counts):
            raise RuntimeError("warm diagnostic and detailed readout paths differ")
        if not np.array_equal(compiled.counts, compiled_counts):
            raise RuntimeError("compiled diagnostic and detailed readout paths differ")
        _validate_output(warm_counts, config)
        _validate_output(compiled_counts, config)
        ledgers = (
            warm.adjusted_inputs,
            warm.residuals,
            compiled.adjusted_inputs,
            compiled.residuals,
        )
        if not all(np.all(np.isfinite(values)) for values in ledgers):
            raise RuntimeError("simplex readout returned a nonfinite adjusted or residual ledger")

        maximum_output_difference = max(
            maximum_output_difference, float(np.max(np.abs(compiled_counts - warm_counts)))
        )
        maximum_adjusted_difference = max(
            maximum_adjusted_difference,
            float(np.max(np.abs(compiled.adjusted_inputs - warm.adjusted_inputs))),
        )
        maximum_residual_difference = max(
            maximum_residual_difference, float(np.max(np.abs(compiled.residuals - warm.residuals)))
        )
        maximum_compiled_residual_sum = max(
            maximum_compiled_residual_sum, float(np.max(np.abs(compiled.residuals.sum(axis=1))))
        )
        maximum_warm_residual_sum = max(
            maximum_warm_residual_sum, float(np.max(np.abs(warm.residuals.sum(axis=1))))
        )
        maximum_mass_error = max(
            maximum_mass_error,
            float(np.max(np.abs(compiled_counts.sum(axis=1) - config.n_particles))),
        )
        minimum_output = min(minimum_output, float(np.min(compiled_counts)))
        support_mismatches = (compiled_counts > 0.0) != (warm_counts > 0.0)
        strict_support_mismatches += int(np.count_nonzero(support_mismatches))
        tolerant_support_mismatches += int(
            np.count_nonzero(
                support_mismatches & (np.maximum(compiled_counts, warm_counts) > _SUPPORT_TOLERANCE)
            )
        )
        warm_fallback_rows += warm_diagnostics.fallback_rows
        compiled_fallback_rows += compiled_diagnostics.fallback_rows
        compiled_fallback_trajectories += int(compiled_diagnostics.used_fallback)
        if compiled_diagnostics.fallback_row is not None:
            first_fallback_rows.append(compiled_diagnostics.fallback_row)

    return {
        "cached_trajectories": raw_fields.shape[0],
        "total_rows": raw_fields.shape[0] * config.n_steps,
        "maximum_output_absolute_difference": maximum_output_difference,
        "maximum_adjusted_absolute_difference": maximum_adjusted_difference,
        "maximum_residual_absolute_difference": maximum_residual_difference,
        "maximum_compiled_residual_sum_error": maximum_compiled_residual_sum,
        "maximum_warm_residual_sum_error": maximum_warm_residual_sum,
        "maximum_mass_error": maximum_mass_error,
        "minimum_output": minimum_output,
        "strict_support_mismatch_entries": strict_support_mismatches,
        "support_mismatch_entries_above_tolerance": tolerant_support_mismatches,
        "strict_support_identical": strict_support_mismatches == 0,
        "support_identical_above_tolerance": tolerant_support_mismatches == 0,
        "warm_fallback_rows": warm_fallback_rows,
        "compiled_fallback_rows": compiled_fallback_rows,
        "compiled_fallback_trajectories": compiled_fallback_trajectories,
        "compiled_first_fallback_rows": first_fallback_rows,
        "correctness_pass": (
            maximum_output_difference <= _CORRECTNESS_TOLERANCE
            and maximum_adjusted_difference <= _CORRECTNESS_TOLERANCE
            and maximum_residual_difference <= _CORRECTNESS_TOLERANCE
            and maximum_compiled_residual_sum <= _CORRECTNESS_TOLERANCE
            and maximum_warm_residual_sum <= _CORRECTNESS_TOLERANCE
            and maximum_mass_error <= _CORRECTNESS_TOLERANCE
            and minimum_output >= 0.0
            and tolerant_support_mismatches == 0
        ),
    }


def _end_to_end(
    method: str,
    config: DiffusionConfig,
    completer: InnovationConditionedModalCompleter,
    rng: np.random.Generator,
) -> np.ndarray:
    if method == EXACT_METHOD:
        return vectorized_multinomial_random_walk_diffusion(config, rng)
    raw = completer.run(rng).spatial_counts
    return _READOUTS[method](raw, config.n_particles)


def _benchmark_case(
    spec: CompiledSimplexBenchmarkSpec,
    case: CompiledSimplexBenchmarkCase,
    case_index: int,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    config = _configuration(spec, case)
    columns = _columns(config, case)
    gc.collect()
    start = perf_counter()
    completer = InnovationConditionedModalCompleter(
        config, retained_modes=case.retained_modes, conditional_rank=case.completion_rank
    )
    setup_seconds = perf_counter() - start

    sequences = np.random.SeedSequence([spec.seed, case_index, 0]).spawn(
        spec.cached_trajectories
    )
    start = perf_counter()
    raw_fields = np.stack(
        [completer.run(np.random.default_rng(sequence)).spatial_counts for sequence in sequences]
    )
    cache_generation_seconds = perf_counter() - start
    timing_rows: list[dict[str, object]] = []

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for raw in raw_fields[: spec.readout_warmups]:
            for readout in _READOUTS.values():
                readout(raw, config.n_particles)

        for repeat in range(spec.readout_repeats):
            raw = raw_fields[spec.readout_warmups + repeat]
            order = _READOUT_ORDERS[repeat % len(_READOUT_ORDERS)]
            for position, method in enumerate(order):
                readout = _READOUTS[method]
                start = perf_counter()
                output = readout(raw, config.n_particles)
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

        for warmup in range(spec.end_to_end_warmups):
            for method in _END_TO_END_ORDERS[0]:
                sequence = np.random.SeedSequence(
                    [spec.seed, case_index, 1, warmup, int(method == EXACT_METHOD)]
                )
                output = _end_to_end(method, config, completer, np.random.default_rng(sequence))
                _validate_output(output, config)

        for repeat in range(spec.end_to_end_repeats):
            order = _END_TO_END_ORDERS[repeat % len(_END_TO_END_ORDERS)]
            for position, method in enumerate(order):
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [spec.seed, case_index, 2, repeat, int(method == EXACT_METHOD)]
                    )
                )
                start = perf_counter()
                output = _end_to_end(method, config, completer, rng)
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
    }
    return timing_rows, diagnostics, resources


def _timing_report(
    rows: Sequence[dict[str, object]], spec: CompiledSimplexBenchmarkSpec
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    samples_by_group: dict[tuple[int, str, str], list[tuple[int, float]]] = {}
    for row in rows:
        key = (int(row["n_nodes"]), str(row["component"]), str(row["method"]))
        samples_by_group.setdefault(key, []).append((int(row["repeat"]), float(row["seconds"])))

    timings = []
    samples: dict[tuple[int, str, str], np.ndarray] = {}
    for key, paired_values in samples_by_group.items():
        values = np.array([seconds for _, seconds in sorted(paired_values)])
        samples[key] = values
        n_nodes, component, method = key
        timings.append(
            {
                "n_nodes": n_nodes,
                "component": component,
                "method": method,
                "samples": values.size,
                "median_seconds": float(np.median(values)),
                "mean_seconds": float(np.mean(values)),
                "std_seconds": float(np.std(values)),
            }
        )

    comparisons = []
    protocols = (
        ("readout_only", WARM_METHOD, COMPILED_METHOD, "compiled_over_warm"),
        ("end_to_end", WARM_METHOD, COMPILED_METHOD, "compiled_over_warm"),
        ("end_to_end", EXACT_METHOD, COMPILED_METHOD, "compiled_over_exact"),
        ("end_to_end", EXACT_METHOD, WARM_METHOD, "warm_over_exact"),
    )
    for case in spec.cases:
        n_nodes = case.n_nodes
        for offset, (component, baseline, candidate, comparison) in enumerate(protocols):
            interval = _paired_ratio_interval(
                samples[(n_nodes, component, baseline)],
                samples[(n_nodes, component, candidate)],
                n_resamples=spec.bootstrap_resamples,
                seed=spec.seed + n_nodes * 100 + offset,
            )
            comparisons.append(
                {
                    "n_nodes": n_nodes,
                    "component": component,
                    "comparison": comparison,
                    "baseline_method": baseline,
                    "candidate_method": candidate,
                    **interval,
                }
            )
    return timings, comparisons


def run_compiled_simplex_benchmark(
    spec: CompiledSimplexBenchmarkSpec, output_dir: str | Path
) -> dict[str, object]:
    """Compare the compiled readout with warm exact and the exact walk."""

    numba_preimported = "numba" in sys.modules
    start = perf_counter()
    if not compiled_warm_simplex_available():
        raise RuntimeError("the compiled simplex benchmark requires an available Numba backend")
    compile_warm_simplex_kernel()
    startup_seconds = perf_counter() - start

    timing_rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    resources: list[dict[str, object]] = []
    for case_index, case in enumerate(spec.cases):
        rows, case_diagnostics, case_resources = _benchmark_case(spec, case, case_index)
        timing_rows.extend(rows)
        diagnostics.append(case_diagnostics)
        resources.append(case_resources)

    timings, comparisons = _timing_report(timing_rows, spec)
    comparison_lookup = {
        (row["n_nodes"], row["component"], row["comparison"]): row for row in comparisons
    }
    gates = []
    for diagnostic in diagnostics:
        n_nodes = diagnostic["n_nodes"]
        readout = comparison_lookup[(n_nodes, "readout_only", "compiled_over_warm")]
        exact = comparison_lookup[(n_nodes, "end_to_end", "compiled_over_exact")]
        gates.append(
            {
                "n_nodes": n_nodes,
                "correctness_pass": diagnostic["correctness_pass"],
                "readout_speed_pass": (
                    readout["median_ratio"] <= _READOUT_TARGET_RATIO
                    and readout["ratio_ci_upper"] < 1.0
                ),
                "exact_crossover_pass": (
                    exact["median_ratio"] <= _CROSSOVER_TARGET_RATIO
                    and exact["ratio_ci_upper"] < 1.0
                ),
            }
        )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    timing_path = destination / "timing_samples.csv"
    diagnostics_path = destination / "diagnostics.csv"
    summary_path = destination / "summary.json"
    _write_csv(timing_rows, timing_path)
    _write_csv(diagnostics, diagnostics_path)
    thread_limits = {name: os.environ.get(name) for name in _THREAD_VARIABLES}
    report: dict[str, object] = {
        "schema_version": 1,
        "study_contract": {
            "workload": "full_field_every_timestep",
            "regime": "fixed_duration",
            "model_setup_in_end_to_end_timing": False,
            "model_setup_reused_across_trajectories": True,
            "compiler_startup_in_timing_samples": False,
            "readout_baseline": WARM_METHOD,
            "candidate": COMPILED_METHOD,
            "end_to_end_comparator": EXACT_METHOD,
            "readout_order_period": len(_READOUT_ORDERS),
            "end_to_end_order_period": len(_END_TO_END_ORDERS),
            "end_to_end_position_balanced": True,
            "paired_cached_fields": True,
            "diagnostics_outside_timing": True,
            "count_and_ledger_absolute_tolerance": _CORRECTNESS_TOLERANCE,
            "support_zero_tolerance": _SUPPORT_TOLERANCE,
            "readout_target_median_ratio": _READOUT_TARGET_RATIO,
            "exact_crossover_target_median_ratio": _CROSSOVER_TARGET_RATIO,
            "paired_bootstrap_confidence_level": 0.95,
            "accuracy_basis": "numerical equivalence on identical completed fields",
        },
        "specification": asdict(spec),
        "runtime_environment": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "numba_version": getattr(sys.modules.get("numba"), "__version__", None),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "thread_limits": thread_limits,
            "one_thread_protocol": all(thread_limits[name] == "1" for name in _THREAD_VARIABLES),
        },
        "compiler_startup": {
            "import_and_jit_seconds": startup_seconds,
            "numba_preimported": numba_preimported,
            "numba_disk_cache_enabled": (
                compiled_warm_simplex_disk_cache_enabled()
            ),
            "cold_measurement_requires_fresh_process": True,
        },
        "timing_summaries": timings,
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
    parser = _build_warm_parser()
    parser.description = "Benchmark compiled exact simplex projection with one BLAS thread."
    parser.set_defaults(output_dir="outputs/compiled_simplex_benchmark")
    parser.epilog = (
        "Set OPENBLAS_NUM_THREADS, OMP_NUM_THREADS, MKL_NUM_THREADS, "
        "NUMEXPR_NUM_THREADS and NUMBA_NUM_THREADS to 1 before starting Python."
    )
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
        spec = CompiledSimplexBenchmarkSpec(
            cases=tuple(
                CompiledSimplexBenchmarkCase(n_nodes, modes, completion_rank)
                for n_nodes in node_counts
            ),
            **options,
        )
        report = run_compiled_simplex_benchmark(spec, output_dir)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"gates": report["gates"], "artifacts": report["artifacts"]}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
