"""Compare streaming and full-history construction of the compact innovation runtime."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
from collections.abc import Callable
from dataclasses import asdict
from itertools import permutations
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np

from .benchmarking import deep_numpy_nbytes
from .compact_runtime_benchmark import (
    EXACT_METHOD,
    _COMPLETION,
    _END_TO_END,
    _THREAD_VARIABLES,
    CompactRuntimeBenchmarkCase as StreamingConstructorBenchmarkCase,
    CompactRuntimeBenchmarkSpec as StreamingConstructorBenchmarkSpec,
    _diagnose_paths,
    _isolated_rss,
    _print_setup_rss,
    _summarize_timings,
    _time_trajectories,
    build_parser as _build_compact_parser,
)
from .compiled_simplex import compile_warm_simplex_kernel, compiled_warm_simplex_available
from .config import DiffusionConfig
from .innovation_completion import PreparedInnovationCompletion
from .joint_innovation_benchmark import _break_even, _passes_speed_gate
from .warm_simplex_benchmark import _columns, _configuration, _write_csv

FULL_HISTORY_METHOD = "full_history_compact_constructor"
STREAMING_METHOD = "streaming_compact_constructor"
_SETUP = "cold_prepared_setup"
_SETUP_ORDERS = tuple(permutations((FULL_HISTORY_METHOD, STREAMING_METHOD)))
_PREPARED_METHODS = (("full_history", FULL_HISTORY_METHOD), ("streaming", STREAMING_METHOD))
_SETUP_MAXIMUM_MEDIAN_RATIO = 1.10
_SETUP_MAXIMUM_CI_UPPER = 1.25
_RUNTIME_MAXIMUM_CI_UPPER = 1.10
_EXACT_TARGET_RATIO = 0.95
_PEAK_RSS_GATE_NODES = 201
_PEAK_RSS_LIMIT_BYTES = 150 * 2**20
_PEAK_RSS_MAXIMUM_RATIO = 0.40
_COMPARISONS = (
    (_SETUP, FULL_HISTORY_METHOD, STREAMING_METHOD, "streaming_over_full_history"),
    (_COMPLETION, FULL_HISTORY_METHOD, STREAMING_METHOD, "streaming_over_full_history"),
    (_END_TO_END, FULL_HISTORY_METHOD, STREAMING_METHOD, "streaming_over_full_history"),
    (_END_TO_END, EXACT_METHOD, STREAMING_METHOD, "streaming_over_exact"),
    (_END_TO_END, EXACT_METHOD, FULL_HISTORY_METHOD, "full_history_over_exact"),
)


def _constructor(method: str) -> Callable[..., PreparedInnovationCompletion]:
    return {
        FULL_HISTORY_METHOD: PreparedInnovationCompletion.from_full_history_config,
        STREAMING_METHOD: PreparedInnovationCompletion.from_config,
    }[method]


def _rss_worker() -> None:
    request = json.load(sys.stdin)
    config = DiffusionConfig(**request["config"])
    prepare = _constructor(request["method"])
    options = request["options"]
    _print_setup_rss(lambda: prepare(config, **options))


def _time_setup(
    spec: StreamingConstructorBenchmarkSpec,
    case: StreamingConstructorBenchmarkCase,
    config: DiffusionConfig,
) -> list[dict[str, object]]:
    rows = []
    for repeat in range(spec.setup_repeats):
        for position, method in enumerate(_SETUP_ORDERS[repeat % len(_SETUP_ORDERS)]):
            prepare = _constructor(method)
            gc.collect()
            start = perf_counter()
            runtime = prepare(
                config, retained_modes=case.retained_modes, conditional_rank=case.completion_rank
            )
            seconds = perf_counter() - start
            del runtime
            rows.append({
                "component": _SETUP,
                "method": method,
                "repeat": repeat,
                "execution_position": position,
                "seconds": seconds,
            })
    return rows


def _diagnose_state(
    reference: PreparedInnovationCompletion, candidate: PreparedInnovationCompletion
) -> dict[str, object]:
    reference_sampler = reference.sampler
    candidate_sampler = candidate.sampler
    reference_factors = reference_sampler.joint_factors
    candidate_factors = candidate_sampler.joint_factors
    matches = {
        "joint_factors_match": np.array_equal(reference_factors, candidate_factors),
        "selected_ranks_match": np.array_equal(
            reference_sampler.selected_ranks, candidate_sampler.selected_ranks
        ),
        "active_columns_match": np.array_equal(
            reference_sampler._active_columns, candidate_sampler._active_columns
        ),
        "latent_count_matches": reference_sampler.latent_count == candidate_sampler.latent_count,
        "retained_modes_match": reference.retained_modes == candidate.retained_modes,
        "initial_modal_matches": np.array_equal(reference.initial_modal, candidate.initial_modal),
        "one_step_factors_match": np.array_equal(
            reference.one_step_factors, candidate.one_step_factors
        ),
        "basis_eigenvalues_match": np.array_equal(
            reference.basis.eigenvalues, candidate.basis.eigenvalues
        ),
        "basis_eigenvectors_match": np.array_equal(
            reference.basis.eigenvectors, candidate.basis.eigenvectors
        ),
    }
    factor_error = None
    if reference_factors.shape == candidate_factors.shape:
        factor_error = float(np.max(np.abs(reference_factors - candidate_factors), initial=0.0))
    return {
        **matches,
        "joint_factors_maximum_absolute_difference": factor_error,
        "runtime_state_matches": all(matches.values()),
    }


def run_streaming_constructor_benchmark(
    spec: StreamingConstructorBenchmarkSpec, output_dir: str | Path
) -> dict[str, object]:
    """Measure compact setup, memory, prepared draws, and exact path agreement."""

    start = perf_counter()
    if not compiled_warm_simplex_available():
        raise RuntimeError("the streaming constructor benchmark requires Numba")
    compile_warm_simplex_kernel()
    compiler_startup_seconds = perf_counter() - start

    artifacts_rows: dict[str, list[dict[str, object]]] = {
        name: [] for name in ("setup_samples", "timing_samples", "diagnostics", "resources")
    }
    timings: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    gates: list[dict[str, object]] = []
    for case in spec.cases:
        config = _configuration(spec, case)
        columns = _columns(config, case)
        setup_rows = _time_setup(spec, case, config)
        options = {"retained_modes": case.retained_modes, "conditional_rank": case.completion_rank}
        reference = PreparedInnovationCompletion.from_full_history_config(config, **options)
        candidate = PreparedInnovationCompletion.from_config(config, **options)
        diagnostics = _diagnose_paths(spec, config, reference.run, candidate.run)
        diagnostics.update(_diagnose_state(reference, candidate))
        diagnostics["correctness_pass"] = bool(
            diagnostics["correctness_pass"] and diagnostics["runtime_state_matches"]
        )
        reference_bytes = deep_numpy_nbytes(reference, include_array_bases=True)
        candidate_bytes = deep_numpy_nbytes(candidate, include_array_bases=True)
        resources = {
            "full_history_retained_array_bytes": reference_bytes,
            "streaming_retained_array_bytes": candidate_bytes,
            "full_history_reported_array_bytes": reference.nbytes,
            "streaming_reported_array_bytes": candidate.nbytes,
        }
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            timing_rows = _time_trajectories(spec, config, {
                FULL_HISTORY_METHOD: reference.run,
                STREAMING_METHOD: candidate.run,
            })
        finally:
            if gc_was_enabled:
                gc.enable()
        del reference, candidate
        resources.update(_isolated_rss(
            config,
            case,
            methods=_PREPARED_METHODS,
            worker_module="eigendiffusion.streaming_constructor_benchmark",
        ))
        case_timings, case_comparisons, medians = _summarize_timings(
            spec, config, setup_rows + timing_rows, _COMPARISONS
        )
        setup, completion, end_to_end, exact, _ = case_comparisons
        case_gates = {
            "n_nodes": case.n_nodes,
            "correctness_pass": diagnostics["correctness_pass"],
            "retained_storage_pass": len({
                resources["full_history_retained_array_bytes"],
                resources["streaming_retained_array_bytes"],
                resources["full_history_reported_array_bytes"],
                resources["streaming_reported_array_bytes"],
            }) == 1,
            "setup_noninferiority_pass": (
                float(setup["median_ratio"]) <= _SETUP_MAXIMUM_MEDIAN_RATIO
                and float(setup["ratio_ci_upper"]) <= _SETUP_MAXIMUM_CI_UPPER
            ),
            "completion_noninferiority_pass": (
                float(completion["ratio_ci_upper"]) <= _RUNTIME_MAXIMUM_CI_UPPER
            ),
            "end_to_end_noninferiority_pass": (
                float(end_to_end["ratio_ci_upper"]) <= _RUNTIME_MAXIMUM_CI_UPPER
            ),
            "exact_crossover_pass": _passes_speed_gate(exact, _EXACT_TARGET_RATIO),
        }
        reference_peak = resources.get("full_history_rss_peak_through_setup_bytes")
        candidate_peak = resources.get("streaming_rss_peak_through_setup_bytes")
        peak_ratio = candidate_peak / reference_peak if candidate_peak and reference_peak else None
        resources["streaming_over_full_history_peak_rss_ratio"] = peak_ratio
        if case.n_nodes == _PEAK_RSS_GATE_NODES:
            case_gates["peak_rss_pass"] = (
                candidate_peak is not None
                and candidate_peak < _PEAK_RSS_LIMIT_BYTES
                and peak_ratio is not None
                and peak_ratio <= _PEAK_RSS_MAXIMUM_RATIO
            )
        for label, method in _PREPARED_METHODS:
            setup_seconds = medians[_SETUP, method]
            exact_seconds = medians[_END_TO_END, EXACT_METHOD]
            runtime_seconds = medians[_END_TO_END, method]
            resources[f"{label}_cold_prepared_setup_seconds"] = setup_seconds
            resources[f"{label}_vs_exact_break_even_trajectories"] = _break_even(
                setup_seconds, exact_seconds, runtime_seconds
            )
            resources[f"{label}_vs_exact_break_even_with_compiler_startup_trajectories"] = (
                _break_even(
                    setup_seconds + compiler_startup_seconds, exact_seconds, runtime_seconds
                )
            )
        artifacts_rows["setup_samples"].extend({**columns, **row} for row in setup_rows)
        artifacts_rows["timing_samples"].extend({**columns, **row} for row in timing_rows)
        artifacts_rows["diagnostics"].append({**columns, **diagnostics})
        artifacts_rows["resources"].append({**columns, **resources})
        timings.extend(case_timings)
        comparisons.extend(case_comparisons)
        gates.append(case_gates)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name, rows in artifacts_rows.items():
        path = destination / f"{name}.csv"
        _write_csv(rows, path)
        artifacts[f"{name}_csv"] = str(path)
    summary_path = destination / "summary.json"
    artifacts["summary_json"] = str(summary_path)
    thread_limits = {name: os.environ.get(name) for name in _THREAD_VARIABLES}
    report = {
        "schema_version": 1,
        "study_contract": {
            "regime": "fixed_duration",
            "workload": "full_field_every_timestep",
            "completion_only_includes_spatial_reconstruction": True,
            "modal_pipeline_readout": "compiled_exact_simplex_bank",
            "exact_comparator": EXACT_METHOD,
            "shared_modal_seeds_within_repeat": True,
            "setup_and_compiler_startup_in_trajectory_timing": False,
            "setup_order_period": len(_SETUP_ORDERS),
            "completion_order_period": 2,
            "end_to_end_order_period": 6,
            "paired_bootstrap_confidence_level": 0.95,
            "setup_maximum_median_ratio": _SETUP_MAXIMUM_MEDIAN_RATIO,
            "setup_maximum_ratio_ci_upper": _SETUP_MAXIMUM_CI_UPPER,
            "runtime_maximum_ratio_ci_upper": _RUNTIME_MAXIMUM_CI_UPPER,
            "streaming_over_exact_target_median_ratio": _EXACT_TARGET_RATIO,
            "peak_rss_gate_nodes": _PEAK_RSS_GATE_NODES,
            "peak_rss_absolute_limit_bytes": _PEAK_RSS_LIMIT_BYTES,
            "streaming_over_full_history_peak_rss_maximum_ratio": _PEAK_RSS_MAXIMUM_RATIO,
            "memory": "unique reachable NumPy backing bytes after setup",
            "rss": "Linux process RSS after GC and high-water RSS through isolated setup",
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
        "diagnostics": artifacts_rows["diagnostics"],
        "resources": artifacts_rows["resources"],
        "gates": gates,
        "all_gates_pass": all(
            value for row in gates for name, value in row.items() if name != "n_nodes"
        ),
        "artifacts": artifacts,
    }
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = _build_compact_parser()
    parser.description = __doc__
    parser.set_defaults(output_dir="outputs/streaming_constructor_benchmark")
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
        spec = StreamingConstructorBenchmarkSpec(
            cases=tuple(
                StreamingConstructorBenchmarkCase(n, modes, completion_rank) for n in node_counts
            ),
            **options,
        )
        report = run_streaming_constructor_benchmark(spec, output_dir)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps({"gates": report["gates"], "artifacts": report["artifacts"]}, indent=2))
    return int(not report["all_gates_pass"])


if __name__ == "__main__":
    raise SystemExit(main())
