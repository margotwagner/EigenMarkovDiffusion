"""Compare the full builder and compact prepared innovation runtime."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from itertools import permutations
from pathlib import Path
from time import perf_counter
from typing import Sequence
from weakref import ReferenceType, ref

import numpy as np

from .baselines import vectorized_multinomial_random_walk_diffusion
from .benchmarking import deep_numpy_nbytes
from .compiled_simplex import (
    compile_warm_simplex_kernel,
    compiled_warm_simplex_available,
    compiled_warm_simplex_bank_counts,
)
from .config import DiffusionConfig
from .innovation_completion import (
    InnovationConditionedCompletionResult,
    InnovationConditionedModalCompleter,
    PreparedInnovationCompletion,
)
from .innovation_completion_benchmark import _validate_spatial
from .joint_innovation_benchmark import (
    BATCHED_METHOD,
    EXACT_METHOD,
    JointInnovationBenchmarkCase as CompactRuntimeBenchmarkCase,
    JointInnovationBenchmarkSpec,
    _break_even,
    build_parser as _build_joint_parser,
)
from .warm_simplex_benchmark import (
    _columns,
    _configuration,
    _paired_ratio_interval,
    _validate_output,
    _write_csv,
)

COMPACT_METHOD = "compact_innovation_completion"
_COMPLETION = "completion_only"
_END_TO_END = "end_to_end"
_MODAL_ORDERS = tuple(permutations((BATCHED_METHOD, COMPACT_METHOD)))
_END_TO_END_ORDERS = tuple(permutations((BATCHED_METHOD, COMPACT_METHOD, EXACT_METHOD)))
_SUPPORT_TOLERANCE = 1.0e-12
_STORAGE_MAXIMUM_RATIO = 0.25
_NONINFERIORITY_MAXIMUM_RATIO = 1.10
_EXACT_TARGET_RATIO = 0.95
_THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)
_RunTrajectory = Callable[[np.random.Generator], InnovationConditionedCompletionResult]


@dataclass(frozen=True, slots=True)
class CompactRuntimeBenchmarkSpec(JointInnovationBenchmarkSpec):
    """Full-field paired timings and independent cold prepared setups."""

    setup_repeats: int = 4

    def __post_init__(self) -> None:
        JointInnovationBenchmarkSpec.__post_init__(self)
        if self.setup_repeats <= 0 or self.setup_repeats % len(_MODAL_ORDERS):
            raise ValueError("setup repeats must balance both method orders")


def _prepare_case(
    spec: CompactRuntimeBenchmarkSpec,
    case: CompactRuntimeBenchmarkCase,
    config: DiffusionConfig,
) -> tuple[
    InnovationConditionedModalCompleter,
    PreparedInnovationCompletion,
    PreparedInnovationCompletion,
    list[dict[str, object]],
    dict[str, object],
]:
    options = {
        "retained_modes": case.retained_modes,
        "conditional_rank": case.completion_rank,
    }
    samples: list[dict[str, object]] = []
    for repeat in range(spec.setup_repeats):
        for position, method in enumerate(_MODAL_ORDERS[repeat % len(_MODAL_ORDERS)]):
            gc.collect()
            row: dict[str, object] = {
                "repeat": repeat,
                "execution_position": position,
                "method": method,
                "cold_prepared_setup_seconds": None,
                "builder_setup_seconds": None,
                "batch_preparation_seconds": None,
                "compact_conversion_seconds": None,
            }
            if method == BATCHED_METHOD:
                start = perf_counter()
                builder = InnovationConditionedModalCompleter(config, **options)
                built = perf_counter()
                builder.prepare_batched_sampler()
                packed = perf_counter()
                row["cold_prepared_setup_seconds"] = packed - start
                row["builder_setup_seconds"] = built - start
                row["batch_preparation_seconds"] = packed - built

                start = perf_counter()
                converted = builder.prepare_runtime()
                row["compact_conversion_seconds"] = perf_counter() - start
            else:
                start = perf_counter()
                runtime = PreparedInnovationCompletion.from_full_history_config(config, **options)
                row["cold_prepared_setup_seconds"] = perf_counter() - start
            samples.append(row)
        if repeat + 1 < spec.setup_repeats:
            del builder, runtime, converted

    # The factory's builder has gone out of scope. Count the graph reachable
    # from each retained object and charge NumPy views to their backing arrays.
    gc.collect()
    builder_bytes = deep_numpy_nbytes(builder, include_array_bases=True)
    compact_bytes = deep_numpy_nbytes(runtime, include_array_bases=True)
    batch_samples = [row for row in samples if row["method"] == BATCHED_METHOD]
    compact_samples = [row for row in samples if row["method"] == COMPACT_METHOD]

    def median(rows: list[dict[str, object]], key: str) -> float:
        return float(np.median([float(row[key]) for row in rows]))

    resources = {
        "builder_setup_seconds": median(batch_samples, "builder_setup_seconds"),
        "batch_preparation_seconds": median(batch_samples, "batch_preparation_seconds"),
        "batched_cold_prepared_setup_seconds": median(batch_samples, "cold_prepared_setup_seconds"),
        "compact_conversion_seconds": median(batch_samples, "compact_conversion_seconds"),
        "compact_cold_prepared_setup_seconds": median(
            compact_samples, "cold_prepared_setup_seconds"
        ),
        "batched_retained_array_bytes": builder_bytes,
        "compact_retained_array_bytes": compact_bytes,
        "compact_reported_array_bytes": runtime.nbytes,
        "packed_sampler_array_bytes": runtime.sampler.nbytes,
        "compact_over_batched_retained_array_ratio": compact_bytes / builder_bytes,
    }
    return builder, runtime, converted, samples, resources


def _release_references(
    builder: InnovationConditionedModalCompleter,
) -> dict[str, tuple[ReferenceType[object], ...]]:
    reference = builder.full_reference
    initial_modal_base = builder._initial_modal.base
    return {
        "builder_released": (ref(builder),),
        "full_reference_released": (ref(reference),),
        "full_reference_histories_released": (
            ref(reference._single_particle_probabilities),
            ref(reference._modal_means),
            ref(reference._modal_covariances),
            ref(reference._innovation_covariances),
            ref(reference._innovation_factors),
        ),
        "target_innovation_history_released": (ref(builder._target_innovation_covariances),),
        "initial_modal_backing_released": (
            (ref(initial_modal_base),) if isinstance(initial_modal_base, np.ndarray) else ()
        ),
        "conditioning_history_released": tuple(
            ref(array)
            for step in builder.conditioning_steps
            for array in (
                step.retained_covariance,
                step.retained_factor,
                step.conditional_gain,
                step.conditional_covariance,
                step.conditional_factor,
            )
        ),
    }


def _diagnose_paths(
    spec: CompactRuntimeBenchmarkSpec,
    config: DiffusionConfig,
    run_reference: _RunTrajectory,
    run_candidate: _RunTrajectory,
) -> dict[str, object]:
    maximum_errors = dict.fromkeys(("modal", "spatial", "physical_count"), 0.0)
    exact_matches = dict.fromkeys(maximum_errors, 0)
    maximum_raw_mass_error = 0.0
    maximum_physical_mass_error = 0.0
    minimum_physical_count = float("inf")
    strict_support_mismatches = 0
    tolerant_support_mismatches = 0
    matching_rng_states = 0
    for repeat in range(spec.diagnostic_trajectories):
        sequence = np.random.SeedSequence([spec.seed, config.n_nodes, 2, repeat])
        reference_rng = np.random.default_rng(sequence)
        candidate_rng = np.random.default_rng(sequence)
        reference = run_reference(reference_rng)
        candidate = run_candidate(candidate_rng)
        matching_rng_states += int(
            reference_rng.bit_generator.state == candidate_rng.bit_generator.state
        )
        physical_counts = []
        for result in (reference, candidate):
            raw = result.spatial_counts
            _validate_spatial(raw, config, physical=False)
            counts = compiled_warm_simplex_bank_counts(raw, config.n_particles)
            _validate_output(counts, config)
            maximum_raw_mass_error = max(
                maximum_raw_mass_error,
                float(np.max(np.abs(raw.sum(axis=1) - config.n_particles))),
            )
            maximum_physical_mass_error = max(
                maximum_physical_mass_error,
                float(np.max(np.abs(counts.sum(axis=1) - config.n_particles))),
            )
            minimum_physical_count = min(minimum_physical_count, float(np.min(counts)))
            physical_counts.append(counts)

        reference_counts, candidate_counts = physical_counts
        for label, left, right in (
            ("modal", reference.modal_amplitudes, candidate.modal_amplitudes),
            ("spatial", reference.spatial_counts, candidate.spatial_counts),
            ("physical_count", reference_counts, candidate_counts),
        ):
            maximum_errors[label] = max(maximum_errors[label], float(np.max(np.abs(left - right))))
            exact_matches[label] += int(np.array_equal(left, right))

        support_mismatch = (reference_counts > 0) != (candidate_counts > 0)
        strict_support_mismatches += int(np.count_nonzero(support_mismatch))
        tolerant_support_mismatches += int(
            np.count_nonzero(
                support_mismatch
                & (np.maximum(reference_counts, candidate_counts) > _SUPPORT_TOLERANCE)
            )
        )

    return {
        "diagnostic_trajectories": spec.diagnostic_trajectories,
        "matching_rng_states": matching_rng_states,
        **{
            f"{label}_maximum_absolute_difference": error
            for label, error in maximum_errors.items()
        },
        **{f"{label}_exact_match_trajectories": count for label, count in exact_matches.items()},
        "raw_maximum_mass_error": maximum_raw_mass_error,
        "physical_maximum_mass_error": maximum_physical_mass_error,
        "physical_minimum_count": minimum_physical_count,
        "strict_support_mismatch_entries": strict_support_mismatches,
        "support_mismatch_entries_above_tolerance": tolerant_support_mismatches,
        "correctness_pass": (
            all(count == spec.diagnostic_trajectories for count in exact_matches.values())
            and matching_rng_states == spec.diagnostic_trajectories
            and tolerant_support_mismatches == 0
            and minimum_physical_count >= 0.0
        ),
    }


def _resident_bytes() -> tuple[int | None, int | None]:
    try:
        status = Path("/proc/self/status").read_text(encoding="utf-8")
    except OSError:
        return None, None
    values = {
        name: int(value.split()[0]) * 1024
        for name, _, value in (line.partition(":") for line in status.splitlines())
        if name in {"VmRSS", "VmHWM"}
    }
    return values.get("VmRSS"), values.get("VmHWM")


def _rss_worker() -> None:
    """Measure one prepared setup in a fresh process."""

    request = json.load(sys.stdin)
    config = DiffusionConfig(**request["config"])
    options = request["options"]

    def prepare() -> InnovationConditionedModalCompleter | PreparedInnovationCompletion:
        if request["method"] == BATCHED_METHOD:
            builder = InnovationConditionedModalCompleter(config, **options)
            builder.prepare_batched_sampler()
            return builder
        return PreparedInnovationCompletion.from_full_history_config(config, **options)

    _print_setup_rss(prepare)


def _print_setup_rss(prepare: Callable[[], object]) -> None:
    """Print process memory before and after preparation while its result is live."""

    gc.collect()
    baseline_rss, _ = _resident_bytes()
    retained = prepare()
    gc.collect()
    post_setup_rss, peak_setup_rss = _resident_bytes()
    del retained

    def increase(value: int | None) -> int | None:
        return None if value is None or baseline_rss is None else value - baseline_rss

    print(json.dumps({
        "rss_baseline_bytes": baseline_rss,
        "rss_after_setup_bytes": post_setup_rss,
        "rss_peak_through_setup_bytes": peak_setup_rss,
        "rss_after_setup_increase_bytes": increase(post_setup_rss),
        "rss_peak_through_setup_increase_bytes": increase(peak_setup_rss),
    }))


def _isolated_rss(
    config: DiffusionConfig,
    case: CompactRuntimeBenchmarkCase,
    *,
    methods: tuple[tuple[str, str], ...] = (
        ("batched", BATCHED_METHOD),
        ("compact", COMPACT_METHOD),
    ),
    worker_module: str = "eigendiffusion.compact_runtime_benchmark",
) -> dict[str, object]:
    if not Path("/proc/self/status").is_file():
        return {}
    environment = os.environ.copy()
    source = str(Path(__file__).resolve().parents[1])
    paths = (source, environment.get("PYTHONPATH", ""))
    environment["PYTHONPATH"] = os.pathsep.join(path for path in paths if path)
    command = [
        sys.executable,
        "-c",
        f"from {worker_module} import _rss_worker; _rss_worker()",
    ]
    resources: dict[str, object] = {}
    for label, method in methods:
        request = {
            "config": asdict(config),
            "options": {
                "retained_modes": case.retained_modes,
                "conditional_rank": case.completion_rank,
            },
            "method": method,
        }
        result = subprocess.run(
            command, input=json.dumps(request), capture_output=True, text=True, env=environment
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"{label} RSS worker exited with code {result.returncode}: {result.stderr}"
            )
        resources.update({
            f"{label}_{name}": value for name, value in json.loads(result.stdout).items()
        })
    return resources


def _time_trajectories(
    spec: CompactRuntimeBenchmarkSpec,
    config: DiffusionConfig,
    samplers: dict[str, _RunTrajectory],
) -> list[dict[str, object]]:
    protocols = (
        (
            _COMPLETION,
            tuple(permutations(samplers)),
            spec.completion_warmups,
            spec.completion_repeats,
        ),
        (
            _END_TO_END,
            tuple(permutations((*samplers, EXACT_METHOD))),
            spec.end_to_end_warmups,
            spec.end_to_end_repeats,
        ),
    )
    rows: list[dict[str, object]] = []
    for component_index, (component, orders, warmups, repeats) in enumerate(protocols):
        for repeat in range(-warmups, repeats):
            for position, method in enumerate(orders[repeat % len(orders)]):
                exact = method == EXACT_METHOD
                rng = np.random.default_rng(
                    np.random.SeedSequence([
                        spec.seed, config.n_nodes, component_index, repeat + warmups, int(exact)
                    ])
                )
                start = perf_counter()
                if exact:
                    output = vectorized_multinomial_random_walk_diffusion(config, rng)
                else:
                    output = samplers[method](rng).spatial_counts
                    if component == _END_TO_END:
                        output = compiled_warm_simplex_bank_counts(output, config.n_particles)
                seconds = perf_counter() - start
                if component == _END_TO_END:
                    _validate_output(output, config)
                    if exact and not np.issubdtype(output.dtype, np.integer):
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


def _summarize_timings(
    spec: CompactRuntimeBenchmarkSpec,
    config: DiffusionConfig,
    timing_rows: list[dict[str, object]],
    protocols: tuple[tuple[str, str, str, str], ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[tuple[str, str], float]]:
    groups: dict[tuple[str, str], list[float]] = {}
    for row in timing_rows:
        key = str(row["component"]), str(row["method"])
        groups.setdefault(key, []).append(float(row["seconds"]))
    samples = {key: np.asarray(seconds) for key, seconds in groups.items()}
    medians = {key: float(np.median(seconds)) for key, seconds in samples.items()}
    timings = [
        {
            "n_nodes": config.n_nodes,
            "component": component,
            "method": method,
            "samples": int(seconds.size),
            "median_seconds": medians[component, method],
            "mean_seconds": float(np.mean(seconds)),
            "std_seconds": float(np.std(seconds)),
        }
        for (component, method), seconds in samples.items()
    ]

    comparisons = []
    for offset, (component, baseline, candidate, label) in enumerate(protocols):
        interval = _paired_ratio_interval(
            samples[component, baseline],
            samples[component, candidate],
            n_resamples=spec.bootstrap_resamples,
            seed=spec.seed + config.n_nodes * len(protocols) + offset,
        )
        comparisons.append(
            {
                "n_nodes": config.n_nodes,
                "component": component,
                "comparison": label,
                "baseline_method": baseline,
                "candidate_method": candidate,
                **interval,
                "speedup": 1.0 / interval["median_ratio"],
            }
        )
    return timings, comparisons, medians


def _summarize_case(
    spec: CompactRuntimeBenchmarkSpec,
    config: DiffusionConfig,
    timing_rows: list[dict[str, object]],
    resources: dict[str, object],
    diagnostics: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    protocols = (
        (_COMPLETION, BATCHED_METHOD, COMPACT_METHOD, "compact_over_batched"),
        (_END_TO_END, BATCHED_METHOD, COMPACT_METHOD, "compact_over_batched"),
        (_END_TO_END, EXACT_METHOD, COMPACT_METHOD, "compact_over_exact"),
        (_END_TO_END, EXACT_METHOD, BATCHED_METHOD, "batched_over_exact"),
    )
    timings, comparisons, medians = _summarize_timings(spec, config, timing_rows, protocols)
    completion, end_to_end, exact, _ = comparisons
    gates = {
        "n_nodes": config.n_nodes,
        "correctness_pass": diagnostics["correctness_pass"],
        "builder_release_pass": diagnostics["builder_release_pass"],
        "storage_reduction_pass": (
            float(resources["compact_over_batched_retained_array_ratio"]) <= _STORAGE_MAXIMUM_RATIO
        ),
        "completion_noninferiority_pass": (
            completion["ratio_ci_upper"] <= _NONINFERIORITY_MAXIMUM_RATIO
        ),
        "end_to_end_noninferiority_pass": (
            end_to_end["ratio_ci_upper"] <= _NONINFERIORITY_MAXIMUM_RATIO
        ),
        "exact_crossover_pass": (
            exact["median_ratio"] <= _EXACT_TARGET_RATIO and exact["ratio_ci_upper"] < 1.0
        ),
    }
    batched_setup = float(resources["batched_cold_prepared_setup_seconds"])
    compact_setup = float(resources["compact_cold_prepared_setup_seconds"])
    conversion = float(resources["compact_conversion_seconds"])
    exact_seconds = medians[_END_TO_END, EXACT_METHOD]
    batched_seconds = medians[_END_TO_END, BATCHED_METHOD]
    compact_seconds = medians[_END_TO_END, COMPACT_METHOD]
    resources.update(
        {
            "compact_vs_batched_cold_break_even_trajectories": _break_even(
                max(0.0, compact_setup - batched_setup), batched_seconds, compact_seconds
            ),
            "compact_conversion_vs_batched_break_even_trajectories": _break_even(
                conversion, batched_seconds, compact_seconds
            ),
            "compact_vs_exact_break_even_trajectories": _break_even(
                compact_setup, exact_seconds, compact_seconds
            ),
            "batched_vs_exact_break_even_trajectories": _break_even(
                batched_setup, exact_seconds, batched_seconds
            ),
        }
    )
    return timings, comparisons, gates


def run_compact_runtime_benchmark(
    spec: CompactRuntimeBenchmarkSpec, output_dir: str | Path
) -> dict[str, object]:
    """Measure setup, retained storage, trajectory timings, and path agreement."""

    start = perf_counter()
    if not compiled_warm_simplex_available():
        raise RuntimeError("the compact runtime benchmark requires an available Numba backend")
    compile_warm_simplex_kernel()
    compiler_startup_seconds = perf_counter() - start

    timing_rows: list[dict[str, object]] = []
    setup_rows: list[dict[str, object]] = []
    resources: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    timings: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    gates: list[dict[str, object]] = []
    for case in spec.cases:
        config = _configuration(spec, case)
        columns = _columns(config, case)
        builder, runtime, converted, case_setups, case_resources = _prepare_case(
            spec, case, config
        )
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            case_samples = _time_trajectories(
                spec, config, {BATCHED_METHOD: builder.run_batched, COMPACT_METHOD: runtime.run}
            )
        finally:
            if gc_was_enabled:
                gc.enable()
        case_diagnostics = _diagnose_paths(spec, config, builder.run_batched, runtime.run)
        release_references = _release_references(builder)
        del builder
        gc.collect()
        release_checks = {
            name: all(reference() is None for reference in references)
            for name, references in release_references.items()
        }
        converted_bytes = deep_numpy_nbytes(converted, include_array_bases=True)
        case_resources["converted_retained_array_bytes_after_builder_release"] = converted_bytes
        case_diagnostics.update(release_checks)
        case_diagnostics["builder_release_pass"] = (
            all(release_checks.values())
            and converted_bytes == case_resources["compact_retained_array_bytes"]
            and case_resources["compact_retained_array_bytes"]
            == case_resources["compact_reported_array_bytes"]
        )
        del runtime, converted
        case_resources.update(_isolated_rss(config, case))
        case_timings, case_comparisons, case_gates = _summarize_case(
            spec, config, case_samples, case_resources, case_diagnostics
        )
        setup_rows.extend({**columns, **row} for row in case_setups)
        timing_rows.extend({**columns, **row} for row in case_samples)
        diagnostics.append({**columns, **case_diagnostics})
        resources.append({**columns, **case_resources})
        timings.extend(case_timings)
        comparisons.extend(case_comparisons)
        gates.append(case_gates)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    artifact_rows = {
        "setup_samples": setup_rows,
        "timing_samples": timing_rows,
        "diagnostics": diagnostics,
        "resources": resources,
    }
    artifacts = {}
    for name, rows in artifact_rows.items():
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
            "setup_order_period": len(_MODAL_ORDERS),
            "completion_order_period": len(_MODAL_ORDERS),
            "end_to_end_order_period": len(_END_TO_END_ORDERS),
            "paired_bootstrap_confidence_level": 0.95,
            "requires_exact_modal_spatial_and_physical_paths": True,
            "support_zero_tolerance": _SUPPORT_TOLERANCE,
            "compact_over_batched_maximum_retained_storage_ratio": _STORAGE_MAXIMUM_RATIO,
            "compact_over_batched_maximum_ratio_ci_upper": _NONINFERIORITY_MAXIMUM_RATIO,
            "compact_over_exact_target_median_ratio": _EXACT_TARGET_RATIO,
            "memory": "unique reachable NumPy backing bytes after setup, excluding transients",
            "rss": "Linux process RSS after GC and high-water RSS through isolated setup",
            "break_even": "ceil(incremental setup / end-to-end savings) if savings are positive",
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
        "artifacts": artifacts,
    }
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = _build_joint_parser()
    parser.description = __doc__
    parser.set_defaults(output_dir="outputs/compact_runtime_benchmark")
    parser.add_argument(
        "--setup-repeats", type=int, default=CompactRuntimeBenchmarkSpec().setup_repeats
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
        spec = CompactRuntimeBenchmarkSpec(
            cases=tuple(
                CompactRuntimeBenchmarkCase(n, modes, completion_rank) for n in node_counts
            ),
            **options,
        )
        report = run_compact_runtime_benchmark(spec, output_dir)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps({"gates": report["gates"], "artifacts": report["artifacts"]}, indent=2))
    return int(
        any(
            not row["correctness_pass"] or not row["builder_release_pass"]
            for row in report["gates"]
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
