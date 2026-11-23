"""Component-level benchmark for the full-field completed modal pipeline.

The benchmark deliberately leaves the model mathematics unchanged.  It times
construction, raw dynamics, persistent completion, and physical readout as
separate stages, while also retaining an independently timed end-to-end run.
Every measured pipeline materializes the complete time-by-node field.
"""

from __future__ import annotations

import csv
import gc
import json
import os
import selectors
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal, Sequence

import numpy as np

from .benchmarking import (
    SELECTED_OPERATING_POINTS,
    OperatingPoint,
    deep_numpy_nbytes,
    stable_diffusion_config,
)
from .completion import PersistentUnresolvedGaussianCompleter
from .config import DiffusionConfig
from .correlated_modal import HandoffCorrelatedModalDiffusion
from .readouts import (
    ReadoutResult,
    readout_constraint_diagnostics,
    simplex_bank_readout,
    temporal_delta_sigma_readout,
)

ComponentReadoutName = Literal["simplex_bank", "delta_sigma_temporal"]
COMPONENT_READOUT_NAMES: tuple[ComponentReadoutName, ...] = (
    "simplex_bank",
    "delta_sigma_temporal",
)

ComponentBenchmarkRegime = Literal["fixed_steps", "fixed_duration"]
COMPONENT_BENCHMARK_REGIMES: tuple[ComponentBenchmarkRegime, ...] = (
    "fixed_steps",
    "fixed_duration",
)

ComponentName = Literal[
    "simulator_setup",
    "completion_setup",
    "setup_total",
    "raw_dynamics",
    "completion_application",
    "physical_readout",
    "trajectory_total",
]


@dataclass(frozen=True, slots=True)
class FullFieldComponentPipeline:
    """Reusable simulator and completion plan for one physical configuration."""

    config: DiffusionConfig
    simulator: HandoffCorrelatedModalDiffusion
    completer: PersistentUnresolvedGaussianCompleter


@dataclass(frozen=True, slots=True)
class ComponentTimingSample:
    """One wall-clock sample for one benchmark stage."""

    component: ComponentName
    repeat: int
    seconds: float
    readout: str = ""

    def as_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FullFieldComponentBenchmark:
    """All component samples and retained-memory diagnostics for one pipeline."""

    config: DiffusionConfig
    regime: ComponentBenchmarkRegime
    retained_modes: int
    handoff_time: float
    completion_rank: int
    completion_ridge: float
    readouts: tuple[ComponentReadoutName, ...]
    setup_repeats: int
    trajectory_repeats: int
    warmup_runs: int
    seed: int
    samples: tuple[ComponentTimingSample, ...]
    setup_array_bytes: int
    resident_array_bytes: int
    output_array_bytes: int
    output_dtype: str
    constraint_diagnostics: dict[str, dict[str, float]]

    def sample_rows(self) -> list[dict[str, str | int | float]]:
        """Return long-format timing rows with configuration columns."""

        configuration: dict[str, str | int | float] = {
            "regime": self.regime,
            "output_workload": "full_field_every_timestep",
            "n_particles": self.config.n_particles,
            "n_nodes": self.config.n_nodes,
            "n_steps": self.config.n_steps,
            "dt": self.config.dt,
            "length": self.config.length,
            "diffusion_coefficient": self.config.diffusion_coefficient,
            "impulse_index": self.config.impulse_index,
            "retained_modes": self.retained_modes,
            "handoff_time": self.handoff_time,
            "completion_rank": self.completion_rank,
            "completion_ridge": self.completion_ridge,
            "setup_repeats": self.setup_repeats,
            "trajectory_repeats": self.trajectory_repeats,
            "warmup_runs": self.warmup_runs,
        }
        return [configuration | sample.as_dict() for sample in self.samples]

    def summary(self) -> dict[str, object]:
        """Summarize timing distributions, accounting, and amortized totals."""

        component_summaries: dict[str, dict[str, float | int]] = {}
        keys = sorted({(sample.component, sample.readout) for sample in self.samples})
        for component, readout in keys:
            values = np.asarray(
                [
                    sample.seconds
                    for sample in self.samples
                    if sample.component == component and sample.readout == readout
                ],
                dtype=float,
            )
            label = component if not readout else f"{component}:{readout}"
            component_summaries[label] = {
                "samples": int(values.size),
                "median_seconds": float(np.median(values)),
                "mean_seconds": float(np.mean(values)),
                "std_seconds": float(np.std(values, ddof=0)),
                "min_seconds": float(np.min(values)),
                "max_seconds": float(np.max(values)),
            }

        def median(component: str, readout: str = "") -> float:
            label = component if not readout else f"{component}:{readout}"
            return float(component_summaries[label]["median_seconds"])

        setup_seconds = median("setup_total")
        raw_seconds = median("raw_dynamics")
        completion_seconds = median("completion_application")
        readout_summaries: dict[str, dict[str, object]] = {}
        for readout in self.readouts:
            readout_seconds = median("physical_readout", readout)
            total_seconds = median("trajectory_total", readout)
            accounted = raw_seconds + completion_seconds + readout_seconds
            readout_summaries[readout] = {
                "component_median_seconds": {
                    "raw_dynamics": raw_seconds,
                    "completion_application": completion_seconds,
                    "physical_readout": readout_seconds,
                },
                "accounted_trajectory_seconds": accounted,
                "measured_end_to_end_seconds": total_seconds,
                "accounted_to_end_to_end_ratio": (
                    accounted / total_seconds if total_seconds > 0.0 else 0.0
                ),
                "amortized_seconds_per_trajectory": {
                    str(count): total_seconds + setup_seconds / count
                    for count in (1, 10, 100, 1_000)
                },
                "constraints": self.constraint_diagnostics[readout],
            }

        return {
            "configuration": {
                "n_particles": self.config.n_particles,
                "n_nodes": self.config.n_nodes,
                "n_steps": self.config.n_steps,
                "dt": self.config.dt,
                "total_time": float(self.config.times[-1]),
                "length": self.config.length,
                "diffusion_coefficient": self.config.diffusion_coefficient,
                "impulse_index": self.config.impulse_index,
                "retained_modes": self.retained_modes,
                "handoff_time": self.handoff_time,
                "completion_rank": self.completion_rank,
                "completion_ridge": self.completion_ridge,
                "readouts": list(self.readouts),
                "setup_repeats": self.setup_repeats,
                "trajectory_repeats": self.trajectory_repeats,
                "warmup_runs": self.warmup_runs,
                "seed": self.seed,
                "regime": self.regime,
                "output_workload": "full_field_every_timestep",
            },
            "components": component_summaries,
            "readouts": readout_summaries,
            "memory": {
                "setup_array_bytes": self.setup_array_bytes,
                "setup_array_megabytes": self.setup_array_bytes / 1024**2,
                "resident_array_bytes": self.resident_array_bytes,
                "resident_array_megabytes": self.resident_array_bytes / 1024**2,
                "one_output_array_bytes": self.output_array_bytes,
                "one_output_array_megabytes": self.output_array_bytes / 1024**2,
                "output_dtype": self.output_dtype,
            },
        }


@dataclass(frozen=True, slots=True)
class ComponentMemoryRecord:
    """Fresh-process memory measurements for one component configuration."""

    regime: ComponentBenchmarkRegime
    operating_point: str
    n_particles: int
    n_nodes: int
    n_steps: int
    dt: float
    total_time: float
    retained_modes: int
    handoff_time: float
    completion_rank: int
    readout: ComponentReadoutName
    baseline_rss_bytes: int
    baseline_peak_rss_bytes: int
    setup_peak_rss_bytes: int
    steady_setup_rss_bytes: int
    process_peak_after_setup_rss_bytes: int
    trajectory_peak_rss_bytes: int
    process_peak_after_trajectory_rss_bytes: int
    final_rss_bytes: int
    setup_array_bytes: int
    resident_array_bytes: int
    output_array_bytes: int
    returned_result_array_bytes: int
    setup_phase_samples: int
    trajectory_phase_samples: int
    sampler_interval_ms: float

    @property
    def setup_peak_incremental_bytes(self) -> int:
        return max(0, self.setup_peak_rss_bytes - self.baseline_rss_bytes)

    @property
    def steady_setup_incremental_bytes(self) -> int:
        return max(0, self.steady_setup_rss_bytes - self.baseline_rss_bytes)

    @property
    def trajectory_peak_over_setup_bytes(self) -> int:
        return max(0, self.trajectory_peak_rss_bytes - self.steady_setup_rss_bytes)

    @property
    def setup_hwm_incremental_bytes(self) -> int:
        return max(
            0,
            self.process_peak_after_setup_rss_bytes - self.baseline_peak_rss_bytes,
        )

    @property
    def trajectory_hwm_incremental_bytes(self) -> int:
        return max(
            0,
            self.process_peak_after_trajectory_rss_bytes
            - self.process_peak_after_setup_rss_bytes,
        )

    def as_dict(self) -> dict[str, str | int | float]:
        row: dict[str, str | int | float] = asdict(self)
        row["setup_peak_incremental_bytes"] = self.setup_peak_incremental_bytes
        row["steady_setup_incremental_bytes"] = self.steady_setup_incremental_bytes
        row["trajectory_peak_over_setup_bytes"] = self.trajectory_peak_over_setup_bytes
        row["setup_hwm_incremental_bytes"] = self.setup_hwm_incremental_bytes
        row["trajectory_hwm_incremental_bytes"] = (
            self.trajectory_hwm_incremental_bytes
        )
        for key, value in tuple(row.items()):
            if key.endswith("_bytes"):
                row[key.removesuffix("_bytes") + "_megabytes"] = int(value) / 1024**2
        return row


@dataclass(frozen=True, slots=True)
class ComponentScalingBenchmark:
    """Two-regime timing and isolated-memory component scaling study."""

    operating_point: str
    benchmarks: tuple[FullFieldComponentBenchmark, ...]
    memory_records: tuple[ComponentMemoryRecord, ...]

    def summary(self) -> dict[str, object]:
        """Return all configuration summaries plus log-log scaling slopes."""

        slopes: dict[str, dict[str, float | None]] = {}
        for regime in COMPONENT_BENCHMARK_REGIMES:
            selected = sorted(
                (item for item in self.benchmarks if item.regime == regime),
                key=lambda item: item.config.n_nodes,
            )
            if len(selected) < 2:
                continue
            metrics: dict[str, list[float]] = {
                "setup_total": [],
                "simulator_setup": [],
                "completion_setup": [],
                "raw_dynamics": [],
                "completion_application": [],
                "setup_array_bytes": [],
                "resident_array_bytes": [],
                "output_array_bytes": [],
            }
            for readout in selected[0].readouts:
                metrics[f"physical_readout:{readout}"] = []
                metrics[f"trajectory_total:{readout}"] = []
            for item in selected:
                report = item.summary()
                components = report["components"]
                memory = report["memory"]
                for key in (
                    "setup_total",
                    "simulator_setup",
                    "completion_setup",
                    "raw_dynamics",
                    "completion_application",
                ):
                    metrics[key].append(float(components[key]["median_seconds"]))
                metrics["setup_array_bytes"].append(
                    float(memory["setup_array_bytes"])
                )
                metrics["resident_array_bytes"].append(
                    float(memory["resident_array_bytes"])
                )
                metrics["output_array_bytes"].append(
                    float(memory["one_output_array_bytes"])
                )
                for readout in item.readouts:
                    for component in ("physical_readout", "trajectory_total"):
                        label = f"{component}:{readout}"
                        metrics[label].append(
                            float(components[label]["median_seconds"])
                        )
            nodes = [float(item.config.n_nodes) for item in selected]
            slopes[regime] = {
                name: _log_log_slope(nodes, values)
                for name, values in metrics.items()
            }

            for readout in selected[0].readouts:
                regime_memory = sorted(
                    (
                        item
                        for item in self.memory_records
                        if item.regime == regime and item.readout == readout
                    ),
                    key=lambda item: item.n_nodes,
                )
                if len(regime_memory) < 2:
                    continue
                memory_nodes = [float(item.n_nodes) for item in regime_memory]
                slopes[regime].update(
                    {
                        f"isolated_setup_peak_increment:{readout}": _log_log_slope(
                            memory_nodes,
                            [
                                float(item.setup_peak_incremental_bytes)
                                for item in regime_memory
                            ],
                        ),
                        f"isolated_steady_setup_increment:{readout}": (
                            _log_log_slope(
                                memory_nodes,
                                [
                                    float(item.steady_setup_incremental_bytes)
                                    for item in regime_memory
                                ],
                            )
                        ),
                        f"isolated_trajectory_peak_over_setup:{readout}": (
                            _log_log_slope(
                                memory_nodes,
                                [
                                    float(item.trajectory_peak_over_setup_bytes)
                                    for item in regime_memory
                                ],
                            )
                        ),
                    }
                )
        return {
            "operating_point": self.operating_point,
            "study_contract": {
                "output_workload": "full_field_every_timestep",
                "operating_point_parameters_scale_with_nodes": True,
                "fixed_steps": (
                    "fixed update count with a stability-limited timestep; "
                    "physical duration may change"
                ),
                "fixed_duration": (
                    "fixed physical duration with a stability-limited timestep; "
                    "update count may change"
                ),
                "isolated_memory": (
                    "fresh process per configuration and readout; phase RSS is "
                    "sampled separately from timing"
                ),
            },
            "runtime_environment": {
                name: os.environ.get(name)
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "configurations": [item.summary() for item in self.benchmarks],
            "memory_records": [item.as_dict() for item in self.memory_records],
            "scaling_slopes": slopes,
        }


def _log_log_slope(
    x_values: Sequence[float],
    y_values: Sequence[float],
) -> float | None:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    positive = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if np.count_nonzero(positive) < 2:
        return None
    return float(np.polyfit(np.log(x[positive]), np.log(y[positive]), 1)[0])


def _validate_arguments(
    config: DiffusionConfig,
    *,
    retained_modes: int,
    handoff_time: float,
    completion_rank: int,
    setup_repeats: int,
    trajectory_repeats: int,
    warmup_runs: int,
    readouts: Sequence[str],
) -> tuple[ComponentReadoutName, ...]:
    if not 1 <= retained_modes <= config.n_nodes:
        raise ValueError("retained_modes must lie in [1, n_nodes]")
    if not config.times[0] <= handoff_time <= config.times[-1]:
        raise ValueError("handoff_time must lie within the simulated time range")
    if completion_rank <= 0:
        raise ValueError("completion_rank must be positive")
    if setup_repeats <= 0 or trajectory_repeats <= 0:
        raise ValueError("setup_repeats and trajectory_repeats must be positive")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be nonnegative")
    if not readouts:
        raise ValueError("at least one readout is required")
    unknown = set(readouts) - set(COMPONENT_READOUT_NAMES)
    if unknown:
        raise ValueError(f"unknown component readouts: {sorted(unknown)}")
    return tuple(dict.fromkeys(readouts))  # type: ignore[return-value]


def _construct_simulator(
    config: DiffusionConfig,
    *,
    retained_modes: int,
    handoff_time: float,
) -> HandoffCorrelatedModalDiffusion:
    return HandoffCorrelatedModalDiffusion(
        config=config,
        initial_n_modes=config.n_nodes,
        final_n_modes=retained_modes,
        handoff_time=handoff_time,
    )


def _construct_completer(
    config: DiffusionConfig,
    simulator: HandoffCorrelatedModalDiffusion,
    *,
    retained_modes: int,
    completion_rank: int,
    completion_ridge: float,
) -> PersistentUnresolvedGaussianCompleter:
    return PersistentUnresolvedGaussianCompleter(
        config,
        retained_modes=retained_modes,
        completion_start_time=simulator.handoff_time,
        completion_rank=completion_rank,
        ridge=completion_ridge,
    )


def build_full_field_component_pipeline(
    config: DiffusionConfig,
    *,
    retained_modes: int,
    handoff_time: float,
    completion_rank: int,
    completion_ridge: float = 1.0e-2,
) -> FullFieldComponentPipeline:
    """Construct the unmodified handoff plus persistent-completion pipeline."""

    simulator = _construct_simulator(
        config,
        retained_modes=retained_modes,
        handoff_time=handoff_time,
    )
    completer = _construct_completer(
        config,
        simulator,
        retained_modes=retained_modes,
        completion_rank=completion_rank,
        completion_ridge=completion_ridge,
    )
    return FullFieldComponentPipeline(
        config=config,
        simulator=simulator,
        completer=completer,
    )


def _apply_readout(
    completed: np.ndarray,
    total_mass: int,
    readout: ComponentReadoutName,
) -> ReadoutResult:
    if readout == "simplex_bank":
        return simplex_bank_readout(completed, total_mass)
    if readout == "delta_sigma_temporal":
        return temporal_delta_sigma_readout(completed, total_mass)
    raise ValueError(f"unknown component readout: {readout}")


def validate_full_field_output(
    counts: np.ndarray,
    config: DiffusionConfig,
    *,
    require_integer: bool = False,
) -> None:
    """Validate the common full-field physical output contract."""

    values = np.asarray(counts)
    if values.shape != (config.n_steps, config.n_nodes):
        raise RuntimeError("component benchmark returned an unexpected shape")
    if not np.all(np.isfinite(values)):
        raise RuntimeError("component benchmark returned nonfinite values")
    if np.min(values) < -1.0e-10:
        raise RuntimeError("component benchmark returned negative physical counts")
    if not np.allclose(
        values.sum(axis=1),
        config.n_particles,
        atol=1.0e-7,
        rtol=0.0,
    ):
        raise RuntimeError("component benchmark failed mass conservation")
    if require_integer and not np.allclose(values, np.rint(values), atol=1.0e-9):
        raise RuntimeError("integer component readout returned noninteger counts")


def run_full_field_component_pipeline(
    pipeline: FullFieldComponentPipeline,
    *,
    readout: ComponentReadoutName,
    seed: int | Sequence[int] = 0,
) -> ReadoutResult:
    """Run one complete full-field trajectory without component instrumentation."""

    simulation_sequence, completion_sequence = np.random.SeedSequence(seed).spawn(2)
    result = _execute_full_field_component_pipeline(
        pipeline,
        readout=readout,
        simulation_sequence=simulation_sequence,
        completion_sequence=completion_sequence,
    )
    validate_full_field_output(
        result.counts,
        pipeline.config,
        require_integer=readout == "delta_sigma_temporal",
    )
    return result


def _execute_full_field_component_pipeline(
    pipeline: FullFieldComponentPipeline,
    *,
    readout: ComponentReadoutName,
    simulation_sequence: np.random.SeedSequence,
    completion_sequence: np.random.SeedSequence,
) -> ReadoutResult:
    """Execute one pipeline with preassigned random streams and no validation."""

    raw = pipeline.simulator.run(rng=np.random.default_rng(simulation_sequence))
    completed, _ = pipeline.completer.complete(
        raw.spatial_counts,
        rng=np.random.default_rng(completion_sequence),
    )
    return _apply_readout(completed, pipeline.config.n_particles, readout)


def benchmark_full_field_components(
    config: DiffusionConfig,
    *,
    regime: ComponentBenchmarkRegime = "fixed_steps",
    retained_modes: int = 75,
    handoff_time: float = 30.0,
    completion_rank: int = 5,
    completion_ridge: float = 1.0e-2,
    readouts: Sequence[ComponentReadoutName] = COMPONENT_READOUT_NAMES,
    setup_repeats: int = 5,
    trajectory_repeats: int = 10,
    warmup_runs: int = 1,
    seed: int = 0,
) -> FullFieldComponentBenchmark:
    """Measure setup, trajectory, completion, and full-field readout costs."""

    if regime not in COMPONENT_BENCHMARK_REGIMES:
        raise ValueError(f"unknown component benchmark regime: {regime}")

    selected_readouts = _validate_arguments(
        config,
        retained_modes=retained_modes,
        handoff_time=handoff_time,
        completion_rank=completion_rank,
        setup_repeats=setup_repeats,
        trajectory_repeats=trajectory_repeats,
        warmup_runs=warmup_runs,
        readouts=readouts,
    )
    samples: list[ComponentTimingSample] = []
    pipeline: FullFieldComponentPipeline | None = None

    for repeat in range(setup_repeats):
        if repeat:
            pipeline = None
            del simulator, completer
        gc.collect()
        total_start = perf_counter()
        simulator_start = perf_counter()
        simulator = _construct_simulator(
            config,
            retained_modes=retained_modes,
            handoff_time=handoff_time,
        )
        simulator_seconds = perf_counter() - simulator_start
        completion_start = perf_counter()
        completer = _construct_completer(
            config,
            simulator,
            retained_modes=retained_modes,
            completion_rank=completion_rank,
            completion_ridge=completion_ridge,
        )
        completion_seconds = perf_counter() - completion_start
        total_seconds = perf_counter() - total_start
        samples.extend(
            [
                ComponentTimingSample(
                    "simulator_setup", repeat, simulator_seconds
                ),
                ComponentTimingSample(
                    "completion_setup", repeat, completion_seconds
                ),
                ComponentTimingSample("setup_total", repeat, total_seconds),
            ]
        )
        pipeline = FullFieldComponentPipeline(config, simulator, completer)

    if pipeline is None:  # pragma: no cover - guarded by argument validation
        raise RuntimeError("component benchmark failed to construct a pipeline")

    for warmup in range(warmup_runs):
        for readout in selected_readouts:
            run_full_field_component_pipeline(
                pipeline,
                readout=readout,
                seed=(seed, 0, warmup),
            )

    setup_array_bytes = deep_numpy_nbytes(pipeline)
    resident_array_bytes = setup_array_bytes
    output_array_bytes = 0
    output_dtype = ""
    diagnostic_accumulator: dict[str, list[dict[str, float]]] = {
        readout: [] for readout in selected_readouts
    }

    garbage_collection_enabled = gc.isenabled()
    if garbage_collection_enabled:
        gc.disable()
    try:
        for repeat in range(trajectory_repeats):
            simulation_sequence, completion_sequence = np.random.SeedSequence(
                [seed, 1, repeat]
            ).spawn(2)

            start = perf_counter()
            raw = pipeline.simulator.run(
                rng=np.random.default_rng(simulation_sequence)
            )
            raw_seconds = perf_counter() - start
            samples.append(
                ComponentTimingSample("raw_dynamics", repeat, raw_seconds)
            )

            start = perf_counter()
            completed, additions = pipeline.completer.complete(
                raw.spatial_counts,
                rng=np.random.default_rng(completion_sequence),
            )
            completion_seconds = perf_counter() - start
            samples.append(
                ComponentTimingSample(
                    "completion_application", repeat, completion_seconds
                )
            )

            readout_order = list(selected_readouts)
            if repeat % 2:
                readout_order.reverse()
            for readout in readout_order:
                start = perf_counter()
                physical = _apply_readout(
                    completed,
                    config.n_particles,
                    readout,
                )
                readout_seconds = perf_counter() - start
                validate_full_field_output(
                    physical.counts,
                    config,
                    require_integer=readout == "delta_sigma_temporal",
                )
                samples.append(
                    ComponentTimingSample(
                        "physical_readout",
                        repeat,
                        readout_seconds,
                        readout,
                    )
                )
                diagnostic_accumulator[readout].append(
                    readout_constraint_diagnostics(
                        physical.counts,
                        config.n_particles,
                    )
                )
                output_array_bytes = max(output_array_bytes, physical.counts.nbytes)
                output_dtype = str(physical.counts.dtype)
                resident_array_bytes = max(
                    resident_array_bytes,
                    deep_numpy_nbytes(
                        (pipeline, raw, completed, additions, physical)
                    ),
                )

            total_order = list(selected_readouts)
            if repeat % 2:
                total_order.reverse()
            for readout in total_order:
                total_simulation_sequence, total_completion_sequence = (
                    np.random.SeedSequence([seed, 2, repeat]).spawn(2)
                )
                start = perf_counter()
                physical = _execute_full_field_component_pipeline(
                    pipeline,
                    readout=readout,
                    simulation_sequence=total_simulation_sequence,
                    completion_sequence=total_completion_sequence,
                )
                total_seconds = perf_counter() - start
                validate_full_field_output(
                    physical.counts,
                    config,
                    require_integer=readout == "delta_sigma_temporal",
                )
                samples.append(
                    ComponentTimingSample(
                        "trajectory_total",
                        repeat,
                        total_seconds,
                        readout,
                    )
                )
                resident_array_bytes = max(
                    resident_array_bytes,
                    deep_numpy_nbytes((pipeline, physical)),
                )
    finally:
        if garbage_collection_enabled:
            gc.enable()

    constraints: dict[str, dict[str, float]] = {}
    for readout, diagnostics in diagnostic_accumulator.items():
        constraints[readout] = {
            key: float(max(item[key] for item in diagnostics))
            for key in diagnostics[0]
        }

    return FullFieldComponentBenchmark(
        config=config,
        regime=regime,
        retained_modes=retained_modes,
        handoff_time=float(pipeline.simulator.handoff_time),
        completion_rank=min(completion_rank, config.n_nodes - retained_modes),
        completion_ridge=completion_ridge,
        readouts=selected_readouts,
        setup_repeats=setup_repeats,
        trajectory_repeats=trajectory_repeats,
        warmup_runs=warmup_runs,
        seed=seed,
        samples=tuple(samples),
        setup_array_bytes=setup_array_bytes,
        resident_array_bytes=resident_array_bytes,
        output_array_bytes=output_array_bytes,
        output_dtype=output_dtype,
        constraint_diagnostics=constraints,
    )


def fixed_step_component_config(
    *,
    n_particles: int,
    n_nodes: int,
    n_steps: int,
    requested_dt: float,
    length: float,
    diffusion_coefficient: float,
    impulse_fraction: float,
    stability_safety: float = 0.95,
) -> DiffusionConfig:
    """Build a fixed-update config with the full-rank modal stability bound."""

    if n_steps < 2:
        raise ValueError("n_steps must be at least two")
    if requested_dt <= 0.0:
        raise ValueError("requested_dt must be positive")
    if not 0.0 < stability_safety <= 1.0:
        raise ValueError("stability_safety must lie in (0, 1]")
    if not 0.0 <= impulse_fraction <= 1.0:
        raise ValueError("impulse_fraction must lie in [0, 1]")
    dx = length / (n_nodes - 1)
    maximum_dt = dx**2 / (4.0 * diffusion_coefficient)
    dt = min(requested_dt, stability_safety * maximum_dt)
    return DiffusionConfig(
        n_particles=n_particles,
        n_nodes=n_nodes,
        n_steps=n_steps,
        impulse_index=int(round(impulse_fraction * (n_nodes - 1))),
        dt=dt,
        length=length,
        diffusion_coefficient=diffusion_coefficient,
    )


def _selected_operating_point(name: str) -> OperatingPoint:
    for point in SELECTED_OPERATING_POINTS:
        if point.name == name:
            return point
    allowed = ", ".join(point.name for point in SELECTED_OPERATING_POINTS)
    raise ValueError(f"unknown operating point {name!r}; choose one of: {allowed}")


def _read_process_rss_bytes(process_id: int) -> int:
    try:
        with open(f"/proc/{process_id}/statm", encoding="utf-8") as handle:
            resident_pages = int(handle.read().split()[1])
        return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError):
        return 0


def measure_component_memory(
    config: DiffusionConfig,
    *,
    regime: ComponentBenchmarkRegime,
    operating_point: str,
    retained_modes: int,
    handoff_time: float,
    completion_rank: int,
    completion_ridge: float,
    readout: ComponentReadoutName,
    seed: int,
    sampler_interval_ms: float = 1.0,
    worker_timeout_seconds: float = 300.0,
) -> ComponentMemoryRecord:
    """Measure setup and trajectory RSS in a synchronized fresh subprocess."""

    if sampler_interval_ms <= 0.0:
        raise ValueError("sampler_interval_ms must be positive")
    if worker_timeout_seconds <= 0.0:
        raise ValueError("worker_timeout_seconds must be positive")
    spec = {
        "config": {
            "n_particles": config.n_particles,
            "n_nodes": config.n_nodes,
            "n_steps": config.n_steps,
            "impulse_index": config.impulse_index,
            "dt": config.dt,
            "length": config.length,
            "diffusion_coefficient": config.diffusion_coefficient,
        },
        "retained_modes": retained_modes,
        "handoff_time": handoff_time,
        "completion_rank": completion_rank,
        "completion_ridge": completion_ridge,
        "readout": readout,
        "seed": seed,
    }
    environment = os.environ.copy()
    for variable in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[variable] = "1"
    source_root = str(Path(__file__).resolve().parents[1])
    inherited_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_root, inherited_pythonpath) if value
    )
    stderr_file = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "eigendiffusion.component_memory_worker",
                "--spec",
                json.dumps(spec, separators=(",", ":")),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            text=True,
            bufsize=1,
            env=environment,
        )
    except BaseException:
        stderr_file.close()
        raise
    if process.stdin is None or process.stdout is None:
        process.kill()
        stderr_file.close()
        raise RuntimeError("failed to open component-memory worker pipes")

    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    phase = ""
    phase_peaks = {"setup": 0, "trajectory": 0}
    phase_samples = {"setup": 0, "trajectory": 0}
    messages: dict[str, dict[str, object]] = {}
    deadline = perf_counter() + worker_timeout_seconds

    def acknowledge() -> None:
        process.stdin.write("continue\n")
        process.stdin.flush()

    try:
        while True:
            remaining_seconds = deadline - perf_counter()
            if remaining_seconds <= 0.0:
                raise TimeoutError(
                    "component-memory worker exceeded "
                    f"{worker_timeout_seconds:g} seconds"
                )
            events = selector.select(
                timeout=min(
                    sampler_interval_ms / 1_000.0,
                    remaining_seconds,
                )
            )
            if not events:
                if phase:
                    rss = _read_process_rss_bytes(process.pid)
                    phase_peaks[phase] = max(phase_peaks[phase], rss)
                    phase_samples[phase] += 1
                if process.poll() is not None:
                    break
                continue
            line = process.stdout.readline()
            if not line:
                if process.poll() is not None:
                    break
                continue
            message = json.loads(line)
            event = str(message["event"])
            if event == "error":
                raise RuntimeError(str(message["message"]))
            messages[event] = message
            if event == "setup_start":
                phase = "setup"
                phase_peaks[phase] = max(
                    phase_peaks[phase],
                    _read_process_rss_bytes(process.pid),
                )
            elif event == "setup_end":
                phase_peaks["setup"] = max(
                    phase_peaks["setup"],
                    int(message["rss_bytes"]),
                )
                phase = ""
            elif event == "trajectory_start":
                phase = "trajectory"
                phase_peaks[phase] = max(
                    phase_peaks[phase],
                    _read_process_rss_bytes(process.pid),
                )
            elif event == "trajectory_end":
                phase_peaks["trajectory"] = max(
                    phase_peaks["trajectory"],
                    int(message["rss_bytes"]),
                )
                phase = ""
            if event == "done":
                break
            acknowledge()
        return_code = process.wait(timeout=max(0.1, deadline - perf_counter()))
        stderr_file.seek(0)
        stderr = stderr_file.read()
        if return_code != 0:
            raise RuntimeError(
                f"component-memory worker exited with code {return_code}: {stderr}"
            )
    except BaseException:
        if process.poll() is None:
            process.kill()
        process.wait()
        raise
    finally:
        selector.close()
        process.stdin.close()
        process.stdout.close()
        stderr_file.close()

    required = {"ready", "setup_end", "trajectory_end", "done"}
    missing = required - set(messages)
    if missing:
        raise RuntimeError(f"component-memory worker omitted events: {sorted(missing)}")
    ready = messages["ready"]
    setup = messages["setup_end"]
    trajectory = messages["trajectory_end"]
    return ComponentMemoryRecord(
        regime=regime,
        operating_point=operating_point,
        n_particles=config.n_particles,
        n_nodes=config.n_nodes,
        n_steps=config.n_steps,
        dt=config.dt,
        total_time=float(config.times[-1]),
        retained_modes=retained_modes,
        handoff_time=float(setup["handoff_time"]),
        completion_rank=completion_rank,
        readout=readout,
        baseline_rss_bytes=int(ready["rss_bytes"]),
        baseline_peak_rss_bytes=int(ready["hwm_bytes"]),
        setup_peak_rss_bytes=phase_peaks["setup"],
        steady_setup_rss_bytes=int(setup["rss_bytes"]),
        process_peak_after_setup_rss_bytes=int(setup["hwm_bytes"]),
        trajectory_peak_rss_bytes=phase_peaks["trajectory"],
        process_peak_after_trajectory_rss_bytes=int(trajectory["hwm_bytes"]),
        final_rss_bytes=int(trajectory["rss_bytes"]),
        setup_array_bytes=int(setup["setup_array_bytes"]),
        resident_array_bytes=int(trajectory["resident_array_bytes"]),
        output_array_bytes=int(trajectory["output_array_bytes"]),
        returned_result_array_bytes=int(trajectory["returned_result_array_bytes"]),
        setup_phase_samples=phase_samples["setup"],
        trajectory_phase_samples=phase_samples["trajectory"],
        sampler_interval_ms=sampler_interval_ms,
    )


def benchmark_component_scaling(
    *,
    node_counts: Sequence[int] = (51, 101, 151, 201),
    regimes: Sequence[ComponentBenchmarkRegime] = COMPONENT_BENCHMARK_REGIMES,
    operating_point: str = "accuracy_first",
    n_particles: int = 5_275,
    fixed_steps: int = 101,
    total_time: float = 100.0,
    requested_dt: float = 1.0,
    length: float = 4.0,
    diffusion_coefficient: float = 2.20e-4,
    impulse_fraction: float = 0.59,
    stability_safety: float = 0.95,
    completion_ridge: float = 1.0e-2,
    readouts: Sequence[ComponentReadoutName] = COMPONENT_READOUT_NAMES,
    setup_repeats: int = 5,
    trajectory_repeats: int = 10,
    warmup_runs: int = 1,
    seed: int = 0,
    isolate_memory: bool = True,
    memory_sampler_interval_ms: float = 1.0,
) -> ComponentScalingBenchmark:
    """Benchmark the selected pipeline in fixed-step and fixed-duration regimes."""

    if not node_counts or any(value < 2 for value in node_counts):
        raise ValueError("node_counts must contain integers >= 2")
    unknown_regimes = set(regimes) - set(COMPONENT_BENCHMARK_REGIMES)
    if unknown_regimes:
        raise ValueError(f"unknown component benchmark regimes: {unknown_regimes}")
    point = _selected_operating_point(operating_point)
    benchmarks: list[FullFieldComponentBenchmark] = []
    memories: list[ComponentMemoryRecord] = []
    for regime_index, regime in enumerate(dict.fromkeys(regimes)):
        for n_nodes in sorted(set(int(value) for value in node_counts)):
            if regime == "fixed_steps":
                config = fixed_step_component_config(
                    n_particles=n_particles,
                    n_nodes=n_nodes,
                    n_steps=fixed_steps,
                    requested_dt=requested_dt,
                    length=length,
                    diffusion_coefficient=diffusion_coefficient,
                    impulse_fraction=impulse_fraction,
                    stability_safety=stability_safety,
                )
            else:
                config = stable_diffusion_config(
                    n_particles=n_particles,
                    n_nodes=n_nodes,
                    total_time=total_time,
                    requested_dt=requested_dt,
                    length=length,
                    diffusion_coefficient=diffusion_coefficient,
                    impulse_fraction=impulse_fraction,
                    stability_safety=stability_safety,
                )
            retained_modes, handoff_time, completion_rank = point.scaled(
                n_nodes,
                float(config.times[-1]),
            )
            configuration_seed = seed + regime_index * 1_000_000 + n_nodes * 1_000
            benchmark = benchmark_full_field_components(
                config,
                regime=regime,
                retained_modes=retained_modes,
                handoff_time=handoff_time,
                completion_rank=completion_rank,
                completion_ridge=completion_ridge,
                readouts=readouts,
                setup_repeats=setup_repeats,
                trajectory_repeats=trajectory_repeats,
                warmup_runs=warmup_runs,
                seed=configuration_seed,
            )
            benchmarks.append(benchmark)
            if isolate_memory:
                for readout_index, readout in enumerate(benchmark.readouts):
                    memories.append(
                        measure_component_memory(
                            config,
                            regime=regime,
                            operating_point=operating_point,
                            retained_modes=retained_modes,
                            handoff_time=benchmark.handoff_time,
                            completion_rank=benchmark.completion_rank,
                            completion_ridge=completion_ridge,
                            readout=readout,
                            seed=configuration_seed + readout_index,
                            sampler_interval_ms=memory_sampler_interval_ms,
                        )
                    )
    return ComponentScalingBenchmark(
        operating_point=operating_point,
        benchmarks=tuple(benchmarks),
        memory_records=tuple(memories),
    )


def write_component_benchmark_csv(
    benchmark: FullFieldComponentBenchmark,
    output_path: str | Path,
) -> Path:
    """Write long-format component samples."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = benchmark.sample_rows()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_component_benchmark_summary(
    benchmark: FullFieldComponentBenchmark,
    output_path: str | Path,
) -> Path:
    """Write the aggregate benchmark report as JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(benchmark.summary(), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def write_component_scaling_csv(
    scaling: ComponentScalingBenchmark,
    output_path: str | Path,
) -> Path:
    """Write every raw timing sample from a component scaling study."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        row
        for benchmark in scaling.benchmarks
        for row in benchmark.sample_rows()
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_component_memory_csv(
    scaling: ComponentScalingBenchmark,
    output_path: str | Path,
) -> Path:
    """Write isolated memory records for a component scaling study."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [record.as_dict() for record in scaling.memory_records]
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_component_scaling_summary(
    scaling: ComponentScalingBenchmark,
    output_path: str | Path,
) -> Path:
    """Write timing, memory, and slope summaries as JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(scaling.summary(), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path
