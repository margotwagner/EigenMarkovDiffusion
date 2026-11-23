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
from itertools import permutations
from pathlib import Path
from time import perf_counter
from typing import Literal, Sequence

import numpy as np

from .baselines import (
    deterministic_markov_diffusion,
    multinomial_random_walk_diffusion,
    vectorized_multinomial_random_walk_diffusion,
)
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

StandardComparatorMethod = Literal[
    "aggregated_multinomial",
    "aggregated_multinomial_vectorized",
    "deterministic_markov",
]
STANDARD_COMPARATOR_METHOD = "aggregated_multinomial"
VECTORIZED_STANDARD_COMPARATOR_METHOD = "aggregated_multinomial_vectorized"
DETERMINISTIC_MARKOV_METHOD = "deterministic_markov"
PRIMARY_STANDARD_COMPARATOR_METHOD = VECTORIZED_STANDARD_COMPARATOR_METHOD
STANDARD_COMPARATOR_METHODS: tuple[StandardComparatorMethod, ...] = (
    STANDARD_COMPARATOR_METHOD,
    VECTORIZED_STANDARD_COMPARATOR_METHOD,
    DETERMINISTIC_MARKOV_METHOD,
)
STANDARD_COMPARATOR_VARIANT = "native_integer_counts"
DETERMINISTIC_MARKOV_VARIANT = "expected_real_counts"
STANDARD_COMPARATOR_VARIANTS = {
    STANDARD_COMPARATOR_METHOD: STANDARD_COMPARATOR_VARIANT,
    VECTORIZED_STANDARD_COMPARATOR_METHOD: STANDARD_COMPARATOR_VARIANT,
    DETERMINISTIC_MARKOV_METHOD: DETERMINISTIC_MARKOV_VARIANT,
}
STANDARD_COMPARATOR_SEED_CHANNELS = {
    STANDARD_COMPARATOR_METHOD: 3,
    VECTORIZED_STANDARD_COMPARATOR_METHOD: 4,
}

ComponentName = Literal[
    "simulator_setup",
    "completion_setup",
    "setup_total",
    "raw_dynamics",
    "completion_application",
    "physical_readout",
    "trajectory_total",
]


def _run_standard_comparator(
    method: str,
    config: DiffusionConfig,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if method == STANDARD_COMPARATOR_METHOD:
        if rng is None:
            raise ValueError("aggregated multinomial comparator requires an RNG")
        return multinomial_random_walk_diffusion(config, rng=rng)
    if method == VECTORIZED_STANDARD_COMPARATOR_METHOD:
        if rng is None:
            raise ValueError("vectorized multinomial comparator requires an RNG")
        return vectorized_multinomial_random_walk_diffusion(config, rng=rng)
    if method == DETERMINISTIC_MARKOV_METHOD:
        return deterministic_markov_diffusion(config)
    raise ValueError(f"unknown standard comparator method: {method}")


def _standard_comparator_variant(method: str) -> str:
    try:
        return STANDARD_COMPARATOR_VARIANTS[method]
    except KeyError as error:
        raise ValueError(f"unknown standard comparator method: {method}") from error


def _standard_comparator_requires_integer(method: str) -> bool:
    if method not in STANDARD_COMPARATOR_METHODS:
        raise ValueError(f"unknown standard comparator method: {method}")
    return method != DETERMINISTIC_MARKOV_METHOD


def _standard_comparator_is_stochastic(method: str) -> bool:
    if method not in STANDARD_COMPARATOR_METHODS:
        raise ValueError(f"unknown standard comparator method: {method}")
    return method != DETERMINISTIC_MARKOV_METHOD


def _standard_transient_core_array_bytes(
    method: str,
    config: DiffusionConfig,
) -> int:
    output_bytes = config.n_steps * config.n_nodes * np.dtype(np.float64).itemsize
    node_vector_bytes = config.n_nodes * np.dtype(np.float64).itemsize
    if method == STANDARD_COMPARATOR_METHOD:
        return output_bytes + node_vector_bytes
    if method == VECTORIZED_STANDARD_COMPARATOR_METHOD:
        # Movers/right integer vectors plus move-probability/right-share vectors.
        return output_bytes + 4 * node_vector_bytes
    if method == DETERMINISTIC_MARKOV_METHOD:
        return output_bytes + node_vector_bytes
    raise ValueError(f"unknown standard comparator method: {method}")


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
    method: str = "completed_modal"
    execution_position: int = -1
    seed_id: str = ""

    def as_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StandardTimingSample:
    """One aligned end-to-end sample from the standard simulation."""

    method: str
    repeat: int
    seconds: float
    execution_position: int
    seed_id: str

    def as_dict(self) -> dict[str, str | int | float]:
        return {
            "component": "trajectory_total",
            "repeat": self.repeat,
            "seconds": self.seconds,
            "readout": _standard_comparator_variant(self.method),
            "method": self.method,
            "execution_position": self.execution_position,
            "seed_id": self.seed_id,
        }


@dataclass(frozen=True, slots=True)
class StandardSimulationBenchmark:
    """Aligned full-state baseline timing for one configuration."""

    method: str
    config: DiffusionConfig
    regime: ComponentBenchmarkRegime
    trajectory_repeats: int
    warmup_runs: int
    seed: int
    samples: tuple[StandardTimingSample, ...]
    output_array_bytes: int
    output_dtype: str
    constraint_diagnostics: dict[str, float]

    def sample_rows(self) -> list[dict[str, str | int | float]]:
        configuration: dict[str, str | int | float] = {
            "pipeline": self.method,
            "variant": _standard_comparator_variant(self.method),
            "output_semantics": (
                "stochastic_trajectory"
                if _standard_comparator_is_stochastic(self.method)
                else "deterministic_expected_field"
            ),
            "regime": self.regime,
            "output_workload": "full_field_every_timestep",
            "n_particles": self.config.n_particles,
            "n_nodes": self.config.n_nodes,
            "n_steps": self.config.n_steps,
            "dt": self.config.dt,
            "length": self.config.length,
            "diffusion_coefficient": self.config.diffusion_coefficient,
            "impulse_index": self.config.impulse_index,
            "retained_modes": "",
            "handoff_time": "",
            "completion_rank": "",
            "completion_ridge": "",
            "setup_repeats": 0,
            "trajectory_repeats": self.trajectory_repeats,
            "warmup_runs": self.warmup_runs,
        }
        return [configuration | sample.as_dict() for sample in self.samples]

    def summary(self) -> dict[str, object]:
        values = np.asarray([sample.seconds for sample in self.samples], dtype=float)
        median_seconds = float(np.median(values))
        transitions = self.config.n_steps - 1
        return {
            "configuration": {
                "method": self.method,
                "variant": _standard_comparator_variant(self.method),
                "output_semantics": (
                    "stochastic_trajectory"
                    if _standard_comparator_is_stochastic(self.method)
                    else "deterministic_expected_field"
                ),
                "stochastic": _standard_comparator_is_stochastic(self.method),
                "regime": self.regime,
                "n_particles": self.config.n_particles,
                "n_nodes": self.config.n_nodes,
                "n_steps": self.config.n_steps,
                "dt": self.config.dt,
                "total_time": float(self.config.times[-1]),
                "length": self.config.length,
                "diffusion_coefficient": self.config.diffusion_coefficient,
                "impulse_index": self.config.impulse_index,
                "trajectory_repeats": self.trajectory_repeats,
                "warmup_runs": self.warmup_runs,
                "seed": self.seed,
                "seed_applicable": _standard_comparator_is_stochastic(
                    self.method
                ),
                "output_workload": "full_field_every_timestep",
                "setup_applicable": False,
            },
            "timing": {
                "samples": int(values.size),
                "median_seconds": median_seconds,
                "mean_seconds": float(np.mean(values)),
                "std_seconds": float(np.std(values, ddof=0)),
                "min_seconds": float(np.min(values)),
                "max_seconds": float(np.max(values)),
                "median_seconds_per_transition": median_seconds / transitions,
                "median_seconds_per_node_transition": (
                    median_seconds / (transitions * self.config.n_nodes)
                ),
                "amortized_seconds_per_trajectory": {
                    str(count): median_seconds for count in (1, 10, 100, 1_000)
                },
            },
            "memory": {
                "setup_array_bytes": 0,
                "setup_array_megabytes": 0.0,
                "output_array_bytes": self.output_array_bytes,
                "output_array_megabytes": self.output_array_bytes / 1024**2,
                "returned_result_array_bytes": self.output_array_bytes,
                "output_dtype": self.output_dtype,
            },
            "constraints": self.constraint_diagnostics,
        }


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
    standard_benchmarks: tuple[StandardSimulationBenchmark, ...] = ()

    @property
    def standard_benchmark(self) -> StandardSimulationBenchmark | None:
        """Return the original loop comparator for backward compatibility."""

        return next(
            (
                item
                for item in self.standard_benchmarks
                if item.method == STANDARD_COMPARATOR_METHOD
            ),
            None,
        )

    def sample_rows(self) -> list[dict[str, str | int | float]]:
        """Return long-format timing rows with configuration columns."""

        configuration: dict[str, str | int | float] = {
            "pipeline": "completed_modal",
            "variant": "",
            "output_semantics": "stochastic_trajectory",
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
        rows: list[dict[str, str | int | float]] = []
        for sample in self.samples:
            row = configuration | sample.as_dict()
            row["pipeline"] = sample.method
            row["variant"] = sample.readout
            rows.append(row)
        return rows

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

        report: dict[str, object] = {
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
        if self.standard_benchmarks:
            report["standard_comparators"] = [
                item.summary() for item in self.standard_benchmarks
            ]
            primary = next(
                (
                    item
                    for item in self.standard_benchmarks
                    if item.method == PRIMARY_STANDARD_COMPARATOR_METHOD
                ),
                self.standard_benchmarks[0],
            )
            report["standard_comparator"] = primary.summary()
        return report


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

    @property
    def overall_peak_incremental_bytes(self) -> int:
        return max(
            0,
            max(self.setup_peak_rss_bytes, self.trajectory_peak_rss_bytes)
            - self.baseline_rss_bytes,
        )

    def as_dict(self) -> dict[str, str | int | float | bool]:
        row: dict[str, str | int | float | bool] = asdict(self)
        row["method"] = "completed_modal"
        row["pipeline"] = "completed_modal"
        row["variant"] = self.readout
        row["output_semantics"] = "stochastic_trajectory"
        row["setup_applicable"] = True
        row["setup_peak_incremental_bytes"] = self.setup_peak_incremental_bytes
        row["steady_setup_incremental_bytes"] = self.steady_setup_incremental_bytes
        row["trajectory_peak_over_setup_bytes"] = self.trajectory_peak_over_setup_bytes
        row["setup_hwm_incremental_bytes"] = self.setup_hwm_incremental_bytes
        row["trajectory_hwm_incremental_bytes"] = (
            self.trajectory_hwm_incremental_bytes
        )
        row["overall_peak_incremental_bytes"] = self.overall_peak_incremental_bytes
        for key, value in tuple(row.items()):
            if key.endswith("_bytes"):
                row[key.removesuffix("_bytes") + "_megabytes"] = int(value) / 1024**2
        return row


@dataclass(frozen=True, slots=True)
class StandardSimulationMemoryRecord:
    """Fresh-process memory for one full-state baseline."""

    regime: ComponentBenchmarkRegime
    operating_point: str
    method: str
    variant: str
    n_particles: int
    n_nodes: int
    n_steps: int
    dt: float
    total_time: float
    setup_applicable: bool
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
    transient_core_array_bytes: int
    output_dtype: str
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
    def trajectory_hwm_incremental_bytes(self) -> int:
        return max(
            0,
            self.process_peak_after_trajectory_rss_bytes
            - self.process_peak_after_setup_rss_bytes,
        )

    @property
    def overall_peak_incremental_bytes(self) -> int:
        return max(
            0,
            max(self.setup_peak_rss_bytes, self.trajectory_peak_rss_bytes)
            - self.baseline_rss_bytes,
        )

    def as_dict(self) -> dict[str, str | int | float | bool]:
        row: dict[str, str | int | float | bool] = asdict(self)
        row["pipeline"] = self.method
        row["output_semantics"] = (
            "stochastic_trajectory"
            if _standard_comparator_is_stochastic(self.method)
            else "deterministic_expected_field"
        )
        row["setup_peak_incremental_bytes"] = self.setup_peak_incremental_bytes
        row["steady_setup_incremental_bytes"] = self.steady_setup_incremental_bytes
        row["trajectory_peak_over_setup_bytes"] = self.trajectory_peak_over_setup_bytes
        row["trajectory_hwm_incremental_bytes"] = (
            self.trajectory_hwm_incremental_bytes
        )
        row["overall_peak_incremental_bytes"] = self.overall_peak_incremental_bytes
        for key, value in tuple(row.items()):
            if key.endswith("_bytes"):
                row[key.removesuffix("_bytes") + "_megabytes"] = (
                    int(value) / 1024**2
                )
        return row


@dataclass(frozen=True, slots=True)
class ComponentScalingBenchmark:
    """Two-regime timing and isolated-memory component scaling study."""

    operating_point: str
    benchmarks: tuple[FullFieldComponentBenchmark, ...]
    memory_records: tuple[ComponentMemoryRecord, ...]
    standard_memory_records: tuple[StandardSimulationMemoryRecord, ...] = ()

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
        standard_benchmarks = [
            standard
            for item in self.benchmarks
            for standard in item.standard_benchmarks
        ]
        standard_slopes = _standard_scaling_slopes(
            standard_benchmarks,
            self.standard_memory_records,
        )
        comparisons = [
            _configuration_comparison(
                benchmark,
                self.memory_records,
                self.standard_memory_records,
            )
            for benchmark in self.benchmarks
            if benchmark.standard_benchmarks
        ]

        return {
            "schema_version": 4,
            "operating_point": self.operating_point,
            "study_contract": {
                "output_workload": "full_field_every_timestep",
                "common_time_grid": True,
                "standard_comparators": list(STANDARD_COMPARATOR_METHODS),
                "primary_standard_comparator": PRIMARY_STANDARD_COMPARATOR_METHOD,
                "standard_comparator_semantics": {
                    STANDARD_COMPARATOR_METHOD: "exact_stochastic_trajectory",
                    VECTORIZED_STANDARD_COMPARATOR_METHOD: (
                        "exact_stochastic_trajectory"
                    ),
                    DETERMINISTIC_MARKOV_METHOD: (
                        "exact_deterministic_expected_field"
                    ),
                },
                "standard_setup_applicable": False,
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
            "standard_comparators": [
                item.summary() for item in standard_benchmarks
            ],
            "standard_memory_records": [
                item.as_dict() for item in self.standard_memory_records
            ],
            "comparisons": comparisons,
            "scaling_slopes": slopes,
            "standard_scaling_slopes": standard_slopes,
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


def _safe_ratio(numerator: float | int, denominator: float | int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _standard_scaling_slopes(
    benchmarks: Sequence[StandardSimulationBenchmark],
    memory_records: Sequence[StandardSimulationMemoryRecord],
) -> dict[str, dict[str, dict[str, float | None]]]:
    slopes: dict[str, dict[str, dict[str, float | None]]] = {}
    for method in STANDARD_COMPARATOR_METHODS:
        method_slopes: dict[str, dict[str, float | None]] = {}
        for regime in COMPONENT_BENCHMARK_REGIMES:
            selected = sorted(
                (
                    item
                    for item in benchmarks
                    if item.method == method and item.regime == regime
                ),
                key=lambda item: item.config.n_nodes,
            )
            if len(selected) < 2:
                continue
            nodes = [float(item.config.n_nodes) for item in selected]
            output_cells = [
                float(item.config.n_nodes * item.config.n_steps)
                for item in selected
            ]
            reports = [item.summary() for item in selected]
            times = [float(item["timing"]["median_seconds"]) for item in reports]
            per_transition = [
                float(item["timing"]["median_seconds_per_transition"])
                for item in reports
            ]
            output_bytes = [
                float(item["memory"]["output_array_bytes"]) for item in reports
            ]
            regime_slopes: dict[str, float | None] = {
                "trajectory_total_vs_nodes": _log_log_slope(nodes, times),
                "trajectory_total_vs_output_cells": _log_log_slope(
                    output_cells,
                    times,
                ),
                "seconds_per_transition_vs_nodes": _log_log_slope(
                    nodes,
                    per_transition,
                ),
                "output_array_bytes_vs_nodes": _log_log_slope(
                    nodes,
                    output_bytes,
                ),
            }
            selected_memory = sorted(
                (
                    item
                    for item in memory_records
                    if item.method == method and item.regime == regime
                ),
                key=lambda item: item.n_nodes,
            )
            if len(selected_memory) >= 2:
                memory_nodes = [float(item.n_nodes) for item in selected_memory]
                regime_slopes["isolated_peak_increment_vs_nodes"] = (
                    _log_log_slope(
                        memory_nodes,
                        [
                            float(item.overall_peak_incremental_bytes)
                            for item in selected_memory
                        ],
                    )
                )
                regime_slopes["resident_array_bytes_vs_nodes"] = _log_log_slope(
                    memory_nodes,
                    [float(item.resident_array_bytes) for item in selected_memory],
                )
            method_slopes[regime] = regime_slopes
        if method_slopes:
            slopes[method] = method_slopes
    return slopes


def _configuration_comparison(
    benchmark: FullFieldComponentBenchmark,
    modal_memory_records: Sequence[ComponentMemoryRecord],
    standard_memory_records: Sequence[StandardSimulationMemoryRecord],
) -> dict[str, object]:
    modal_report = benchmark.summary()
    setup_seconds = float(
        modal_report["components"]["setup_total"]["median_seconds"]
    )
    standard_comparisons: dict[str, dict[str, object]] = {}
    standard_seconds_by_method: dict[str, float] = {}

    for standard in benchmark.standard_benchmarks:
        standard_seconds = float(standard.summary()["timing"]["median_seconds"])
        standard_seconds_by_method[standard.method] = standard_seconds
        stochastic_comparison = _standard_comparator_is_stochastic(
            standard.method
        )
        standard_memory = next(
            (
                item
                for item in standard_memory_records
                if item.regime == benchmark.regime
                and item.n_nodes == benchmark.config.n_nodes
                and item.method == standard.method
            ),
            None,
        )
        readout_comparisons: dict[str, dict[str, object]] = {}
        for readout in benchmark.readouts:
            modal_seconds = float(
                modal_report["readouts"][readout]["measured_end_to_end_seconds"]
            )
            modal_by_repeat = {
                sample.repeat: sample.seconds
                for sample in benchmark.samples
                if sample.component == "trajectory_total"
                and sample.readout == readout
            }
            standard_by_repeat = {
                sample.repeat: sample.seconds for sample in standard.samples
            }
            paired_ratios = [
                modal_by_repeat[repeat] / standard_by_repeat[repeat]
                for repeat in sorted(modal_by_repeat.keys() & standard_by_repeat.keys())
            ]
            break_even = None
            if stochastic_comparison and modal_seconds < standard_seconds:
                break_even = max(
                    1,
                    int(
                        np.ceil(
                            setup_seconds / (standard_seconds - modal_seconds)
                        )
                    ),
                )
            modal_memory = next(
                (
                    item
                    for item in modal_memory_records
                    if item.regime == benchmark.regime
                    and item.n_nodes == benchmark.config.n_nodes
                    and item.readout == readout
                ),
                None,
            )
            memory_comparison: dict[str, float | None] | None = None
            if modal_memory is not None and standard_memory is not None:
                memory_comparison = {
                    "overall_peak_increment_ratio_modal_over_standard": (
                        _safe_ratio(
                            modal_memory.overall_peak_incremental_bytes,
                            standard_memory.overall_peak_incremental_bytes,
                        )
                    ),
                    "logical_peak_array_ratio_modal_over_standard": _safe_ratio(
                        modal_memory.resident_array_bytes,
                        standard_memory.transient_core_array_bytes,
                    ),
                    "returned_result_array_ratio_modal_over_standard": _safe_ratio(
                        modal_memory.returned_result_array_bytes,
                        standard_memory.returned_result_array_bytes,
                    ),
                    "output_array_ratio_modal_over_standard": _safe_ratio(
                        modal_memory.output_array_bytes,
                        standard_memory.output_array_bytes,
                    ),
                }
            readout_comparisons[readout] = {
                "statistically_aligned_stochastic_workload": (
                    stochastic_comparison
                ),
                "steady_runtime_ratio_modal_over_standard": _safe_ratio(
                    modal_seconds,
                    standard_seconds,
                ),
                "median_paired_repeat_ratio_modal_over_standard": (
                    float(np.median(paired_ratios)) if paired_ratios else None
                ),
                "amortized_runtime_ratio_modal_over_standard": {
                    str(count): _safe_ratio(
                        modal_seconds + setup_seconds / count,
                        standard_seconds,
                    )
                    for count in (1, 10, 100, 1_000)
                },
                "break_even_trajectories": break_even,
                "break_even_applicable": stochastic_comparison,
                "memory": memory_comparison,
            }
        standard_comparisons[standard.method] = {
            "standard_median_seconds": standard_seconds,
            "output_semantics": (
                "stochastic_trajectory"
                if stochastic_comparison
                else "deterministic_expected_field"
            ),
            "readouts": readout_comparisons,
        }

    primary_method = (
        PRIMARY_STANDARD_COMPARATOR_METHOD
        if PRIMARY_STANDARD_COMPARATOR_METHOD in standard_comparisons
        else next(iter(standard_comparisons))
    )
    primary = standard_comparisons[primary_method]
    loop_seconds = standard_seconds_by_method.get(STANDARD_COMPARATOR_METHOD)
    vectorized_seconds = standard_seconds_by_method.get(
        VECTORIZED_STANDARD_COMPARATOR_METHOD
    )
    implementation_speedup = None
    if loop_seconds is not None and vectorized_seconds is not None:
        implementation_speedup = _safe_ratio(loop_seconds, vectorized_seconds)
    deterministic_seconds = standard_seconds_by_method.get(
        DETERMINISTIC_MARKOV_METHOD
    )
    stochastic_over_deterministic = None
    if vectorized_seconds is not None and deterministic_seconds is not None:
        stochastic_over_deterministic = _safe_ratio(
            vectorized_seconds,
            deterministic_seconds,
        )
    return {
        "regime": benchmark.regime,
        "n_nodes": benchmark.config.n_nodes,
        "n_steps": benchmark.config.n_steps,
        "dt": benchmark.config.dt,
        "total_time": float(benchmark.config.times[-1]),
        "ratio_direction": (
            "modal_over_standard; values above one mean modal is slower or uses "
            "more memory"
        ),
        "primary_standard_method": primary_method,
        "standard_implementation_speedup_loop_over_vectorized": (
            implementation_speedup
        ),
        "standard_runtime_ratio_vectorized_stochastic_over_deterministic_mean": (
            stochastic_over_deterministic
        ),
        "modal_setup_median_seconds": setup_seconds,
        "standards": standard_comparisons,
        # Compatibility aliases point to the faster production comparator.
        "standard_median_seconds": primary["standard_median_seconds"],
        "readouts": primary["readouts"],
    }


def _balanced_execution_orders(
    tasks: Sequence[tuple[str, str]],
) -> tuple[tuple[tuple[str, str], ...], ...]:
    """Return a repeatable schedule balanced for pairwise order by ten runs."""

    values = tuple(tasks)
    if len(values) == 3:
        # Two Latin-square cycles cover all six permutations. The final four
        # are two reverse pairs, so every task pair precedes each other five
        # times over the default ten repeats.
        index_orders = (
            (0, 1, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
            (0, 2, 1),
            (1, 0, 2),
            (0, 1, 2),
            (2, 1, 0),
            (1, 2, 0),
            (0, 2, 1),
        )
        return tuple(tuple(values[index] for index in order) for order in index_orders)
    if len(values) == 4:
        # Five orders paired with their reverses balance every pair over the
        # default ten repeats while spreading absolute execution positions.
        base_orders = (
            (0, 1, 2, 3),
            (1, 2, 3, 0),
            (2, 3, 0, 1),
            (3, 0, 1, 2),
            (0, 2, 3, 1),
        )
        index_orders = tuple(
            order
            for base in base_orders
            for order in (base, tuple(reversed(base)))
        )
        return tuple(tuple(values[index] for index in order) for order in index_orders)
    if len(values) == 5:
        # Five cyclic rotations and their reverses give ten orders. Every task
        # occupies each absolute position twice, and each pair appears in both
        # relative orders five times.
        indices = tuple(range(5))
        base_orders = tuple(
            indices[offset:] + indices[:offset] for offset in range(5)
        )
        index_orders = tuple(
            order
            for base in base_orders
            for order in (base, tuple(reversed(base)))
        )
        return tuple(tuple(values[index] for index in order) for order in index_orders)
    return tuple(permutations(values))


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
    include_standard_comparator: bool = False,
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
        if include_standard_comparator:
            for method in STANDARD_COMPARATOR_METHODS:
                channel = STANDARD_COMPARATOR_SEED_CHANNELS.get(method)
                standard_rng = (
                    None
                    if channel is None
                    else np.random.default_rng(
                        np.random.SeedSequence([seed, 0, channel, warmup])
                    )
                )
                standard_warmup = _run_standard_comparator(
                    method,
                    config,
                    standard_rng,
                )
                validate_full_field_output(
                    standard_warmup,
                    config,
                    require_integer=_standard_comparator_requires_integer(
                        method
                    ),
                )
                del standard_warmup

    setup_array_bytes = deep_numpy_nbytes(pipeline)
    resident_array_bytes = setup_array_bytes
    output_array_bytes = 0
    output_dtype = ""
    diagnostic_accumulator: dict[str, list[dict[str, float]]] = {
        readout: [] for readout in selected_readouts
    }
    standard_samples: dict[str, list[StandardTimingSample]] = {
        method: [] for method in STANDARD_COMPARATOR_METHODS
    }
    standard_diagnostics: dict[str, list[dict[str, float]]] = {
        method: [] for method in STANDARD_COMPARATOR_METHODS
    }
    standard_output_array_bytes = {
        method: 0 for method in STANDARD_COMPARATOR_METHODS
    }
    standard_output_dtype = {
        method: "" for method in STANDARD_COMPARATOR_METHODS
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

            del raw, completed, additions, physical

            total_tasks = [
                ("completed_modal", readout) for readout in selected_readouts
            ]
            if include_standard_comparator:
                total_tasks.extend(
                    (method, _standard_comparator_variant(method))
                    for method in STANDARD_COMPARATOR_METHODS
                )
            task_orders = _balanced_execution_orders(total_tasks)
            total_tasks = list(task_orders[repeat % len(task_orders)])
            for execution_position, (method, variant) in enumerate(total_tasks):
                if method in STANDARD_COMPARATOR_METHODS:
                    channel = STANDARD_COMPARATOR_SEED_CHANNELS.get(method)
                    standard_rng = (
                        None
                        if channel is None
                        else np.random.default_rng(
                            np.random.SeedSequence([seed, channel, repeat])
                        )
                    )
                    start = perf_counter()
                    standard_counts = _run_standard_comparator(
                        method,
                        config,
                        standard_rng,
                    )
                    total_seconds = perf_counter() - start
                    validate_full_field_output(
                        standard_counts,
                        config,
                        require_integer=_standard_comparator_requires_integer(
                            method
                        ),
                    )
                    standard_samples[method].append(
                        StandardTimingSample(
                            method=method,
                            repeat=repeat,
                            seconds=total_seconds,
                            execution_position=execution_position,
                            seed_id=(
                                f"{seed}:{channel}:{repeat}"
                                if channel is not None
                                else ""
                            ),
                        )
                    )
                    standard_diagnostics[method].append(
                        readout_constraint_diagnostics(
                            standard_counts,
                            config.n_particles,
                        )
                    )
                    standard_output_array_bytes[method] = max(
                        standard_output_array_bytes[method],
                        standard_counts.nbytes,
                    )
                    standard_output_dtype[method] = str(standard_counts.dtype)
                    del standard_counts
                    continue

                total_simulation_sequence, total_completion_sequence = (
                    np.random.SeedSequence([seed, 2, repeat]).spawn(2)
                )
                start = perf_counter()
                physical = _execute_full_field_component_pipeline(
                    pipeline,
                    readout=variant,  # type: ignore[arg-type]
                    simulation_sequence=total_simulation_sequence,
                    completion_sequence=total_completion_sequence,
                )
                total_seconds = perf_counter() - start
                validate_full_field_output(
                    physical.counts,
                    config,
                    require_integer=variant == "delta_sigma_temporal",
                )
                samples.append(
                    ComponentTimingSample(
                        "trajectory_total",
                        repeat,
                        total_seconds,
                        variant,
                        execution_position=execution_position,
                        seed_id=f"{seed}:2:{repeat}",
                    )
                )
                resident_array_bytes = max(
                    resident_array_bytes,
                    deep_numpy_nbytes((pipeline, physical)),
                )
                del physical
    finally:
        if garbage_collection_enabled:
            gc.enable()

    constraints: dict[str, dict[str, float]] = {}
    for readout, diagnostics in diagnostic_accumulator.items():
        constraints[readout] = {
            key: float(max(item[key] for item in diagnostics))
            for key in diagnostics[0]
        }

    standard_benchmarks: tuple[StandardSimulationBenchmark, ...] = ()
    if include_standard_comparator:
        standard_benchmarks = tuple(
            StandardSimulationBenchmark(
                method=method,
                config=config,
                regime=regime,
                trajectory_repeats=trajectory_repeats,
                warmup_runs=warmup_runs,
                seed=seed,
                samples=tuple(standard_samples[method]),
                output_array_bytes=standard_output_array_bytes[method],
                output_dtype=standard_output_dtype[method],
                constraint_diagnostics={
                    key: float(
                        max(item[key] for item in standard_diagnostics[method])
                    )
                    for key in standard_diagnostics[method][0]
                },
            )
            for method in STANDARD_COMPARATOR_METHODS
        )

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
        standard_benchmarks=standard_benchmarks,
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


def _run_synchronized_memory_worker(
    spec: dict[str, object],
    *,
    sampler_interval_ms: float,
    worker_timeout_seconds: float,
) -> tuple[
    dict[str, dict[str, object]],
    dict[str, int],
    dict[str, int],
]:
    """Run the shared memory worker and sample phase-local current RSS."""

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
    return messages, phase_peaks, phase_samples


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
    spec["method"] = "completed_modal"
    messages, phase_peaks, phase_samples = _run_synchronized_memory_worker(
        spec,
        sampler_interval_ms=sampler_interval_ms,
        worker_timeout_seconds=worker_timeout_seconds,
    )
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


def measure_standard_comparator_memory(
    config: DiffusionConfig,
    *,
    regime: ComponentBenchmarkRegime,
    operating_point: str,
    seed: int,
    method: StandardComparatorMethod = STANDARD_COMPARATOR_METHOD,
    sampler_interval_ms: float = 1.0,
    worker_timeout_seconds: float = 300.0,
) -> StandardSimulationMemoryRecord:
    """Measure one full-state baseline's memory in a fresh process."""

    if sampler_interval_ms <= 0.0:
        raise ValueError("sampler_interval_ms must be positive")
    if worker_timeout_seconds <= 0.0:
        raise ValueError("worker_timeout_seconds must be positive")
    if method not in STANDARD_COMPARATOR_METHODS:
        raise ValueError(f"unknown standard comparator method: {method}")
    spec: dict[str, object] = {
        "method": method,
        "config": {
            "n_particles": config.n_particles,
            "n_nodes": config.n_nodes,
            "n_steps": config.n_steps,
            "impulse_index": config.impulse_index,
            "dt": config.dt,
            "length": config.length,
            "diffusion_coefficient": config.diffusion_coefficient,
        },
        "seed": seed,
    }
    messages, phase_peaks, phase_samples = _run_synchronized_memory_worker(
        spec,
        sampler_interval_ms=sampler_interval_ms,
        worker_timeout_seconds=worker_timeout_seconds,
    )
    ready = messages["ready"]
    setup = messages["setup_end"]
    trajectory = messages["trajectory_end"]
    return StandardSimulationMemoryRecord(
        regime=regime,
        operating_point=operating_point,
        method=method,
        variant=_standard_comparator_variant(method),
        n_particles=config.n_particles,
        n_nodes=config.n_nodes,
        n_steps=config.n_steps,
        dt=config.dt,
        total_time=float(config.times[-1]),
        setup_applicable=False,
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
        transient_core_array_bytes=int(trajectory["transient_core_array_bytes"]),
        output_dtype=str(trajectory["output_dtype"]),
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
    include_standard_comparator: bool = True,
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
    standard_memories: list[StandardSimulationMemoryRecord] = []
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
                include_standard_comparator=include_standard_comparator,
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
                if include_standard_comparator:
                    for method in STANDARD_COMPARATOR_METHODS:
                        standard_memories.append(
                            measure_standard_comparator_memory(
                                config,
                                regime=regime,
                                operating_point=operating_point,
                                method=method,
                                seed=(
                                    configuration_seed
                                    + 10_000
                                    + STANDARD_COMPARATOR_SEED_CHANNELS.get(
                                        method,
                                        0,
                                    )
                                ),
                                sampler_interval_ms=memory_sampler_interval_ms,
                            )
                        )
    return ComponentScalingBenchmark(
        operating_point=operating_point,
        benchmarks=tuple(benchmarks),
        memory_records=tuple(memories),
        standard_memory_records=tuple(standard_memories),
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
    rows: list[dict[str, str | int | float]] = []
    for benchmark in scaling.benchmarks:
        rows.extend(benchmark.sample_rows())
        for standard in benchmark.standard_benchmarks:
            rows.extend(standard.sample_rows())
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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
    rows.extend(record.as_dict() for record in scaling.standard_memory_records)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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
