"""Reproducible runtime and resident-array memory benchmarks.

Two benchmark families are provided:

``benchmark_random_walks``
    Retains the original naive-versus-multinomial random-walk benchmark.

``benchmark_diffusion_methods``
    Compares the multinomial random walk, the full correlated modal model,
    the raw adaptive handoff model, and the independent and persistent
    unresolved Gaussian completion pipelines.

The reported memory values are the NumPy array bytes retained by setup objects
and returned results. They are more informative than hand-written estimates,
but they are not process peak resident memory and do not include short-lived
BLAS/LAPACK workspaces.
"""

from __future__ import annotations

import csv
import multiprocessing as mp
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Callable, Literal

import numpy as np

from .anchored_closure import AnchoredFittedClosureModel, AnchoredFittedLocalCompleter
from .baselines import (
    multinomial_random_walk_diffusion,
    naive_random_walk_diffusion,
)
from .completion import (
    LocalPersistentUnresolvedGaussianCompleter,
    PersistentUnresolvedGaussianCompleter,
    UnresolvedGaussianCompleter,
)
from .config import DiffusionConfig
from .conditioned_closure import (
    PhysicsConditionedClosureModel,
    PhysicsConditionedLocalCompleter,
)
from .correlated_modal import (
    CorrelatedModalDiffusion,
    HandoffCorrelatedModalDiffusion,
)
from .fitted_closure import FittedLocalClosureModel, FittedLocalPersistentCompleter
from .readouts import simplex_bank_readout

BenchmarkMethodName = Literal[
    "multinomial_random_walk",
    "full_correlated_modal",
    "handoff_raw",
    "handoff_independent_completion",
    "handoff_persistent_completion",
]
BENCHMARK_METHOD_NAMES: tuple[BenchmarkMethodName, ...] = (
    "multinomial_random_walk",
    "full_correlated_modal",
    "handoff_raw",
    "handoff_independent_completion",
    "handoff_persistent_completion",
)


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    """One selected persistent-completion pipeline at the 101-node reference."""

    name: str
    reference_modes: int
    reference_handoff_time: float
    reference_completion_rank: int

    def scaled(self, n_nodes: int, total_time: float) -> tuple[int, float, int]:
        """Scale mode count, handoff time, and completion rank proportionally."""

        if n_nodes < 2:
            raise ValueError("n_nodes must be at least two")
        modes = min(
            n_nodes - 1,
            max(1, int(round(n_nodes * self.reference_modes / 101.0))),
        )
        omitted = n_nodes - modes
        rank = min(
            omitted,
            max(1, int(round(n_nodes * self.reference_completion_rank / 101.0))),
        )
        handoff_time = total_time * self.reference_handoff_time / 100.0
        return modes, handoff_time, rank


SELECTED_OPERATING_POINTS: tuple[OperatingPoint, ...] = (
    OperatingPoint("compute_first", 35, 20.0, 20),
    OperatingPoint("balanced", 50, 10.0, 10),
    OperatingPoint("accuracy_first", 75, 30.0, 5),
)
# The hardened held-out gate rejects the local closure on long-domain lag
# covariance. Keep the exact dense analytic closure as the production default;
# local variants remain explicit performance experiments.
DEFAULT_COMPLETION_CLOSURE = "dense_persistent"
DEFAULT_LOCAL_MODAL_BANDWIDTH = 12


@dataclass(frozen=True, slots=True)
class OperatingPointBenchmark:
    """Isolated runtime and peak-memory result for one operating point."""

    operating_point: str
    closure: str
    modal_bandwidth: int
    n_particles: int
    n_nodes: int
    n_steps: int
    final_modes: int
    handoff_time: float
    completion_rank: int
    repeats: int
    amortization_runs: int
    setup_seconds: float
    median_run_seconds: float
    mean_run_seconds: float
    std_run_seconds: float
    amortized_seconds_per_run: float
    setup_array_bytes: int
    resident_array_bytes: int
    peak_rss_bytes: int

    @property
    def setup_array_megabytes(self) -> float:
        return self.setup_array_bytes / 1024**2

    @property
    def resident_array_megabytes(self) -> float:
        return self.resident_array_bytes / 1024**2

    @property
    def peak_rss_megabytes(self) -> float:
        return self.peak_rss_bytes / 1024**2

    def as_dict(self) -> dict[str, str | int | float]:
        row = asdict(self)
        row["setup_array_megabytes"] = self.setup_array_megabytes
        row["resident_array_megabytes"] = self.resident_array_megabytes
        row["peak_rss_megabytes"] = self.peak_rss_megabytes
        return row


@dataclass(frozen=True, slots=True)
class RandomWalkBenchmark:
    """Summary for one random-walk method and one particle count."""

    method: str
    n_particles: int
    n_nodes: int
    n_steps: int
    repeats: int
    median_seconds: float
    mean_seconds: float
    std_seconds: float
    min_seconds: float
    core_array_bytes: int

    @property
    def core_array_megabytes(self) -> float:
        return self.core_array_bytes / 1024**2

    def as_dict(self) -> dict[str, str | int | float]:
        return {
            "method": self.method,
            "n_particles": self.n_particles,
            "n_nodes": self.n_nodes,
            "n_steps": self.n_steps,
            "repeats": self.repeats,
            "median_seconds": self.median_seconds,
            "mean_seconds": self.mean_seconds,
            "std_seconds": self.std_seconds,
            "min_seconds": self.min_seconds,
            "core_array_bytes": self.core_array_bytes,
            "core_array_megabytes": self.core_array_megabytes,
        }


@dataclass(frozen=True, slots=True)
class DiffusionMethodBenchmark:
    """Runtime and resident-array summary for one end-to-end method."""

    method: str
    sweep_axis: str
    n_particles: int
    n_nodes: int
    n_steps: int
    dt: float
    total_time: float
    initial_modes: int
    final_modes: int
    completion_rank: int
    handoff_time: float
    repeats: int
    amortization_runs: int
    setup_seconds: float
    median_run_seconds: float
    mean_run_seconds: float
    std_run_seconds: float
    min_run_seconds: float
    one_run_total_seconds: float
    amortized_seconds_per_run: float
    setup_array_bytes: int
    resident_array_bytes: int

    @property
    def setup_array_megabytes(self) -> float:
        return self.setup_array_bytes / 1024**2

    @property
    def resident_array_megabytes(self) -> float:
        return self.resident_array_bytes / 1024**2

    def as_dict(self) -> dict[str, str | int | float]:
        return {
            "method": self.method,
            "sweep_axis": self.sweep_axis,
            "n_particles": self.n_particles,
            "n_nodes": self.n_nodes,
            "n_steps": self.n_steps,
            "dt": self.dt,
            "total_time": self.total_time,
            "initial_modes": self.initial_modes,
            "final_modes": self.final_modes,
            "completion_rank": self.completion_rank,
            "handoff_time": self.handoff_time,
            "repeats": self.repeats,
            "amortization_runs": self.amortization_runs,
            "setup_seconds": self.setup_seconds,
            "median_run_seconds": self.median_run_seconds,
            "mean_run_seconds": self.mean_run_seconds,
            "std_run_seconds": self.std_run_seconds,
            "min_run_seconds": self.min_run_seconds,
            "one_run_total_seconds": self.one_run_total_seconds,
            "amortized_seconds_per_run": self.amortized_seconds_per_run,
            "setup_array_bytes": self.setup_array_bytes,
            "setup_array_megabytes": self.setup_array_megabytes,
            "resident_array_bytes": self.resident_array_bytes,
            "resident_array_megabytes": self.resident_array_megabytes,
        }


@dataclass(frozen=True, slots=True)
class _MethodSetup:
    method: BenchmarkMethodName
    simulator: object | None
    completer: object | None


@dataclass(frozen=True, slots=True)
class _RunPayload:
    counts: np.ndarray
    retained_objects: tuple[object, ...]


def deep_numpy_nbytes(value: object, *, include_array_bases: bool = False) -> int:
    """Return unique NumPy array bytes reachable from ``value``.

    References are deduplicated by object identity, so a basis shared by a
    simulator and result is counted once. With ``include_array_bases``, direct
    NumPy-array base chains count their shared backing array once instead of
    their logical sizes. Views backed through non-array wrappers retain their
    logical-size accounting. This intentionally ignores Python object overhead
    and temporary numerical workspaces.
    """

    seen_objects: set[int] = set()
    seen_arrays: set[int] = set()

    def visit(item: object) -> int:
        identity = id(item)
        if identity in seen_objects:
            return 0
        seen_objects.add(identity)

        if isinstance(item, np.ndarray):
            if include_array_bases and isinstance(item.base, np.ndarray):
                return visit(item.base)
            if identity in seen_arrays:
                return 0
            seen_arrays.add(identity)
            return int(item.nbytes)
        if is_dataclass(item) and not isinstance(item, type):
            return sum(visit(getattr(item, field.name)) for field in fields(item))
        if isinstance(item, dict):
            return sum(visit(key) + visit(entry) for key, entry in item.items())
        if isinstance(item, (tuple, list, set, frozenset)):
            return sum(visit(entry) for entry in item)
        if hasattr(item, "__dict__"):
            return sum(visit(entry) for entry in vars(item).values())
        if hasattr(item, "__slots__"):
            slots = item.__slots__
            if isinstance(slots, str):
                slots = (slots,)
            return sum(
                visit(getattr(item, slot))
                for slot in slots
                if hasattr(item, slot)
            )
        return 0

    return visit(value)


def estimate_random_walk_core_bytes(
    config: DiffusionConfig,
    method: str,
) -> int:
    """Estimate principal arrays in the two random-walk implementations."""

    int64_bytes = np.dtype(np.int64).itemsize
    counts_history = config.n_steps * config.n_nodes * int64_bytes

    if method == "naive":
        trajectories = config.n_particles * config.n_steps * int64_bytes
        return trajectories + counts_history
    if method == "multinomial":
        temporary_node_counts = config.n_nodes * int64_bytes
        return counts_history + temporary_node_counts
    raise ValueError("method must be 'naive' or 'multinomial'")


def benchmark_random_walks(
    base_config: DiffusionConfig,
    particle_counts: list[int],
    repeats: int = 3,
    seed: int = 0,
) -> list[RandomWalkBenchmark]:
    """Benchmark naive and multinomial random walks over particle counts."""

    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if not particle_counts or any(count <= 0 for count in particle_counts):
        raise ValueError("particle_counts must contain positive integers")

    methods: dict[str, Callable] = {
        "naive": naive_random_walk_diffusion,
        "multinomial": multinomial_random_walk_diffusion,
    }
    records: list[RandomWalkBenchmark] = []

    for n_particles in particle_counts:
        config = replace(base_config, n_particles=n_particles)
        timings: dict[str, list[float]] = {name: [] for name in methods}
        repeat_sequences = np.random.SeedSequence(seed + n_particles).spawn(repeats * 2)

        for repeat_index in range(repeats):
            order = ("naive", "multinomial")
            if repeat_index % 2:
                order = tuple(reversed(order))

            for order_index, method in enumerate(order):
                sequence_index = 2 * repeat_index + order_index
                rng = np.random.default_rng(repeat_sequences[sequence_index])
                start = perf_counter()
                output = methods[method](config, rng=rng)
                elapsed = perf_counter() - start

                if output.shape != (config.n_steps, config.n_nodes):
                    raise RuntimeError(f"unexpected output shape from {method}")
                if not np.all(output.sum(axis=1) == config.n_particles):
                    raise RuntimeError(f"mass conservation failed for {method}")
                timings[method].append(elapsed)

        for method in methods:
            values = np.asarray(timings[method], dtype=float)
            records.append(
                RandomWalkBenchmark(
                    method=method,
                    n_particles=config.n_particles,
                    n_nodes=config.n_nodes,
                    n_steps=config.n_steps,
                    repeats=repeats,
                    median_seconds=float(np.median(values)),
                    mean_seconds=float(np.mean(values)),
                    std_seconds=float(np.std(values, ddof=0)),
                    min_seconds=float(np.min(values)),
                    core_array_bytes=estimate_random_walk_core_bytes(config, method),
                )
            )

    return records


def stable_diffusion_config(
    *,
    n_particles: int,
    n_nodes: int,
    total_time: float,
    requested_dt: float,
    length: float,
    diffusion_coefficient: float,
    impulse_fraction: float,
    stability_safety: float = 0.95,
) -> DiffusionConfig:
    """Build a config with fixed physical duration and an automatically stable dt."""

    if total_time <= 0.0:
        raise ValueError("total_time must be positive")
    if requested_dt <= 0.0:
        raise ValueError("requested_dt must be positive")
    if not 0.0 < stability_safety <= 1.0:
        raise ValueError("stability_safety must lie in (0, 1]")
    if not 0.0 <= impulse_fraction <= 1.0:
        raise ValueError("impulse_fraction must lie in [0, 1]")

    dx = length / (n_nodes - 1)

    # The benchmark includes correlated modal models whose explicit one-step
    # factors are ``1 - lambda_k * dt`` and are required to remain
    # nonnegative. For the 1D Neumann Laplacian, ``lambda_max < 4 D / dx^2``.
    # Therefore ``dt <= dx^2 / (4 D)`` is a conservative bound that is valid
    # for every benchmarked method. This is twice as strict as the ordinary
    # nearest-neighbour random-walk constraint ``2 D dt / dx^2 <= 1``.
    maximum_dt = dx**2 / (4.0 * diffusion_coefficient)
    provisional_dt = min(requested_dt, stability_safety * maximum_dt)
    intervals = max(1, int(np.ceil(total_time / provisional_dt)))
    actual_dt = total_time / intervals
    impulse_index = int(round(impulse_fraction * (n_nodes - 1)))
    return DiffusionConfig(
        n_particles=n_particles,
        n_nodes=n_nodes,
        n_steps=intervals + 1,
        impulse_index=impulse_index,
        dt=actual_dt,
        length=length,
        diffusion_coefficient=diffusion_coefficient,
    )


def _resolve_final_modes(
    n_nodes: int,
    *,
    final_modes: int | None,
    retained_fraction: float,
) -> int:
    if final_modes is not None:
        if final_modes <= 0:
            raise ValueError("final_modes must be positive")
        return min(int(final_modes), n_nodes)
    if not 0.0 < retained_fraction <= 1.0:
        raise ValueError("retained_fraction must lie in (0, 1]")
    return min(n_nodes, max(1, int(round(retained_fraction * n_nodes))))


def _setup_method(
    method: BenchmarkMethodName,
    config: DiffusionConfig,
    *,
    final_modes: int,
    handoff_time: float,
    completion_rank: int,
    completion_ridge: float,
) -> _MethodSetup:
    if method == "multinomial_random_walk":
        return _MethodSetup(method=method, simulator=None, completer=None)
    if method == "full_correlated_modal":
        simulator = CorrelatedModalDiffusion(config=config, n_modes=config.n_nodes)
        return _MethodSetup(method=method, simulator=simulator, completer=None)

    simulator = HandoffCorrelatedModalDiffusion(
        config=config,
        initial_n_modes=config.n_nodes,
        final_n_modes=final_modes,
        handoff_time=handoff_time,
    )
    if method == "handoff_raw":
        return _MethodSetup(method=method, simulator=simulator, completer=None)

    completion_start = simulator.handoff_time
    if method == "handoff_independent_completion":
        completer = UnresolvedGaussianCompleter(
            config,
            retained_modes=final_modes,
            completion_start_time=completion_start,
            completion_rank=completion_rank,
            ridge=completion_ridge,
        )
    elif method == "handoff_persistent_completion":
        completer = PersistentUnresolvedGaussianCompleter(
            config,
            retained_modes=final_modes,
            completion_start_time=completion_start,
            completion_rank=completion_rank,
            ridge=completion_ridge,
        )
    else:  # pragma: no cover - guarded by public validation
        raise ValueError(f"unknown benchmark method: {method}")
    return _MethodSetup(method=method, simulator=simulator, completer=completer)


def _run_method(
    setup: _MethodSetup,
    config: DiffusionConfig,
    seed_sequence: np.random.SeedSequence,
) -> _RunPayload:
    simulation_sequence, completion_sequence = seed_sequence.spawn(2)
    simulation_rng = np.random.default_rng(simulation_sequence)

    if setup.method == "multinomial_random_walk":
        counts = multinomial_random_walk_diffusion(config, rng=simulation_rng)
        return _RunPayload(counts=counts, retained_objects=(counts,))

    result = setup.simulator.run(rng=simulation_rng)
    if setup.completer is None:
        return _RunPayload(
            counts=result.spatial_counts,
            retained_objects=(result,),
        )

    completion_rng = np.random.default_rng(completion_sequence)
    completed, additions = setup.completer.complete(
        result.spatial_counts,
        rng=completion_rng,
    )
    return _RunPayload(
        counts=completed,
        retained_objects=(result, completed, additions),
    )


def _validate_benchmark_output(payload: _RunPayload, config: DiffusionConfig) -> None:
    counts = np.asarray(payload.counts)
    if counts.shape != (config.n_steps, config.n_nodes):
        raise RuntimeError("benchmark method returned an unexpected shape")
    masses = counts.sum(axis=1)
    if not np.allclose(masses, config.n_particles, atol=1e-7, rtol=0.0):
        raise RuntimeError("benchmark method failed mass conservation")
    if not np.all(np.isfinite(counts)):
        raise RuntimeError("benchmark method returned nonfinite values")


def _benchmark_one_configuration(
    config: DiffusionConfig,
    *,
    sweep_axis: str,
    methods: tuple[BenchmarkMethodName, ...],
    final_modes: int,
    handoff_time: float,
    completion_rank: int,
    completion_ridge: float,
    repeats: int,
    amortization_runs: int,
    seed: int,
) -> list[DiffusionMethodBenchmark]:
    setups: dict[BenchmarkMethodName, _MethodSetup] = {}
    setup_seconds: dict[BenchmarkMethodName, float] = {}
    setup_bytes: dict[BenchmarkMethodName, int] = {}

    for method in methods:
        start = perf_counter()
        setup = _setup_method(
            method,
            config,
            final_modes=final_modes,
            handoff_time=handoff_time,
            completion_rank=completion_rank,
            completion_ridge=completion_ridge,
        )
        setup_seconds[method] = perf_counter() - start
        setups[method] = setup
        setup_bytes[method] = deep_numpy_nbytes(setup)

    timings: dict[BenchmarkMethodName, list[float]] = {
        method: [] for method in methods
    }
    resident_bytes: dict[BenchmarkMethodName, int] = {
        method: setup_bytes[method] for method in methods
    }
    sequences = np.random.SeedSequence(
        [seed, config.n_nodes, config.n_particles]
    ).spawn(repeats * len(methods))

    for repeat_index in range(repeats):
        order = list(methods)
        if repeat_index % 2:
            order.reverse()
        for order_index, method in enumerate(order):
            sequence_index = repeat_index * len(methods) + order_index
            start = perf_counter()
            payload = _run_method(setups[method], config, sequences[sequence_index])
            elapsed = perf_counter() - start
            _validate_benchmark_output(payload, config)
            timings[method].append(elapsed)
            resident_bytes[method] = max(
                resident_bytes[method],
                deep_numpy_nbytes((setups[method], payload.retained_objects)),
            )

    records: list[DiffusionMethodBenchmark] = []
    effective_completion_rank = min(completion_rank, config.n_nodes - final_modes)
    for method in methods:
        values = np.asarray(timings[method], dtype=float)
        median = float(np.median(values))
        setup_time = setup_seconds[method]
        method_completion_rank = (
            effective_completion_rank
            if "completion" in method
            else 0
        )
        records.append(
            DiffusionMethodBenchmark(
                method=method,
                sweep_axis=sweep_axis,
                n_particles=config.n_particles,
                n_nodes=config.n_nodes,
                n_steps=config.n_steps,
                dt=config.dt,
                total_time=float(config.times[-1]),
                initial_modes=config.n_nodes,
                final_modes=(
                    config.n_nodes
                    if method in {"multinomial_random_walk", "full_correlated_modal"}
                    else final_modes
                ),
                completion_rank=method_completion_rank,
                handoff_time=(
                    0.0
                    if method in {"multinomial_random_walk", "full_correlated_modal"}
                    else float(handoff_time)
                ),
                repeats=repeats,
                amortization_runs=amortization_runs,
                setup_seconds=setup_time,
                median_run_seconds=median,
                mean_run_seconds=float(np.mean(values)),
                std_run_seconds=float(np.std(values, ddof=0)),
                min_run_seconds=float(np.min(values)),
                one_run_total_seconds=setup_time + median,
                amortized_seconds_per_run=median + setup_time / amortization_runs,
                setup_array_bytes=setup_bytes[method],
                resident_array_bytes=resident_bytes[method],
            )
        )
    return records


def benchmark_diffusion_methods(
    *,
    node_counts: list[int],
    particle_counts: list[int],
    base_nodes: int = 101,
    base_particles: int = 5_275,
    total_time: float = 100.0,
    requested_dt: float = 1.0,
    length: float = 4.0,
    diffusion_coefficient: float = 2.20e-4,
    impulse_fraction: float = 0.59,
    methods: tuple[BenchmarkMethodName, ...] = BENCHMARK_METHOD_NAMES,
    final_modes: int | None = None,
    retained_fraction: float = 0.5,
    handoff_time: float = 10.0,
    completion_rank: int = 10,
    completion_ridge: float = 1.0e-2,
    repeats: int = 3,
    amortization_runs: int = 100,
    stability_safety: float = 0.95,
    seed: int = 0,
) -> list[DiffusionMethodBenchmark]:
    """Benchmark all main pipelines across spatial and particle scaling.

    Node sweeps keep the physical duration fixed and reduce ``dt`` when needed
    to satisfy the nearest-neighbour stability constraint. Particle sweeps use
    ``base_nodes`` and otherwise the same physical setup.
    """

    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if amortization_runs <= 0:
        raise ValueError("amortization_runs must be positive")
    if completion_rank <= 0:
        raise ValueError("completion_rank must be positive")
    if not node_counts or any(value < 2 for value in node_counts):
        raise ValueError("node_counts must contain integers >= 2")
    if not particle_counts or any(value <= 0 for value in particle_counts):
        raise ValueError("particle_counts must contain positive integers")
    unknown = set(methods) - set(BENCHMARK_METHOD_NAMES)
    if unknown:
        raise ValueError(f"unknown benchmark methods: {sorted(unknown)}")
    if handoff_time < 0.0 or handoff_time > total_time:
        raise ValueError("handoff_time must lie within [0, total_time]")

    records: list[DiffusionMethodBenchmark] = []
    unique_nodes = sorted(set(int(value) for value in node_counts))
    unique_particles = sorted(set(int(value) for value in particle_counts))

    for n_nodes in unique_nodes:
        config = stable_diffusion_config(
            n_particles=base_particles,
            n_nodes=n_nodes,
            total_time=total_time,
            requested_dt=requested_dt,
            length=length,
            diffusion_coefficient=diffusion_coefficient,
            impulse_fraction=impulse_fraction,
            stability_safety=stability_safety,
        )
        resolved_modes = _resolve_final_modes(
            n_nodes,
            final_modes=final_modes,
            retained_fraction=retained_fraction,
        )
        records.extend(
            _benchmark_one_configuration(
                config,
                sweep_axis="nodes",
                methods=methods,
                final_modes=resolved_modes,
                handoff_time=handoff_time,
                completion_rank=completion_rank,
                completion_ridge=completion_ridge,
                repeats=repeats,
                amortization_runs=amortization_runs,
                seed=seed,
            )
        )

    particle_config_nodes = int(base_nodes)
    particle_final_modes = _resolve_final_modes(
        particle_config_nodes,
        final_modes=final_modes,
        retained_fraction=retained_fraction,
    )
    for n_particles in unique_particles:
        config = stable_diffusion_config(
            n_particles=n_particles,
            n_nodes=particle_config_nodes,
            total_time=total_time,
            requested_dt=requested_dt,
            length=length,
            diffusion_coefficient=diffusion_coefficient,
            impulse_fraction=impulse_fraction,
            stability_safety=stability_safety,
        )
        records.extend(
            _benchmark_one_configuration(
                config,
                sweep_axis="particles",
                methods=methods,
                final_modes=particle_final_modes,
                handoff_time=handoff_time,
                completion_rank=completion_rank,
                completion_ridge=completion_ridge,
                repeats=repeats,
                amortization_runs=amortization_runs,
                seed=seed,
            )
        )

    return records


def _write_rows_csv(rows: list[dict[str, object]], output_path: str | Path) -> Path:
    if not rows:
        raise ValueError("records must not be empty")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def write_benchmark_csv(
    records: list[RandomWalkBenchmark],
    output_path: str | Path,
) -> Path:
    """Write random-walk benchmark summaries to CSV."""

    return _write_rows_csv([record.as_dict() for record in records], output_path)


def write_diffusion_method_benchmark_csv(
    records: list[DiffusionMethodBenchmark],
    output_path: str | Path,
) -> Path:
    """Write end-to-end method benchmark summaries to CSV."""

    return _write_rows_csv([record.as_dict() for record in records], output_path)


def _peak_rss_bytes() -> int:
    """Return this process's peak RSS in bytes on Unix-like systems."""

    import resource
    import sys

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux and the project's execution environment report KiB.
    return value if sys.platform == "darwin" else value * 1024


def _measure_operating_point(
    config: DiffusionConfig,
    point: OperatingPoint,
    *,
    completion_ridge: float,
    repeats: int,
    amortization_runs: int,
    seed: int,
    closure: str = DEFAULT_COMPLETION_CLOSURE,
    modal_bandwidth: int = DEFAULT_LOCAL_MODAL_BANDWIDTH,
    fitted_model: FittedLocalClosureModel | None = None,
    anchored_model: AnchoredFittedClosureModel | None = None,
    conditioned_model: PhysicsConditionedClosureModel | None = None,
) -> OperatingPointBenchmark:
    """Measure one completed and simplex-constrained pipeline in this process."""

    import os
    import threading
    import time

    stop_monitor = threading.Event()
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    sampled_rss = [0]

    def monitor_rss() -> None:
        while not stop_monitor.is_set():
            try:
                with open("/proc/self/statm", encoding="utf-8") as handle:
                    resident_pages = int(handle.read().split()[1])
                sampled_rss[0] = max(sampled_rss[0], resident_pages * page_size)
            except (OSError, ValueError, IndexError):
                sampled_rss[0] = max(sampled_rss[0], _peak_rss_bytes())
            time.sleep(0.001)

    monitor = threading.Thread(target=monitor_rss, daemon=True)
    monitor.start()

    final_modes, handoff_time, completion_rank = point.scaled(
        config.n_nodes,
        float(config.times[-1]),
    )
    setup_start = perf_counter()
    simulator = HandoffCorrelatedModalDiffusion(
        config=config,
        initial_n_modes=config.n_nodes,
        final_n_modes=final_modes,
        handoff_time=handoff_time,
    )
    completer_kwargs = {
        "retained_modes": final_modes,
        "completion_start_time": simulator.handoff_time,
        "completion_rank": completion_rank,
        "ridge": completion_ridge,
    }
    if closure == "dense_persistent":
        completer = PersistentUnresolvedGaussianCompleter(config, **completer_kwargs)
    elif closure == "local_persistent":
        completer_kwargs["modal_bandwidth"] = modal_bandwidth
        completer = LocalPersistentUnresolvedGaussianCompleter(
            config,
            **completer_kwargs,
        )
    elif closure == "fitted_local":
        if fitted_model is None:
            raise ValueError("fitted_local closure requires fitted_model")
        completer = FittedLocalPersistentCompleter(
            config,
            fitted_model,
            retained_modes=final_modes,
            completion_start_time=simulator.handoff_time,
        )
    elif closure == "anchored_fitted":
        if anchored_model is None:
            raise ValueError("anchored_fitted closure requires anchored_model")
        completer = AnchoredFittedLocalCompleter(
            config,
            anchored_model,
            retained_modes=final_modes,
            completion_start_time=simulator.handoff_time,
        )
    elif closure == "conditioned_fitted":
        if conditioned_model is None:
            raise ValueError("conditioned_fitted closure requires conditioned_model")
        completer = PhysicsConditionedLocalCompleter(
            config,
            conditioned_model,
            retained_modes=final_modes,
            completion_start_time=simulator.handoff_time,
        )
    else:
        raise ValueError(
            "closure must be dense_persistent, local_persistent, fitted_local, "
            "anchored_fitted, or conditioned_fitted"
        )
    setup_seconds = perf_counter() - setup_start
    setup_objects = (simulator, completer)
    setup_array_bytes = deep_numpy_nbytes(setup_objects)

    sequences = np.random.SeedSequence(
        [seed, config.n_nodes, point.reference_modes]
    ).spawn(repeats + 1)

    def run_once(sequence: np.random.SeedSequence) -> tuple[float, tuple[object, ...]]:
        simulation_sequence, completion_sequence = sequence.spawn(2)
        start = perf_counter()
        result = simulator.run(rng=np.random.default_rng(simulation_sequence))
        completed, additions = completer.complete(
            result.spatial_counts,
            rng=np.random.default_rng(completion_sequence),
        )
        constrained = simplex_bank_readout(completed, config.n_particles)
        elapsed = perf_counter() - start
        payload = _RunPayload(
            counts=constrained.counts,
            retained_objects=(result, completed, additions, constrained),
        )
        _validate_benchmark_output(payload, config)
        return elapsed, payload.retained_objects

    # Warm up lazy BLAS paths before collecting timings.
    run_once(sequences[0])
    timings: list[float] = []
    resident_array_bytes = setup_array_bytes
    for sequence in sequences[1:]:
        elapsed, retained = run_once(sequence)
        timings.append(elapsed)
        resident_array_bytes = max(
            resident_array_bytes,
            deep_numpy_nbytes((setup_objects, retained)),
        )

    values = np.asarray(timings, dtype=float)
    median = float(np.median(values))
    stop_monitor.set()
    monitor.join()
    return OperatingPointBenchmark(
        operating_point=point.name,
        closure=closure,
        modal_bandwidth=(
            modal_bandwidth
            if closure
            in {
                "local_persistent",
                "fitted_local",
                "anchored_fitted",
                "conditioned_fitted",
            }
            else 0
        ),
        n_particles=config.n_particles,
        n_nodes=config.n_nodes,
        n_steps=config.n_steps,
        final_modes=final_modes,
        handoff_time=handoff_time,
        completion_rank=completion_rank,
        repeats=repeats,
        amortization_runs=amortization_runs,
        setup_seconds=setup_seconds,
        median_run_seconds=median,
        mean_run_seconds=float(np.mean(values)),
        std_run_seconds=float(np.std(values, ddof=0)),
        amortized_seconds_per_run=median + setup_seconds / amortization_runs,
        setup_array_bytes=setup_array_bytes,
        resident_array_bytes=resident_array_bytes,
        peak_rss_bytes=max(sampled_rss[0], 1),
    )


def _operating_point_worker(
    queue,
    config: DiffusionConfig,
    point: OperatingPoint,
    completion_ridge: float,
    repeats: int,
    amortization_runs: int,
    seed: int,
    closure: str,
    modal_bandwidth: int,
    fitted_model: FittedLocalClosureModel | None,
    anchored_model: AnchoredFittedClosureModel | None,
    conditioned_model: PhysicsConditionedClosureModel | None,
) -> None:
    """Spawn-process entry point used to isolate peak RSS measurements."""

    try:
        queue.put(
            (
                True,
                _measure_operating_point(
                    config,
                    point,
                    completion_ridge=completion_ridge,
                    repeats=repeats,
                    amortization_runs=amortization_runs,
                    seed=seed,
                    closure=closure,
                    modal_bandwidth=modal_bandwidth,
                    fitted_model=fitted_model,
                    anchored_model=anchored_model,
                    conditioned_model=conditioned_model,
                ),
            )
        )
    except BaseException as error:  # pragma: no cover - child failure path
        queue.put((False, f"{type(error).__name__}: {error}"))


def benchmark_operating_points(
    *,
    node_counts: list[int],
    n_particles: int = 5_275,
    total_time: float = 100.0,
    requested_dt: float = 1.0,
    length: float = 4.0,
    diffusion_coefficient: float = 2.20e-4,
    impulse_fraction: float = 0.59,
    operating_points: tuple[OperatingPoint, ...] = SELECTED_OPERATING_POINTS,
    completion_ridge: float = 1.0e-2,
    repeats: int = 3,
    amortization_runs: int = 100,
    stability_safety: float = 0.95,
    seed: int = 0,
    isolate_peak_rss: bool = True,
    closure: str = DEFAULT_COMPLETION_CLOSURE,
    modal_bandwidth: int = DEFAULT_LOCAL_MODAL_BANDWIDTH,
    fitted_model: FittedLocalClosureModel | None = None,
    anchored_model: AnchoredFittedClosureModel | None = None,
    conditioned_model: PhysicsConditionedClosureModel | None = None,
) -> list[OperatingPointBenchmark]:
    """Benchmark selected pipelines across spatial sizes.

    With ``isolate_peak_rss=True``, every configuration runs in a fresh spawned
    process.  This prevents an earlier, larger allocation from contaminating a
    later process-wide peak-RSS measurement.
    """

    if not node_counts or any(value < 2 for value in node_counts):
        raise ValueError("node_counts must contain integers >= 2")
    if repeats <= 0 or amortization_runs <= 0:
        raise ValueError("repeats and amortization_runs must be positive")
    records: list[OperatingPointBenchmark] = []
    for n_nodes in sorted(set(int(value) for value in node_counts)):
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
        for point_index, point in enumerate(operating_points):
            point_seed = seed + 10_000 * n_nodes + point_index
            if not isolate_peak_rss:
                record = _measure_operating_point(
                    config,
                    point,
                    completion_ridge=completion_ridge,
                    repeats=repeats,
                    amortization_runs=amortization_runs,
                    seed=point_seed,
                    closure=closure,
                    modal_bandwidth=modal_bandwidth,
                    fitted_model=fitted_model,
                    anchored_model=anchored_model,
                    conditioned_model=conditioned_model,
                )
            else:
                context = mp.get_context("spawn")
                queue = context.Queue()
                process = context.Process(
                    target=_operating_point_worker,
                    args=(
                        queue,
                        config,
                        point,
                        completion_ridge,
                        repeats,
                        amortization_runs,
                        point_seed,
                        closure,
                        modal_bandwidth,
                        fitted_model,
                        anchored_model,
                        conditioned_model,
                    ),
                )
                process.start()
                success, payload = queue.get()
                process.join()
                queue.close()
                if not success:
                    raise RuntimeError(
                        f"operating-point benchmark failed for {point.name}: {payload}"
                    )
                if process.exitcode != 0:
                    raise RuntimeError(
                        f"operating-point benchmark exited with code {process.exitcode}"
                    )
                record = payload
            records.append(record)
    return records


def write_operating_point_benchmark_csv(
    records: list[OperatingPointBenchmark],
    output_path: str | Path,
) -> Path:
    """Write selected operating-point benchmark records."""

    return _write_rows_csv([record.as_dict() for record in records], output_path)
