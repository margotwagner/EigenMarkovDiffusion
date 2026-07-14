"""Reproducible runtime and core-array memory benchmarks."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np

from .baselines import (
    multinomial_random_walk_diffusion,
    naive_random_walk_diffusion,
)
from .config import DiffusionConfig


@dataclass(frozen=True, slots=True)
class RandomWalkBenchmark:
    """Summary for one method and one particle count."""

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


def estimate_random_walk_core_bytes(
    config: DiffusionConfig,
    method: str,
) -> int:
    """Estimate memory occupied by the principal arrays in each implementation.

    This is intentionally an implementation-level estimate rather than a claim
    about full process resident memory. Both implementations save the complete
    node-count history. The naive method additionally stores every particle's
    complete trajectory; the multinomial method uses one temporary node vector.
    """

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

        # Alternate execution order across repeats to reduce systematic ordering bias.
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


def write_benchmark_csv(
    records: list[RandomWalkBenchmark],
    output_path: str | Path,
) -> Path:
    """Write benchmark summaries to CSV."""

    if not records:
        raise ValueError("records must not be empty")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [record.as_dict() for record in records]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output
