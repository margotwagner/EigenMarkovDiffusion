"""Plotting helpers for validation, sweeps, and benchmarks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .config import DiffusionConfig

FloatArray = NDArray[np.float64]


def plot_validation(
    config: DiffusionConfig,
    continuous_reference: FloatArray,
    discrete_reference: FloatArray,
    analytic_variance: FloatArray,
    eigenmarkov_runs: FloatArray,
    random_walk_runs: FloatArray,
    output_path: str | Path,
    profile_step: int | None = None,
) -> Path:
    """Plot mean, variance, and nonnegativity diagnostics."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    step = min(config.n_steps - 1, 20) if profile_step is None else profile_step
    if not 0 <= step < config.n_steps:
        raise ValueError("profile_step must index an available timestep")

    em_mean = np.mean(eigenmarkov_runs, axis=0)
    em_std = np.std(eigenmarkov_runs, axis=0, ddof=0)
    em_variance = np.var(eigenmarkov_runs, axis=0, ddof=0)
    rw_mean = np.mean(random_walk_runs, axis=0)
    rw_std = np.std(random_walk_runs, axis=0, ddof=0)
    rw_variance = np.var(random_walk_runs, axis=0, ddof=0)

    em_negative_mass = (
        np.clip(-eigenmarkov_runs, 0.0, None).sum(axis=-1) / config.n_particles
    )
    rw_negative_mass = (
        np.clip(-random_walk_runs, 0.0, None).sum(axis=-1) / config.n_particles
    )

    x = config.positions
    t = config.times
    impulse = config.impulse_index

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    ax = axes[0, 0]
    ax.plot(x, continuous_reference[step], "--", label="continuous reference", linewidth=1.8)
    ax.plot(x, discrete_reference[step], label="discrete expected mean", linewidth=2.2)
    ax.plot(x, rw_mean[step], label="random-walk mean", linewidth=1.8)
    ax.fill_between(
        x,
        rw_mean[step] - rw_std[step],
        rw_mean[step] + rw_std[step],
        alpha=0.15,
    )
    ax.plot(x, em_mean[step], label="EigenMarkov mean", linewidth=1.8)
    ax.fill_between(
        x,
        em_mean[step] - em_std[step],
        em_mean[step] + em_std[step],
        alpha=0.15,
    )
    ax.set_title(f"Spatial mean at t = {t[step]:g} µs")
    ax.set_xlabel("distance (µm)")
    ax.set_ylabel("particle count")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(
        t,
        continuous_reference[:, impulse],
        "--",
        label="continuous reference",
        linewidth=1.8,
    )
    ax.plot(
        t,
        discrete_reference[:, impulse],
        label="discrete expected mean",
        linewidth=2.2,
    )
    ax.plot(t, rw_mean[:, impulse], label="random-walk mean", linewidth=1.8)
    ax.fill_between(
        t,
        rw_mean[:, impulse] - rw_std[:, impulse],
        rw_mean[:, impulse] + rw_std[:, impulse],
        alpha=0.15,
    )
    ax.plot(t, em_mean[:, impulse], label="EigenMarkov mean", linewidth=1.8)
    ax.fill_between(
        t,
        em_mean[:, impulse] - em_std[:, impulse],
        em_mean[:, impulse] + em_std[:, impulse],
        alpha=0.15,
    )
    ax.set_title("Impulse-node mean time course")
    ax.set_xlabel("time (µs)")
    ax.set_ylabel("particle count")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(x, analytic_variance[step], label="analytic multinomial variance", linewidth=2.2)
    ax.plot(x, rw_variance[step], label="random-walk sample variance", linewidth=1.8)
    ax.plot(x, em_variance[step], label="EigenMarkov sample variance", linewidth=1.8)
    ax.set_title(f"Spatial variance at t = {t[step]:g} µs")
    ax.set_xlabel("distance (µm)")
    ax.set_ylabel("particle-count variance")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(t, np.mean(rw_negative_mass, axis=0), label="random walk", linewidth=1.8)
    ax.plot(t, np.mean(em_negative_mass, axis=0), label="EigenMarkov", linewidth=1.8)
    ax.set_title("Mean negative mass fraction")
    ax.set_xlabel("time (µs)")
    ax.set_ylabel("negative mass / total particles")
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_mode_sweep(
    modes: list[int],
    time_window_errors: dict[float, list[float]],
    variance_errors: list[float],
    negative_mass_fractions: list[float],
    negative_entry_fractions: list[float],
    output_path: str | Path,
) -> Path:
    """Plot time-resolved mean error and physicality diagnostics."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))

    ax = axes[0, 0]
    for start_time, errors in sorted(time_window_errors.items()):
        ax.plot(modes, errors, marker="o", label=f"t ≥ {start_time:g} µs")
    ax.set_xlabel("retained eigenmodes")
    ax.set_ylabel("relative L2 mean error")
    ax.set_title("Mean accuracy by evaluation window")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(modes, variance_errors, marker="o")
    ax.set_xlabel("retained eigenmodes")
    ax.set_ylabel("relative L2 variance error")
    ax.set_title("Variance accuracy")

    ax = axes[1, 0]
    ax.plot(modes, negative_mass_fractions, marker="o")
    ax.set_xlabel("retained eigenmodes")
    ax.set_ylabel("mean negative mass fraction")
    ax.set_title("Negative reconstructed mass")

    ax = axes[1, 1]
    ax.plot(modes, negative_entry_fractions, marker="o")
    ax.set_xlabel("retained eigenmodes")
    ax.set_ylabel("fraction of entries below zero")
    ax.set_title("Nonnegativity diagnostic")

    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_random_walk_benchmark(
    records,
    output_path: str | Path,
) -> Path:
    """Plot runtime and estimated core-array memory for both random walks."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    methods = ("naive", "multinomial")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for method in methods:
        selected = sorted(
            (record for record in records if record.method == method),
            key=lambda record: record.n_particles,
        )
        if not selected:
            continue
        particles = [record.n_particles for record in selected]
        runtimes = [record.median_seconds for record in selected]
        memory = [record.core_array_megabytes for record in selected]
        axes[0].plot(particles, runtimes, marker="o", label=method)
        axes[1].plot(particles, memory, marker="o", label=method)

    axes[0].set_xlabel("number of particles")
    axes[0].set_ylabel("median runtime (seconds)")
    axes[0].set_title("Random-walk runtime")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].legend()

    axes[1].set_xlabel("number of particles")
    axes[1].set_ylabel("core-array memory (MiB)")
    axes[1].set_title("Implementation storage")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output
