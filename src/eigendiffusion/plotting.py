"""Small plotting helpers used by the command-line examples."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .config import DiffusionConfig

FloatArray = NDArray[np.float64]


def plot_validation(
    config: DiffusionConfig,
    deterministic: FloatArray,
    eigenmarkov_mean: FloatArray,
    eigenmarkov_std: FloatArray,
    random_walk_mean: FloatArray,
    random_walk_std: FloatArray,
    output_path: str | Path,
    profile_step: int | None = None,
) -> Path:
    """Plot one spatial profile and the impulse-node time course."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    step = min(config.n_steps - 1, 20) if profile_step is None else profile_step
    if not 0 <= step < config.n_steps:
        raise ValueError("profile_step must index an available timestep")

    x = config.positions
    t = config.times
    impulse = config.impulse_index

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(x, deterministic[step], label="deterministic mean", linewidth=2.2)
    axes[0].plot(x, random_walk_mean[step], label="random-walk mean", linewidth=1.8)
    axes[0].fill_between(
        x,
        random_walk_mean[step] - random_walk_std[step],
        random_walk_mean[step] + random_walk_std[step],
        alpha=0.18,
    )
    axes[0].plot(x, eigenmarkov_mean[step], label="EigenMarkov mean", linewidth=1.8)
    axes[0].fill_between(
        x,
        eigenmarkov_mean[step] - eigenmarkov_std[step],
        eigenmarkov_mean[step] + eigenmarkov_std[step],
        alpha=0.18,
    )
    axes[0].set_title(f"Spatial profile at t = {t[step]:g} µs")
    axes[0].set_xlabel("distance (µm)")
    axes[0].set_ylabel("particle count")
    axes[0].legend(fontsize=9)

    axes[1].plot(t, deterministic[:, impulse], label="deterministic mean", linewidth=2.2)
    axes[1].plot(t, random_walk_mean[:, impulse], label="random-walk mean", linewidth=1.8)
    axes[1].fill_between(
        t,
        random_walk_mean[:, impulse] - random_walk_std[:, impulse],
        random_walk_mean[:, impulse] + random_walk_std[:, impulse],
        alpha=0.18,
    )
    axes[1].plot(t, eigenmarkov_mean[:, impulse], label="EigenMarkov mean", linewidth=1.8)
    axes[1].fill_between(
        t,
        eigenmarkov_mean[:, impulse] - eigenmarkov_std[:, impulse],
        eigenmarkov_mean[:, impulse] + eigenmarkov_std[:, impulse],
        alpha=0.18,
    )
    axes[1].set_title("Impulse-node time course")
    axes[1].set_xlabel("time (µs)")
    axes[1].set_ylabel("particle count")
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_mode_sweep(
    modes: list[int],
    relative_errors: list[float],
    negative_fractions: list[float],
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(modes, relative_errors, marker="o")
    axes[0].set_xlabel("retained eigenmodes")
    axes[0].set_ylabel("relative L2 error")
    axes[0].set_title("Mean accuracy")

    axes[1].plot(modes, negative_fractions, marker="o")
    axes[1].set_xlabel("retained eigenmodes")
    axes[1].set_ylabel("fraction below zero")
    axes[1].set_title("Nonnegativity diagnostic")

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
