"""Plotting helpers for validation, model comparison, sweeps, and benchmarks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .config import DiffusionConfig

FloatArray = NDArray[np.float64]


def _display_name(model_name: str) -> str:
    names = {
        "independent_modal": "independent modal",
        "correlated_modal": "correlated modal",
        "banked_correlated_modal": "banked correlated modal",
        "handoff_correlated_modal": "handoff correlated modal",
        "random_walk_multinomial": "multinomial random walk",
        "random_walk_naive": "naive random walk",
        "raw": "raw readout",
        "simplex_bank": "simplex-bank readout",
        "delta_sigma_temporal": "temporal ΔΣ readout",
        "delta_sigma_neighbor": "neighbor ΔΣ readout",
        "unresolved_gaussian_completion": "unresolved Gaussian completion",
        "analytic_reference": "analytic reference",
        "full_correlated_modal": "full correlated modal",
        "handoff_raw": "101→50 handoff + raw",
        "handoff_completion": "101→50 handoff + rank-10 completion",
    }
    if "+" in model_name:
        model, readout = model_name.split("+", maxsplit=1)
        model_text = names.get(model, model.replace("_", " "))
        readout_text = names.get(readout, readout.replace("_", " "))
        return f"{model_text} + {readout_text}"
    return names.get(model_name, model_name.replace("_", " "))


def plot_validation(
    config: DiffusionConfig,
    continuous_reference: FloatArray,
    discrete_reference: FloatArray,
    analytic_variance: FloatArray,
    modal_runs: FloatArray,
    random_walk_runs: FloatArray,
    output_path: str | Path,
    profile_step: int | None = None,
    modal_label: str = "independent modal",
) -> Path:
    """Plot mean, variance, and nonnegativity diagnostics for one modal model."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    step = min(config.n_steps - 1, 20) if profile_step is None else profile_step
    if not 0 <= step < config.n_steps:
        raise ValueError("profile_step must index an available timestep")

    modal_mean = np.mean(modal_runs, axis=0)
    modal_std = np.std(modal_runs, axis=0, ddof=0)
    modal_variance = np.var(modal_runs, axis=0, ddof=0)
    rw_mean = np.mean(random_walk_runs, axis=0)
    rw_std = np.std(random_walk_runs, axis=0, ddof=0)
    rw_variance = np.var(random_walk_runs, axis=0, ddof=0)

    modal_negative_mass = (
        np.clip(-modal_runs, 0.0, None).sum(axis=-1) / config.n_particles
    )
    rw_negative_mass = (
        np.clip(-random_walk_runs, 0.0, None).sum(axis=-1) / config.n_particles
    )

    x = config.positions
    t = config.times
    impulse = config.impulse_index
    display_label = _display_name(modal_label)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    ax = axes[0, 0]
    ax.plot(
        x,
        continuous_reference[step],
        "--",
        label="continuous reference",
        linewidth=1.8,
    )
    ax.plot(x, discrete_reference[step], label="discrete expected mean", linewidth=2.2)
    ax.plot(x, rw_mean[step], label="random-walk mean", linewidth=1.8)
    ax.fill_between(
        x,
        rw_mean[step] - rw_std[step],
        rw_mean[step] + rw_std[step],
        alpha=0.15,
    )
    ax.plot(x, modal_mean[step], label=f"{display_label} mean", linewidth=1.8)
    ax.fill_between(
        x,
        modal_mean[step] - modal_std[step],
        modal_mean[step] + modal_std[step],
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
    ax.plot(t, modal_mean[:, impulse], label=f"{display_label} mean", linewidth=1.8)
    ax.fill_between(
        t,
        modal_mean[:, impulse] - modal_std[:, impulse],
        modal_mean[:, impulse] + modal_std[:, impulse],
        alpha=0.15,
    )
    ax.set_title("Impulse-node mean time course")
    ax.set_xlabel("time (µs)")
    ax.set_ylabel("particle count")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(
        x,
        analytic_variance[step],
        label="analytic multinomial variance",
        linewidth=2.2,
    )
    ax.plot(x, rw_variance[step], label="random-walk sample variance", linewidth=1.8)
    ax.plot(
        x,
        modal_variance[step],
        label=f"{display_label} sample variance",
        linewidth=1.8,
    )
    ax.set_title(f"Spatial variance at t = {t[step]:g} µs")
    ax.set_xlabel("distance (µm)")
    ax.set_ylabel("particle-count variance")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(t, np.mean(rw_negative_mass, axis=0), label="random walk", linewidth=1.8)
    ax.plot(
        t,
        np.mean(modal_negative_mass, axis=0),
        label=display_label,
        linewidth=1.8,
    )
    ax.set_title("Mean negative mass fraction")
    ax.set_xlabel("time (µs)")
    ax.set_ylabel("negative mass / total particles")
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_model_comparison(
    config: DiffusionConfig,
    discrete_reference: FloatArray,
    analytic_variance: FloatArray,
    model_runs: dict[str, FloatArray],
    diagnostics: dict[str, dict[str, float | None]],
    output_path: str | Path,
    profile_step: int | None = None,
) -> Path:
    """Compare all modal formulations in one figure."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    step = min(config.n_steps - 1, 20) if profile_step is None else profile_step
    if not 0 <= step < config.n_steps:
        raise ValueError("profile_step must index an available timestep")

    t = config.times
    x = config.positions
    impulse = config.impulse_index
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    ax = axes[0, 0]
    ax.plot(
        t,
        discrete_reference[:, impulse],
        "--",
        linewidth=2.2,
        label="discrete expected mean",
    )
    for name, runs in model_runs.items():
        ax.plot(t, runs.mean(axis=0)[:, impulse], label=_display_name(name), linewidth=1.7)
    ax.set_title("Impulse-node mean")
    ax.set_xlabel("time (µs)")
    ax.set_ylabel("particle count")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(x, analytic_variance[step], "--", linewidth=2.2, label="analytic variance")
    for name, runs in model_runs.items():
        ax.plot(
            x,
            np.var(runs, axis=0, ddof=0)[step],
            label=_display_name(name),
            linewidth=1.7,
        )
    ax.set_title(f"Spatial variance at t = {t[step]:g} µs")
    ax.set_xlabel("distance (µm)")
    ax.set_ylabel("particle-count variance")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for name, runs in model_runs.items():
        negative_mass = np.clip(-runs, 0.0, None).sum(axis=-1) / config.n_particles
        ax.plot(t, negative_mass.mean(axis=0), label=_display_name(name), linewidth=1.7)
    ax.set_title("Mean negative mass fraction")
    ax.set_xlabel("time (µs)")
    ax.set_ylabel("negative mass / total particles")
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    model_names = list(model_runs)
    x_positions = np.arange(len(model_names), dtype=float)
    width = 0.25
    metric_specs = (
        ("mean_relative_l2_error", "mean"),
        ("variance_relative_l2_error", "variance"),
        ("covariance_relative_frobenius_error", "covariance"),
    )
    for metric_index, (key, label) in enumerate(metric_specs):
        values = [
            np.nan if diagnostics[name].get(key) is None else float(diagnostics[name][key])
            for name in model_names
        ]
        ax.bar(x_positions + (metric_index - 1) * width, values, width, label=label)
    ax.set_xticks(x_positions, [_display_name(name) for name in model_names], rotation=15)
    ax.set_ylabel("relative error (lower is better)")
    ax.set_title("Ensemble diagnostics")
    ax.set_yscale("log")
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
    model_label: str = "modal model",
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
    ax.set_title("Variance relative error — lower is better")

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

    fig.suptitle(_display_name(model_label), y=1.01)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output



def plot_handoff_sweep(
    rows: list[dict[str, float | int | str | None]],
    final_modes: list[int],
    handoff_times: list[float],
    late_start_time: float,
    output_path: str | Path,
    readout_label: str = "raw",
) -> Path:
    """Plot full-to-reduced handoff accuracy over times and final dimensions."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    late_key = (
        f"mean_relative_l2_t_ge_{late_start_time:g}us"
        .replace("-", "m")
        .replace(".", "p")
    )

    by_pair = {
        (float(row["requested_handoff_time"]), int(row["final_n_modes"])): row
        for row in rows
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))

    panels = (
        ("mean_relative_l2_error", "All-time mean error", "relative L2 mean error"),
        (late_key, f"Mean error for t ≥ {late_start_time:g} µs", "relative L2 mean error"),
        ("variance_relative_l2_error", "Variance error", "relative L2 variance error"),
        (
            "mean_handoff_projection_relative_l2",
            "One-time handoff projection error",
            "relative L2 projection error",
        ),
    )

    for ax, (metric, title, ylabel) in zip(axes.ravel(), panels, strict=True):
        for handoff_time in handoff_times:
            values = []
            for mode_count in final_modes:
                row = by_pair[(float(handoff_time), int(mode_count))]
                value = row.get(metric)
                values.append(np.nan if value is None else float(value))
            ax.plot(final_modes, values, marker="o", label=f"handoff {handoff_time:g} µs")
        ax.set_xlabel("final retained eigenmodes")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Full-to-reduced correlated modal handoff + {_display_name(readout_label)}",
        y=1.01,
    )
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


def plot_completion_rank_sweep(
    ranks: list[int],
    mean_errors: list[float],
    variance_errors: list[float],
    covariance_errors: list[float],
    negative_mass_fractions: list[float],
    completion_l1_fractions: list[float],
    output_path: str | Path,
) -> Path:
    """Plot accuracy and physicality against stochastic completion rank."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))

    axes[0, 0].plot(ranks, mean_errors, marker="o")
    axes[0, 0].set_title("Mean relative error")
    axes[0, 0].set_xlabel("completion rank (0 = raw)")
    axes[0, 0].set_ylabel("relative L2 error")

    axes[0, 1].plot(ranks, variance_errors, marker="o", label="variance")
    axes[0, 1].plot(ranks, covariance_errors, marker="o", label="covariance")
    axes[0, 1].set_title("Stochastic moment error")
    axes[0, 1].set_xlabel("completion rank (0 = raw)")
    axes[0, 1].set_ylabel("relative error")
    axes[0, 1].legend()

    axes[1, 0].plot(ranks, negative_mass_fractions, marker="o")
    axes[1, 0].set_title("Negative reconstructed mass")
    axes[1, 0].set_xlabel("completion rank (0 = raw)")
    axes[1, 0].set_ylabel("mean negative mass fraction")

    axes[1, 1].plot(ranks, completion_l1_fractions, marker="o")
    axes[1, 1].set_title("Added unresolved field")
    axes[1, 1].set_xlabel("completion rank (0 = raw)")
    axes[1, 1].set_ylabel("mean L1 addition / total mass")

    fig.suptitle("Unresolved Gaussian completion rank sweep", y=1.01)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_temporal_correlation_comparison(
    *,
    lag_times: list[float],
    covariance_errors: dict[str, list[float]],
    impulse_correlations: dict[str, list[float]],
    window_correlations: dict[str, list[float]],
    profile_positions: FloatArray,
    profile_covariances: dict[str, FloatArray],
    profile_lag_time: float,
    output_path: str | Path,
) -> Path:
    """Plot cross-time covariance and lag-correlation diagnostics."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))

    ax = axes[0, 0]
    for name, values in covariance_errors.items():
        ax.plot(lag_times, values, marker="o", label=_display_name(name))
    ax.set_xlabel("time lag (µs)")
    ax.set_ylabel("relative Frobenius error")
    ax.set_title("Cross-time covariance error")
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for name, values in impulse_correlations.items():
        ax.plot(lag_times, values, marker="o", label=_display_name(name))
    ax.set_xlabel("time lag (µs)")
    ax.set_ylabel("mean lag correlation")
    ax.set_title("Impulse-node temporal correlation")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for name, values in window_correlations.items():
        ax.plot(lag_times, values, marker="o", label=_display_name(name))
    ax.set_xlabel("time lag (µs)")
    ax.set_ylabel("mean lag correlation")
    ax.set_title("Central-window temporal correlation")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for name, values in profile_covariances.items():
        linestyle = "--" if name == "analytic_reference" else "-"
        ax.plot(
            profile_positions,
            values,
            linestyle,
            linewidth=1.8,
            label=_display_name(name),
        )
    ax.set_xlabel("distance (µm)")
    ax.set_ylabel("mean diagonal cross-time covariance")
    ax.set_title(f"Spatial persistence profile at lag {profile_lag_time:g} µs")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output
