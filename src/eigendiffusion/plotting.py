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
        "persistent_unresolved_completion": "persistent unresolved completion",
        "unresolved_gaussian_completion_simplex_bank": (
            "unresolved completion + simplex bank"
        ),
        "persistent_unresolved_completion_simplex_bank": (
            "persistent completion + simplex bank"
        ),
        "multinomial_random_walk": "multinomial random walk",
        "full_correlated_modal": "full correlated modal",
        "handoff_raw": "101→reduced handoff + raw",
        "handoff_independent_completion": "handoff + independent completion",
        "handoff_persistent_completion": "handoff + persistent completion",
        "analytic_reference": "analytic reference",
        "handoff_completion": "handoff + rank-10 completion",
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
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    ax = axes[0, 0]
    for name, runs in model_runs.items():
        observed_mean = runs.mean(axis=0)
        numerator = np.linalg.norm(observed_mean - discrete_reference, axis=1)
        denominator = np.linalg.norm(discrete_reference, axis=1)
        error = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator > 0.0,
        )
        ax.plot(t, error, label=_display_name(name), linewidth=1.7)
    ax.set_title("Mean-profile error by time")
    ax.set_xlabel("time (µs)")
    ax.set_ylabel("relative L2 mean error")
    ax.set_yscale("log")
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


def plot_trial_convergence(
    rows: list[dict[str, float | int | str | None]],
    times: FloatArray,
    mean_error_by_time: dict[str, FloatArray],
    output_path: str | Path,
) -> Path:
    """Plot moment-error convergence and expose time-resolved mean bias."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model_names = list(mean_error_by_time)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    metric_specs = (
        ("mean_relative_l2_error", "Mean error convergence", "relative L2 mean error"),
        (
            "variance_relative_l2_error",
            "Variance error convergence",
            "relative L2 variance error",
        ),
        (
            "covariance_relative_frobenius_error",
            "Covariance error convergence",
            "relative Frobenius covariance error",
        ),
    )

    for ax, (metric, title, ylabel) in zip(axes.ravel()[:3], metric_specs, strict=True):
        for name in model_names:
            selected = sorted(
                (row for row in rows if row["model"] == name),
                key=lambda row: int(row["trial_count"]),
            )
            counts = np.asarray([int(row["trial_count"]) for row in selected])
            values = np.asarray([float(row[metric]) for row in selected])
            lower = np.asarray(
                [float(row[f"{metric}_ci_lower"]) for row in selected]
            )
            upper = np.asarray(
                [float(row[f"{metric}_ci_upper"]) for row in selected]
            )
            ax.plot(counts, values, marker="o", label=_display_name(name))
            ax.fill_between(counts, lower, upper, alpha=0.13)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("number of trials")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.2)

    ax = axes[1, 1]
    for name, error in mean_error_by_time.items():
        ax.plot(times, error, label=_display_name(name), linewidth=1.7)
    ax.set_yscale("log")
    ax.set_xlabel("time (µs)")
    ax.set_ylabel("relative L2 mean error")
    ax.set_title("Mean-profile error at maximum trial count")
    ax.grid(alpha=0.2)

    axes[0, 0].legend(fontsize=7)
    ax.legend(fontsize=7)
    fig.suptitle(
        "Trial-count convergence\nshading: block-bootstrap confidence interval",
        y=1.01,
    )
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


def plot_handoff_selection(
    rows: list[dict[str, float | int | str | bool | None]],
    final_modes: list[int],
    handoff_times: list[float],
    output_path: str | Path,
) -> Path:
    """Plot analytic handoff predictions and optional validation results."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    by_pair = {
        (float(row["requested_handoff_time"]), int(row["final_n_modes"])): row
        for row in rows
    }
    mean_grid = np.asarray(
        [
            [
                float(by_pair[(time, mode)]["predicted_mean_relative_l2_error"])
                for mode in final_modes
            ]
            for time in handoff_times
        ]
    )
    covariance_grid = np.asarray(
        [
            [
                float(
                    by_pair[(time, mode)][
                        "predicted_covariance_relative_frobenius_error"
                    ]
                )
                for mode in final_modes
            ]
            for time in handoff_times
        ]
    )
    work_grid = np.asarray(
        [
            [
                float(by_pair[(time, mode)]["modal_update_fraction_of_full"])
                for mode in final_modes
            ]
            for time in handoff_times
        ]
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    grids = (
        (mean_grid, "Predicted mean truncation error"),
        (covariance_grid, "Predicted covariance truncation error"),
        (work_grid, "Modal-update work fraction"),
    )
    for ax, (grid, title) in zip(axes.ravel()[:3], grids, strict=True):
        image = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(final_modes)), final_modes)
        ax.set_yticks(range(len(handoff_times)), [f"{value:g}" for value in handoff_times])
        ax.set_xlabel("retained modes")
        ax.set_ylabel("handoff time (µs)")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    validated = [
        row
        for row in rows
        if row.get("observed_covariance_relative_frobenius_error") is not None
    ]
    if validated:
        specs = (
            (
                "predicted_mean_with_sampling_error",
                "observed_mean_relative_l2_error",
                "mean",
            ),
            (
                "predicted_variance_with_sampling_error",
                "observed_variance_relative_l2_error",
                "variance",
            ),
            (
                "predicted_covariance_with_sampling_error",
                "observed_covariance_relative_frobenius_error",
                "covariance",
            ),
        )
        all_values: list[float] = []
        for predicted_key, observed_key, label in specs:
            predicted = np.asarray([float(row[predicted_key]) for row in validated])
            observed = np.asarray([float(row[observed_key]) for row in validated])
            ax.scatter(predicted, observed, label=label, alpha=0.8)
            all_values.extend(predicted.tolist())
            all_values.extend(observed.tolist())
        positive = np.asarray([value for value in all_values if value > 0.0])
        lower = float(np.min(positive)) * 0.8
        upper = float(np.max(positive)) * 1.2
        ax.plot([lower, upper], [lower, upper], "--", color="black", linewidth=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("analytic prediction + sampling floor")
        ax.set_ylabel("observed ensemble error")
        ax.set_title("Held-out validation")
        ax.legend(fontsize=8)
    else:
        frontier = [row for row in rows if bool(row.get("pareto_optimal"))]
        ax.scatter(
            [float(row["modal_update_fraction_of_full"]) for row in rows],
            [float(row["predicted_covariance_relative_frobenius_error"]) for row in rows],
            alpha=0.45,
            label="candidates",
        )
        ax.scatter(
            [float(row["modal_update_fraction_of_full"]) for row in frontier],
            [float(row["predicted_covariance_relative_frobenius_error"]) for row in frontier],
            label="Pareto frontier",
        )
        ax.set_xlabel("modal-update work fraction")
        ax.set_ylabel("predicted covariance error")
        ax.set_title("Accuracy/work frontier")
        ax.legend(fontsize=8)

    fig.suptitle("Predictive handoff selection", y=1.01)
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
    rows: list[dict[str, float | int | str | None]],
    output_path: str | Path,
) -> Path:
    """Plot completion accuracy, persistence, and physicality by rank."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.0))
    readouts = []
    for row in rows:
        name = str(row["readout"])
        if name != "raw" and name not in readouts:
            readouts.append(name)

    panels = (
        ("mean_relative_l2_error", "Mean error", "relative L2 error"),
        ("variance_relative_l2_error", "Variance error", "relative L2 error"),
        (
            "covariance_relative_frobenius_error",
            "Same-time covariance error",
            "relative Frobenius error",
        ),
        (
            "cross_time_covariance_relative_frobenius_error",
            "Cross-time covariance error",
            "relative Frobenius error",
        ),
        (
            "mean_negative_mass_fraction",
            "Negative mass",
            "mean negative mass / total",
        ),
        (
            "mean_constraint_residual_l1_fraction",
            "Constraint residual",
            "mean residual L1 / total",
        ),
    )
    raw = next(row for row in rows if str(row["readout"]) == "raw")
    for ax, (metric, title, ylabel) in zip(axes.ravel(), panels, strict=True):
        for readout in readouts:
            selected = sorted(
                (row for row in rows if str(row["readout"]) == readout),
                key=lambda row: int(row["completion_rank"]),
            )
            ranks = [0] + [int(row["completion_rank"]) for row in selected]
            raw_value = raw.get(metric)
            values = [0.0 if raw_value is None else float(raw_value)] + [
                0.0 if row.get(metric) is None else float(row[metric])
                for row in selected
            ]
            ax.plot(ranks, values, marker="o", label=_display_name(readout))
        ax.set_title(title)
        ax.set_xlabel("completion rank (0 = raw)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Completion rank, temporal persistence, and physical constraints", y=1.01)
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


def plot_diffusion_method_benchmark(
    records,
    output_path: str | Path,
) -> Path:
    """Plot runtime, setup cost, and resident-array scaling for main methods."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    methods = []
    for record in records:
        if record.method not in methods:
            methods.append(record.method)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    node_records = [record for record in records if record.sweep_axis == "nodes"]
    particle_records = [record for record in records if record.sweep_axis == "particles"]

    for method in methods:
        selected = sorted(
            (record for record in node_records if record.method == method),
            key=lambda record: record.n_nodes,
        )
        if selected:
            nodes = [record.n_nodes for record in selected]
            run_time = [record.median_run_seconds for record in selected]
            setup_time = [record.setup_seconds for record in selected]
            memory = [record.resident_array_megabytes for record in selected]
            label = _display_name(method)
            axes[0, 0].plot(nodes, run_time, marker="o", label=label)
            axes[0, 1].plot(nodes, setup_time, marker="o", label=label)
            axes[1, 0].plot(nodes, memory, marker="o", label=label)

        selected_particles = sorted(
            (record for record in particle_records if record.method == method),
            key=lambda record: record.n_particles,
        )
        if selected_particles:
            particles = [record.n_particles for record in selected_particles]
            run_time = [record.median_run_seconds for record in selected_particles]
            axes[1, 1].plot(
                particles,
                run_time,
                marker="o",
                label=_display_name(method),
            )

    axes[0, 0].set_title("Per-run runtime vs spatial resolution")
    axes[0, 0].set_xlabel("number of spatial nodes")
    axes[0, 0].set_ylabel("median runtime (seconds)")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend(fontsize=7)

    axes[0, 1].set_title("One-time setup cost vs spatial resolution")
    axes[0, 1].set_xlabel("number of spatial nodes")
    axes[0, 1].set_ylabel("setup time (seconds)")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("symlog", linthresh=1.0e-5)

    axes[1, 0].set_title("Retained NumPy arrays vs spatial resolution")
    axes[1, 0].set_xlabel("number of spatial nodes")
    axes[1, 0].set_ylabel("resident arrays (MiB)")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")

    axes[1, 1].set_title("Per-run runtime vs particle count")
    axes[1, 1].set_xlabel("number of particles")
    axes[1, 1].set_ylabel("median runtime (seconds)")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")

    fig.suptitle(
        "End-to-end diffusion method performance\n"
        "runtime excludes one-time setup; memory counts retained NumPy arrays",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_operating_point_study(
    records,
    accuracy_rows: list[dict[str, float | int | str | None]],
    output_path: str | Path,
) -> Path:
    """Plot exact pipeline scaling and distribution-level accuracy."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    names = []
    for record in records:
        if record.operating_point not in names:
            names.append(record.operating_point)

    labels = {
        "compute_first": "compute-first",
        "balanced": "balanced",
        "accuracy_first": "accuracy-first",
        "sampling_floor": "reference sampling floor",
    }
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2))
    benchmark_panels = (
        ("median_run_seconds", "Per-trajectory runtime", "seconds"),
        ("setup_seconds", "One-time setup", "seconds"),
        ("peak_rss_megabytes", "Isolated peak RSS", "MiB"),
        ("resident_array_megabytes", "Retained NumPy arrays", "MiB"),
    )
    for ax, (field, title, ylabel) in zip(
        axes.ravel()[:4], benchmark_panels, strict=True
    ):
        for name in names:
            selected = sorted(
                (row for row in records if row.operating_point == name),
                key=lambda row: row.n_nodes,
            )
            ax.plot(
                [row.n_nodes for row in selected],
                [float(getattr(row, field)) for row in selected],
                marker="o",
                label=labels.get(name, name),
            )
        ax.set_title(title)
        ax.set_xlabel("spatial nodes")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)

    accuracy_panels = (
        (
            "marginal_normalized_wasserstein",
            "Selected-node marginals",
            "normalized Wasserstein-1",
        ),
        (
            "sliced_normalized_wasserstein",
            "Full spatial vectors",
            "normalized sliced Wasserstein-1",
        ),
    )
    for ax, (field, title, ylabel) in zip(
        axes.ravel()[4:], accuracy_panels, strict=True
    ):
        rows = [row for row in accuracy_rows if row.get(field) is not None]
        positions = np.arange(len(rows))
        ax.bar(
            positions,
            [float(row[field]) for row in rows],
            color=["0.65" if row["operating_point"] == "sampling_floor" else "C0" for row in rows],
        )
        ax.set_xticks(
            positions,
            [labels.get(str(row["operating_point"]), str(row["operating_point"])) for row in rows],
            rotation=18,
            ha="right",
        )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle(
        "Selected completed pipelines: measured cost and distribution accuracy",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output
