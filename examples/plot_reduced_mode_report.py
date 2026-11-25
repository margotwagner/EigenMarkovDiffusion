"""Plot the matched handoff comparison and analytic mode sweep for the report."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


METRICS = (
    ("mean_error", "Mean", "#287271", "o"),
    ("variance_error", "Variance", "#D97706", "s"),
    ("covariance_error", "Same-time covariance", "#6D597A", "^"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("outputs/reduced_mode_report"),
        help="directory containing matched_same_time.csv and analytic_mode_sweep.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/figures/reduced_modes_loss.png"),
    )
    return parser


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _required_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        raise ValueError(f"missing {key!r} in row {row!r}")
    return float(value)


def _matched_series(
    rows: list[dict[str, str]],
) -> tuple[dict[str, str], list[dict[str, str]]]:
    exact_rows = [row for row in rows if row.get("model") == "exact_sampling_floor"]
    raw_rows = [row for row in rows if row.get("model") == "raw_handoff"]
    if len(exact_rows) != 1:
        raise ValueError("matched data must contain exactly one exact_sampling_floor row")
    if not raw_rows:
        raise ValueError("matched data must contain at least one raw_handoff row")
    raw_rows.sort(key=lambda row: int(row["retained_modes"]))
    return exact_rows[0], raw_rows


def _first_floor_crossing(
    rows: list[dict[str, str]], metric: str, floor: float
) -> int | None:
    for row in rows:
        if _required_float(row, metric) <= floor:
            return int(row["retained_modes"])
    return None


def _format_crossings(
    analytic_rows: list[dict[str, str]], exact_row: dict[str, str]
) -> str:
    crossings: list[str] = []
    short_names = {
        "mean_error": "mean",
        "variance_error": "variance",
        "covariance_error": "covariance",
    }
    for metric, _, _, _ in METRICS:
        crossing = _first_floor_crossing(
            analytic_rows,
            metric,
            _required_float(exact_row, metric),
        )
        crossing_text = "none" if crossing is None else str(crossing)
        crossings.append(f"{short_names[metric]} {crossing_text}")
    return "First at or below exact floor: " + " · ".join(crossings)


def _plot_matched(
    ax: plt.Axes,
    exact_row: dict[str, str],
    raw_rows: list[dict[str, str]],
    initial_modes: int,
) -> None:
    x = np.arange(len(METRICS), dtype=float)
    labels = [
        label.replace(" ", "\n", 1) if label.startswith("Same-time") else label
        for _, label, _, _ in METRICS
    ]
    exact_values = [_required_float(exact_row, metric) for metric, _, _, _ in METRICS]
    ax.plot(
        x,
        exact_values,
        color="#525B70",
        linestyle="--",
        marker="D",
        markerfacecolor="white",
        markeredgewidth=1.8,
        markersize=7.5,
        linewidth=1.8,
        label="Exact walk (sampling floor)",
        zorder=4,
    )

    raw_styles = [
        ("#B5461E", "^"),
        ("#2A6F97", "o"),
        ("#7A5195", "s"),
    ]
    for index, row in enumerate(raw_rows):
        color, marker = raw_styles[min(index, len(raw_styles) - 1)]
        values = [_required_float(row, metric) for metric, _, _, _ in METRICS]
        retained_modes = int(row["retained_modes"])
        ax.plot(
            x,
            values,
            color=color,
            marker=marker,
            markersize=8.5,
            linewidth=2.2,
            label=f"Raw {initial_modes}→{retained_modes}",
            zorder=3,
        )

    ax.set_yscale("log")
    ax.set_ylim(8.0e-4, 1.0)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Relative error against analytic moments")
    ax.set_title("Matched stochastic handoffs", loc="left", fontweight="bold")
    ax.grid(axis="y", which="both", color="#D8DEE7", linewidth=0.75, alpha=0.8)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)


def _plot_sweep(
    ax: plt.Axes,
    analytic_rows: list[dict[str, str]],
    exact_row: dict[str, str],
) -> None:
    modes = np.asarray([int(row["retained_modes"]) for row in analytic_rows])
    for metric, label, color, marker in METRICS:
        values = np.asarray([_required_float(row, metric) for row in analytic_rows])
        positive = values > 1.0e-12
        ax.plot(
            modes[positive],
            values[positive],
            color=color,
            marker=marker,
            markevery=8,
            markersize=4.3,
            linewidth=2.0,
            label=label,
        )
        ax.axhline(
            _required_float(exact_row, metric),
            color=color,
            linestyle=":",
            linewidth=1.35,
            alpha=0.72,
        )
        for retained_modes, fillstyle in ((50, "none"), (75, "full")):
            selected = modes == retained_modes
            if np.any(selected):
                ax.plot(
                    retained_modes,
                    values[selected][0],
                    color=color,
                    marker=marker,
                    fillstyle=fillstyle,
                    markeredgewidth=1.5,
                    markersize=7.0,
                    linestyle="none",
                    zorder=5,
                )

    for retained_modes in (50, 75):
        ax.axvline(retained_modes, color="#AAB2BF", linewidth=0.8, alpha=0.55)

    metric_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            marker=marker,
            markersize=5,
            linewidth=2,
            label="Same-time cov." if label == "Same-time covariance" else label,
        )
        for _, label, color, marker in METRICS
    ]
    metric_handles.append(
        Line2D(
            [0],
            [0],
            color="#525B70",
            linestyle=":",
            linewidth=1.4,
            label="Exact sampling floor",
        )
    )
    ax.legend(
        handles=metric_handles,
        loc="center right",
        bbox_to_anchor=(0.98, 0.58),
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="#D8DEE7",
        framealpha=0.94,
        fontsize=9.6,
    )

    ax.set_yscale("log")
    ax.set_xlim(1, 102)
    ax.set_ylim(7.0e-6, 1.25)
    ax.set_xlabel("Retained modes after the 10 µs handoff")
    ax.set_title("Analytic truncation loss", loc="left", fontweight="bold")
    ax.grid(axis="y", which="both", color="#D8DEE7", linewidth=0.75, alpha=0.8)
    ax.text(
        0.025,
        0.035,
        _format_crossings(analytic_rows, exact_row)
        + "\nAll analytic losses are zero at full rank (M=101).",
        transform=ax.transAxes,
        fontsize=8.2,
        color="#414B5A",
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#D8DEE7",
            "alpha": 0.93,
        },
    )


def main() -> None:
    args = build_parser().parse_args()
    matched_path = args.data_dir / "matched_same_time.csv"
    analytic_path = args.data_dir / "analytic_mode_sweep.csv"
    exact_row, raw_rows = _matched_series(_read_csv(matched_path))
    analytic_rows = _read_csv(analytic_path)
    analytic_rows.sort(key=lambda row: int(row["retained_modes"]))
    if not analytic_rows:
        raise ValueError("analytic mode sweep is empty")

    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.labelcolor": "#27313F",
            "axes.edgecolor": "#7D8998",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": "#526171",
            "ytick.color": "#526171",
            "figure.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    initial_modes = max(int(row["retained_modes"]) for row in analytic_rows)
    _plot_matched(axes[0], exact_row, raw_rows, initial_modes)
    _plot_sweep(axes[1], analytic_rows, exact_row)
    figure.tight_layout(w_pad=2.7)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
