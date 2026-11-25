"""Generate the three accuracy tables used by the reduced-mode report.

The stochastic tables reproduce the archived 500-trajectory protocols. The
same-time comparison uses exact-walk seed 1 and raw-handoff seed 2; the lag
comparison uses exact-walk seed 3 and raw-handoff seed 5. Both 50 and 75 modes
reuse the same raw seed so their random streams are aligned.

The broad mode curve is different: it analytically projects the exact mean and
covariance after the 10 us handoff. It measures irreducible truncation loss,
not sampled reduced dynamics or Gaussian-closure error.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterator

import numpy as np

from eigendiffusion.config import DiffusionConfig
from eigendiffusion.ensemble import run_modal_ensemble, run_random_walk_ensemble
from eigendiffusion.handoff import predict_handoff_grid
from eigendiffusion.metrics import relative_frobenius_error, summarize_ensemble
from eigendiffusion.references import (
    discrete_expected_diffusion,
    multinomial_marginal_covariance,
    multinomial_marginal_variance,
)
from eigendiffusion.temporal import (
    empirical_mean_cross_time_covariance,
    multinomial_mean_cross_time_covariance,
    valid_time_origins,
)

RETAINED_MODES = (50, 75)
HANDOFF_TIME_US = 10.0
COVARIANCE_TIMES_US = (1.0, 5.0, 20.0, 50.0, 100.0)
LAG_TIMES_US = (1.0, 2.0, 5.0, 10.0, 20.0)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--output", default="outputs/reduced_mode_report")
    return parser


def report_config() -> DiffusionConfig:
    return DiffusionConfig(
        n_particles=5_275,
        n_nodes=101,
        n_steps=101,
        impulse_index=59,
        dt=1.0,
        length=4.0,
        diffusion_coefficient=2.20e-4,
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report_ensembles(
    config: DiffusionConfig,
    n_runs: int,
    *,
    exact_seed: int,
    raw_seed: int,
) -> Iterator[tuple[str, int | None, np.ndarray]]:
    exact = run_random_walk_ensemble(
        config,
        n_runs=n_runs,
        seed=exact_seed,
        method="multinomial",
    )
    yield "exact_sampling_floor", None, exact.runs
    for modes in RETAINED_MODES:
        raw = run_modal_ensemble(
            config,
            n_runs=n_runs,
            model="handoff_correlated_modal",
            n_modes=modes,
            initial_n_modes=config.n_nodes,
            handoff_time=HANDOFF_TIME_US,
            seed=raw_seed,
        )
        yield "raw_handoff", modes, raw.runs


def same_time_rows(config: DiffusionConfig, n_runs: int) -> list[dict[str, object]]:
    mean = discrete_expected_diffusion(config)
    variance = multinomial_marginal_variance(config)
    covariance = multinomial_marginal_covariance(config)
    covariance_indices = [
        int(np.argmin(np.abs(config.times - time)))
        for time in COVARIANCE_TIMES_US
    ]
    rows: list[dict[str, object]] = []
    for model, modes, runs in report_ensembles(
        config,
        n_runs,
        exact_seed=1,
        raw_seed=2,
    ):
        metrics = summarize_ensemble(
            mean,
            variance,
            runs,
            config.n_particles,
            reference_covariance=covariance,
            covariance_time_indices=covariance_indices,
        )
        rows.append(
            {
                "model": model,
                "retained_modes": modes,
                "mean_error": metrics.mean_relative_l2_error,
                "variance_error": metrics.variance_relative_l2_error,
                "covariance_error": metrics.covariance_relative_frobenius_error,
            }
        )
    return rows


def analytic_rows(config: DiffusionConfig) -> list[dict[str, object]]:
    covariance_indices = [
        int(np.argmin(np.abs(config.times - time)))
        for time in COVARIANCE_TIMES_US
    ]
    predictions = predict_handoff_grid(
        config,
        final_modes=list(range(1, config.n_nodes + 1)),
        handoff_times=[HANDOFF_TIME_US],
        covariance_time_indices=covariance_indices,
        initial_n_modes=config.n_nodes,
    )
    return [
        {
            "model": "analytic_projected_truncation",
            "retained_modes": row.final_n_modes,
            "mean_error": row.predicted_mean_relative_l2_error,
            "variance_error": row.predicted_variance_relative_l2_error,
            "covariance_error": row.predicted_covariance_relative_frobenius_error,
            "modal_update_fraction_of_full": row.modal_update_fraction_of_full,
        }
        for row in predictions
    ]


def lag_rows(config: DiffusionConfig, n_runs: int) -> list[dict[str, object]]:
    start_step = int(np.argmin(np.abs(config.times - 10.0)))
    end_step = int(np.argmin(np.abs(config.times - 100.0)))
    lag_steps = [int(round(time / config.dt)) for time in LAG_TIMES_US]
    references: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for lag in lag_steps:
        origins = valid_time_origins(
            config.n_steps,
            lag,
            start_step=start_step,
            end_step=end_step,
        )
        references[lag] = (
            origins,
            multinomial_mean_cross_time_covariance(
                config,
                lag,
                start_indices=origins,
            ),
        )

    rows: list[dict[str, object]] = []
    for model, modes, runs in report_ensembles(
        config,
        n_runs,
        exact_seed=3,
        raw_seed=5,
    ):
        for lag in lag_steps:
            origins, reference = references[lag]
            observed = empirical_mean_cross_time_covariance(
                runs,
                lag,
                start_indices=origins,
            )
            rows.append(
                {
                    "model": model,
                    "retained_modes": modes,
                    "lag_time_us": lag * config.dt,
                    "cross_time_covariance_error": relative_frobenius_error(
                        reference,
                        observed,
                    ),
                }
            )
    return rows


def main() -> None:
    args = build_parser().parse_args()
    if args.runs <= 1:
        raise ValueError("runs must exceed one")
    output = Path(args.output)
    config = report_config()

    moments = same_time_rows(config, args.runs)
    analytic = analytic_rows(config)
    lags = lag_rows(config, args.runs)
    write_csv(output / "matched_same_time.csv", moments)
    write_csv(output / "analytic_mode_sweep.csv", analytic)
    write_csv(output / "matched_lag.csv", lags)

    for row in moments:
        print(
            f"{row['model']} M={row['retained_modes']}: "
            f"mean={float(row['mean_error']):.8g}, "
            f"variance={float(row['variance_error']):.8g}, "
            f"covariance={float(row['covariance_error']):.8g}"
        )
    print(f"Saved report accuracy data to {output}")


if __name__ == "__main__":
    main()
