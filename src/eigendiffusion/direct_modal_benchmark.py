"""Reproducible study runner for direct retained Gaussian diffusion.

The study deliberately separates three questions:

* whether the prepared modal transition satisfies its exact moment algebra;
* how much full-field moment information is lost by retaining ``M`` modes;
* how setup, modal evolution, reconstruction, and a physical simplex readout
  compare with the exact vectorized multinomial walk.

Only deterministic analytic accuracy diagnostics are used here.  This keeps
the sweep compact and avoids mistaking finite-ensemble noise for truncation
error.  Distribution-level validation belongs in a separate stochastic study.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal, Sequence

import numpy as np

from .baselines import vectorized_multinomial_random_walk_diffusion
from .benchmarking import deep_numpy_nbytes, stable_diffusion_config
from .component_benchmark import fixed_step_component_config
from .config import DiffusionConfig
from .direct_modal import DirectModalGaussianDiffusion, DirectModalGaussianResult
from .readouts import simplex_bank_counts

StudyRegime = Literal["fixed_steps", "fixed_duration"]
STUDY_REGIMES: tuple[StudyRegime, ...] = ("fixed_steps", "fixed_duration")

DIRECT_MODAL_METHOD = "direct_modal"
DIRECT_RAW_METHOD = "direct_raw_spatial"
DIRECT_SIMPLEX_METHOD = "direct_simplex_bank"
EXACT_METHOD = "exact_vectorized_multinomial"
TIMED_METHODS = (
    DIRECT_MODAL_METHOD,
    DIRECT_RAW_METHOD,
    DIRECT_SIMPLEX_METHOD,
    EXACT_METHOD,
)


@dataclass(frozen=True, slots=True)
class DirectModalStudySpec:
    """Configuration for the direct-modal accuracy and performance sweep."""

    node_counts: tuple[int, ...] = (101, 201)
    mode_counts: tuple[int, ...] = (20, 35, 50, 75)
    regimes: tuple[StudyRegime, ...] = STUDY_REGIMES
    n_particles: int = 5_275
    fixed_steps: int = 101
    total_time: float = 100.0
    requested_dt: float = 1.0
    length: float = 4.0
    diffusion_coefficient: float = 2.20e-4
    impulse_fraction: float = 0.59
    stability_safety: float = 0.95
    setup_repeats: int = 5
    trajectory_repeats: int = 30
    warmup_runs: int = 1
    seed: int = 20_261_123

    def __post_init__(self) -> None:
        if not self.node_counts or any(value < 2 for value in self.node_counts):
            raise ValueError("node_counts must contain integers >= 2")
        if not self.mode_counts or any(value <= 0 for value in self.mode_counts):
            raise ValueError("mode_counts must contain positive integers")
        unknown = set(self.regimes) - set(STUDY_REGIMES)
        if unknown:
            raise ValueError(f"unknown study regimes: {sorted(unknown)}")
        if self.fixed_steps < 2:
            raise ValueError("fixed_steps must be at least 2")
        if self.setup_repeats <= 0 or self.trajectory_repeats <= 0:
            raise ValueError("setup_repeats and trajectory_repeats must be positive")
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs must be nonnegative")


def _configuration(
    spec: DirectModalStudySpec,
    regime: StudyRegime,
    n_nodes: int,
) -> DiffusionConfig:
    common = {
        "n_particles": spec.n_particles,
        "n_nodes": n_nodes,
        "requested_dt": spec.requested_dt,
        "length": spec.length,
        "diffusion_coefficient": spec.diffusion_coefficient,
        "impulse_fraction": spec.impulse_fraction,
        "stability_safety": spec.stability_safety,
    }
    if regime == "fixed_steps":
        return fixed_step_component_config(n_steps=spec.fixed_steps, **common)
    return stable_diffusion_config(total_time=spec.total_time, **common)


def _configuration_columns(
    config: DiffusionConfig,
    regime: StudyRegime,
    n_modes: int,
) -> dict[str, str | int | float]:
    return {
        "regime": regime,
        "n_particles": config.n_particles,
        "n_nodes": config.n_nodes,
        "n_steps": config.n_steps,
        "dt": config.dt,
        "total_time": float(config.times[-1]),
        "length": config.length,
        "diffusion_coefficient": config.diffusion_coefficient,
        "impulse_index": int(config.impulse_index),
        "n_modes": n_modes,
    }


def _relative_norm(numerator: np.ndarray, denominator: np.ndarray) -> float:
    denominator_norm = float(np.linalg.norm(np.asarray(denominator).ravel()))
    numerator_norm = float(np.linalg.norm(np.asarray(numerator).ravel()))
    if denominator_norm == 0.0:
        return 0.0 if numerator_norm == 0.0 else float("inf")
    return numerator_norm / denominator_norm


def analytic_direct_modal_diagnostics(
    model: DirectModalGaussianDiffusion,
    *,
    regime: StudyRegime,
) -> dict[str, str | int | float | bool]:
    """Return retained algebra and raw full-field truncation diagnostics."""

    config = model.config
    vectors = model.basis.eigenvectors
    factors = model.one_step_factors
    probabilities = model.single_particle_probabilities
    means = model.modal_means
    covariances = model.modal_covariances
    innovations = model.innovation_covariances
    innovation_factors = model.innovation_factors

    propagated_covariances = (
        factors[None, :, None]
        * covariances[:-1]
        * factors[None, None, :]
    )
    expected_innovations = covariances[1:] - propagated_covariances
    expected_innovations = 0.5 * (
        expected_innovations + np.swapaxes(expected_innovations, 1, 2)
    )
    mean_residual = means[1:] - factors[None, :] * means[:-1]
    covariance_residual = covariances[1:] - (
        propagated_covariances + innovations
    )
    factor_residual = innovations - np.einsum(
        "tij,tkj->tik",
        innovation_factors,
        innovation_factors,
        optimize=True,
    )

    minimum_eigenvalue = min(
        float(np.linalg.eigvalsh(covariance)[0])
        for covariance in innovations
    )

    particle_count = float(config.n_particles)
    full_mean = particle_count * probabilities
    projected_mean = means @ vectors.T
    exact_variance = particle_count * probabilities * (1.0 - probabilities)
    projected_variance = np.einsum(
        "ni,tij,nj->tn",
        vectors,
        covariances,
        vectors,
        optimize=True,
    )

    # P C P is the Frobenius-orthogonal projection of C into the retained
    # spatial subspace. Hence ||C - P C P||_F^2 = ||C||_F^2 - ||V.T C V||_F^2.
    probability_squares = np.sum(probabilities**2, axis=1)
    full_covariance_norm_squared = particle_count**2 * (
        probability_squares
        - 2.0 * np.sum(probabilities**3, axis=1)
        + probability_squares**2
    )
    retained_covariance_norm_squared = np.sum(covariances**2, axis=(1, 2))
    full_covariance_total = float(np.sum(full_covariance_norm_squared))
    omitted_covariance_total = max(
        0.0,
        full_covariance_total - float(np.sum(retained_covariance_norm_squared)),
    )
    covariance_truncation_error = (
        0.0
        if full_covariance_total == 0.0
        else float(np.sqrt(omitted_covariance_total / full_covariance_total))
    )

    algebra_scale = max(float(np.linalg.norm(covariances[1:].ravel())), 1.0)
    innovation_scale = max(float(np.linalg.norm(innovations.ravel())), 1.0)
    result: dict[str, str | int | float | bool] = {
        **_configuration_columns(config, regime, model.n_modes),
        "retained_mean_recursion_relative_l2_error": _relative_norm(
            mean_residual,
            means[1:],
        ),
        "retained_covariance_recursion_relative_frobenius_error": (
            float(np.linalg.norm(covariance_residual.ravel())) / algebra_scale
        ),
        "innovation_covariance_identity_relative_frobenius_error": (
            float(np.linalg.norm((innovations - expected_innovations).ravel()))
            / innovation_scale
        ),
        "innovation_factor_relative_frobenius_error": (
            float(np.linalg.norm(factor_residual.ravel())) / innovation_scale
        ),
        "minimum_innovation_eigenvalue": minimum_eigenvalue,
        "maximum_constant_mode_covariance": float(
            max(
                np.max(np.abs(covariances[:, 0, :])),
                np.max(np.abs(covariances[:, :, 0])),
            )
        ),
        "maximum_constant_mode_factor": float(
            np.max(np.abs(innovation_factors[:, 0, :]))
        ),
        "raw_mean_relative_l2_truncation_error": _relative_norm(
            projected_mean - full_mean,
            full_mean,
        ),
        "raw_variance_relative_l2_truncation_error": _relative_norm(
            projected_variance - exact_variance,
            exact_variance,
        ),
        "raw_covariance_relative_frobenius_truncation_error": (
            covariance_truncation_error
        ),
        "raw_maximum_mean_mass_error": float(
            np.max(np.abs(projected_mean.sum(axis=1) - particle_count))
        ),
        "raw_minimum_projected_mean": float(np.min(projected_mean)),
    }
    result["retained_algebra_pass"] = bool(
        result["retained_mean_recursion_relative_l2_error"] <= 1.0e-10
        and result["retained_covariance_recursion_relative_frobenius_error"]
        <= 1.0e-10
        and result["innovation_covariance_identity_relative_frobenius_error"]
        <= 1.0e-10
        and result["innovation_factor_relative_frobenius_error"] <= 1.0e-10
        and result["minimum_innovation_eigenvalue"] >= -1.0e-10
        and result["maximum_constant_mode_covariance"] <= 1.0e-10
        and result["maximum_constant_mode_factor"] <= 1.0e-10
    )
    return result


def _validate_modal_result(
    result: DirectModalGaussianResult,
    config: DiffusionConfig,
    n_modes: int,
) -> None:
    if result.modal_amplitudes.shape != (config.n_steps, n_modes):
        raise RuntimeError("direct modal benchmark returned an unexpected modal shape")
    if not np.all(np.isfinite(result.modal_amplitudes)):
        raise RuntimeError("direct modal benchmark returned nonfinite modes")


def _validate_spatial(values: np.ndarray, config: DiffusionConfig, *, physical: bool) -> None:
    if values.shape != (config.n_steps, config.n_nodes):
        raise RuntimeError("benchmark returned an unexpected spatial shape")
    if not np.all(np.isfinite(values)):
        raise RuntimeError("benchmark returned nonfinite spatial values")
    if not np.allclose(values.sum(axis=1), config.n_particles, atol=1.0e-7, rtol=0.0):
        raise RuntimeError("benchmark failed mass conservation")
    if physical and np.min(values) < -1.0e-10:
        raise RuntimeError("simplex or exact benchmark returned negative counts")


def _execution_orders() -> tuple[tuple[str, ...], ...]:
    # Five orders and their reverses balance absolute position and pairwise
    # precedence over each ten-repeat cycle.
    base = TIMED_METHODS
    rotations = tuple(base[offset:] + base[:offset] for offset in range(4))
    extra = (base[0], base[2], base[3], base[1])
    return tuple(order for item in (*rotations, extra) for order in (item, item[::-1]))


def _timed_action(
    method: str,
    model: DirectModalGaussianDiffusion,
    config: DiffusionConfig,
    rng: np.random.Generator,
) -> object:
    if method == DIRECT_MODAL_METHOD:
        return model.run(rng=rng, reconstruct_spatial=False)
    if method == DIRECT_RAW_METHOD:
        return model.run(rng=rng, reconstruct_spatial=True)
    if method == DIRECT_SIMPLEX_METHOD:
        raw = model.run(rng=rng, reconstruct_spatial=True)
        if raw.spatial_counts is None:  # pragma: no cover - guarded by call
            raise RuntimeError("direct raw trajectory omitted its spatial field")
        return simplex_bank_counts(raw.spatial_counts, config.n_particles)
    if method == EXACT_METHOD:
        return vectorized_multinomial_random_walk_diffusion(config, rng=rng)
    raise ValueError(f"unknown timed method: {method}")


def _validate_timed_output(
    method: str,
    output: object,
    config: DiffusionConfig,
    n_modes: int,
) -> None:
    if method == DIRECT_MODAL_METHOD:
        if not isinstance(output, DirectModalGaussianResult):
            raise RuntimeError("modal-only benchmark returned an unexpected result type")
        _validate_modal_result(output, config, n_modes)
        if output.spatial_counts is not None:
            raise RuntimeError("modal-only benchmark unexpectedly reconstructed space")
        return
    if method == DIRECT_RAW_METHOD:
        if not isinstance(output, DirectModalGaussianResult):
            raise RuntimeError("raw-spatial benchmark returned an unexpected result type")
        _validate_modal_result(output, config, n_modes)
        if output.spatial_counts is None:
            raise RuntimeError("raw-spatial benchmark omitted its spatial field")
        _validate_spatial(output.spatial_counts, config, physical=False)
        return
    if not isinstance(output, np.ndarray):
        raise RuntimeError("physical benchmark returned an unexpected result type")
    _validate_spatial(output, config, physical=True)


def _benchmark_one(
    spec: DirectModalStudySpec,
    config: DiffusionConfig,
    *,
    regime: StudyRegime,
    regime_index: int,
    n_modes: int,
) -> tuple[
    list[dict[str, object]],
    dict[str, object],
    dict[str, object],
]:
    columns = _configuration_columns(config, regime, n_modes)
    timing_rows: list[dict[str, object]] = []
    model: DirectModalGaussianDiffusion | None = None
    for repeat in range(spec.setup_repeats):
        model = None
        gc.collect()
        start = perf_counter()
        model = DirectModalGaussianDiffusion(config, n_modes=n_modes)
        seconds = perf_counter() - start
        timing_rows.append(
            {
                **columns,
                "method": DIRECT_MODAL_METHOD,
                "component": "setup",
                "repeat": repeat,
                "execution_position": -1,
                "seed_id": "",
                "seconds": seconds,
            }
        )
    if model is None:  # pragma: no cover - validated positive repeats
        raise RuntimeError("direct modal setup did not produce a model")

    for warmup in range(spec.warmup_runs):
        for channel, method in enumerate(TIMED_METHODS):
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [spec.seed, regime_index, config.n_nodes, n_modes, 0, warmup, channel]
                )
            )
            output = _timed_action(method, model, config, rng)
            _validate_timed_output(method, output, config, n_modes)
            del output

    orders = _execution_orders()
    garbage_collection_enabled = gc.isenabled()
    if garbage_collection_enabled:
        gc.disable()
    try:
        for repeat in range(spec.trajectory_repeats):
            order = orders[repeat % len(orders)]
            for position, method in enumerate(order):
                channel = 3 if method == EXACT_METHOD else 2
                seed_values = [
                    spec.seed,
                    regime_index,
                    config.n_nodes,
                    n_modes,
                    1,
                    repeat,
                    channel,
                ]
                rng = np.random.default_rng(np.random.SeedSequence(seed_values))
                start = perf_counter()
                output = _timed_action(method, model, config, rng)
                seconds = perf_counter() - start
                _validate_timed_output(method, output, config, n_modes)
                timing_rows.append(
                    {
                        **columns,
                        "method": method,
                        "component": "trajectory_total",
                        "repeat": repeat,
                        "execution_position": position,
                        "seed_id": ":".join(str(value) for value in seed_values),
                        "seconds": seconds,
                    }
                )
                del output
    finally:
        if garbage_collection_enabled:
            gc.enable()

    memory = _logical_memory_record(model, regime=regime)
    accuracy = analytic_direct_modal_diagnostics(model, regime=regime)
    return timing_rows, memory, accuracy


def _logical_memory_record(
    model: DirectModalGaussianDiffusion,
    *,
    regime: StudyRegime,
) -> dict[str, object]:
    config = model.config
    columns = _configuration_columns(config, regime, model.n_modes)
    setup_bytes = deep_numpy_nbytes(model)
    modal_output_bytes = config.n_steps * model.n_modes * np.dtype(float).itemsize
    spatial_output_bytes = config.n_steps * config.n_nodes * np.dtype(float).itemsize
    exact_output_bytes = config.n_steps * config.n_nodes * np.dtype(np.int64).itemsize
    factor_bytes = (
        (config.n_steps - 1)
        * model.n_modes
        * model.n_modes
        * np.dtype(float).itemsize
    )
    method_values = {
        DIRECT_MODAL_METHOD: {
            "setup_array_bytes": setup_bytes,
            "primary_output_bytes": modal_output_bytes,
            "returned_array_bytes": modal_output_bytes,
            "post_return_resident_array_bytes": setup_bytes + modal_output_bytes,
            "estimated_peak_live_array_bytes": setup_bytes + modal_output_bytes,
        },
        DIRECT_RAW_METHOD: {
            "setup_array_bytes": setup_bytes,
            "primary_output_bytes": spatial_output_bytes,
            "returned_array_bytes": modal_output_bytes + spatial_output_bytes,
            "post_return_resident_array_bytes": (
                setup_bytes + modal_output_bytes + spatial_output_bytes
            ),
            "estimated_peak_live_array_bytes": (
                setup_bytes + modal_output_bytes + spatial_output_bytes
            ),
        },
        DIRECT_SIMPLEX_METHOD: {
            "setup_array_bytes": setup_bytes,
            "primary_output_bytes": spatial_output_bytes,
            "returned_array_bytes": spatial_output_bytes,
            "post_return_resident_array_bytes": setup_bytes + spatial_output_bytes,
            "estimated_peak_live_array_bytes": (
                setup_bytes + modal_output_bytes + 2 * spatial_output_bytes
            ),
        },
        EXACT_METHOD: {
            "setup_array_bytes": 0,
            "primary_output_bytes": exact_output_bytes,
            "returned_array_bytes": exact_output_bytes,
            "post_return_resident_array_bytes": exact_output_bytes,
            # Counts plus movers/right and probability/share node vectors.
            "estimated_peak_live_array_bytes": (
                exact_output_bytes + 4 * config.n_nodes * np.dtype(np.int64).itemsize
            ),
        },
    }
    return {
        "configuration": columns,
        "innovation_factor_bytes": factor_bytes,
        "logical_only": True,
        "methods": method_values,
    }


def _flatten_memory(record: dict[str, object]) -> list[dict[str, object]]:
    configuration = record["configuration"]
    if not isinstance(configuration, dict):  # pragma: no cover - internal contract
        raise RuntimeError("memory record has an invalid configuration")
    methods = record["methods"]
    if not isinstance(methods, dict):  # pragma: no cover - internal contract
        raise RuntimeError("memory record has invalid method data")
    rows: list[dict[str, object]] = []
    for method, values in methods.items():
        if not isinstance(values, dict):  # pragma: no cover - internal contract
            raise RuntimeError("memory record has invalid method values")
        row = {
            **configuration,
            "method": method,
            "innovation_factor_bytes": (
                record["innovation_factor_bytes"]
                if method != EXACT_METHOD
                else 0
            ),
            "logical_only": True,
            **values,
        }
        for key, value in tuple(row.items()):
            if key.endswith("_bytes"):
                row[key.removesuffix("_bytes") + "_megabytes"] = (
                    float(value) / 1024**2
                )
        rows.append(row)
    return rows


def _timing_summaries(
    timing_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[float]] = {}
    for row in timing_rows:
        key = (
            row["regime"],
            row["n_nodes"],
            row["n_steps"],
            row["dt"],
            row["n_modes"],
            row["method"],
            row["component"],
        )
        grouped.setdefault(key, []).append(float(row["seconds"]))

    summaries: list[dict[str, object]] = []
    for key, samples in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        regime, n_nodes, n_steps, dt, n_modes, method, component = key
        values = np.asarray(samples, dtype=float)
        summaries.append(
            {
                "regime": regime,
                "n_nodes": n_nodes,
                "n_steps": n_steps,
                "dt": dt,
                "n_modes": n_modes,
                "method": method,
                "component": component,
                "samples": values.size,
                "median_seconds": float(np.median(values)),
                "mean_seconds": float(np.mean(values)),
                "std_seconds": float(np.std(values, ddof=0)),
                "min_seconds": float(np.min(values)),
                "max_seconds": float(np.max(values)),
            }
        )

    exact_lookup = {
        (row["regime"], row["n_nodes"], row["n_steps"], row["n_modes"]): float(
            row["median_seconds"]
        )
        for row in summaries
        if row["method"] == EXACT_METHOD and row["component"] == "trajectory_total"
    }
    for row in summaries:
        if row["component"] != "trajectory_total":
            row["median_ratio_over_exact"] = None
            continue
        exact = exact_lookup[
            (row["regime"], row["n_nodes"], row["n_steps"], row["n_modes"])
        ]
        row["median_ratio_over_exact"] = (
            float(row["median_seconds"]) / exact if exact > 0.0 else None
        )
    return summaries


def _write_csv(rows: Sequence[dict[str, object]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_direct_modal_study(
    spec: DirectModalStudySpec,
    output_dir: str | Path,
) -> dict[str, object]:
    """Run the complete sweep, write CSV/JSON artifacts, and return the report."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    timing_rows: list[dict[str, object]] = []
    accuracy_rows: list[dict[str, object]] = []
    memory_rows: list[dict[str, object]] = []
    configurations: list[dict[str, object]] = []

    for regime_index, regime in enumerate(dict.fromkeys(spec.regimes)):
        for n_nodes in sorted(set(spec.node_counts)):
            config = _configuration(spec, regime, n_nodes)
            configurations.append(
                {
                    **_configuration_columns(config, regime, 0),
                    "n_modes": list(
                        value for value in sorted(set(spec.mode_counts)) if value <= n_nodes
                    ),
                }
            )
            for n_modes in sorted(set(spec.mode_counts)):
                if n_modes > n_nodes:
                    continue
                samples, memory, accuracy = _benchmark_one(
                    spec,
                    config,
                    regime=regime,
                    regime_index=regime_index,
                    n_modes=n_modes,
                )
                timing_rows.extend(samples)
                memory_rows.extend(_flatten_memory(memory))
                accuracy_rows.append(accuracy)

    timing_summaries = _timing_summaries(timing_rows)
    timing_path = destination / "timing_samples.csv"
    accuracy_path = destination / "analytic_accuracy.csv"
    memory_path = destination / "logical_memory.csv"
    summary_path = destination / "summary.json"
    _write_csv(timing_rows, timing_path)
    _write_csv(accuracy_rows, accuracy_path)
    _write_csv(memory_rows, memory_path)

    report: dict[str, object] = {
        "schema_version": 1,
        "study_contract": {
            "modal_model": "DirectModalGaussianDiffusion",
            "accuracy": "analytic retained algebra and raw projection truncation",
            "performance_regimes": list(dict.fromkeys(spec.regimes)),
            "timed_methods": list(TIMED_METHODS),
            "memory": (
                "unique reachable NumPy arrays and explicit live-array estimates; "
                "not process RSS"
            ),
            "common_time_grid_within_configuration": True,
        },
        "specification": asdict(spec),
        "runtime_environment": {
            name: os.environ.get(name)
            for name in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "configurations": configurations,
        "timing_summaries": timing_summaries,
        "analytic_accuracy": accuracy_rows,
        "logical_memory": memory_rows,
        "artifacts": {
            "timing_samples_csv": str(timing_path),
            "analytic_accuracy_csv": str(accuracy_path),
            "logical_memory_csv": str(memory_path),
            "summary_json": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark direct retained Gaussian diffusion across N and M.",
    )
    parser.add_argument("--node-counts", type=int, nargs="+", default=[101, 201])
    parser.add_argument("--modes", type=int, nargs="+", default=[20, 35, 50, 75])
    parser.add_argument(
        "--regimes",
        nargs="+",
        choices=STUDY_REGIMES,
        default=list(STUDY_REGIMES),
    )
    parser.add_argument("--particles", type=int, default=5_275)
    parser.add_argument("--steps", type=int, default=101)
    parser.add_argument("--total-time", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--length", type=float, default=4.0)
    parser.add_argument("--diffusion", type=float, default=2.20e-4)
    parser.add_argument("--impulse-fraction", type=float, default=0.59)
    parser.add_argument("--stability-safety", type=float, default=0.95)
    parser.add_argument("--setup-repeats", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20_261_123)
    parser.add_argument(
        "--output-dir",
        default="outputs/direct_modal_gaussian",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = DirectModalStudySpec(
        node_counts=tuple(args.node_counts),
        mode_counts=tuple(args.modes),
        regimes=tuple(args.regimes),
        n_particles=args.particles,
        fixed_steps=args.steps,
        total_time=args.total_time,
        requested_dt=args.dt,
        length=args.length,
        diffusion_coefficient=args.diffusion,
        impulse_fraction=args.impulse_fraction,
        stability_safety=args.stability_safety,
        setup_repeats=args.setup_repeats,
        trajectory_repeats=args.repeats,
        warmup_runs=args.warmups,
        seed=args.seed,
    )
    report = run_direct_modal_study(spec, args.output_dir)
    compact = {
        "configurations": report["configurations"],
        "timing_summaries": report["timing_summaries"],
        "artifacts": report["artifacts"],
    }
    print(json.dumps(compact, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI usage
    raise SystemExit(main())
