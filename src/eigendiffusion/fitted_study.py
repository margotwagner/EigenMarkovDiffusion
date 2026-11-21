"""End-to-end training, held-out validation, and scaling for fitted closure."""

from __future__ import annotations

import csv
import gc
import json
from pathlib import Path

import numpy as np

from .anchored_closure import AnchoredFittedClosureModel, complete_anchored_ensemble
from .benchmarking import (
    SELECTED_OPERATING_POINTS,
    benchmark_operating_points,
    benchmark_random_walks,
    stable_diffusion_config,
    write_benchmark_csv,
    write_operating_point_benchmark_csv,
)
from .conditioned_closure import (
    PhysicsConditionedClosureModel,
    PhysicsConditionedRegressionModel,
    complete_conditioned_ensemble,
    cross_validate_conditioned_regression,
    recommend_anchor_expansion,
)
from .distribution_metrics import summarize_distribution_ensemble
from .ensemble import (
    apply_readout_ensemble,
    run_modal_ensemble,
    run_random_walk_ensemble,
)
from .fitted_ablation import (
    run_anchor_innovation_ablation,
    run_exact_anchor_ensemble_validation,
)
from .fitted_closure import (
    FittedClosureTrainingCase,
    FittedLocalClosureModel,
    complete_fitted_ensemble,
)
from .metrics import relative_frobenius_error, summarize_ensemble
from .plotting import plot_fitted_closure_study, plot_random_walk_benchmark
from .references import (
    discrete_expected_diffusion,
    multinomial_marginal_covariance,
    multinomial_marginal_variance,
)
from .temporal import (
    empirical_mean_cross_time_covariance,
    multinomial_mean_cross_time_covariance,
    valid_time_origins,
)


def _write_rows(rows: list[dict[str, object]], output: Path) -> Path:
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


def _nearest_indices(times: np.ndarray, requested: list[float]) -> list[int]:
    indices = sorted(
        {
            int(np.argmin(np.abs(times - value)))
            for value in requested
            if times[0] <= value <= times[-1]
        }
    )
    return [index for index in indices if index > 0] or [len(times) - 1]


def run_fitted_closure_study(args) -> int:
    """Train once, freeze the model, and evaluate unseen physical cases."""

    physics_conditioned = bool(getattr(args, "physics_conditioned", False))
    anchored = bool(getattr(args, "anchor_conditioned", False)) or physics_conditioned
    persistent_innovations = bool(getattr(args, "persistent_innovations", False))
    if persistent_innovations and not anchored:
        raise ValueError("persistent innovations require --anchor-conditioned")
    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    point = SELECTED_OPERATING_POINTS[2]

    def config_for(spec: tuple[int, int, float, float, float]):
        nodes, particles, impulse, diffusion, length = spec
        return stable_diffusion_config(
            n_particles=particles,
            n_nodes=nodes,
            total_time=args.total_time,
            requested_dt=args.dt,
            length=length,
            diffusion_coefficient=diffusion,
            impulse_fraction=impulse,
            stability_safety=args.stability_safety,
        )

    training_specs = (
        (
            (51, 2_500, 0.35, 1.5e-4, 3.5),
            (51, 8_000, 0.72, 3.5e-4, 5.5),
            (81, 4_000, 0.50, 1.8e-4, 3.8),
            (81, 6_500, 0.68, 2.7e-4, 4.5),
            (101, 3_000, 0.25, 2.0e-4, 4.0),
            (101, 4_500, 0.45, 2.0e-4, 4.2),
            (101, 6_000, 0.65, 2.4e-4, 3.8),
            (101, 8_000, 0.78, 3.4e-4, 5.0),
            (126, 4_000, 0.48, 1.8e-4, 4.5),
            (126, 7_000, 0.70, 2.7e-4, 5.5),
            (151, 5_000, 0.55, 2.2e-4, 4.0),
            (201, 5_500, 0.60, 2.2e-4, 4.0),
        )
        if anchored
        else (
            (51, 2_000, 0.20, 1.1e-4, 3.0),
            (51, 9_000, 0.72, 4.4e-4, 5.0),
            (101, 3_000, 0.18, 2.2e-4, 4.5),
            (101, 8_000, 0.48, 1.1e-4, 5.0),
            (101, 5_275, 0.82, 4.4e-4, 3.0),
            (151, 2_500, 0.30, 2.2e-4, 4.0),
            (151, 8_500, 0.67, 3.3e-4, 5.5),
            (201, 5_000, 0.55, 1.65e-4, 4.5),
        )
    )
    if physics_conditioned and args.use_bridge_anchors:
        training_specs += (
            (126, 5_292, 0.592, 2.2045407685e-4, 4.9749371855),
            (114, 5_612, 0.577, 2.3237900077e-4, 4.8062459363),
            (176, 5_244, 0.5733333333, 2.2e-4, 4.0),
            (151, 5_745, 0.625, 2.2978250586e-4, 3.8987177379),
        )
    training_cases: list[FittedClosureTrainingCase] = []
    training_manifest: list[dict[str, object]] = []
    for case_index, spec in enumerate(training_specs):
        config = config_for(spec)
        modes, handoff_time, _ = point.scaled(
            config.n_nodes,
            float(config.times[-1]),
        )
        ensemble = run_random_walk_ensemble(
            config,
            n_runs=args.training_runs,
            seed=args.seed + case_index,
            method="multinomial",
        )
        training_cases.append(
            FittedClosureTrainingCase(
                config=config,
                runs=ensemble.runs,
                retained_modes=modes,
                handoff_time=handoff_time,
            )
        )
        training_manifest.append(
            {
                "case": f"train_{case_index}",
                "nodes": config.n_nodes,
                "particles": config.n_particles,
                "impulse_index": config.impulse_index,
                "diffusion": config.diffusion_coefficient,
                "length": config.length,
                "runs": args.training_runs,
                "retained_modes": modes,
                "handoff_time": handoff_time,
            }
        )
    _write_rows(training_manifest, output_directory / "training_manifest.csv")

    fit_kwargs = {
        "bandwidth": args.bandwidth,
        "n_frequency_bins": args.frequency_bins,
        "n_time_bins": args.time_bins,
        "residual_grid_size": args.residual_grid_size,
        "ridge": args.fit_ridge,
        "max_residual_samples": args.max_residual_samples,
        "max_regression_samples_per_target": args.regression_samples_per_target,
        "seed": args.seed,
    }
    ablation_path: Path | None = None
    if anchored and args.run_ablation:
        ablation_path = run_anchor_innovation_ablation(
            training_cases,
            output_directory / "anchor_innovation_ablation.csv",
            ranks=args.ablation_ranks,
            bandwidth=args.bandwidth,
            n_frequency_bins=args.frequency_bins,
            n_time_bins=args.time_bins,
            residual_grid_size=args.residual_grid_size,
            nearest_anchors=args.nearest_anchors,
            ridge=args.fit_ridge,
            max_residual_samples=args.max_residual_samples,
            max_regression_samples_per_target=args.regression_samples_per_target,
            seed=args.seed,
        )
    conditioning_cv_path: Path | None = None
    anchor_recommendations_path: Path | None = None
    selected_conditioning_degree = args.conditioning_degree
    if physics_conditioned and args.run_conditioning_cv:
        conditioning_cv_path, selected_conditioning_degree = (
            cross_validate_conditioned_regression(
                training_cases,
                output_directory / "conditioning_cross_validation.csv",
                degrees=args.conditioning_degrees,
                folds=args.conditioning_folds,
                bandwidth=args.bandwidth,
                n_frequency_bins=args.frequency_bins,
                ridge=args.conditioning_ridge,
                max_regression_samples_per_target=(
                    args.regression_samples_per_target
                ),
                seed=args.seed,
            )
        )
        anchor_recommendations_path = recommend_anchor_expansion(
            training_cases,
            conditioning_cv_path,
            output_directory / "recommended_anchor_expansion.csv",
            count=args.recommended_anchor_count,
        )
        with conditioning_cv_path.open(newline="", encoding="utf-8") as handle:
            cv_rows = list(csv.DictReader(handle))
        selected_cv_errors = [
            float(row["combined_prediction_error"])
            for row in cv_rows
            if int(row["degree"]) == selected_conditioning_degree
        ]
        cv_mean_error = float(np.mean(selected_cv_errors))
        cv_max_error = float(np.max(selected_cv_errors))
        cv_gate_pass = (
            cv_mean_error <= args.cv_mean_threshold
            and cv_max_error <= args.cv_max_threshold
        )
        if args.gate_before_lockbox and not cv_gate_pass:
            summary = {
                "training_cases": len(training_specs),
                "training_runs_per_case": args.training_runs,
                "conditioning_degree": selected_conditioning_degree,
                "cv_mean_error": cv_mean_error,
                "cv_max_error": cv_max_error,
                "cv_mean_threshold": args.cv_mean_threshold,
                "cv_max_threshold": args.cv_max_threshold,
                "cv_gate_pass": False,
                "lockbox_opened": False,
                "production_default": "analytic_local_bandwidth_12",
                "conditioning_cross_validation_csv": str(conditioning_cv_path),
                "recommended_anchor_expansion_csv": str(
                    anchor_recommendations_path
                ),
            }
            summary_path = output_directory / "prelockbox_summary.json"
            summary_path.write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(summary, indent=2))
            return 0
    if physics_conditioned:
        innovations = AnchoredFittedClosureModel.fit(
            training_cases,
            innovation_rank=args.innovation_rank,
            nearest_anchors=args.nearest_anchors,
            persistent_innovations=persistent_innovations,
            **fit_kwargs,
        )
        regression = PhysicsConditionedRegressionModel.fit(
            training_cases,
            bandwidth=args.bandwidth,
            n_frequency_bins=args.frequency_bins,
            degree=selected_conditioning_degree,
            ridge=args.conditioning_ridge,
            max_regression_samples_per_target=args.regression_samples_per_target,
            seed=args.seed,
        )
        model = PhysicsConditionedClosureModel(regression, innovations)
        model_path = model.save(output_directory / "physics_conditioned_closure.npz")
        candidate_model = "physics_conditioned_fitted"
        candidate_closure = "conditioned_fitted"
    elif anchored:
        model = AnchoredFittedClosureModel.fit(
            training_cases,
            innovation_rank=args.innovation_rank,
            nearest_anchors=args.nearest_anchors,
            persistent_innovations=persistent_innovations,
            **fit_kwargs,
        )
        artifact_name = (
            "anchored_persistent_fitted_closure.npz"
            if persistent_innovations
            else "anchored_fitted_closure.npz"
        )
        model_path = model.save(output_directory / artifact_name)
        candidate_model = (
            "anchored_persistent_fitted"
            if persistent_innovations
            else "anchored_fitted"
        )
        candidate_closure = "anchored_fitted"
    else:
        model = FittedLocalClosureModel.fit(training_cases, **fit_kwargs)
        model_path = model.save(output_directory / "fitted_local_closure.npz")
        candidate_model = "fitted_local"
        candidate_closure = "fitted_local"
    exact_anchor_path: Path | None = None
    if anchored and not physics_conditioned and args.run_ablation:
        exact_anchor_path = run_exact_anchor_ensemble_validation(
            training_cases,
            model,
            output_directory / "exact_anchor_validation.csv",
            n_runs=args.exact_anchor_runs,
            bandwidth=args.bandwidth,
            seed=args.seed + 70_000,
        )
    del training_cases
    gc.collect()

    if args.run_lockbox:
        with Path(args.lockbox_manifest).open(newline="", encoding="utf-8") as handle:
            lockbox_rows = list(csv.DictReader(handle))
        validation_specs = tuple(
            (
                row["case"],
                (
                    int(row["nodes"]),
                    int(row["particles"]),
                    float(row["impulse_fraction"]),
                    float(row["diffusion"]),
                    float(row["length"]),
                ),
                int(row["runs"]),
                int(row["seed"]),
            )
            for row in lockbox_rows
        )
    else:
        validation_specs = (
            (
                "base_unseen",
                (101, 5_275, 0.59, 2.2e-4, 4.0),
                args.validation_runs,
                args.seed + 10_000,
            ),
            (
                "unseen_impulse",
                (101, 5_275, 0.37, 2.2e-4, 4.0),
                args.heldout_runs,
                args.seed + 10_001,
            ),
            (
                "unseen_particles",
                (101, 6_500, 0.59, 2.2e-4, 4.0),
                args.heldout_runs,
                args.seed + 10_002,
            ),
            (
                "unseen_diffusion",
                (101, 5_275, 0.59, 3.0e-4, 4.0),
                args.heldout_runs,
                args.seed + 10_003,
            ),
            (
                "unseen_grid",
                (126, 5_275, 0.59, 2.2e-4, 4.0),
                args.heldout_runs,
                args.seed + 10_004,
            ),
            (
                "unseen_domain",
                (101, 5_275, 0.59, 2.2e-4, 6.0),
                args.heldout_runs,
                args.seed + 10_005,
            ),
        )
    validation_rows: list[dict[str, object]] = []
    floor_metrics: dict[str, float | int] | None = None

    def evaluate(
        config,
        reference_runs: np.ndarray,
        candidate_runs: np.ndarray,
        *,
        modes: int,
        handoff_time: float,
        seed: int,
    ) -> dict[str, float | int]:
        time_indices = _nearest_indices(config.times, args.distribution_times)
        node_indices = sorted(
            {
                min(config.n_nodes - 1, max(0, config.impulse_index + offset))
                for offset in args.distribution_node_offsets
            }
        )
        distribution = summarize_distribution_ensemble(
            reference_runs,
            candidate_runs,
            time_indices=time_indices,
            node_indices=node_indices,
            n_projections=args.projections,
            histogram_bins=args.histogram_bins,
            seed=seed,
        )
        analytic_mean = discrete_expected_diffusion(config)
        analytic_variance = multinomial_marginal_variance(config)
        analytic_covariance = multinomial_marginal_covariance(config)
        moments = summarize_ensemble(
            analytic_mean,
            analytic_variance,
            candidate_runs,
            config.n_particles,
            reference_covariance=analytic_covariance,
            covariance_time_indices=time_indices,
        )
        start = int(np.argmin(np.abs(config.times - handoff_time)))
        origins = valid_time_origins(config.n_steps, 1, start_step=start)
        exact_cross = multinomial_mean_cross_time_covariance(
            config,
            1,
            start_indices=origins,
        )
        empirical_cross = empirical_mean_cross_time_covariance(
            candidate_runs,
            1,
            start_indices=origins,
        )
        return {
            **distribution.as_dict(),
            "mean_relative_l2_error": moments.mean_relative_l2_error,
            "variance_relative_l2_error": moments.variance_relative_l2_error,
            "covariance_relative_frobenius_error": float(
                moments.covariance_relative_frobenius_error
            ),
            "cross_time_covariance_relative_frobenius_error": (
                relative_frobenius_error(exact_cross, empirical_cross)
            ),
            "mean_negative_mass_fraction": moments.mean_negative_mass_fraction,
            "maximum_mass_error": moments.maximum_mass_error,
            "retained_modes": modes,
        }

    for case_index, (case_name, spec, n_runs, case_seed) in enumerate(
        validation_specs
    ):
        config = config_for(spec)
        modes, handoff_time, completion_rank = point.scaled(
            config.n_nodes,
            float(config.times[-1]),
        )
        reference = run_random_walk_ensemble(
            config,
            n_runs=n_runs,
            seed=case_seed,
            method="multinomial",
        )
        if case_index == 0:
            second_reference = run_random_walk_ensemble(
                config,
                n_runs=n_runs,
                seed=case_seed + 10_000,
                method="multinomial",
            )
            floor_metrics = evaluate(
                config,
                reference.runs,
                second_reference.runs,
                modes=config.n_nodes,
                handoff_time=handoff_time,
                seed=args.seed,
            )
            validation_rows.append(
                {
                    "case": case_name,
                    "model": "sampling_floor",
                    "runs": n_runs,
                    **floor_metrics,
                }
            )
            del second_reference
        raw = run_modal_ensemble(
            config,
            n_runs=n_runs,
            model="handoff_correlated_modal",
            n_modes=modes,
            initial_n_modes=config.n_nodes,
            handoff_time=handoff_time,
            seed=case_seed + 20_000,
        )
        analytic = apply_readout_ensemble(
            raw,
            config,
            readout="local_persistent_unresolved_completion_simplex_bank",
            retained_modes=modes,
            completion_start_time=handoff_time,
            completion_rank=completion_rank,
            completion_bandwidth=args.bandwidth,
            seed=case_seed + 30_000,
        )
        if physics_conditioned:
            fitted = complete_conditioned_ensemble(
                raw.runs,
                config,
                model,
                retained_modes=modes,
                completion_start_time=handoff_time,
                seed=case_seed + 30_000,
                apply_simplex_bank=True,
            )
        elif anchored:
            fitted = complete_anchored_ensemble(
                raw.runs,
                config,
                model,
                retained_modes=modes,
                completion_start_time=handoff_time,
                seed=case_seed + 30_000,
                apply_simplex_bank=True,
            )
        else:
            fitted = complete_fitted_ensemble(
                raw.runs,
                config,
                model,
                retained_modes=modes,
                completion_start_time=handoff_time,
                seed=case_seed + 30_000,
                apply_simplex_bank=True,
            )
        for model_name, runs in (
            ("analytic_local", analytic.runs),
            (candidate_model, fitted),
        ):
            metrics = evaluate(
                config,
                reference.runs,
                runs,
                modes=modes,
                handoff_time=handoff_time,
                seed=args.seed,
            )
            validation_rows.append(
                {
                    "case": case_name,
                    "model": model_name,
                    "nodes": config.n_nodes,
                    "particles": config.n_particles,
                    "impulse_index": config.impulse_index,
                    "diffusion": config.diffusion_coefficient,
                    "length": config.length,
                    "runs": n_runs,
                    **metrics,
                }
            )
        del reference, raw, analytic, fitted
        gc.collect()

    validation_csv = _write_rows(
        validation_rows,
        output_directory / "heldout_validation.csv",
    )
    if floor_metrics is None:
        raise RuntimeError("base sampling floor was not evaluated")
    acceptance_metrics = (
        "mean_relative_l2_error",
        "variance_relative_l2_error",
        "covariance_relative_frobenius_error",
        "cross_time_covariance_relative_frobenius_error",
        "marginal_normalized_wasserstein",
        "sliced_normalized_wasserstein",
    )
    cases = sorted({str(row["case"]) for row in validation_rows})
    accuracy_pass = True
    failures: list[dict[str, object]] = []
    for case_name in cases:
        analytic = next(
            row
            for row in validation_rows
            if row["case"] == case_name and row["model"] == "analytic_local"
        )
        fitted = next(
            row
            for row in validation_rows
            if row["case"] == case_name and row["model"] == candidate_model
        )
        for metric in acceptance_metrics:
            threshold = float(analytic[metric]) + float(floor_metrics[metric])
            if float(fitted[metric]) > threshold:
                accuracy_pass = False
                failures.append(
                    {
                        "case": case_name,
                        "metric": metric,
                        "fitted": fitted[metric],
                        "threshold": threshold,
                    }
                )

    analytic_benchmark = benchmark_operating_points(
        node_counts=args.node_counts,
        n_particles=args.particles,
        total_time=args.total_time,
        requested_dt=args.dt,
        length=4.0,
        diffusion_coefficient=2.2e-4,
        impulse_fraction=0.59,
        operating_points=(point,),
        repeats=args.repeats,
        amortization_runs=args.amortization_runs,
        seed=args.seed,
        closure="local_persistent",
        modal_bandwidth=args.bandwidth,
    )
    candidate_benchmark = benchmark_operating_points(
        node_counts=args.node_counts,
        n_particles=args.particles,
        total_time=args.total_time,
        requested_dt=args.dt,
        length=4.0,
        diffusion_coefficient=2.2e-4,
        impulse_fraction=0.59,
        operating_points=(point,),
        repeats=args.repeats,
        amortization_runs=args.amortization_runs,
        seed=args.seed,
        closure=candidate_closure,
        modal_bandwidth=args.bandwidth,
        fitted_model=None if anchored else model,
        anchored_model=model if anchored and not physics_conditioned else None,
        conditioned_model=model if physics_conditioned else None,
    )
    benchmark_records = [*analytic_benchmark, *candidate_benchmark]
    scaling_csv = write_operating_point_benchmark_csv(
        benchmark_records,
        output_directory / "scaling.csv",
    )
    largest = max(args.node_counts)
    analytic_largest = next(row for row in analytic_benchmark if row.n_nodes == largest)
    fitted_largest = next(row for row in candidate_benchmark if row.n_nodes == largest)
    scaling_pass = (
        fitted_largest.setup_seconds < analytic_largest.setup_seconds
        and fitted_largest.peak_rss_bytes <= analytic_largest.peak_rss_bytes
        and fitted_largest.median_run_seconds <= 1.25 * analytic_largest.median_run_seconds
    )

    base_config = config_for((101, args.particles, 0.59, 2.2e-4, 4.0))
    random_walk_records = benchmark_random_walks(
        base_config,
        particle_counts=args.random_walk_particle_counts,
        repeats=args.repeats,
        seed=args.seed,
    )
    write_benchmark_csv(
        random_walk_records,
        output_directory / "random_walk_baselines.csv",
    )
    plot_random_walk_benchmark(
        random_walk_records,
        output_directory / "random_walk_baselines.png",
    )
    figure = plot_fitted_closure_study(
        validation_rows,
        benchmark_records,
        output_directory / "fitted_closure.png",
        candidate_model=candidate_model,
    )
    accepted = accuracy_pass and scaling_pass
    summary = {
        "training_cases": len(training_specs),
        "model_kind": candidate_model,
        "training_runs_per_case": args.training_runs,
        "validation_runs": args.validation_runs,
        "heldout_runs": args.heldout_runs,
        "accuracy_pass": accuracy_pass,
        "scaling_pass": scaling_pass,
        "accepted_as_default": accepted,
        "lockbox_opened": bool(args.run_lockbox),
        "production_default": (
            candidate_model if accepted else "analytic_local_bandwidth_12"
        ),
        "failures": failures,
        "model": str(model_path),
        "conditioning_degree": (
            selected_conditioning_degree if physics_conditioned else None
        ),
        "conditioning_cross_validation_csv": (
            str(conditioning_cv_path) if conditioning_cv_path is not None else None
        ),
        "recommended_anchor_expansion_csv": (
            str(anchor_recommendations_path)
            if anchor_recommendations_path is not None
            else None
        ),
        "ablation_csv": str(ablation_path) if ablation_path is not None else None,
        "exact_anchor_validation_csv": (
            str(exact_anchor_path) if exact_anchor_path is not None else None
        ),
        "validation_csv": str(validation_csv),
        "scaling_csv": str(scaling_csv),
        "figure": str(figure),
        "largest_grid": {
            "nodes": largest,
            "analytic_setup_seconds": analytic_largest.setup_seconds,
            "fitted_setup_seconds": fitted_largest.setup_seconds,
            "analytic_run_seconds": analytic_largest.median_run_seconds,
            "fitted_run_seconds": fitted_largest.median_run_seconds,
            "analytic_peak_rss_mib": analytic_largest.peak_rss_megabytes,
            "fitted_peak_rss_mib": fitted_largest.peak_rss_megabytes,
        },
    }
    (output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0
