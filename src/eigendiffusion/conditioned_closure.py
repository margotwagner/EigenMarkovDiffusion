"""Physics-conditioned sparse fitted closure.

The regression is fitted jointly across physical configurations.  Local modal
features are interacted with dimensionless configuration coordinates, so the
model transfers predictions rather than interpolating independently estimated
ridge coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .anchored_closure import (
    AnchoredFittedClosureModel,
    AnchoredFittedLocalCompleter,
)
from .config import DiffusionConfig
from .fitted_closure import FittedClosureTrainingCase, _canonical_basis
from .readouts import simplex_bank_readout
from .references import discrete_expected_diffusion

FloatArray = NDArray[np.float64]


def physics_coordinates(config: DiffusionConfig) -> FloatArray:
    """Coordinates that control normalized one-dimensional diffusion dynamics."""

    dimensionless_time = (
        config.diffusion_coefficient * float(config.times[-1]) / config.length**2
    )
    return np.asarray(
        [
            np.log(float(config.n_nodes)),
            config.impulse_index / max(config.n_nodes - 1, 1),
            np.log(max(dimensionless_time, np.finfo(float).tiny)),
        ],
        dtype=float,
    )


def _condition_basis(coordinates: FloatArray, degree: int) -> FloatArray:
    values = np.asarray(coordinates, dtype=float)
    terms = [1.0, *values]
    if degree == 2:
        terms.extend(values**2)
        terms.extend(
            values[left] * values[right]
            for left in range(values.size)
            for right in range(left + 1, values.size)
        )
    elif degree != 1:
        raise ValueError("conditioning degree must be 1 or 2")
    return np.asarray(terms, dtype=float)


@dataclass(frozen=True, slots=True)
class PhysicsConditionedRegressionModel:
    """Joint sparse regression whose coefficients vary with physical coordinates."""

    bandwidth: int
    n_frequency_bins: int
    degree: int
    descriptor_center: FloatArray
    descriptor_scale: FloatArray
    initial_coefficient_basis: FloatArray
    transition_coefficient_basis: FloatArray

    @property
    def feature_width(self) -> int:
        return 2 * self.bandwidth + 1

    def condition(self, config: DiffusionConfig) -> FloatArray:
        normalized = (
            physics_coordinates(config) - self.descriptor_center
        ) / self.descriptor_scale
        return _condition_basis(normalized, self.degree)

    def coefficients(self, config: DiffusionConfig) -> tuple[FloatArray, FloatArray]:
        condition = self.condition(config)
        return (
            np.einsum("p,bpw->bw", condition, self.initial_coefficient_basis),
            np.einsum("p,bpw->bw", condition, self.transition_coefficient_basis),
        )

    @classmethod
    def fit(
        cls,
        cases: list[FittedClosureTrainingCase],
        *,
        bandwidth: int = 12,
        n_frequency_bins: int = 8,
        degree: int = 1,
        ridge: float = 1.0e-2,
        max_regression_samples_per_target: int = 2_000,
        seed: int = 0,
    ) -> PhysicsConditionedRegressionModel:
        """Fit one descriptor-conditioned regression directly from all trajectories."""

        if not cases:
            raise ValueError("cases must not be empty")
        descriptors = np.stack([physics_coordinates(case.config) for case in cases])
        center = np.mean(descriptors, axis=0)
        scale = np.std(descriptors, axis=0, ddof=0)
        scale[scale < 1.0e-12] = 1.0
        conditions = [
            _condition_basis((descriptor - center) / scale, degree)
            for descriptor in descriptors
        ]
        condition_width = conditions[0].size
        width = 2 * bandwidth + 1
        transition_width = 3 * width
        initial_size = condition_width * width
        transition_size = condition_width * transition_width
        initial_xx = np.zeros(
            (n_frequency_bins, initial_size, initial_size), dtype=float
        )
        initial_xy = np.zeros((n_frequency_bins, initial_size), dtype=float)
        transition_xx = np.zeros(
            (n_frequency_bins, transition_size, transition_size), dtype=float
        )
        transition_xy = np.zeros((n_frequency_bins, transition_size), dtype=float)
        generator = np.random.default_rng(seed)

        for case, condition in zip(cases, conditions, strict=True):
            config = case.config
            runs = np.asarray(case.runs, dtype=float)
            retained = case.retained_modes
            start = int(np.argmin(np.abs(config.times - case.handoff_time)))
            basis = _canonical_basis(config)
            modal = runs @ basis.eigenvectors
            modal_mean = discrete_expected_diffusion(config) @ basis.eigenvectors
            deviation = (modal - modal_mean[None, :, :]) / np.sqrt(
                config.n_particles
            )
            for target in range(retained, config.n_nodes):
                frequency_bin = min(
                    n_frequency_bins - 1,
                    int(target / max(config.n_nodes - 1, 1) * n_frequency_bins),
                )
                sources = target + np.arange(-bandwidth, bandwidth + 1)
                initial_features = np.zeros((runs.shape[0], width), dtype=float)
                valid_retained = (sources >= 0) & (sources < retained)
                initial_features[:, valid_retained] = deviation[
                    :, start, sources[valid_retained]
                ]
                conditioned_initial = (
                    condition[None, :, None] * initial_features[:, None, :]
                ).reshape(runs.shape[0], initial_size)
                initial_target = deviation[:, start, target]
                initial_xx[frequency_bin] += conditioned_initial.T @ conditioned_initial
                initial_xy[frequency_bin] += conditioned_initial.T @ initial_target

                transition_count = config.n_steps - start - 1
                if transition_count <= 0:
                    continue
                transition_features = np.zeros(
                    (runs.shape[0], transition_count, transition_width), dtype=float
                )
                valid_unresolved = (sources >= retained) & (sources < config.n_nodes)
                transition_features[:, :, :width][:, :, valid_unresolved] = deviation[
                    :, start:-1, sources[valid_unresolved]
                ]
                transition_features[:, :, width : 2 * width][
                    :, :, valid_retained
                ] = deviation[:, start:-1, sources[valid_retained]]
                transition_features[:, :, 2 * width :][
                    :, :, valid_retained
                ] = deviation[:, start + 1 :, sources[valid_retained]]
                flattened = transition_features.reshape(-1, transition_width)
                target_values = deviation[:, start + 1 :, target].reshape(-1)
                if flattened.shape[0] > max_regression_samples_per_target:
                    selected = generator.choice(
                        flattened.shape[0],
                        size=max_regression_samples_per_target,
                        replace=False,
                    )
                    flattened = flattened[selected]
                    target_values = target_values[selected]
                conditioned_transition = (
                    condition[None, :, None] * flattened[:, None, :]
                ).reshape(flattened.shape[0], transition_size)
                transition_xx[frequency_bin] += (
                    conditioned_transition.T @ conditioned_transition
                )
                transition_xy[frequency_bin] += (
                    conditioned_transition.T @ target_values
                )

        def solve(
            normal: FloatArray,
            target: FloatArray,
            local_width: int,
        ) -> FloatArray:
            result = np.zeros(
                (n_frequency_bins, condition_width, local_width), dtype=float
            )
            for frequency_bin in range(n_frequency_bins):
                matrix = normal[frequency_bin]
                scale_value = float(np.trace(matrix) / max(matrix.shape[0], 1))
                if scale_value <= 1.0e-15:
                    continue
                coefficients = np.linalg.solve(
                    matrix + ridge * scale_value * np.eye(matrix.shape[0]),
                    target[frequency_bin],
                )
                result[frequency_bin] = coefficients.reshape(
                    condition_width, local_width
                )
            return result

        return cls(
            bandwidth=bandwidth,
            n_frequency_bins=n_frequency_bins,
            degree=degree,
            descriptor_center=center,
            descriptor_scale=scale,
            initial_coefficient_basis=solve(initial_xx, initial_xy, width),
            transition_coefficient_basis=solve(
                transition_xx, transition_xy, transition_width
            ),
        )

    def save(self, output_path: str | Path) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            bandwidth=np.asarray(self.bandwidth),
            n_frequency_bins=np.asarray(self.n_frequency_bins),
            degree=np.asarray(self.degree),
            descriptor_center=self.descriptor_center,
            descriptor_scale=self.descriptor_scale,
            initial_coefficient_basis=self.initial_coefficient_basis,
            transition_coefficient_basis=self.transition_coefficient_basis,
        )
        return output


@dataclass(frozen=True, slots=True)
class PhysicsConditionedClosureModel:
    """Joint conditioned regression paired with validated anchor innovations."""

    regression: PhysicsConditionedRegressionModel
    innovations: AnchoredFittedClosureModel

    def save(self, output_path: str | Path) -> Path:
        output = Path(output_path)
        self.regression.save(output)
        self.innovations.save(output.with_name(f"{output.stem}_innovations.npz"))
        return output


class PhysicsConditionedLocalCompleter:
    """Apply conditioned coefficients with anchor-based Gaussian innovations."""

    def __init__(
        self,
        config: DiffusionConfig,
        model: PhysicsConditionedClosureModel,
        *,
        retained_modes: int,
        completion_start_time: float,
    ) -> None:
        initial, transition = model.regression.coefficients(config)
        self._completer = AnchoredFittedLocalCompleter(
            config,
            model.innovations,
            retained_modes=retained_modes,
            completion_start_time=completion_start_time,
            initial_coefficients=initial,
            transition_coefficients=transition,
        )

    def complete(
        self,
        raw_counts: FloatArray,
        rng: np.random.Generator | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        return self._completer.complete(raw_counts, rng=rng)

    def complete_batch(
        self,
        raw_counts: FloatArray,
        rng: np.random.Generator | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        return self._completer.complete_batch(raw_counts, rng=rng)


def complete_conditioned_ensemble(
    runs: FloatArray,
    config: DiffusionConfig,
    model: PhysicsConditionedClosureModel,
    *,
    retained_modes: int,
    completion_start_time: float,
    seed: int = 0,
    apply_simplex_bank: bool = True,
) -> FloatArray:
    """Apply the conditioned closure to a complete trajectory ensemble."""

    values = np.asarray(runs, dtype=float)
    completer = PhysicsConditionedLocalCompleter(
        config,
        model,
        retained_modes=retained_modes,
        completion_start_time=completion_start_time,
    )
    completed, _ = completer.complete_batch(values, rng=np.random.default_rng(seed))
    if not apply_simplex_bank:
        return completed
    return np.stack(
        [simplex_bank_readout(run, config.n_particles).counts for run in completed]
    )


def conditioned_prediction_error(
    model: PhysicsConditionedRegressionModel,
    cases: list[FittedClosureTrainingCase],
) -> tuple[float, float]:
    """Return teacher-forced initial and transition normalized prediction errors."""

    initial_numerator = 0.0
    initial_denominator = 0.0
    transition_numerator = 0.0
    transition_denominator = 0.0
    for case in cases:
        config = case.config
        retained = case.retained_modes
        start = int(np.argmin(np.abs(config.times - case.handoff_time)))
        basis = _canonical_basis(config)
        modal = np.asarray(case.runs, dtype=float) @ basis.eigenvectors
        modal_mean = discrete_expected_diffusion(config) @ basis.eigenvectors
        deviation = (modal - modal_mean[None, :, :]) / np.sqrt(config.n_particles)
        initial_coefficients, transition_coefficients = model.coefficients(config)
        from .fitted_closure import FittedLocalClosureModel

        prototype = FittedLocalClosureModel(
            bandwidth=model.bandwidth,
            n_frequency_bins=model.n_frequency_bins,
            n_time_bins=1,
            residual_grid_size=1,
            initial_coefficients=initial_coefficients,
            transition_coefficients=transition_coefficients,
            initial_residual_bank=np.empty((0, 1)),
            transition_residual_banks=(np.empty((0, 1)),),
        )
        initial_target = deviation[:, start, retained:]
        initial_prediction = prototype._predict_initial(deviation[:, start], retained)
        initial_numerator += float(np.sum((initial_target - initial_prediction) ** 2))
        initial_denominator += float(np.sum(initial_target**2))
        for time_index in range(start, config.n_steps - 1):
            target = deviation[:, time_index + 1, retained:]
            prediction = prototype._predict_transition(
                deviation[:, time_index],
                deviation[:, time_index + 1, :retained],
                retained,
            )
            transition_numerator += float(np.sum((target - prediction) ** 2))
            transition_denominator += float(np.sum(target**2))
    return (
        initial_numerator / max(initial_denominator, np.finfo(float).tiny),
        transition_numerator / max(transition_denominator, np.finfo(float).tiny),
    )


def cross_validate_conditioned_regression(
    cases: list[FittedClosureTrainingCase],
    output_path: str | Path,
    *,
    degrees: list[int],
    folds: int,
    bandwidth: int,
    n_frequency_bins: int,
    ridge: float,
    max_regression_samples_per_target: int,
    seed: int,
) -> tuple[Path, int]:
    """Select conditioning degree using complete held-out physical configurations."""

    if folds < 2 or folds > len(cases):
        raise ValueError("folds must be between two and the number of cases")
    fold_indices = np.array_split(np.arange(len(cases)), folds)
    rows: list[dict[str, object]] = []
    for degree in sorted(set(degrees)):
        for fold_index, heldout_indices in enumerate(fold_indices):
            heldout_set = set(int(index) for index in heldout_indices)
            training = [case for index, case in enumerate(cases) if index not in heldout_set]
            heldout = [case for index, case in enumerate(cases) if index in heldout_set]
            model = PhysicsConditionedRegressionModel.fit(
                training,
                bandwidth=bandwidth,
                n_frequency_bins=n_frequency_bins,
                degree=degree,
                ridge=ridge,
                max_regression_samples_per_target=max_regression_samples_per_target,
                seed=seed + 100 * degree + fold_index,
            )
            initial_error, transition_error = conditioned_prediction_error(model, heldout)
            rows.append(
                {
                    "degree": degree,
                    "fold": fold_index,
                    "heldout_cases": " ".join(str(index) for index in heldout_indices),
                    "initial_normalized_prediction_error": initial_error,
                    "transition_normalized_prediction_error": transition_error,
                    "combined_prediction_error": 0.5 * (initial_error + transition_error),
                }
            )
    means = {
        degree: float(
            np.mean(
                [
                    float(row["combined_prediction_error"])
                    for row in rows
                    if row["degree"] == degree
                ]
            )
        )
        for degree in sorted(set(degrees))
    }
    selected_degree = min(means, key=means.get)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output, selected_degree


def recommend_anchor_expansion(
    cases: list[FittedClosureTrainingCase],
    cross_validation_path: str | Path,
    output_path: str | Path,
    *,
    count: int = 4,
) -> Path:
    """Propose bridge anchors around the worst configuration-level CV regions."""

    if count <= 0:
        raise ValueError("count must be positive")
    with Path(cross_validation_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    degrees = sorted({int(row["degree"]) for row in rows})
    mean_errors = {
        degree: np.mean(
            [
                float(row["combined_prediction_error"])
                for row in rows
                if int(row["degree"]) == degree
            ]
        )
        for degree in degrees
    }
    selected_degree = min(mean_errors, key=mean_errors.get)
    case_errors = np.zeros(len(cases), dtype=float)
    for row in rows:
        if int(row["degree"]) != selected_degree:
            continue
        error = float(row["combined_prediction_error"])
        for index in str(row["heldout_cases"]).split():
            case_errors[int(index)] = error

    descriptors = np.stack([physics_coordinates(case.config) for case in cases])
    center = np.mean(descriptors, axis=0)
    scale = np.std(descriptors, axis=0, ddof=0)
    scale[scale < 1.0e-12] = 1.0
    normalized = (descriptors - center) / scale
    proposals: list[dict[str, object]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for source_index in np.argsort(-case_errors):
        distances = np.linalg.norm(normalized - normalized[source_index], axis=1)
        distances[source_index] = np.inf
        source = cases[int(source_index)].config
        for neighbor_index in np.argsort(distances):
            neighbor = cases[int(neighbor_index)].config
            nodes = max(2, int(round(0.5 * (source.n_nodes + neighbor.n_nodes))))
            impulse_fraction = 0.5 * (
                source.impulse_index / max(source.n_nodes - 1, 1)
                + neighbor.impulse_index / max(neighbor.n_nodes - 1, 1)
            )
            diffusion = float(
                np.sqrt(
                    source.diffusion_coefficient * neighbor.diffusion_coefficient
                )
            )
            length = float(np.sqrt(source.length * neighbor.length))
            key = (
                nodes,
                int(round(1_000 * impulse_fraction)),
                int(round(1.0e8 * diffusion)),
                int(round(1_000 * length)),
            )
            if key not in seen:
                seen.add(key)
                break
        else:  # pragma: no cover - requires every midpoint to be duplicated
            continue
        proposals.append(
            {
                "source_anchor": int(source_index),
                "bridge_neighbor": int(neighbor_index),
                "source_cv_error": case_errors[source_index],
                "suggested_nodes": nodes,
                "suggested_particles": int(
                    round(np.sqrt(source.n_particles * neighbor.n_particles))
                ),
                "suggested_impulse_fraction": impulse_fraction,
                "suggested_diffusion": diffusion,
                "suggested_length": length,
                "reason": "bridge worst configuration-level CV region",
            }
        )
        if len(proposals) >= count:
            break
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(proposals[0]))
        writer.writeheader()
        writer.writerows(proposals)
    return output
