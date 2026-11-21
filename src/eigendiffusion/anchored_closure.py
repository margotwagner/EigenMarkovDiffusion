"""Configuration-conditioned fitted closure with low-rank innovations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .config import DiffusionConfig
from .fitted_closure import (
    FittedClosureTrainingCase,
    FittedLocalClosureModel,
    _canonical_basis,
    _interpolation_matrix,
)
from .operators import Eigenbasis
from .readouts import simplex_bank_readout
from .references import discrete_expected_diffusion

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def configuration_descriptor(config: DiffusionConfig) -> FloatArray:
    """Return scale-aware coordinates used to locate nearby fitted anchors."""

    impulse_fraction = config.impulse_index / max(config.n_nodes - 1, 1)
    dimensionless_time = (
        config.diffusion_coefficient * float(config.times[-1]) / config.length**2
    )
    return np.asarray(
        [
            np.log(float(config.n_nodes)),
            impulse_fraction,
            np.log(config.diffusion_coefficient),
            np.log(config.length),
            np.log(max(dimensionless_time, np.finfo(float).tiny)),
            np.log(float(config.n_particles)),
        ],
        dtype=float,
    )


def _low_rank_factor(
    residuals: FloatArray,
    rank: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Fit a low-rank-plus-diagonal Gaussian while preserving marginal variance."""

    values = np.asarray(residuals, dtype=float)
    mean = np.mean(values, axis=0)
    centered = values - mean
    factor = np.zeros((values.shape[1], rank), dtype=float)
    if values.shape[0] > 1 and rank > 0:
        _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
        effective_rank = min(rank, singular_values.size)
        factor[:, :effective_rank] = (
            right_vectors[:effective_rank].T
            * singular_values[:effective_rank][None, :]
            / np.sqrt(values.shape[0] - 1)
        )
        empirical_variance = np.var(values, axis=0, ddof=1)
    else:
        empirical_variance = np.zeros(values.shape[1], dtype=float)
    diagonal_variance = np.maximum(empirical_variance - np.sum(factor**2, axis=1), 0.0)
    return mean, factor, np.sqrt(diagonal_variance)


def _residual_sequences(
    case: FittedClosureTrainingCase,
    model: FittedLocalClosureModel,
) -> tuple[FloatArray, FloatArray, IntArray]:
    """Return initial and time-ordered transition residuals on the fixed grid."""

    config = case.config
    runs = np.asarray(case.runs, dtype=float)
    retained = case.retained_modes
    start = int(np.argmin(np.abs(config.times - case.handoff_time)))
    basis = _canonical_basis(config)
    modal = runs @ basis.eigenvectors
    modal_mean = discrete_expected_diffusion(config) @ basis.eigenvectors
    deviation = (modal - modal_mean[None, :, :]) / np.sqrt(config.n_particles)
    to_grid = _interpolation_matrix(config.n_nodes - retained, model.residual_grid_size)

    initial_prediction = model._predict_initial(deviation[:, start], retained)
    initial = (deviation[:, start, retained:] - initial_prediction) @ to_grid.T
    transitions: list[FloatArray] = []
    time_bins: list[int] = []
    for time_index in range(start, config.n_steps - 1):
        prediction = model._predict_transition(
            deviation[:, time_index],
            deviation[:, time_index + 1, :retained],
            retained,
        )
        transitions.append(
            (deviation[:, time_index + 1, retained:] - prediction) @ to_grid.T
        )
        fraction = (time_index - start) / max(config.n_steps - start - 2, 1)
        time_bins.append(min(model.n_time_bins - 1, int(fraction * model.n_time_bins)))
    return initial, np.stack(transitions, axis=1), np.asarray(time_bins, dtype=np.int64)


def _conditional_innovation_statistics(
    initial: FloatArray,
    transitions: FloatArray,
    time_bins: IntArray,
    *,
    n_time_bins: int,
    rank: int,
    ridge: float,
    max_samples: int,
    seed: int,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Fit stable Gaussian residual transitions in normalized-grid coordinates."""

    generator = np.random.default_rng(seed)
    previous = np.concatenate((initial[:, None, :], transitions[:, :-1, :]), axis=1)
    means: list[FloatArray] = []
    previous_means: list[FloatArray] = []
    gains: list[FloatArray] = []
    factors: list[FloatArray] = []
    diagonals: list[FloatArray] = []
    grid_size = initial.shape[1]
    for time_bin in range(n_time_bins):
        selected_times = np.flatnonzero(time_bins == time_bin)
        current_values = transitions[:, selected_times, :].reshape(-1, grid_size)
        previous_values = previous[:, selected_times, :].reshape(-1, grid_size)
        if current_values.shape[0] > max_samples:
            selected = generator.choice(
                current_values.shape[0], size=max_samples, replace=False
            )
            current_values = current_values[selected]
            previous_values = previous_values[selected]
        current_mean = np.mean(current_values, axis=0)
        previous_mean = np.mean(previous_values, axis=0)
        current_centered = current_values - current_mean
        previous_centered = previous_values - previous_mean
        normal = previous_centered.T @ previous_centered
        scale = max(float(np.trace(normal) / grid_size), 1.0e-12)
        coefficients = np.linalg.solve(
            normal + ridge * scale * np.eye(grid_size),
            previous_centered.T @ current_centered,
        )
        gain = coefficients.T
        left, singular_values, right = np.linalg.svd(gain, full_matrices=False)
        gain = (left * np.minimum(singular_values, 0.95)[None, :]) @ right
        conditional = current_centered - previous_centered @ gain.T
        conditional_mean, factor, diagonal = _low_rank_factor(conditional, rank)
        means.append(current_mean + conditional_mean)
        previous_means.append(previous_mean)
        gains.append(gain)
        factors.append(factor)
        diagonals.append(diagonal)
    return (
        np.stack(means),
        np.stack(previous_means),
        np.stack(gains),
        np.stack(factors),
        np.stack(diagonals),
    )


@dataclass(frozen=True, slots=True)
class AnchoredFittedClosureModel:
    """Sparse anchor regressions interpolated in physical-configuration space."""

    bandwidth: int
    n_frequency_bins: int
    n_time_bins: int
    residual_grid_size: int
    innovation_rank: int
    nearest_anchors: int
    persistent_innovations: bool
    descriptor_center: FloatArray
    descriptor_scale: FloatArray
    anchor_descriptors: FloatArray
    initial_coefficients: FloatArray
    transition_coefficients: FloatArray
    initial_innovation_mean: FloatArray
    initial_innovation_factor: FloatArray
    initial_innovation_diagonal: FloatArray
    transition_innovation_mean: FloatArray
    transition_previous_mean: FloatArray
    transition_innovation_gain: FloatArray
    transition_innovation_factor: FloatArray
    transition_innovation_diagonal: FloatArray

    @classmethod
    def fit(
        cls,
        cases: list[FittedClosureTrainingCase],
        *,
        bandwidth: int = 12,
        n_frequency_bins: int = 8,
        n_time_bins: int = 5,
        residual_grid_size: int = 32,
        innovation_rank: int = 12,
        nearest_anchors: int = 3,
        persistent_innovations: bool = False,
        ridge: float = 1.0e-3,
        max_residual_samples: int = 4_000,
        max_regression_samples_per_target: int = 2_000,
        seed: int = 0,
    ) -> AnchoredFittedClosureModel:
        """Fit one sparse closure per physical anchor and compress its innovations."""

        if not cases:
            raise ValueError("cases must not be empty")
        if innovation_rank <= 0 or nearest_anchors <= 0:
            raise ValueError("innovation_rank and nearest_anchors must be positive")
        models = [
            FittedLocalClosureModel.fit(
                [case],
                bandwidth=bandwidth,
                n_frequency_bins=n_frequency_bins,
                n_time_bins=n_time_bins,
                residual_grid_size=residual_grid_size,
                ridge=ridge,
                max_residual_samples=max_residual_samples,
                max_regression_samples_per_target=max_regression_samples_per_target,
                seed=seed + index,
            )
            for index, case in enumerate(cases)
        ]
        descriptors = np.stack([configuration_descriptor(case.config) for case in cases])
        center = np.mean(descriptors, axis=0)
        scale = np.std(descriptors, axis=0, ddof=0)
        scale[scale < 1.0e-12] = 1.0

        initial_statistics = [
            _low_rank_factor(model.initial_residual_bank, innovation_rank)
            for model in models
        ]
        transition_statistics = []
        for index, (case, model) in enumerate(zip(cases, models, strict=True)):
            if persistent_innovations:
                initial, transitions, time_bins = _residual_sequences(case, model)
                transition_statistics.append(
                    _conditional_innovation_statistics(
                        initial,
                        transitions,
                        time_bins,
                        n_time_bins=n_time_bins,
                        rank=innovation_rank,
                        ridge=ridge,
                        max_samples=max_residual_samples,
                        seed=seed + 10_000 + index,
                    )
                )
            else:
                statistics = [
                    _low_rank_factor(bank, innovation_rank)
                    for bank in model.transition_residual_banks
                ]
                transition_statistics.append(
                    (
                        np.stack([item[0] for item in statistics]),
                        np.zeros((n_time_bins, residual_grid_size)),
                        np.zeros((n_time_bins, residual_grid_size, residual_grid_size)),
                        np.stack([item[1] for item in statistics]),
                        np.stack([item[2] for item in statistics]),
                    )
                )
        return cls(
            bandwidth=bandwidth,
            n_frequency_bins=n_frequency_bins,
            n_time_bins=n_time_bins,
            residual_grid_size=residual_grid_size,
            innovation_rank=innovation_rank,
            nearest_anchors=min(nearest_anchors, len(cases)),
            persistent_innovations=persistent_innovations,
            descriptor_center=center,
            descriptor_scale=scale,
            anchor_descriptors=descriptors,
            initial_coefficients=np.stack([model.initial_coefficients for model in models]),
            transition_coefficients=np.stack(
                [model.transition_coefficients for model in models]
            ),
            initial_innovation_mean=np.stack([item[0] for item in initial_statistics]),
            initial_innovation_factor=np.stack([item[1] for item in initial_statistics]),
            initial_innovation_diagonal=np.stack([item[2] for item in initial_statistics]),
            transition_innovation_mean=np.stack(
                [item[0] for item in transition_statistics]
            ),
            transition_previous_mean=np.stack(
                [item[1] for item in transition_statistics]
            ),
            transition_innovation_gain=np.stack(
                [item[2] for item in transition_statistics]
            ),
            transition_innovation_factor=np.stack(
                [item[3] for item in transition_statistics]
            ),
            transition_innovation_diagonal=np.stack(
                [item[4] for item in transition_statistics]
            ),
        )

    def interpolation_weights(self, config: DiffusionConfig) -> tuple[IntArray, FloatArray]:
        """Return inverse-distance weights for the nearest physical anchors."""

        query = (configuration_descriptor(config) - self.descriptor_center) / self.descriptor_scale
        anchors = (self.anchor_descriptors - self.descriptor_center) / self.descriptor_scale
        distances = np.linalg.norm(anchors - query[None, :], axis=1)
        indices = np.argsort(distances)[: self.nearest_anchors].astype(np.int64)
        selected = distances[indices]
        if selected[0] < 1.0e-12:
            return indices[:1], np.ones(1, dtype=float)
        weights = 1.0 / np.maximum(selected, 5.0e-2) ** 2
        return indices, weights / np.sum(weights)

    def save(self, output_path: str | Path) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            bandwidth=np.asarray(self.bandwidth),
            n_frequency_bins=np.asarray(self.n_frequency_bins),
            n_time_bins=np.asarray(self.n_time_bins),
            residual_grid_size=np.asarray(self.residual_grid_size),
            innovation_rank=np.asarray(self.innovation_rank),
            nearest_anchors=np.asarray(self.nearest_anchors),
            persistent_innovations=np.asarray(self.persistent_innovations),
            descriptor_center=self.descriptor_center,
            descriptor_scale=self.descriptor_scale,
            anchor_descriptors=self.anchor_descriptors,
            initial_coefficients=self.initial_coefficients,
            transition_coefficients=self.transition_coefficients,
            initial_innovation_mean=self.initial_innovation_mean,
            initial_innovation_factor=self.initial_innovation_factor,
            initial_innovation_diagonal=self.initial_innovation_diagonal,
            transition_innovation_mean=self.transition_innovation_mean,
            transition_previous_mean=self.transition_previous_mean,
            transition_innovation_gain=self.transition_innovation_gain,
            transition_innovation_factor=self.transition_innovation_factor,
            transition_innovation_diagonal=self.transition_innovation_diagonal,
        )
        return output

    @classmethod
    def load(cls, input_path: str | Path) -> AnchoredFittedClosureModel:
        with np.load(input_path) as payload:
            scalar_names = (
                "bandwidth",
                "n_frequency_bins",
                "n_time_bins",
                "residual_grid_size",
                "innovation_rank",
                "nearest_anchors",
            )
            array_names = (
                "descriptor_center",
                "descriptor_scale",
                "anchor_descriptors",
                "initial_coefficients",
                "transition_coefficients",
                "initial_innovation_mean",
                "initial_innovation_factor",
                "initial_innovation_diagonal",
                "transition_innovation_mean",
                "transition_previous_mean",
                "transition_innovation_gain",
                "transition_innovation_factor",
                "transition_innovation_diagonal",
            )
            values = {name: int(payload[name]) for name in scalar_names}
            values["persistent_innovations"] = bool(
                payload["persistent_innovations"]
                if "persistent_innovations" in payload
                else False
            )
            anchor_count = payload["anchor_descriptors"].shape[0]
            n_time_bins = values["n_time_bins"]
            grid_size = values["residual_grid_size"]
            optional_arrays = {
                "transition_previous_mean": np.zeros(
                    (anchor_count, n_time_bins, grid_size)
                ),
                "transition_innovation_gain": np.zeros(
                    (anchor_count, n_time_bins, grid_size, grid_size)
                ),
            }
            array_names = tuple(
                name for name in array_names if name not in optional_arrays
            )
            values.update({name: payload[name].copy() for name in array_names})
            values.update(
                {
                    name: payload[name].copy() if name in payload else default
                    for name, default in optional_arrays.items()
                }
            )
            return cls(**values)


class AnchoredFittedLocalCompleter:
    """Complete one reduced trajectory with interpolated anchor parameters."""

    def __init__(
        self,
        config: DiffusionConfig,
        model: AnchoredFittedClosureModel,
        *,
        retained_modes: int,
        completion_start_time: float,
        initial_coefficients: FloatArray | None = None,
        transition_coefficients: FloatArray | None = None,
    ) -> None:
        if not 1 <= retained_modes < config.n_nodes:
            raise ValueError("retained_modes must omit at least one mode")
        self.config = config
        self.model = model
        self.retained_modes = int(retained_modes)
        self.completion_start_step = int(
            np.argmin(np.abs(config.times - completion_start_time))
        )
        self.completion_start_time = float(config.times[self.completion_start_step])
        self.full_basis = _canonical_basis(config)
        self.retained_basis = Eigenbasis(
            eigenvalues=self.full_basis.eigenvalues[:retained_modes].copy(),
            eigenvectors=self.full_basis.eigenvectors[:, :retained_modes].copy(),
        )
        self.omitted_eigenvectors = self.full_basis.eigenvectors[:, retained_modes:]
        self.modal_mean = discrete_expected_diffusion(config) @ self.full_basis.eigenvectors
        self.anchor_indices, self.anchor_weights = model.interpolation_weights(config)
        self.initial_coefficients = (
            np.tensordot(
                self.anchor_weights,
                model.initial_coefficients[self.anchor_indices],
                axes=(0, 0),
            )
            if initial_coefficients is None
            else np.asarray(initial_coefficients, dtype=float)
        )
        self.transition_coefficients = (
            np.tensordot(
                self.anchor_weights,
                model.transition_coefficients[self.anchor_indices],
                axes=(0, 0),
            )
            if transition_coefficients is None
            else np.asarray(transition_coefficients, dtype=float)
        )
        self._prototype = FittedLocalClosureModel(
            bandwidth=model.bandwidth,
            n_frequency_bins=model.n_frequency_bins,
            n_time_bins=model.n_time_bins,
            residual_grid_size=model.residual_grid_size,
            initial_coefficients=self.initial_coefficients,
            transition_coefficients=self.transition_coefficients,
            initial_residual_bank=np.empty((0, model.residual_grid_size)),
            transition_residual_banks=tuple(
                np.empty((0, model.residual_grid_size))
                for _ in range(model.n_time_bins)
            ),
        )
        self._from_grid = _interpolation_matrix(
            model.residual_grid_size,
            config.n_nodes - retained_modes,
        )
        selected = self.anchor_indices
        weights = self.anchor_weights
        square_root_weights = np.sqrt(weights)

        def combined_factor(factors: FloatArray) -> FloatArray:
            return np.concatenate(
                [
                    weight * factor
                    for weight, factor in zip(
                        square_root_weights, factors, strict=True
                    )
                ],
                axis=1,
            )

        self._initial_mean = np.tensordot(
            weights, model.initial_innovation_mean[selected], axes=(0, 0)
        )
        self._initial_factor = combined_factor(
            model.initial_innovation_factor[selected]
        )
        self._initial_diagonal = np.sqrt(
            np.tensordot(
                weights,
                model.initial_innovation_diagonal[selected] ** 2,
                axes=(0, 0),
            )
        )
        self._transition_means = np.tensordot(
            weights, model.transition_innovation_mean[selected], axes=(0, 0)
        )
        self._transition_previous_means = np.tensordot(
            weights, model.transition_previous_mean[selected], axes=(0, 0)
        )
        self._transition_gains = np.tensordot(
            weights, model.transition_innovation_gain[selected], axes=(0, 0)
        )
        self._transition_factors = tuple(
            combined_factor(model.transition_innovation_factor[selected, time_bin])
            for time_bin in range(model.n_time_bins)
        )
        self._transition_diagonals = np.sqrt(
            np.tensordot(
                weights,
                model.transition_innovation_diagonal[selected] ** 2,
                axes=(0, 0),
            )
        )

    def _sample_innovations(
        self,
        generator: np.random.Generator,
        *,
        time_bin: int | None,
        batch_size: int,
        previous: FloatArray | None = None,
    ) -> FloatArray:
        if time_bin is None:
            mean = self._initial_mean
            factor = self._initial_factor
            diagonal = self._initial_diagonal
        else:
            mean = self._transition_means[time_bin]
            factor = self._transition_factors[time_bin]
            diagonal = self._transition_diagonals[time_bin]
        innovation = np.broadcast_to(mean, (batch_size, mean.size)).copy()
        if time_bin is not None and self.model.persistent_innovations:
            if previous is None:
                raise ValueError("persistent innovations require a previous state")
            innovation += (
                previous - self._transition_previous_means[time_bin]
            ) @ self._transition_gains[time_bin].T
        innovation += (
            generator.standard_normal((batch_size, factor.shape[1])) @ factor.T
            + generator.standard_normal((batch_size, diagonal.size)) * diagonal
        )
        return innovation

    def complete_batch(
        self,
        raw_counts: FloatArray,
        rng: np.random.Generator | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        """Complete a run-by-time-by-node batch with one vectorized time loop."""

        values = np.asarray(raw_counts, dtype=float)
        expected = (self.config.n_steps, self.config.n_nodes)
        if values.ndim != 3 or values.shape[1:] != expected:
            raise ValueError("raw_counts batch shape does not match configuration")
        generator = np.random.default_rng() if rng is None else rng
        completed = values.copy()
        additions = np.zeros_like(values)
        retained_states = values @ self.retained_basis.eigenvectors
        scale = np.sqrt(self.config.n_particles)
        retained_deviation = (
            retained_states - self.modal_mean[None, :, : self.retained_modes]
        ) / scale
        start = self.completion_start_step
        n_nodes = self.config.n_nodes
        batch_size = values.shape[0]
        full_deviation = np.zeros((batch_size, n_nodes), dtype=float)
        full_deviation[:, : self.retained_modes] = retained_deviation[:, start]
        innovation = self._sample_innovations(
            generator,
            time_bin=None,
            batch_size=batch_size,
        )
        unresolved = self._prototype._predict_initial(
            full_deviation, self.retained_modes
        ) + innovation @ self._from_grid.T

        def store(time_index: int) -> None:
            modal_state = self.modal_mean[time_index, self.retained_modes :] + (
                scale * unresolved
            )
            addition = modal_state @ self.omitted_eigenvectors.T
            addition -= np.mean(addition, axis=1, keepdims=True)
            additions[:, time_index] = addition
            completed[:, time_index] = values[:, time_index] + addition

        store(start)
        for time_index in range(start, self.config.n_steps - 1):
            full_deviation[:, : self.retained_modes] = retained_deviation[:, time_index]
            full_deviation[:, self.retained_modes :] = unresolved
            unresolved = self._prototype._predict_transition(
                full_deviation,
                retained_deviation[:, time_index + 1],
                self.retained_modes,
            )
            fraction = (time_index - start) / max(self.config.n_steps - start - 2, 1)
            time_bin = min(
                self.model.n_time_bins - 1,
                int(fraction * self.model.n_time_bins),
            )
            innovation = self._sample_innovations(
                generator,
                time_bin=time_bin,
                batch_size=batch_size,
                previous=innovation,
            )
            unresolved += innovation @ self._from_grid.T
            store(time_index + 1)
        return completed, additions

    def complete(
        self,
        raw_counts: FloatArray,
        rng: np.random.Generator | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        values = np.asarray(raw_counts, dtype=float)
        if values.shape != (self.config.n_steps, self.config.n_nodes):
            raise ValueError("raw_counts shape does not match configuration")
        completed, additions = self.complete_batch(values[None, :, :], rng=rng)
        return completed[0], additions[0]


def complete_anchored_ensemble(
    runs: FloatArray,
    config: DiffusionConfig,
    model: AnchoredFittedClosureModel,
    *,
    retained_modes: int,
    completion_start_time: float,
    seed: int = 0,
    apply_simplex_bank: bool = True,
) -> FloatArray:
    """Apply one anchored completer to a stack of reduced trajectories."""

    values = np.asarray(runs, dtype=float)
    if values.ndim != 3 or values.shape[1:] != (config.n_steps, config.n_nodes):
        raise ValueError("runs shape does not match configuration")
    completer = AnchoredFittedLocalCompleter(
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
