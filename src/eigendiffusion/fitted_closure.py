"""Trajectory-fitted sparse persistent unresolved-mode closure.

The fitted model uses no analytic covariance at inference time.  Sparse local
regression coefficients and empirical innovation banks are learned from full
multinomial trajectories spanning several physical configurations.  Modal
deviations are normalized by ``sqrt(n_particles)`` and regression parameters
are shared in normalized-frequency bins so one frozen model can be applied to
unseen particle counts, grids, domains, impulses, and diffusivities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .config import DiffusionConfig
from .operators import Eigenbasis, eigendecompose, neumann_laplacian_1d
from .references import discrete_expected_diffusion
from .readouts import simplex_bank_readout

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class FittedClosureTrainingCase:
    """Full reference trajectories and handoff definition for one configuration."""

    config: DiffusionConfig
    runs: FloatArray
    retained_modes: int
    handoff_time: float


def _canonical_basis(config: DiffusionConfig) -> Eigenbasis:
    operator = neumann_laplacian_1d(
        config.n_nodes,
        config.length,
        config.diffusion_coefficient,
    )
    basis = eigendecompose(operator, n_modes=config.n_nodes)
    vectors = basis.eigenvectors.copy()
    pivot_rows = np.argmax(np.abs(vectors), axis=0)
    signs = np.sign(vectors[pivot_rows, np.arange(config.n_nodes)])
    signs[signs == 0.0] = 1.0
    vectors *= signs[None, :]
    return Eigenbasis(eigenvalues=basis.eigenvalues.copy(), eigenvectors=vectors)


def _interpolation_matrix(source_size: int, target_size: int) -> FloatArray:
    """Return a linear map from a normalized source grid to a target grid."""

    if source_size <= 0 or target_size <= 0:
        raise ValueError("interpolation sizes must be positive")
    if source_size == 1:
        return np.ones((target_size, 1), dtype=float)
    source = np.linspace(0.0, 1.0, source_size)
    target = np.linspace(0.0, 1.0, target_size)
    matrix = np.zeros((target_size, source_size), dtype=float)
    upper = np.searchsorted(source, target, side="right")
    upper = np.clip(upper, 1, source_size - 1)
    lower = upper - 1
    fraction = (target - source[lower]) / (source[upper] - source[lower])
    rows = np.arange(target_size)
    matrix[rows, lower] = 1.0 - fraction
    matrix[rows, upper] += fraction
    return matrix


@dataclass(frozen=True, slots=True)
class FittedLocalClosureModel:
    """Frozen cross-configuration sparse regression and innovation model."""

    bandwidth: int
    n_frequency_bins: int
    n_time_bins: int
    residual_grid_size: int
    initial_coefficients: FloatArray
    transition_coefficients: FloatArray
    initial_residual_bank: FloatArray
    transition_residual_banks: tuple[FloatArray, ...]

    @property
    def feature_width(self) -> int:
        return 2 * self.bandwidth + 1

    def frequency_bin(self, mode_index: int, n_nodes: int) -> int:
        fraction = mode_index / max(n_nodes - 1, 1)
        return min(self.n_frequency_bins - 1, int(fraction * self.n_frequency_bins))

    def save(self, output_path: str | Path) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {
            "bandwidth": np.asarray(self.bandwidth),
            "n_frequency_bins": np.asarray(self.n_frequency_bins),
            "n_time_bins": np.asarray(self.n_time_bins),
            "residual_grid_size": np.asarray(self.residual_grid_size),
            "initial_coefficients": self.initial_coefficients,
            "transition_coefficients": self.transition_coefficients,
            "initial_residual_bank": self.initial_residual_bank,
        }
        for index, bank in enumerate(self.transition_residual_banks):
            payload[f"transition_residual_bank_{index}"] = bank
        np.savez_compressed(output, **payload)
        return output

    @classmethod
    def load(cls, input_path: str | Path) -> FittedLocalClosureModel:
        with np.load(input_path) as payload:
            n_time_bins = int(payload["n_time_bins"])
            return cls(
                bandwidth=int(payload["bandwidth"]),
                n_frequency_bins=int(payload["n_frequency_bins"]),
                n_time_bins=n_time_bins,
                residual_grid_size=int(payload["residual_grid_size"]),
                initial_coefficients=payload["initial_coefficients"].copy(),
                transition_coefficients=payload["transition_coefficients"].copy(),
                initial_residual_bank=payload["initial_residual_bank"].copy(),
                transition_residual_banks=tuple(
                    payload[f"transition_residual_bank_{index}"].copy()
                    for index in range(n_time_bins)
                ),
            )

    @classmethod
    def fit(
        cls,
        cases: list[FittedClosureTrainingCase],
        *,
        bandwidth: int = 12,
        n_frequency_bins: int = 8,
        n_time_bins: int = 5,
        residual_grid_size: int = 32,
        ridge: float = 1.0e-3,
        max_residual_samples: int = 4_000,
        max_regression_samples_per_target: int = 2_000,
        seed: int = 0,
    ) -> FittedLocalClosureModel:
        """Fit shared local regressions and empirical innovation banks."""

        if not cases:
            raise ValueError("cases must not be empty")
        if bandwidth < 0 or n_frequency_bins <= 0 or n_time_bins <= 0:
            raise ValueError("bandwidth and bin counts must be valid")
        width = 2 * bandwidth + 1
        transition_width = 3 * width
        initial_xx = np.zeros((n_frequency_bins, width, width), dtype=float)
        initial_xy = np.zeros((n_frequency_bins, width), dtype=float)
        transition_xx = np.zeros(
            (n_frequency_bins, transition_width, transition_width),
            dtype=float,
        )
        transition_xy = np.zeros((n_frequency_bins, transition_width), dtype=float)

        rng = np.random.default_rng(seed)
        prepared: list[tuple[FittedClosureTrainingCase, FloatArray, FloatArray, int]] = []
        for case in cases:
            config = case.config
            runs = np.asarray(case.runs, dtype=float)
            if runs.ndim != 3 or runs.shape[1:] != (config.n_steps, config.n_nodes):
                raise ValueError("training runs do not match their configuration")
            r = int(case.retained_modes)
            if not 1 <= r < config.n_nodes:
                raise ValueError("training retained_modes must omit at least one mode")
            start = int(np.argmin(np.abs(config.times - case.handoff_time)))
            basis = _canonical_basis(config)
            modal = runs @ basis.eigenvectors
            modal_mean = discrete_expected_diffusion(config) @ basis.eigenvectors
            deviation = (modal - modal_mean[None, :, :]) / np.sqrt(config.n_particles)
            prepared.append((case, deviation, modal_mean, start))

            for target in range(r, config.n_nodes):
                frequency_bin = min(
                    n_frequency_bins - 1,
                    int(target / max(config.n_nodes - 1, 1) * n_frequency_bins),
                )
                initial_features = np.zeros((runs.shape[0], width), dtype=float)
                for offset in range(-bandwidth, bandwidth + 1):
                    source = target + offset
                    if 0 <= source < r:
                        initial_features[:, offset + bandwidth] = deviation[
                            :, start, source
                        ]
                initial_target = deviation[:, start, target]
                initial_xx[frequency_bin] += initial_features.T @ initial_features
                initial_xy[frequency_bin] += initial_features.T @ initial_target

                n_transitions = config.n_steps - start - 1
                if n_transitions <= 0:
                    continue
                transition_features = np.zeros(
                    (runs.shape[0], n_transitions, transition_width),
                    dtype=float,
                )
                for offset in range(-bandwidth, bandwidth + 1):
                    source = target + offset
                    position = offset + bandwidth
                    if r <= source < config.n_nodes:
                        transition_features[:, :, position] = deviation[
                            :, start:-1, source
                        ]
                    if 0 <= source < r:
                        transition_features[:, :, width + position] = deviation[
                            :, start:-1, source
                        ]
                        transition_features[:, :, 2 * width + position] = deviation[
                            :, start + 1 :, source
                        ]
                flattened = transition_features.reshape(-1, transition_width)
                target_values = deviation[:, start + 1 :, target].reshape(-1)
                if flattened.shape[0] > max_regression_samples_per_target:
                    selected = rng.choice(
                        flattened.shape[0],
                        size=max_regression_samples_per_target,
                        replace=False,
                    )
                    flattened = flattened[selected]
                    target_values = target_values[selected]
                transition_xx[frequency_bin] += flattened.T @ flattened
                transition_xy[frequency_bin] += flattened.T @ target_values

        def solve(normal: FloatArray, target: FloatArray) -> FloatArray:
            coefficients = np.zeros_like(target)
            for bin_index in range(normal.shape[0]):
                matrix = normal[bin_index]
                scale = float(np.trace(matrix) / max(matrix.shape[0], 1))
                if scale <= 1.0e-15:
                    continue
                coefficients[bin_index] = np.linalg.solve(
                    matrix + ridge * scale * np.eye(matrix.shape[0]),
                    target[bin_index],
                )
            return coefficients

        initial_coefficients = solve(initial_xx, initial_xy)
        transition_coefficients = solve(transition_xx, transition_xy)
        initial_residuals: list[FloatArray] = []
        transition_residuals: list[list[FloatArray]] = [
            [] for _ in range(n_time_bins)
        ]

        prototype = cls(
            bandwidth=bandwidth,
            n_frequency_bins=n_frequency_bins,
            n_time_bins=n_time_bins,
            residual_grid_size=residual_grid_size,
            initial_coefficients=initial_coefficients,
            transition_coefficients=transition_coefficients,
            initial_residual_bank=np.empty((0, residual_grid_size)),
            transition_residual_banks=tuple(
                np.empty((0, residual_grid_size)) for _ in range(n_time_bins)
            ),
        )
        for case, deviation, _, start in prepared:
            config = case.config
            r = case.retained_modes
            u = config.n_nodes - r
            to_grid = _interpolation_matrix(u, residual_grid_size)
            initial_prediction = prototype._predict_initial(deviation[:, start, :], r)
            initial_error = deviation[:, start, r:] - initial_prediction
            initial_residuals.append(initial_error @ to_grid.T)

            for time_index in range(start, config.n_steps - 1):
                prediction = prototype._predict_transition(
                    deviation[:, time_index, :],
                    deviation[:, time_index + 1, :r],
                    r,
                )
                error = deviation[:, time_index + 1, r:] - prediction
                fraction = (time_index - start) / max(config.n_steps - start - 2, 1)
                time_bin = min(n_time_bins - 1, int(fraction * n_time_bins))
                transition_residuals[time_bin].append(error @ to_grid.T)

        def capped(values: list[FloatArray]) -> FloatArray:
            combined = np.concatenate(values, axis=0)
            if combined.shape[0] <= max_residual_samples:
                return combined
            selected = rng.choice(
                combined.shape[0],
                size=max_residual_samples,
                replace=False,
            )
            return combined[selected]

        return cls(
            bandwidth=bandwidth,
            n_frequency_bins=n_frequency_bins,
            n_time_bins=n_time_bins,
            residual_grid_size=residual_grid_size,
            initial_coefficients=initial_coefficients,
            transition_coefficients=transition_coefficients,
            initial_residual_bank=capped(initial_residuals),
            transition_residual_banks=tuple(
                capped(values) for values in transition_residuals
            ),
        )

    def _predict_initial(self, deviations: FloatArray, retained_modes: int) -> FloatArray:
        values = np.asarray(deviations, dtype=float)
        n_nodes = values.shape[-1]
        width = self.feature_width
        result = np.zeros((values.shape[0], n_nodes - retained_modes), dtype=float)
        for output_index, target in enumerate(range(retained_modes, n_nodes)):
            features = np.zeros((values.shape[0], width), dtype=float)
            for offset in range(-self.bandwidth, self.bandwidth + 1):
                source = target + offset
                if 0 <= source < retained_modes:
                    features[:, offset + self.bandwidth] = values[:, source]
            frequency_bin = self.frequency_bin(target, n_nodes)
            result[:, output_index] = features @ self.initial_coefficients[frequency_bin]
        return result

    def _predict_transition(
        self,
        current_deviations: FloatArray,
        next_retained_deviations: FloatArray,
        retained_modes: int,
    ) -> FloatArray:
        current = np.asarray(current_deviations, dtype=float)
        next_retained = np.asarray(next_retained_deviations, dtype=float)
        n_nodes = current.shape[-1]
        width = self.feature_width
        result = np.zeros((current.shape[0], n_nodes - retained_modes), dtype=float)
        for output_index, target in enumerate(range(retained_modes, n_nodes)):
            features = np.zeros((current.shape[0], 3 * width), dtype=float)
            for offset in range(-self.bandwidth, self.bandwidth + 1):
                source = target + offset
                position = offset + self.bandwidth
                if retained_modes <= source < n_nodes:
                    features[:, position] = current[:, source]
                if 0 <= source < retained_modes:
                    features[:, width + position] = current[:, source]
                    features[:, 2 * width + position] = next_retained[:, source]
            frequency_bin = self.frequency_bin(target, n_nodes)
            result[:, output_index] = (
                features @ self.transition_coefficients[frequency_bin]
            )
        return result


class FittedLocalPersistentCompleter:
    """Apply one frozen fitted closure to a reduced spatial trajectory."""

    def __init__(
        self,
        config: DiffusionConfig,
        model: FittedLocalClosureModel,
        *,
        retained_modes: int,
        completion_start_time: float,
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
        deterministic = discrete_expected_diffusion(config)
        self.modal_mean = deterministic @ self.full_basis.eigenvectors

    def complete(
        self,
        raw_counts: FloatArray,
        rng: np.random.Generator | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        values = np.asarray(raw_counts, dtype=float)
        if values.shape != (self.config.n_steps, self.config.n_nodes):
            raise ValueError("raw_counts shape does not match configuration")
        generator = np.random.default_rng() if rng is None else rng
        completed = values.copy()
        additions = np.zeros_like(values)
        retained_states = values @ self.retained_basis.eigenvectors
        scale = np.sqrt(self.config.n_particles)
        retained_deviation = (
            retained_states - self.modal_mean[:, : self.retained_modes]
        ) / scale
        start = self.completion_start_step
        n_nodes = self.config.n_nodes
        full_deviation = np.zeros((1, n_nodes), dtype=float)
        full_deviation[0, : self.retained_modes] = retained_deviation[start]
        unresolved = self.model._predict_initial(
            full_deviation,
            self.retained_modes,
        )[0]
        from_grid = _interpolation_matrix(
            self.model.residual_grid_size,
            n_nodes - self.retained_modes,
        )
        initial_index = int(generator.integers(self.model.initial_residual_bank.shape[0]))
        unresolved += from_grid @ self.model.initial_residual_bank[initial_index]

        def store(time_index: int) -> None:
            modal_state = self.modal_mean[time_index, self.retained_modes :] + (
                scale * unresolved
            )
            addition = self.omitted_eigenvectors @ modal_state
            addition -= float(addition.sum()) / n_nodes
            additions[time_index] = addition
            completed[time_index] = values[time_index] + addition

        store(start)
        for time_index in range(start, self.config.n_steps - 1):
            full_deviation[0, : self.retained_modes] = retained_deviation[time_index]
            full_deviation[0, self.retained_modes :] = unresolved
            next_retained = retained_deviation[time_index + 1][None, :]
            unresolved = self.model._predict_transition(
                full_deviation,
                next_retained,
                self.retained_modes,
            )[0]
            fraction = (time_index - start) / max(self.config.n_steps - start - 2, 1)
            time_bin = min(
                self.model.n_time_bins - 1,
                int(fraction * self.model.n_time_bins),
            )
            bank = self.model.transition_residual_banks[time_bin]
            residual_index = int(generator.integers(bank.shape[0]))
            unresolved += from_grid @ bank[residual_index]
            store(time_index + 1)
        return completed, additions


def complete_fitted_ensemble(
    runs: FloatArray,
    config: DiffusionConfig,
    model: FittedLocalClosureModel,
    *,
    retained_modes: int,
    completion_start_time: float,
    seed: int = 0,
    apply_simplex_bank: bool = True,
) -> FloatArray:
    """Apply one fitted completer to a stack of reduced trajectories."""

    values = np.asarray(runs, dtype=float)
    if values.ndim != 3 or values.shape[1:] != (config.n_steps, config.n_nodes):
        raise ValueError("runs shape does not match configuration")
    completer = FittedLocalPersistentCompleter(
        config,
        model,
        retained_modes=retained_modes,
        completion_start_time=completion_start_time,
    )
    sequences = np.random.SeedSequence(seed).spawn(values.shape[0])
    outputs: list[FloatArray] = []
    for run, sequence in zip(values, sequences, strict=True):
        completed, _ = completer.complete(run, rng=np.random.default_rng(sequence))
        if apply_simplex_bank:
            completed = simplex_bank_readout(completed, config.n_particles).counts
        outputs.append(completed)
    return np.stack(outputs)
