import numpy as np

from eigendiffusion.completion import (
    LocalPersistentUnresolvedGaussianCompleter,
    PersistentUnresolvedGaussianCompleter,
    SparseRowGain,
    SparseRowTopology,
    _GaussianCompletionBase,
    _modal_multinomial_covariance_block,
    _modal_multinomial_cross_covariance_block,
)
from eigendiffusion.config import DiffusionConfig
from eigendiffusion.ensemble import apply_readout_ensemble, run_modal_ensemble
from eigendiffusion.operators import eigendecompose, neumann_laplacian_1d
from eigendiffusion.references import (
    multinomial_node_probabilities,
    random_walk_transition_matrix,
)
from eigendiffusion.temporal import (
    mean_node_lag_correlation_from_runs,
    multinomial_mean_node_lag_correlation,
    valid_time_origins,
)


def persistent_config() -> DiffusionConfig:
    return DiffusionConfig(
        n_particles=1000,
        n_nodes=31,
        n_steps=40,
        impulse_index=15,
        dt=1.0,
        length=4.0,
        diffusion_coefficient=2.20e-4,
    )


def test_persistent_completion_is_full_rank_noop() -> None:
    config = persistent_config()
    raw = run_modal_ensemble(
        config,
        n_runs=2,
        model="correlated_modal",
        n_modes=config.n_nodes,
        seed=0,
    )
    completed = apply_readout_ensemble(
        raw,
        config,
        readout="persistent_unresolved_completion",
        retained_modes=config.n_nodes,
        completion_start_time=0.0,
        seed=1,
    )
    np.testing.assert_array_equal(completed.runs, raw.runs)


def test_persistent_completion_preserves_mass_and_pre_handoff_values() -> None:
    config = persistent_config()
    raw = run_modal_ensemble(
        config,
        n_runs=3,
        model="handoff_correlated_modal",
        n_modes=15,
        initial_n_modes=config.n_nodes,
        handoff_time=10.0,
        seed=2,
    )
    completed = apply_readout_ensemble(
        raw,
        config,
        readout="persistent_unresolved_completion",
        retained_modes=15,
        completion_start_time=10.0,
        completion_rank=5,
        seed=3,
    )
    np.testing.assert_array_equal(completed.runs[:, :10], raw.runs[:, :10])
    np.testing.assert_allclose(
        completed.runs.sum(axis=-1),
        config.n_particles,
        atol=1e-9,
    )


def test_persistent_completion_is_reproducible() -> None:
    config = persistent_config()
    raw = run_modal_ensemble(
        config,
        n_runs=2,
        model="handoff_correlated_modal",
        n_modes=15,
        initial_n_modes=config.n_nodes,
        handoff_time=10.0,
        seed=4,
    )
    first = apply_readout_ensemble(
        raw,
        config,
        readout="persistent_unresolved_completion",
        retained_modes=15,
        completion_start_time=10.0,
        completion_rank=5,
        seed=5,
    )
    second = apply_readout_ensemble(
        raw,
        config,
        readout="persistent_unresolved_completion",
        retained_modes=15,
        completion_start_time=10.0,
        completion_rank=5,
        seed=5,
    )
    np.testing.assert_array_equal(first.runs, second.runs)


def test_persistent_completion_simplex_bank_is_physical() -> None:
    config = persistent_config()
    raw = run_modal_ensemble(
        config,
        n_runs=3,
        model="handoff_correlated_modal",
        n_modes=15,
        initial_n_modes=config.n_nodes,
        handoff_time=10.0,
        seed=40,
    )
    constrained = apply_readout_ensemble(
        raw,
        config,
        readout="persistent_unresolved_completion_simplex_bank",
        retained_modes=15,
        completion_start_time=10.0,
        completion_rank=5,
        seed=41,
    )
    assert np.min(constrained.runs) >= 0.0
    np.testing.assert_allclose(
        constrained.runs.sum(axis=-1),
        config.n_particles,
        atol=1.0e-9,
    )
    assert constrained.auxiliary is not None
    assert "completion_l1_fraction" in constrained.auxiliary
    assert "constraint_residual_l1_fraction" in constrained.auxiliary


def test_persistent_completion_improves_one_step_temporal_correlation() -> None:
    config = persistent_config()
    raw = run_modal_ensemble(
        config,
        n_runs=120,
        model="handoff_correlated_modal",
        n_modes=15,
        initial_n_modes=config.n_nodes,
        handoff_time=10.0,
        seed=6,
    )
    independent = apply_readout_ensemble(
        raw,
        config,
        readout="unresolved_gaussian_completion",
        retained_modes=15,
        completion_start_time=10.0,
        completion_rank=5,
        seed=7,
    )
    persistent = apply_readout_ensemble(
        raw,
        config,
        readout="persistent_unresolved_completion",
        retained_modes=15,
        completion_start_time=10.0,
        completion_rank=5,
        seed=8,
    )
    origins = valid_time_origins(config.n_steps, 1, start_step=10)
    reference = multinomial_mean_node_lag_correlation(
        config,
        1,
        [config.impulse_index],
        start_indices=origins,
    )
    independent_value = mean_node_lag_correlation_from_runs(
        independent.runs,
        1,
        [config.impulse_index],
        start_indices=origins,
    )
    persistent_value = mean_node_lag_correlation_from_runs(
        persistent.runs,
        1,
        [config.impulse_index],
        start_indices=origins,
    )
    assert abs(persistent_value - reference) < abs(independent_value - reference)


def test_persistent_completion_rank_is_capped() -> None:
    config = persistent_config()
    completer = PersistentUnresolvedGaussianCompleter(
        config,
        retained_modes=30,
        completion_rank=100,
    )
    assert completer.completion_rank == 1


def test_persistent_completion_counts_only_matches_detailed_path() -> None:
    config = DiffusionConfig(
        n_particles=200,
        n_nodes=11,
        n_steps=8,
        impulse_index=5,
        dt=1.0,
        length=4.0,
        diffusion_coefficient=2.20e-4,
    )
    raw = run_modal_ensemble(
        config,
        n_runs=1,
        model="handoff_correlated_modal",
        n_modes=6,
        initial_n_modes=config.n_nodes,
        handoff_time=2.0,
        seed=17,
    ).runs[0]
    completer = PersistentUnresolvedGaussianCompleter(
        config,
        retained_modes=6,
        completion_start_time=2.0,
        completion_rank=2,
    )

    detailed, additions = completer.complete(raw, rng=np.random.default_rng(18))
    counts_only = completer.complete_counts(raw, rng=np.random.default_rng(18))
    retained = raw @ completer.retained_basis.eigenvectors
    supplied_retained = completer.complete_counts(
        raw,
        rng=np.random.default_rng(18),
        retained_states=retained,
    )

    np.testing.assert_allclose(counts_only, detailed, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        supplied_retained,
        detailed,
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(additions.sum(axis=1), 0.0, atol=1.0e-12)


def test_captured_variance_rank_is_minimal_and_reports_cap_failure() -> None:
    config = DiffusionConfig(
        n_particles=100,
        n_nodes=4,
        n_steps=2,
        impulse_index=1,
    )
    covariance = np.diag([9.0, 4.0, 1.0])
    adaptive = _GaussianCompletionBase(
        config,
        retained_modes=1,
        completion_rank=3,
        residual_variance_tolerance=0.2,
    )
    factor, diagnostics = adaptive._low_rank_psd_factor_with_diagnostics(
        covariance
    )
    assert factor.shape == (3, 2)
    assert diagnostics.selected_rank == 2
    assert np.isclose(diagnostics.captured_variance_fraction, 13.0 / 14.0)
    assert diagnostics.tolerance_met
    assert 9.0 / 14.0 < 0.8 <= 13.0 / 14.0

    capped = _GaussianCompletionBase(
        config,
        retained_modes=1,
        completion_rank=1,
        residual_variance_tolerance=0.2,
    )
    _, capped_diagnostics = capped._low_rank_psd_factor_with_diagnostics(
        covariance
    )
    assert capped_diagnostics.selected_rank == 1
    assert not capped_diagnostics.tolerance_met

    fixed = _GaussianCompletionBase(
        config,
        retained_modes=1,
        completion_rank=2,
    )
    fixed_factor, fixed_diagnostics = fixed._low_rank_psd_factor_with_diagnostics(
        covariance
    )
    np.testing.assert_allclose(
        fixed_factor @ fixed_factor.T,
        np.diag([9.0, 4.0, 0.0]),
        atol=1.0e-12,
    )
    assert fixed_diagnostics.selected_rank == 2
    assert fixed_diagnostics.tolerance_met


def test_sparse_row_gain_csr_products_match_dense_matrix() -> None:
    topology = SparseRowTopology(
        indptr=np.array([0, 2, 2, 3], dtype=np.int64),
        indices=np.array([0, 2, 1], dtype=np.int64),
        row_indices=np.array([0, 0, 2], dtype=np.int64),
        n_columns=3,
    )
    gain = SparseRowGain(
        topology=topology,
        values=np.array([2.0, -1.0, 3.0]),
    )
    dense = np.array(
        [
            [2.0, 0.0, -1.0],
            [0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
        ]
    )
    vector = np.array([1.0, 2.0, 4.0])
    matrix = np.column_stack([vector, 2.0 * vector])
    np.testing.assert_array_equal(gain.to_dense(), dense)
    np.testing.assert_allclose(gain.matvec(vector), dense @ vector)
    np.testing.assert_allclose(gain.matmat(matrix), dense @ matrix)
    np.testing.assert_allclose(gain.rmatvec(vector), dense.T @ vector)


def test_restricted_modal_covariance_blocks_match_dense_formulas() -> None:
    config = DiffusionConfig(
        n_particles=200,
        n_nodes=11,
        n_steps=8,
        impulse_index=5,
        dt=1.0,
        length=4.0,
        diffusion_coefficient=2.20e-4,
    )
    operator = neumann_laplacian_1d(
        config.n_nodes,
        config.length,
        config.diffusion_coefficient,
    )
    basis = eigendecompose(operator)
    vectors = basis.eigenvectors
    transition = random_walk_transition_matrix(config)
    probabilities = multinomial_node_probabilities(config)
    rows = np.array([1, 4, 7, 9], dtype=np.int64)
    columns = np.array([0, 3, 6, 10], dtype=np.int64)
    time_index = 3

    probability = probabilities[time_index]
    next_probability = probabilities[time_index + 1]
    spatial_covariance = config.n_particles * (
        np.diag(probability) - np.outer(probability, probability)
    )
    dense_modal = vectors.T @ spatial_covariance @ vectors
    restricted = _modal_multinomial_covariance_block(
        probability,
        vectors,
        config.n_particles,
        rows,
        columns,
    )
    np.testing.assert_allclose(
        restricted,
        dense_modal[np.ix_(rows, columns)],
        atol=1.0e-12,
        rtol=1.0e-12,
    )

    cross_spatial = config.n_particles * (
        probability[:, None] * transition
        - np.outer(probability, next_probability)
    )
    dense_cross_modal = vectors.T @ cross_spatial @ vectors
    restricted_cross = _modal_multinomial_cross_covariance_block(
        probability,
        next_probability,
        vectors,
        1.0 - config.dt * basis.eigenvalues,
        config.n_particles,
        rows,
        columns,
    )
    np.testing.assert_allclose(
        restricted_cross,
        dense_cross_modal[np.ix_(rows, columns)],
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_local_restricted_parameters_match_legacy_full_assembly() -> None:
    config = DiffusionConfig(
        n_particles=200,
        n_nodes=11,
        n_steps=8,
        impulse_index=5,
        dt=1.0,
        length=4.0,
        diffusion_coefficient=2.20e-4,
    )
    completer = LocalPersistentUnresolvedGaussianCompleter(
        config,
        retained_modes=6,
        completion_start_time=2.0,
        completion_rank=3,
        modal_bandwidth=2,
    )
    probabilities = multinomial_node_probabilities(config)
    transition = random_walk_transition_matrix(config)
    vectors = completer.full_basis.eigenvectors
    transition_vectors = transition @ vectors
    r = completer.retained_modes
    start = completer.completion_start_step

    def modal_covariance(probability: np.ndarray) -> np.ndarray:
        modal_probability = probability @ vectors
        covariance = config.n_particles * (
            vectors.T @ (probability[:, None] * vectors)
            - np.outer(modal_probability, modal_probability)
        )
        return 0.5 * (covariance + covariance.T)

    c_tt = modal_covariance(probabilities[start])
    expected_initial_gain = completer._solve_sparse_gain(
        c_tt[r:, :r],
        c_tt[:r, :r],
        completer._initial_columns(),
    )
    expected_initial_covariance = (
        c_tt[r:, r:]
        - expected_initial_gain.to_dense() @ c_tt[:r, r:]
    )
    initial = completer._initial_step
    assert initial is not None
    np.testing.assert_allclose(
        initial.conditional_gain.to_dense(),
        expected_initial_gain.to_dense(),
        atol=1.0e-11,
        rtol=1.0e-11,
    )
    expected_initial_factor = completer._low_rank_psd_factor(
        expected_initial_covariance
    )
    np.testing.assert_allclose(
        initial.conditional_factor @ initial.conditional_factor.T,
        expected_initial_factor @ expected_initial_factor.T,
        atol=1.0e-10,
        rtol=1.0e-10,
    )

    time_index = start
    p_t = probabilities[time_index]
    p_next = probabilities[time_index + 1]
    c_nn = modal_covariance(p_next)
    q_t = p_t @ vectors
    q_next = p_next @ vectors
    cross_modal = config.n_particles * (
        vectors.T @ (p_t[:, None] * transition_vectors)
        - np.outer(q_t, q_next)
    )
    conditioning_covariance = np.block(
        [
            [c_tt[r:, r:], c_tt[r:, :r], cross_modal[r:, :r]],
            [c_tt[:r, r:], c_tt[:r, :r], cross_modal[:r, :r]],
            [cross_modal[r:, :r].T, cross_modal[:r, :r].T, c_nn[:r, :r]],
        ]
    )
    conditioning_covariance = 0.5 * (
        conditioning_covariance + conditioning_covariance.T
    )
    target_conditioning_covariance = np.concatenate(
        [
            cross_modal[r:, r:].T,
            cross_modal[:r, r:].T,
            c_nn[r:, :r],
        ],
        axis=1,
    )
    expected_gain = completer._solve_sparse_gain(
        target_conditioning_covariance,
        conditioning_covariance,
        completer._transition_columns(),
    )
    expected_covariance = (
        c_nn[r:, r:]
        - expected_gain.to_dense() @ target_conditioning_covariance.T
    )
    transition_step = completer._transition_steps[time_index]
    assert transition_step is not None
    np.testing.assert_allclose(
        transition_step.conditional_gain.to_dense(),
        expected_gain.to_dense(),
        atol=1.0e-11,
        rtol=1.0e-11,
    )
    expected_factor = completer._low_rank_psd_factor(expected_covariance)
    np.testing.assert_allclose(
        transition_step.conditional_factor @ transition_step.conditional_factor.T,
        expected_factor @ expected_factor.T,
        atol=1.0e-10,
        rtol=1.0e-10,
    )


def test_local_transition_gains_share_packed_csr_topology() -> None:
    config = DiffusionConfig(
        n_particles=200,
        n_nodes=15,
        n_steps=10,
        impulse_index=7,
        dt=1.0,
        length=4.0,
        diffusion_coefficient=2.20e-4,
    )
    completer = LocalPersistentUnresolvedGaussianCompleter(
        config,
        retained_modes=7,
        completion_start_time=2.0,
        completion_rank=3,
        modal_bandwidth=2,
    )
    steps = [step for step in completer._transition_steps if step is not None]
    assert len(steps) > 1
    topology = steps[0].conditional_gain.topology
    assert all(step.conditional_gain.topology is topology for step in steps)
    assert all(
        step.conditional_gain.values.size == topology.indices.size for step in steps
    )
    padded_size = completer.omitted_modes * int(np.max(topology.counts))
    assert topology.indices.size < padded_size

    summary = completer.factor_diagnostics_summary()
    assert summary["factor_count"] == len(steps) + 1
    assert summary["selected_rank_min"] <= summary["selected_rank_median"]
    assert summary["selected_rank_median"] <= summary["selected_rank_max"]
    assert 0.0 <= summary["minimum_captured_variance_fraction"] <= 1.0
    assert summary["tolerance_met"] is True


def test_local_persistent_matches_dense_at_full_bandwidth() -> None:
    config = DiffusionConfig(
        n_particles=200,
        n_nodes=11,
        n_steps=8,
        impulse_index=5,
        dt=1.0,
        length=4.0,
        diffusion_coefficient=2.20e-4,
    )
    raw = run_modal_ensemble(
        config,
        n_runs=1,
        model="handoff_correlated_modal",
        n_modes=6,
        initial_n_modes=11,
        handoff_time=2.0,
        seed=50,
    ).runs[0]
    dense = PersistentUnresolvedGaussianCompleter(
        config,
        retained_modes=6,
        completion_start_time=2.0,
        completion_rank=3,
    )
    local = LocalPersistentUnresolvedGaussianCompleter(
        config,
        retained_modes=6,
        completion_start_time=2.0,
        completion_rank=3,
        modal_bandwidth=11,
    )
    dense_counts, _ = dense.complete(raw, rng=np.random.default_rng(51))
    local_counts, _ = local.complete(raw, rng=np.random.default_rng(51))
    np.testing.assert_allclose(local_counts, dense_counts, atol=1.0e-8)


def test_local_persistent_simplex_readout_is_physical() -> None:
    config = persistent_config()
    raw = run_modal_ensemble(
        config,
        n_runs=2,
        model="handoff_correlated_modal",
        n_modes=15,
        initial_n_modes=config.n_nodes,
        handoff_time=10.0,
        seed=52,
    )
    completed = apply_readout_ensemble(
        raw,
        config,
        readout="local_persistent_unresolved_completion_simplex_bank",
        retained_modes=15,
        completion_start_time=10.0,
        completion_rank=5,
        completion_bandwidth=4,
        seed=53,
    )
    assert np.min(completed.runs) >= 0.0
    np.testing.assert_allclose(completed.runs.sum(axis=-1), 1000.0, atol=1.0e-9)
