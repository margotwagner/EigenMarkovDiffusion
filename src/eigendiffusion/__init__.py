"""EigenDiffusion: stochastic diffusion in a Laplacian eigenbasis."""

from .baselines import (
    deterministic_diffusion,
    multinomial_random_walk_diffusion,
    naive_random_walk_diffusion,
    naive_random_walk_trajectories,
    random_walk_diffusion,
    trajectories_to_counts,
)
from .benchmarking import (
    RandomWalkBenchmark,
    benchmark_random_walks,
    estimate_random_walk_core_bytes,
)
from .config import DiffusionConfig
from .eigenmarkov import EigenMarkovDiffusion, EigenMarkovResult
from .ensemble import (
    EnsembleResult,
    run_eigenmarkov_ensemble,
    run_multinomial_random_walk_ensemble,
    run_naive_random_walk_ensemble,
    run_random_walk_ensemble,
)
from .metrics import (
    EnsembleDiagnostics,
    ValidationMetrics,
    empirical_covariance,
    relative_l2_error_from_time,
    summarize,
    summarize_ensemble,
)
from .references import (
    continuous_expected_diffusion,
    discrete_expected_diffusion,
    multinomial_marginal_covariance,
    multinomial_marginal_variance,
    multinomial_node_probabilities,
    random_walk_transition_matrix,
)

__all__ = [
    "DiffusionConfig",
    "EigenMarkovDiffusion",
    "EigenMarkovResult",
    "EnsembleDiagnostics",
    "EnsembleResult",
    "RandomWalkBenchmark",
    "ValidationMetrics",
    "benchmark_random_walks",
    "continuous_expected_diffusion",
    "deterministic_diffusion",
    "discrete_expected_diffusion",
    "empirical_covariance",
    "estimate_random_walk_core_bytes",
    "multinomial_marginal_covariance",
    "multinomial_marginal_variance",
    "multinomial_node_probabilities",
    "multinomial_random_walk_diffusion",
    "naive_random_walk_diffusion",
    "naive_random_walk_trajectories",
    "random_walk_diffusion",
    "random_walk_transition_matrix",
    "relative_l2_error_from_time",
    "run_eigenmarkov_ensemble",
    "run_multinomial_random_walk_ensemble",
    "run_naive_random_walk_ensemble",
    "run_random_walk_ensemble",
    "summarize",
    "summarize_ensemble",
    "trajectories_to_counts",
]

__version__ = "0.3.0"
