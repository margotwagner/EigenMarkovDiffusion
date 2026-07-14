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
from .metrics import ValidationMetrics, summarize

__all__ = [
    "DiffusionConfig",
    "EigenMarkovDiffusion",
    "EigenMarkovResult",
    "EnsembleResult",
    "RandomWalkBenchmark",
    "ValidationMetrics",
    "benchmark_random_walks",
    "deterministic_diffusion",
    "estimate_random_walk_core_bytes",
    "multinomial_random_walk_diffusion",
    "naive_random_walk_diffusion",
    "naive_random_walk_trajectories",
    "random_walk_diffusion",
    "run_eigenmarkov_ensemble",
    "run_multinomial_random_walk_ensemble",
    "run_naive_random_walk_ensemble",
    "run_random_walk_ensemble",
    "summarize",
    "trajectories_to_counts",
]

__version__ = "0.2.0"
