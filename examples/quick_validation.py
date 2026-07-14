"""Run a small validation experiment without using the CLI."""

from eigendiffusion import (
    DiffusionConfig,
    deterministic_diffusion,
    run_eigenmarkov_ensemble,
    run_random_walk_ensemble,
    summarize,
)

config = DiffusionConfig(n_nodes=51, n_steps=51, n_particles=2_000)
reference = deterministic_diffusion(config)
naive = run_random_walk_ensemble(
    config,
    n_runs=10,
    seed=1,
    method="naive",
)
multinomial = run_random_walk_ensemble(
    config,
    n_runs=25,
    seed=2,
    method="multinomial",
)
eigenmarkov = run_eigenmarkov_ensemble(config, n_runs=25, seed=3)

print("Naive random walk:", summarize(reference, naive.mean))
print("Multinomial random walk:", summarize(reference, multinomial.mean))
print("EigenMarkov:", summarize(reference, eigenmarkov.mean))
