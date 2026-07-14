"""Benchmark both random-walk implementations through the Python API."""

from eigendiffusion import DiffusionConfig, benchmark_random_walks

base_config = DiffusionConfig(n_particles=100, n_nodes=101, n_steps=101)
records = benchmark_random_walks(
    base_config,
    particle_counts=[100, 500, 1_000, 5_275],
    repeats=3,
    seed=0,
)

for record in records:
    print(
        record.method,
        record.n_particles,
        f"{record.median_seconds:.4f} s",
        f"{record.core_array_megabytes:.3f} MiB",
    )
