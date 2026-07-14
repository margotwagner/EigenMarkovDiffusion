from eigendiffusion import DiffusionConfig
from eigendiffusion.benchmarking import (
    benchmark_random_walks,
    estimate_random_walk_core_bytes,
)


def test_naive_core_storage_exceeds_multinomial_storage():
    config = DiffusionConfig(n_particles=100, n_nodes=11, n_steps=10)
    naive = estimate_random_walk_core_bytes(config, "naive")
    multinomial = estimate_random_walk_core_bytes(config, "multinomial")
    assert naive > multinomial


def test_benchmark_returns_both_methods_for_each_particle_count():
    config = DiffusionConfig(n_particles=10, n_nodes=7, n_steps=6)
    records = benchmark_random_walks(
        config,
        particle_counts=[10, 20],
        repeats=1,
        seed=3,
    )
    observed = {(record.n_particles, record.method) for record in records}
    assert observed == {
        (10, "naive"),
        (10, "multinomial"),
        (20, "naive"),
        (20, "multinomial"),
    }
    assert all(record.median_seconds >= 0.0 for record in records)
