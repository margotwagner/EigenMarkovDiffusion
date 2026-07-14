"""Minimal Python API example."""

import numpy as np

from eigendiffusion import DiffusionConfig, EigenMarkovDiffusion, deterministic_diffusion

config = DiffusionConfig(
    n_particles=5_275,
    n_nodes=101,
    n_steps=101,
    impulse_index=59,
)

model = EigenMarkovDiffusion(
    config,
    n_modes=101,
    modal_particle_weight=1.0,
)
result = model.run(rng=np.random.default_rng(42))
reference = deterministic_diffusion(config)

print("EigenMarkov shape:", result.spatial_counts.shape)
print("Deterministic shape:", reference.shape)
print("Minimum reconstructed count:", result.minimum_count)
print("Maximum mass error:", np.max(np.abs(result.spatial_counts.sum(axis=1) - config.n_particles)))
