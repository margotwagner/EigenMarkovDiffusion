# Succinct project summary

EigenDiffusion investigates whether stochastic diffusion can be simulated in a
spectral basis rather than by tracking every particle on a spatial grid. A
reflecting-boundary diffusion operator is eigendecomposed into independent
spatial modes. Each nonconstant mode is represented as a two-state Markov
system whose positive and negative populations switch at a rate determined by
the mode's eigenvalue. The modes are then transformed back into physical space.

The current project is a proof of concept. Its strongest result is that the
ensemble mean of the EigenMarkov process can approximate deterministic spectral
diffusion and a spatial random-walk baseline. The principal unresolved issue is
that independently sampled signed modes can produce negative reconstructed
particle counts and have not yet been shown to reproduce the full spatial
covariance of the physical random walk.

The active code is intentionally small:

- `operators.py` defines the diffusion matrix and spatial/modal transforms;
- `eigenmarkov.py` implements the central stochastic modal model;
- `baselines.py` contains deterministic and random-walk references;
- `ensemble.py` runs reproducible repeated simulations;
- `metrics.py` measures mean error, mass conservation, and negative values;
- `cli.py` provides the two canonical experiments: validation and mode sweeps.

The deterministic calcium-calbindin reaction models from the older repository
are not part of the active package because they are not yet integrated with the
stochastic EigenMarkov formulation.
