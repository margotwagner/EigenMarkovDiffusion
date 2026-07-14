# Project status

EigenDiffusion is a research prototype for replacing a spatial particle random
walk with stochastic dynamics over diffusion eigenmodes. The current evidence
supports the central proof of concept: the ensemble mean of the modal process
can approximate deterministic and random-walk diffusion profiles.

The main unresolved scientific issue is the stochastic reconstruction. Signed,
independently sampled modes can generate negative spatial values and do not yet
have a demonstrated guarantee of reproducing the complete spatial covariance of
a physical particle process. The code therefore reports negative values rather
than clipping them.

This repository intentionally focuses on the smallest reproducible diffusion
study:

- the Neumann diffusion operator;
- deterministic spectral propagation;
- an original-style particle-by-particle random walk;
- an exact multinomial node-count random walk;
- the stochastic EigenMarkov model;
- ensemble validation, random-walk benchmarking, and mode sweeps;
- tests of conservation, operator invariants, and consistency of the two
  random-walk formulations.

The naive and multinomial random walks describe the same independent-particle
model at the node-count level. Their comparison isolates the computational gain
from aggregation. EigenMarkov should therefore be benchmarked against both, and
especially against the multinomial implementation as the stronger spatial
baseline.

The calcium-calbindin reaction-diffusion prototypes from the original repository
are not included because they are not yet integrated into the stochastic modal
model. They should return as a separate, tested module only after the diffusion
formulation is settled.
