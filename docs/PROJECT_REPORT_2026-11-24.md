# EigenDiffusion current project report

**Status as of November 24, 2026**

## Abstract

EigenDiffusion investigates whether stochastic diffusion can be simulated in a
Laplacian eigenbasis more efficiently than with an exact spatial count process,
while preserving the statistics and physical constraints that matter for a
particle system. The project has progressed from independent signed modal
populations to correlated Gaussian dynamics, reduced-model completion, and a
direct time-inhomogeneous Gaussian transition whose retained first and second
temporal moments are exact by construction.

The current prepared pipeline uses 75 retained modes, a rank-25 conditional
innovation completion, batched joint innovation sampling, full-field spatial
reconstruction, and a compiled residual-feedback simplex readout. At the tested
101- and 201-node operating points, its 500-trajectory mean, variance,
same-time covariance, and cross-time covariance errors are at approximately the
finite-ensemble error floor of an exact multinomial walk. Once prepared, the
pipeline is 5.28 times and 3.14 times as fast as the exact vectorized
multinomial comparator. Streaming construction reduced the 201-node setup peak
resident memory from 442.5 MiB to 81.0 MiB.

The central limitations have also become clearer. Construction remains
expensive because every transition still uses dense covariance assembly, an
innovation eigendecomposition, and two conditioning singular-value
decompositions. The speed advantage therefore appears only after setup is
amortized over approximately 53 trajectories at 101 nodes or 234 trajectories
at 201 nodes. The current output is continuous rather than integer-valued and
does not have the exact multinomial path law. Marginal Jensen--Shannon
divergence exposes a distribution-shape discrepancy even when first and second
moments and sliced Wasserstein distance are near their sampling floors.

## Research objective and success criteria

The project asks whether a stochastic diffusion process can be represented and
sampled efficiently in a spectral basis rather than by updating every particle
or every spatial transfer directly. A successful formulation should meet four
criteria:

1. **Stochastic accuracy.** Recover the expected field, marginal variance,
   same-time spatial covariance, and cross-time covariance of the exact count
   process.
2. **Physical output.** Preserve total mass and produce nonnegative spatial
   counts, with integer output when the application requires a literal particle
   interpretation.
3. **Computational advantage.** Beat an efficient exact aggregated count
   process, not merely a particle-by-particle implementation, after all relevant
   setup and output costs are included.
4. **Transfer and scaling.** Retain accuracy and favorable cost as the grid,
   domain, physical parameters, impulse location, and duration change.

The strongest current result satisfies the first criterion for tested first and
second moments, satisfies nonnegativity and mass conservation with continuous
outputs, and satisfies the prepared-trajectory performance criterion at 101 and
201 nodes. Exact distributional equivalence, one-off performance, and broader
scaling remain open.

## Exact reference process and evidence hierarchy

The physical reference is a one-dimensional reflecting nearest-neighbor random
walk. All particles start at one grid node. At each step, every particle moves
left, remains in place, or moves right with probabilities determined by the
diffusion coefficient, grid spacing, and timestep.

Three stochastic implementations provide an evidence hierarchy:

- The **naive particle walk** retains every particle identity and trajectory.
- The **aggregated multinomial walk** samples integer left/stay/right transfer
  counts at each source node. It has exactly the same node-count process in
  distribution as the naive walk but discards particle identities.
- The **vectorized multinomial walk** replaces each source-node multinomial with
  two conditional binomial draws. It retains the same exact transition law and
  is the primary performance comparator.

The deterministic Markov stencil provides the exact finite-step expected field
but no stochastic variance. Analytic multinomial formulas provide exact
same-time and cross-time second moments. An independent finite exact ensemble is
used to measure the nonzero Monte Carlo error floor at a fixed trial count.

This hierarchy prevents two misleading conclusions: matching the mean is not
enough to validate a stochastic model, and beating the naive particle walk is
not enough to establish a useful reduced-order speed advantage.

## Model families and their roles

| Family | Model or method | Defining idea | Current role |
| --- | --- | --- | --- |
| Exact reference | Naive particle walk | Update every particle identity | Validation reference only |
| Exact reference | Aggregated multinomial walk | Sample integer left/stay/right transfers at each node | Exact stochastic reference |
| Exact reference | Vectorized multinomial walk | Two conditional binomial draws reproduce the multinomial transfer law | Primary exact performance comparator |
| Deterministic reference | Discrete Markov expectation | Propagate the expected field with the finite-step transition matrix | Mean and computational-work reference |
| Original modal | `independent_modal` | Independent signed two-state populations for each eigenmode | Historical formulation and CLI default; rejected for stochastic fidelity |
| Correlated modal | `correlated_modal` | Generate a shared spatial Gaussian debit-credit fluctuation and project it into modes | Demonstrates that correlated noise repairs first and second moments |
| Physical correlated | `banked_correlated_modal` | Add a spatial simplex bank to correlated modal dynamics | Demonstrates physical output, but retains a full spatial ledger |
| Reduced correlated | `handoff_correlated_modal` | Start at high modal rank and discard fast modes after a selected time | Earlier reduction and handoff-selection studies |
| Earlier completion | Independent unresolved Gaussian completion | Sample unresolved modes from their same-time conditional law at each output time | Restores same-time moments but loses temporal persistence |
| Earlier completion | Dense persistent Gaussian completion | Condition the next unresolved state on its prior state and the retained transition | Trusted analytic reference for the older handoff pipeline |
| Structured completion | Local persistent completion | Replace dense gains with row-local sparse modal gains | Memory experiment; rejected on long-domain lag covariance |
| Adaptive completion | Energy-selected rank | Choose the smallest residual rank capturing at least 90% of conditional innovation variance | Accurate in the older pipeline but rejected by cost gates |
| Learned completion | Fitted, anchored, persistent-innovation, physics-conditioned, and target-conditioned closures | Replace analytic conditioning with fitted regressions and innovation banks | Rejected by transfer, temporal, or runtime gates; lockbox remains unopened |
| Current stochastic core | `direct_modal_gaussian` | Sample a direct time-inhomogeneous Gaussian transition in the eigenbasis | Promoted stochastic core |
| Current completion | `innovation_conditioned_modal_completion` | Condition unresolved transition innovations on retained innovations and add low-rank residual noise | Current 75-mode, rank-25 candidate |
| Physical readout | Simplex bank | Project onto the nonnegative fixed-mass simplex and carry the correction residual in time | Physical output operator |
| Optimized readout | Warm simplex bank | Reuse the previous active-set threshold while preserving the exact projection | Portable exact reference |
| Current readout | Compiled warm simplex bank | Fuse active-set certification, projection, correction, and residual updates in a guarded Numba kernel | Preferred prepared-trajectory readout |
| Approximate readout | Clip and renormalize | Clip negatives, globally rescale positive mass, and carry a residual | Rejected because marginal Jensen--Shannon divergence worsened |
| Integer readout | Temporal and neighbor Delta-Sigma | Apply integer-simplex quantization with temporal or spatial residual feedback | Experimental; integer and physical, but not the exact multinomial law |

### Independent modal dynamics

The original EigenMarkov formulation represents each nonconstant mode with
positive and negative two-state populations. Each signed unit flips with a
probability determined by the mode eigenvalue, so the expected modal update is
correct for the represented modal state. The default initialization rounds
nonzero modal amplitudes to the nearest integer state unit and can introduce a
small initial quantization error. Its central assumption is that modal
fluctuations are independent.
The exact transformed multinomial covariance is not diagonal, so this
assumption strongly overdisperses the physical variance and covariance.

At full rank, the independent model produced variance and covariance relative
errors of approximately 2.88 and 2.84 and negative values in roughly 40% of
reconstructed entries. It remains useful as the original hypothesis and as a
demonstration that correct modal means do not determine the stochastic field.
These figures come from the [earlier full-rank model comparison](PROJECT_REPORT_2026-11-20.md).

### Shared spatial Gaussian debit-credit dynamics

The correlated formulation samples a Gaussian approximation to the categorical
left/stay/right fluctuation at every source node. Each source fluctuation has
zero sum, and the destination contributions are accumulated into one shared
spatial debit-credit field before projection into the eigenbasis. One spatial
event therefore affects many modes and restores the missing cross-mode
correlation.

At full rank, correlated modal dynamics reached the same mean, variance, and
same-time covariance error scale as a finite exact ensemble. The model still
uses continuous Gaussian transfers. Negative reconstructed counts can occur,
and the spatial fluctuation must be generated during every trajectory.

### Banks, handoffs, and analytic completion

Simplex banking established that nonnegative, mass-conserving output can be
obtained without materially changing the recovered moments. Handoff studies
then evolved many modes through the impulse-dominated transient and discarded
fast modes later. Raw truncation preserved much of the mean but lost too much
variance and covariance.

Independent unresolved Gaussian completion repaired same-time statistics but
redrew the omitted field independently at each saved time. Persistent
completion added an unresolved latent state and repaired short-lag covariance.
Dense persistent completion became the trusted analytic reference for the older
handoff pipeline, although its setup and storage were expensive.

Local sparse gains reduced memory but failed a hardened long-domain
lag-covariance gate. Several learned successors reduced analytic setup work but
failed configuration transfer, temporal, or runtime gates. Exact-anchor
ablations showed that regression transfer, rather than innovation rank, was the
dominant learned-model error. The final target-conditioned candidate failed its
pre-lockbox cross-validation gate; the six-case lockbox remains unopened.

### Direct retained-modal Gaussian transition

The direct transition removes node-space Gaussian noise generation from the
trajectory loop. Exact multinomial modal covariances are prepared in advance,
and the retained state follows a diagonal linear Gaussian recurrence. Because
the reflecting transition and cosine basis share eigenvectors, the full direct
Gaussian process has the exact projected mean, same-time covariance, and
pairwise cross-time covariance.

Raw low-mode reconstruction is not an adequate physical model because omitted
modes retain substantial stochastic energy. The direct-transition screen
therefore promoted the fast retained dynamics as a core and used its runtime
margin to support innovation-conditioned completion.

### Innovation-conditioned completion

Each full modal innovation is partitioned into retained and unresolved blocks.
The observed retained innovation determines a conditional mean for the
unresolved innovation. Residual unresolved noise is sampled at a configurable
rank. A square-root null-space construction avoids subtracting nearly equal
covariance matrices at early rank-deficient transitions.

The conditional mean and deterministic unresolved initial state are not
truncated. Only residual conditional noise is rank-limited. At the current
201-node operating point, rank 25 captures 98.99% of residual variance on
average and reduces analytic full-covariance error to 0.0143.

## Current prepared pipeline

The promoted prepared physical pipeline is

```text
diffusion configuration
    -> analytic reflecting-walk cosine basis
    -> exact projected multinomial moment sequence
    -> direct retained Gaussian innovation factors
    -> conditional low-rank unresolved innovation factors
    -> packed joint factors for all transitions
    -> one batched innovation contraction per trajectory
    -> sequential diagonal modal recurrence
    -> full spatial reconstruction at every saved time
    -> compiled exact residual-feedback simplex readout
```

The current benchmarked operating point uses \(M=75\) retained modes and
conditional residual rank \(r=25\). At 201 nodes, the packed transition factor
therefore has 100 latent columns rather than 201. The completed recurrence still
propagates and reconstructs the full modal and spatial fields; the reduction is
primarily in stochastic latent dimension and prepared sampling work, not in the
size of the returned full-field trajectory.

This prepared path is not the earlier `correlated_modal` implementation that
generates a shared spatial Gaussian debit-credit fluctuation inside every
trajectory. It uses the exact walk's analytic first and second moments during
setup to prepare correlated modal innovation factors, then samples those factors
directly during each trajectory.

### Algorithmic changes that produced the current result

1. **Analytic cosine basis.** The reflecting path has a DCT-II eigenbasis, so
   the spatial operator does not require a general numerical eigendecomposition.
2. **Direct modal innovations.** Time-indexed modal covariance factors replace
   node-space Gaussian debit-credit sampling in every trajectory.
3. **Innovation rather than frame completion.** Conditioning transition
   innovations preserves temporal structure that independent output completion
   cannot recover.
4. **Square-root null-space conditioning.** The retained-row SVD separates
   observed and unobserved latent directions without an unstable explicit
   Schur-complement subtraction.
5. **Residual-only rank truncation.** Deterministic unresolved content and the
   conditional mean remain intact; only residual conditional noise is truncated.
6. **Joint factor packing.** Retained noise, its conditional unresolved mean,
   and residual unresolved noise are combined into one factor per transition.
7. **Batched sampling.** All Gaussian transition innovations are generated with
   one batched contraction before the sequential diagonal recurrence.
8. **Compact runtime state.** Diagnostic probability, covariance, gain, and
   factor histories are released; only the initial state, transition diagonal,
   basis, factor layout, and packed factors remain.
9. **Streaming construction.** One transition is constructed and packed at a
   time, eliminating the full-history setup peak without changing the returned
   runtime object.
10. **Compiled exact physical readout.** A guarded Numba kernel fuses warm
    active-set updates, KKT certification, mass correction, and residual
    feedback. A complete NumPy replay is used as a transactional fallback.
11. **Staged promotion gates.** Moment, temporal, distribution, runtime,
    storage, RSS, and held-out transfer gates prevented faster but weaker
    approximations from becoming defaults.

## Quantitative accuracy

### Accuracy protocol

The canonical accuracy experiment used 5,275 particles, domain length 4,
diffusion coefficient \(2.2\times10^{-4}\), physical duration 100, an impulse at
approximately \(0.59L\), 75 retained modes, and conditional rank 25. The
101-node grid saved 101 states with \(\Delta t=1\); the stable 201-node grid
saved 233 states with \(\Delta t\approx0.431034\).

The study generated 500 modal trajectories and an independent 500-trajectory
exact ensemble. A second independent 500-run exact ensemble, sampled at the
selected distribution times, supplied the exact-vs-exact distribution floor.
Same-time covariance was evaluated at physical times 1, 5, 20, 50, and 100.
Cross-time covariance used physical lags 1, 5, and 20 and five time origins.
Distribution diagnostics used 12 scalar marginals formed from three times and
four spatial positions, 30 common histogram bins, and 32 fixed zero-sum
sliced-Wasserstein directions.

The accuracy ensemble used the warm exact simplex operator. The compiled
operator agrees with it to approximately \(10^{-11}\) counts on the benchmarked
trajectories. Joint batched sampling reproduces the stepwise Gaussian law to
roundoff, and the compact streaming constructor matches the full-history
prepared factors and sampled paths exactly in the constructor diagnostic.
These results therefore apply to the current prepared implementation within the
reported numerical tolerances.

### Analytic effect of conditional rank

This comparison isolates model-factor error from finite-ensemble sampling
error for the 201-node, 75-retained-mode candidate.

| Conditional rank | Mean captured residual variance | Full covariance error | Worst-time covariance error | Marginal variance error | Lag-1 covariance error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 0.9700 | 0.0382 | 0.1109 | 0.0251 | 0.0169 |
| **25** | **0.9899** | **0.0143** | **0.0483** | **0.00938** | **0.00639** |
| 30 | 0.9967 | 0.00476 | 0.0183 | 0.00306 | 0.00217 |
| 126, full unresolved rank | 1.0000 | \(1.74\times10^{-7}\) | \(7.66\times10^{-7}\) | \(6.38\times10^{-8}\) | \(8.57\times10^{-8}\) |

Sources: [initial completion screen](../outputs/innovation_completion_screen/analytic_accuracy.csv)
and [rank refinement](../outputs/innovation_completion_rank_refinement/analytic_accuracy.csv).

At rank 25, the minimum, median, and mean captured residual variance fractions
over transitions were 0.8316, 0.9970, and 0.9899. The analytic mean-sequence
relative error was approximately \(1.6\times10^{-14}\) before physical readout
because the rank cap does not truncate the conditional mean or unresolved
deterministic initial condition.

### Ensemble moment and temporal accuracy

Lower values are better for all relative norm errors below. The exact-walk row
is the error obtained from a finite 500-trajectory exact ensemble rather than a
zero-error analytic object. It is therefore the relevant Monte Carlo sampling
floor for interpreting the modal result.

| Nodes | Ensemble | Mean error | Variance error | Same-time covariance error | Lag-1 error | Lag-5 error | Lag-20 error |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 101 | Current modal pipeline | 0.00171 | 0.0642 | 0.1193 | 0.1131 | 0.1995 | 0.4126 |
| 101 | Exact-walk sampling floor | 0.00154 | 0.0636 | 0.1204 | 0.1180 | 0.1922 | 0.4091 |
| 201 | Current modal pipeline | 0.00244 | 0.0638 | 0.1444 | 0.2301 | 0.4093 | 0.7804 |
| 201 | Exact-walk sampling floor | 0.00248 | 0.0622 | 0.1462 | 0.2321 | 0.4390 | 0.7493 |

Source: [500-trajectory accuracy results](../outputs/clip_renormalize_benchmark/accuracy.csv).

At 500 trajectories, current modal mean, variance, same-time covariance, and
tested lag-covariance errors are approximately at the exact finite-ensemble
floor. The large long-lag numbers are not uniquely modal error: the exact
500-run ensemble has almost the same estimation error because long-lag spatial
covariance is difficult to estimate at this trial count.

### Distribution and physical-output accuracy

The exact-vs-exact row compares two independent exact ensembles and therefore
measures the nonzero distribution-distance floor at 500 trajectories.

| Nodes | Ensemble | Normalized marginal \(W_1\) | Marginal JS, nats | Normalized sliced \(W_1\) | Maximum mass error | Negative fraction |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 101 | Current modal pipeline | 0.0731 | 0.0662 | 0.0774 | \(9.1\times10^{-13}\) | 0 |
| 101 | Exact-vs-exact floor | 0.0436 | 0.00711 | 0.0722 | 0 | 0 |
| 201 | Current modal pipeline | 0.0721 | 0.0670 | 0.0793 | \(9.1\times10^{-13}\) | 0 |
| 201 | Exact-vs-exact floor | 0.0342 | 0.00609 | 0.0819 | 0 | 0 |

Source: [500-trajectory accuracy results](../outputs/clip_renormalize_benchmark/accuracy.csv).

The normalized sliced Wasserstein score is within 7% of the observed
exact-vs-exact floor at both grids. The scalar marginal Wasserstein score is
1.68 and 2.11 times its floor at 101 and 201 nodes. Marginal Jensen--Shannon
divergence is 9.3 and 11.0 times its floor, which exposes a clearer
shape/support-sensitive discrepancy without isolating those two causes. The
current formulation therefore captures the tested first and second moments and
broad projected multivariate geometry, but not the exact integer multinomial
law.

## Performance and total cost

### Performance protocol

The canonical performance study used the same physical configuration,
75 retained modes, rank 25, and a full spatial field at every saved time. It ran
with one numerical thread under Python 3.11.2, NumPy 1.24.2, and Numba 0.56.4.
Cold setup used four balanced repeats. Prepared trajectory measurements used 30
balanced repeats, with setup and compiled-readout startup excluded. The exact
comparator was the vectorized multinomial walk, including production of the same
full-field trajectory.

### Setup and prepared simulation time

| Nodes | Streaming setup, s | Completion and full reconstruction, ms/trajectory | Completed field plus compiled readout, ms/trajectory | Exact vectorized walk, ms/trajectory | Prepared speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 101 | 0.2933 | 1.0217 | 1.2979 | 6.8507 | 5.28x |
| 201 | 2.8397 | 4.9280 | 5.6898 | 17.8705 | 3.14x |

Source: [streaming constructor benchmark](../outputs/streaming_constructor_benchmark/summary.json).

The incremental compiled-readout contribution inferred from the difference of
medians is approximately 0.276 ms at 101 nodes and 0.762 ms at 201 nodes. This
difference is not an independently timed component. The separate compiled
[readout study](COMPILED_SIMPLEX_READOUT.md) measured readout-only medians of
0.182 and 0.682 ms on cached trajectories.

The exact comparator has no separately amortized setup in this protocol;
transition-array construction and output allocation are included in every
measured exact trajectory.

Prepared modal setup is reusable only for trajectories with the same grid,
domain, timestep and duration, diffusion coefficient, particle count, impulse
location, retained-mode count, conditional rank, and numerical/readout
configuration. Random seeds and sampled trajectories may change without
repeating setup.

### Setup plus simulation time

For \(K\) independent trajectories on one prepared configuration,

\[
T_{\mathrm{modal}}(K)=t_{\mathrm{setup}}+K t_{\mathrm{modal}},\qquad
T_{\mathrm{exact}}(K)=K t_{\mathrm{exact}}.
\]

The following derived totals add the median streamed setup to \(K\) times the
median trajectory cost; they are not direct timings of \(K\)-trajectory batches.
The startup column adds the separately measured 0.3719-second compiled-readout
startup observed once per process in the canonical run.

| Nodes | Trajectories \(K\) | Modal setup plus simulations, s | Modal plus measured readout startup, s | Exact simulations, s |
| ---: | ---: | ---: | ---: | ---: |
| 101 | 1 | 0.2946 | 0.6665 | 0.0069 |
| 101 | 100 | 0.4231 | 0.7950 | 0.6851 |
| 101 | 250 | 0.6178 | 0.9897 | 1.7127 |
| 101 | 1,000 | 1.5913 | 1.9631 | 6.8507 |
| 201 | 1 | 2.8454 | 3.2173 | 0.0179 |
| 201 | 100 | 3.4087 | 3.7806 | 1.7870 |
| 201 | 250 | 4.2622 | 4.6341 | 4.4676 |
| 201 | 1,000 | 8.5295 | 8.9014 | 17.8705 |

Derived from the [streaming constructor benchmark](../outputs/streaming_constructor_benchmark/summary.json).

Ignoring compiled-readout startup, the modal total becomes lower after
approximately 53 trajectories at 101 nodes and 234 trajectories at 201 nodes.
Including the measured 0.3719-second startup moves those thresholds to 120 and
264 trajectories. A fresh isolated empty-cache run was measured separately at
about 2.59 seconds in the [compiled-readout study](COMPILED_SIMPLEX_READOUT.md).
The 0.3719-second value is the startup observed in the canonical benchmark in
an environment with disk caching enabled; that artifact does not separately
identify how much code was loaded from cache.

### Setup and trajectory memory

Memory measures must remain distinct. Retained array storage is the unique
reachable NumPy backing storage held by a prepared runtime. Resident set size
(RSS) includes the Python interpreter and loaded libraries. Peak RSS is a
process high-water measurement during construction and must not be added to
retained arrays.

| Nodes | Retained prepared arrays, MiB | Baseline RSS, MiB | RSS after setup, MiB | Peak RSS through setup, MiB | Incremental setup peak, MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 101 | 7.79 | 37.49 | 49.51 | 49.61 | 12.12 |
| 201 | 35.89 | 37.48 | 78.43 | 80.97 | 43.49 |

Source: [streaming constructor resources](../outputs/streaming_constructor_benchmark/resources.csv).

Streaming and full-history construction retain the same prepared arrays.
Streaming changes construction lifetime: it reduced absolute peak RSS from
93.11 to 49.61 MiB at 101 nodes and from 442.52 to 80.97 MiB at 201 nodes.

One full \(T\times J\) float64 or int64 output field, where \(T\) is the number
of saved time points, has an exact logical size. The following table separates
that storage from process RSS and explicitly marks the two missing high-water
measurements.

| Memory quantity | 101 nodes, MiB | 201 nodes, MiB | Evidence type |
| --- | ---: | ---: | --- |
| Setup peak process RSS | 49.61 | 80.97 | Measured in isolated setup process |
| Steady prepared-array storage | 7.79 | 35.89 | Measured unique reachable arrays |
| Simulation: one returned physical field | 0.0778 | 0.3573 | Exact logical array size |
| Setup plus one returned field | 7.8644 | 36.2493 | Derived logical retained-array sum |
| Simulation-phase process high-water RSS | Not measured | Not measured | Requires phase-sampled run |
| Combined setup-plus-simulation process peak RSS | Not measured | Not measured | Requires one-process benchmark |

The logical sizes are derived from the prepared-array measurements and the
full-field shapes recorded by the [streaming constructor benchmark](../outputs/streaming_constructor_benchmark/summary.json).
If an application deliberately retains modal and raw-spatial trajectory fields
as well as the physical output, the two intermediate fields occupy 0.1557 and
0.7146 MiB, and the setup-plus-three-field logical totals are 8.0201 and
36.9639 MiB. These are retention scenarios, not measured simultaneous live
sets for the timed counts-only path.

Additional innovation, modal, spatial-reconstruction, and readout temporaries
are allocated during different execution phases. The streaming benchmark did
not isolate trajectory-phase peak RSS or measure setup and trajectory phases in
one process. Its setup-only high-water therefore cannot be converted into a
combined process peak; a phase-sampled measurement is required.

A separate compatible full-field memory run measured the exact vectorized
comparator's returned output at 0.0778 and 0.3573 MiB, transient core arrays at
0.0809 and 0.3634 MiB, and incremental trajectory RSS at 0.453 and 0.762 MiB for
101 and 201 nodes. These exact-path memory values came from a separate process
[benchmark](../outputs/component_benchmark/optimized_101_201_memory.csv) and
are not mixed into the modal setup RSS table.

## Properties captured and not captured

| Property | Exact aggregated random walk | Current prepared modal pipeline |
| --- | --- | --- |
| Expected spatial field | Exact | Exact before physical readout; empirically at the sampling floor after readout |
| Total mass | Exactly \(N\) | Exactly \(N\); constant mode and simplex correction |
| Nonnegative output | Yes | Yes after compiled simplex readout |
| Integer output | Yes | No; output is generally continuous |
| Same-time covariance | Exact | Exact for the full direct Gaussian process; approximate at rank 25 and after readout, empirically at the tested sampling floor |
| Pairwise cross-time covariance | Exact | Exact for the full direct Gaussian process; approximate at rank 25 and after readout, empirically at the tested sampling floor |
| One-time marginal law | Multinomial | Gaussian-derived and then nonlinearly projected |
| One-step conditional law | State-dependent integer multinomial transfers | Time-indexed Gaussian innovations derived from analytic moments |
| Higher moments | Exact multinomial skewness, kurtosis, and tails | Not constructed to match higher moments |
| Empty-node and threshold probabilities | Exact | Not guaranteed |
| Complete multi-time path law | Exact count-valued Markov law | Linear Gaussian modal path plus residual-feedback readout |
| Particle identities | Not retained by the aggregated reference | Not retained |
| Full spatial output | Integer field at every saved time | Continuous physical field at every saved time |
| Reusable setup | Negligible | Expensive but reusable for an identical configuration |
| Prepared trajectory time | Performance comparator | Faster at both tested grids |
| One-off total time | Faster | Slower because setup dominates |
| Exact distributional equivalence | Yes | No |

Exact distributional equivalence would require the same probability for every
possible integer trajectory, not only the same mean and covariance. For
example, a binomial variable and a Gaussian variable can have the same mean and
variance while assigning completely different probabilities to zero counts,
integer values, and tail events. Simplex projection repairs nonnegativity and
mass but does not turn a Gaussian law into a multinomial law.

An exact integer implementation must sample source-node multinomial transfers,
or their conditional-binomial equivalent, from the realized integer spatial
state. Transforming that exact state into modal coordinates afterward is a
lossless change of basis at full rank, but it does not avoid the spatial
stochastic update. With fewer than all modes, different integer spatial states
can share the same retained coordinates while having different next-step
transfer laws; a reduced modal state is therefore not sufficient for an exact
Markov closure in general.

## Scientific and engineering conclusions

1. Correct mean propagation is solved and is no longer a sufficient model
   criterion.
2. Independent modal fluctuations are scientifically inadequate for the exact
   particle-count process.
3. Shared physical noise and direct correlated innovations recover the missing
   first and second moment structure.
4. Unresolved stochastic energy, rather than only unresolved mean content,
   determines the rank needed after modal reduction.
5. Conditioning innovations preserves temporal statistics more effectively
   than independently completing output frames.
6. The current rank-25 candidate is a strong moment model: analytic covariance
   truncation is small, and 500-run moment and temporal errors are at the exact
   sampling floor.
7. Physical simplex banking can preserve those moments while enforcing
   nonnegativity and mass conservation, but it does not provide integer output
   or the exact multinomial law.
8. Compiling the exact readout and batching transition innovations established
   a clear prepared-trajectory crossover against an already efficient exact
   aggregated comparator.
9. Compact state and streamed construction solved the prototype's retained and
   transient-memory duplication, but not the dense factorization cost or packed
   factor scaling.
10. The current benefit is an ensemble benefit: it is compelling when many
    trajectories reuse one physical setup and unfavorable for one-off runs.

## Current limitations and next experiments

### Dense setup factorizations

At 201 nodes, streamed setup still takes 2.84 seconds. A one-thread profile
attributes roughly half of construction to two conditioning SVDs and about one
third to the full innovation eigendecomposition. Construction remains dense at
every saved transition even though only 25 residual directions are retained.

The first implementation experiment should expose the innovation covariance as
a matrix-free operator and compute only the leading spectrum required by an
energy tolerance. A block Lanczos or randomized subspace method can be
warm-started from the preceding transition. It should be validated against the
current dense factors using captured energy, Ritz residuals, factor Gram error,
determinism, and propagated moment error before any speed claim is made. In
particular, the approximation must preserve the retained covariance block,
conditional mean, and pseudoinverse null space to their declared tolerances;
capturing total innovation energy alone is not sufficient.

The retained-row conditioning should be profiled separately. Structured QR,
smaller eigenproblems, reused subspaces, and rank-revealing updates are possible
alternatives to repeated full SVDs. Per-transition adaptive ranks and ragged or
bucketed factor storage should then convert lower numerical rank into lower
runtime storage rather than padding every transition to the maximum width.

### Distribution shape and integer output

The current model matches tested moments but not the exact marginal shape. The
next accuracy experiment should separate four sources of discrepancy:

1. full conditional rank versus rank 25;
2. raw Gaussian field versus exact simplex readout;
3. continuous simplex output versus integer Delta-Sigma output; and
4. Gaussian innovation law versus an integer or non-Gaussian transition.

All variants should use the same exact ensembles, frozen histogram edges,
projections, and seeds. An absolute distribution gate should accompany the
existing readout-noninferiority gates. If integer support is the dominant issue,
randomized dependent rounding on the integer simplex may preserve conditional
means better than deterministic rounding. If Gaussianity remains the dominant
issue at full rank, a higher-moment or non-Gaussian innovation model is needed.
Neither approach can claim an exact law without sampling the realized
state-dependent integer transfers.

### Scaling and transfer

The newest fast pipeline has primarily been tested on one length-4 physical
configuration at 101 and 201 nodes. Its formulas are general within the current
linear one-dimensional setting, but rank-25 accuracy, setup cost, and crossover
cannot be assumed to transfer.

The next staged study should first run analytic factor and timing checks at 401
and 801 nodes in both fixed-step and fixed-duration regimes. It should then
repeat the shifted-domain and shifted-physics cases that exposed earlier closure
failures, especially the length-6 temporal case. Fresh 250-trajectory ensembles
should be reserved for candidates that pass analytic covariance, setup,
storage, and prepared-runtime gates.

### Full-field output cost

The current benchmark reconstructs every node at every saved time. This is a
fair full-field comparison, but it imposes an output-size lower bound and means
the completed recurrence is not a small-state solver. At finer fixed-duration
grids, stability requires more time steps while each output field also grows.

Applications that need only selected nodes, linear observables, terminal fields,
or summary statistics should be benchmarked separately. Computing those
quantities directly from retained modes may reveal a larger and more scalable
advantage than full-field materialization.

### Dependence on analytic reference moments

The current setup is an analytic Gaussian oracle: it uses exact multinomial
moments available for independent linear diffusion. A broader closure for
heterogeneous, reactive, nonlinear, or higher-dimensional systems may not have
that oracle.

If learned completion remains a project goal, the next architecture should be
materially different from the rejected local linear regressions. Candidates
include nonlinear conditional operators, structured covariance operators, or
distillation from the analytic teacher. Whole-configuration cross-validation
must pass before the frozen lockbox is opened.

### Reproducibility and product integration

The research API now promotes the prepared direct pipeline, while the general
CLI still defaults to the historical independent model. A paper-ready release
should expose the current pipeline through one supported configuration and CLI
workflow, name the exact comparator consistently, and distinguish historical,
reference, promoted, and rejected methods in the main documentation.

A combined phase-sampled memory benchmark should measure setup, steady prepared
state, trajectory high-water, and returned-output memory in one isolated process.
Continuous integration should run the unit suite, linting, and a small numerical
smoke benchmark. Canonical experiment manifests should record code revision,
environment, thread limits, and artifact hashes.

## Recommended order of work

1. Add per-transition setup profiles and implement a validated matrix-free
   partial innovation factorization.
2. Re-run the small construction and prepared-runtime gate before any large
   stochastic ensemble.
3. Add adaptive per-transition ranks and compact variable-width storage if the
   partial solver changes the rank profile.
4. Run the full-rank/rank-25 and raw/simplex/integer distribution attribution
   experiment with one predeclared absolute distribution gate.
5. Extend analytic cost and rank scaling to 401 and 801 nodes, followed by the
   length-6 and shifted-physics confirmations.
6. Add a combined memory high-water benchmark and integrate the promoted path
   into the main CLI.
7. Pursue a new learned or non-Gaussian closure only after its intended use case
   and validation gate are specified.

## Reproducibility status

The package version is 0.12.0 and requires Python 3.10 or newer. The canonical
November 23 benchmarks used Python 3.11.2, NumPy 1.24.2, Numba 0.56.4, Linux,
and one thread for OpenBLAS, OpenMP, MKL, NumExpr, and Numba. Generated CSV and
JSON artifacts are under `outputs/`; study protocols and result summaries are
tracked under `docs/`.

The current test run completed with 236 passing tests and one llvmlite
deprecation warning. Ruff is declared as a development dependency but was not
installed in the environment used for this report check.

Primary reproducibility commands are:

```bash
pytest
ruff check .

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=src \
python3 -m eigendiffusion.streaming_constructor_benchmark \
  --output-dir outputs/streaming_constructor_benchmark
```

The accuracy and performance claims in this report are supported by:

- [`outputs/clip_renormalize_benchmark/accuracy.csv`](../outputs/clip_renormalize_benchmark/accuracy.csv)
- [`outputs/innovation_completion_screen/analytic_accuracy.csv`](../outputs/innovation_completion_screen/analytic_accuracy.csv)
- [`outputs/innovation_completion_rank_refinement/analytic_accuracy.csv`](../outputs/innovation_completion_rank_refinement/analytic_accuracy.csv)
- [`outputs/streaming_constructor_benchmark/summary.json`](../outputs/streaming_constructor_benchmark/summary.json)
- [`outputs/component_benchmark/optimized_101_201_memory.csv`](../outputs/component_benchmark/optimized_101_201_memory.csv)
- [Direct retained-modal transition](DIRECT_MODAL_TRANSITION.md)
- [Compiled exact simplex readout](COMPILED_SIMPLEX_READOUT.md)
- [Joint batched innovation sampling](JOINT_INNOVATION_BENCHMARK.md)
- [Compact prepared runtime](COMPACT_INNOVATION_RUNTIME.md)
- [Streaming compact constructor](STREAMING_INNOVATION_CONSTRUCTOR.md)

## Appendix A: notation

| Symbol | Definition |
| --- | --- |
| \(J\) | Number of spatial nodes |
| \(N\) | Number of particles and conserved total mass |
| \(T\) | Number of saved time points |
| \(R\) | Number of independent simulated trajectories in an ensemble |
| \(L\) | Physical domain length |
| \(\Delta x=L/(J-1)\) | Grid spacing |
| \(D_{\mathrm{phys}}\) | Physical diffusion coefficient |
| \(\Delta t\) | Simulation timestep |
| \(q=D_{\mathrm{phys}}\Delta t/\Delta x^2\) | Directed interior jump probability |
| \(i_0\) | Initial impulse node |
| \(A\in\mathbb R^{J\times J}\) | Positive-semidefinite reflecting-boundary diffusion operator |
| \(P=I-\Delta tA\) | One-step spatial transition matrix |
| \(V\) | Orthonormal eigenvector matrix of \(A\) |
| \(\Lambda=\operatorname{diag}(\lambda_0,\ldots,\lambda_{J-1})\) | Diffusion eigenvalues |
| \(\Gamma=I-\Delta t\Lambda\) | Diagonal modal transition matrix |
| \(n_t\in\mathbb Z_+^J\) | Exact spatial count vector at saved step \(t\) |
| \(p_t\in[0,1]^J\) | One-particle node-probability column vector |
| \(\mu_t\) | Exact spatial mean count vector |
| \(\Sigma_t\) | Exact same-time spatial covariance |
| \(\Sigma_{t,s}\) | Exact cross-time spatial covariance |
| \(a_t=V^\top n_t\) | Full modal count vector |
| \(M\) | Number of retained slow modes |
| \(r\) | Retained conditional residual innovation rank |
| \(C_t=V^\top\Sigma_tV\) | Exact modal covariance |
| \(Q_t\) | Modal innovation covariance for transition \(t\to t+1\) |
| \(\mathcal L_t\) | A modal-innovation square-root factor satisfying \(\mathcal L_t\mathcal L_t^\top=Q_t\) |
| \(z_t\) | Standard-normal latent vector |
| \(\xi_t\) | Shared zero-sum spatial Gaussian debit-credit fluctuation |
| \(\eta_t\) | Full modal innovation, partitioned into retained and unresolved components |
| \(\mathcal R,\mathcal U\) | Retained and unresolved modal-index sets |
| \(\mathcal S_N\) | Nonnegative spatial simplex with total mass \(N\) |
| \(y_t\) | Physical output after readout |
| \(e_t\) | Physical-readout residual ledger |
| \(\widehat{\cdot}\) | Finite-ensemble estimate |
| \(\|\cdot\|_2\), \(\|\cdot\|_F\) | Euclidean and Frobenius norms |

## Appendix B: exact random-walk mathematics

### Spatial operator and timestep condition

The semidiscrete deterministic diffusion equation is

\[
\frac{dn}{dt}=-An,
\]

where \(A\) is the positive-semidefinite Neumann Laplacian scaled by
\(D_{\mathrm{phys}}/\Delta x^2\). The implemented finite-step expectation is

\[
\mu_{t+1}=P\mu_t,\qquad P=I-\Delta tA.
\]

For an interior node, a particle moves left with probability \(q\), right with
probability \(q\), and remains with probability \(1-2q\). Reflecting endpoints
have one inward move with probability \(q\) and remain with probability
\(1-q\). Nonnegative probabilities require

\[
0\le q\le\frac12.
\]

### Exact integer transition

Let \((p_{i,L},p_{i,S},p_{i,R})\) be the valid left, stay, and right
destination probabilities at source node \(i\), and let
\((c^L_{t,i},c^S_{t,i},c^R_{t,i})\) be the corresponding integer transfer
counts. Conditional on the realized integer source count,

\[
(c^L_{t,i},c^S_{t,i},c^R_{t,i})
\sim\operatorname{Multinomial}
\left(n_{t,i};p_{i,L},p_{i,S},p_{i,R}\right).
\]

Destination counts are the sums of incoming integer transfers. The vectorized
exact implementation first samples

\[
m_{t,i}\sim\operatorname{Binomial}
\left(n_{t,i},p_{i,L}+p_{i,R}\right)
\]

and then

\[
c^R_{t,i}\mid m_{t,i}\sim\operatorname{Binomial}
\left(m_{t,i},\frac{p_{i,R}}{p_{i,L}+p_{i,R}}\right),
\qquad c^L_{t,i}=m_{t,i}-c^R_{t,i},
\qquad c^S_{t,i}=n_{t,i}-m_{t,i}.
\]

Here \(m_{t,i}\) is the total number of movers. This two-binomial construction
is distributionally identical to the source-node multinomial draw.

### Exact marginal and temporal moments

For a deterministic impulse, let \(p_0\) place probability one at \(i_0\). We
use a column-vector convention; because the implemented transition \(P\) is
symmetric, this is equivalent to the row-vector update used in the code. Then

\[
p_{t+1}=Pp_t,\qquad
n_t\sim\operatorname{Multinomial}(N,p_t).
\]

The exact spatial mean and covariance are

\[
\mu_t=Np_t,
\]

\[
\Sigma_t=N\left[\operatorname{diag}(p_t)-p_t p_t^\top\right].
\]

For \(s\ge t\), the exact cross-time covariance is

\[
\Sigma_{t,s}
=N\left[\operatorname{diag}(p_t)P^{s-t}-p_t p_s^\top\right].
\]

These equations describe first and second moments only; the exact path law is
defined by the conditional integer transition above.

## Appendix C: modal and completion mathematics

### Spectral representation

Because \(A\) is symmetric,

\[
A=V\Lambda V^\top.
\]

Full modal coordinates are \(a_t=V^\top n_t\), and the expected modal update is

\[
\mathbb E[a_{t+1}]=\Gamma\mathbb E[a_t],
\qquad
\Gamma=I-\Delta t\Lambda.
\]

The zero mode has \(\lambda_0=0\), so its amplitude is constant and represents
conserved total mass. Retaining the first \(M\) columns of \(V\) keeps the
slowest spatial modes.

### Original independent EigenMarkov model

For each nonconstant mode \(k\), the original model writes

\[
a_{t,k}=w(u^+_{t,k}-u^-_{t,k}),
\]

where \(w\) is the configured modal unit weight and \(u^+_{t,k}\) and
\(u^-_{t,k}\) are the positive and negative integer unit populations. Each
unit flips sign with probability

\[
\pi_k=\frac{\lambda_k\Delta t}{2}.
\]

Therefore the represented modal state satisfies

\[
\mathbb E[a_{t+1,k}\mid a_{t,k}]
=(1-\lambda_k\Delta t)a_{t,k}.
\]

The conditional expected update is exact for that represented state, but the
default nearest-unit initialization can quantize the exact initial modal
amplitude. Independent mode flips also imply a diagonal modal noise model that
does not equal \(V^\top\Sigma_tV\) in general.

### Shared spatial Gaussian fluctuation

At a source node with nonnegative mass \(x_i\) and categorical destination
probability vector \(\rho_i\), the correlated model uses the conditional
Gaussian covariance

\[
\Sigma_{\xi,i}
=x_i\left[\operatorname{diag}(\rho_i)-\rho_i\rho_i^\top\right].
\]

Source contributions are accumulated into a zero-sum spatial field \(\xi_t\)
and projected as \(V_M^\top\xi_t\). For the nonnegative source state or
simplex-projected proxy supplied to the sampler, this matches the categorical
transfer covariance formula. The draw remains continuous and is not
conditioned on a realized integer count state when the proxy is real-valued.

### Direct Gaussian transition

Project the exact multinomial covariance into the eigenbasis:

\[
C_t=V^\top\Sigma_tV.
\]

The direct Gaussian process is

\[
a_{t+1}=\Gamma a_t+\mathcal L_tz_t,
\qquad z_t\sim\mathcal N(0,I),
\]

with innovation covariance

\[
Q_t=\mathcal L_t\mathcal L_t^\top=C_{t+1}-\Gamma C_t\Gamma.
\]

The implementation symmetrizes \(Q_t\), verifies positive semidefiniteness up
to a relative tolerance, and clips only roundoff-scale negative eigenvalues.
With the exact initial mean and covariance, the full-rank recurrence has modal
covariance \(C_t\). For \(s\ge t\),

\[
\operatorname{Cov}(a_t,a_s)=C_t\Gamma^{s-t}.
\]

This proves exact first and second temporal moments for the Gaussian recurrence;
it does not prove equality with the transformed multinomial distribution.

### Innovation-conditioned unresolved modes

Partition one innovation and its covariance into retained index set
\(\mathcal R\) and unresolved index set \(\mathcal U\):

\[
\eta_t=
\begin{bmatrix}\eta_{\mathcal R,t}\\\eta_{\mathcal U,t}\end{bmatrix},
\qquad
Q_t=
\begin{bmatrix}
Q_{\mathcal R\mathcal R,t} & Q_{\mathcal R\mathcal U,t}\\
Q_{\mathcal U\mathcal R,t} & Q_{\mathcal U\mathcal U,t}
\end{bmatrix}.
\]

The Gaussian conditional law is

\[
\eta_{\mathcal U,t}\mid\eta_{\mathcal R,t}
\sim\mathcal N\left(
Q_{\mathcal U\mathcal R,t}Q_{\mathcal R\mathcal R,t}^{+}
\eta_{\mathcal R,t},
Q_{\mathcal U\mathcal U,t}
-Q_{\mathcal U\mathcal R,t}Q_{\mathcal R\mathcal R,t}^{+}
Q_{\mathcal R\mathcal U,t}
\right),
\]

where \((\cdot)^+\) is the Moore--Penrose pseudoinverse. The explicit
Schur-complement subtraction can be unstable when early transitions are rank
deficient. Instead, partition a joint factor using the same retained and
unresolved row sets,

\[
\mathcal L_t=
\begin{bmatrix}\mathcal L_{\mathcal R,t}\\
\mathcal L_{\mathcal U,t}\end{bmatrix},
\]

and compute an SVD
\(\mathcal L_{\mathcal R,t}=U_tS_tW_t^\top\). The observed right-singular
directions determine the conditional gain, while the null directions of
\(\mathcal L_{\mathcal R,t}\) give a square root of the residual conditional covariance.
A second SVD retains only the leading \(r\) residual directions.

If \(H_t\) is the retained innovation factor, \(G_t\) is the unresolved-on-
retained conditional gain, and \(B_{r,t}\) is the rank-\(r\) residual factor,
the packed joint factor has the block form

\[
F_t=
\begin{bmatrix}
H_t & 0\\
G_tH_t & B_{r,t}
\end{bmatrix}.
\]

One standard-normal vector of width \(M+r\) therefore generates the retained
and completed unresolved innovation together.

## Appendix D: physical readout mathematics

Define the continuous fixed-mass simplex

\[
\mathcal S_N=
\left\{y\in\mathbb R^J:y_i\ge0,\ \mathbf1^\top y=N\right\}.
\]

For raw completed field \(x_t\) and prior residual ledger \(e_{t-1}\), the
simplex bank computes

\[
b_t=x_t+e_{t-1},
\]

\[
y_t=\Pi_{\mathcal S_N}(b_t)
=\arg\min_{y\in\mathcal S_N}\frac12\|y-b_t\|_2^2,
\]

\[
e_t=b_t-y_t.
\]

The residual makes the correction explicit and carries it forward instead of
discarding it. The exact projection has the threshold form

\[
y_{t,i}=\max(b_{t,i}-\tau_t,0),
\]

where \(\tau_t\) is chosen so that \(\sum_i y_{t,i}=N\). The compiled warm
kernel starts from the preceding threshold, updates the active set, and checks
the KKT support inequalities. If a numerical guard fails, the complete
trajectory is replayed through the sorting-based NumPy reference with a clean
ledger.

The output is continuous. Integer Delta-Sigma readouts replace the continuous
projection with an integer-simplex quantizer and carry the quantization residual
through time. Integer constraints alone do not reproduce the multinomial law.

## Appendix E: accuracy and distribution metrics

For ensemble trajectories \(y_t^{(b)}\), \(b=1,\ldots,R\), define

\[
\widehat\mu_t=\frac1R\sum_{b=1}^{R}y_t^{(b)},
\]

\[
\widehat v_{t,i}=\frac1R\sum_{b=1}^{R}
\left(y_{t,i}^{(b)}-\widehat\mu_{t,i}\right)^2,
\]

and

\[
\widehat\Sigma_t=\frac1R\sum_{b=1}^{R}
\left(y_t^{(b)}-\widehat\mu_t\right)
\left(y_t^{(b)}-\widehat\mu_t\right)^\top.
\]

The implementation uses population normalization \(1/R\), not \(1/(R-1)\).

### Analytic rank and covariance errors

Let \(\sigma_{t,j}\) be the singular values of the residual conditional
innovation factor at transition \(t\), in decreasing order. The variance
fraction represented by residual rank \(r\) is

\[
f_t(r)=
\frac{\sum_{j=1}^{r}\sigma_{t,j}^2}
{\sum_j\sigma_{t,j}^2}.
\]

The analytic-rank table reports the mean of \(f_t(r)\) over transitions and
also quotes its minimum and median for the selected rank. Let \(\widetilde C_t\)
be the modal covariance propagated by the truncated packed factors and \(C_t\)
the exact modal covariance. The full covariance error is

\[
E_{C,\mathrm{all}}=
\frac{
\left(\sum_t\|\widetilde C_t-C_t\|_F^2\right)^{1/2}}
{
\left(\sum_t\|C_t\|_F^2\right)^{1/2}},
\]

and the worst-time covariance error is

\[
E_{C,\max}=
\max_{t:\|C_t\|_F>0}
\frac{\|\widetilde C_t-C_t\|_F}{\|C_t\|_F}.
\]

Marginal variance error applies the same stacked relative norm to the diagonals
of the reconstructed spatial covariances
\(V\widetilde C_tV^\top\) and \(VC_tV^\top\). For a lag of \(\ell\) steps, the
candidate and exact modal cross-covariances are
\(\widetilde C_t\Gamma^\ell\) and \(C_t\Gamma^\ell\). The analytic lag error
stacks their relative Frobenius differences over every valid time origin. The
reported physical lags 1, 5, and 20 are converted to the nearest valid step
counts for each grid.

### Mean and variance errors

With time and node axes stacked,

\[
E_\mu=
\frac{\|\widehat\mu-\mu\|_F}{\|\mu\|_F},
\qquad
E_v=
\frac{\|\widehat v-v\|_F}{\|v\|_F}.
\]

The reference-standardized sampled-mean error is

\[
E_{\mu,\mathrm{SE}}=
\frac{\|\widehat\mu-\mu\|_F}
{\|\sqrt{v/R}\|_F}.
\]

A value near one indicates a sampled mean discrepancy on the scale expected
from the reference Monte Carlo standard error.

### Same-time and cross-time covariance errors

For selected covariance times \(\mathcal T\),

\[
E_\Sigma=
\frac{
\left(\sum_{t\in\mathcal T}
\|\widehat\Sigma_t-\Sigma_t\|_F^2\right)^{1/2}}
{
\left(\sum_{t\in\mathcal T}\|\Sigma_t\|_F^2\right)^{1/2}}.
\]

For lag \(\ell\), each time origin is centered separately before empirical
cross-covariances are averaged. The reported lag error is

\[
E_{\Sigma,\ell}=
\frac{\|\widehat\Sigma_\ell-\Sigma_\ell\|_F}
{\|\Sigma_\ell\|_F}.
\]

### Physicality metrics

Maximum mass error is

\[
E_{\mathrm{mass}}=
\max_{b,t}\left|\mathbf1^\top y_t^{(b)}-N\right|.
\]

The diagnostics also record minimum output, negative-entry fraction below a
\(-10^{-12}\) tolerance, negative mass divided by \(N\), and noninteger-entry
fraction for readouts that claim integer output.

### Wasserstein and Jensen--Shannon metrics

For scalar samples \(X\) and \(Y\), empirical Wasserstein-1 distance is

\[
W_1(X,Y)=\int_0^1
\left|F_X^{-1}(u)-F_Y^{-1}(u)\right|\,du.
\]

The normalized marginal score divides each distance by

\[
\max(\operatorname{sd}(X),1),
\]

where \(X\) is the exact reference sample, and then averages over selected
time-node marginals.

For common-bin discrete histograms \(P\) and \(Q\), let \(M=(P+Q)/2\). The
Jensen--Shannon divergence in nats is

\[
\operatorname{JS}(P,Q)=
\frac12\operatorname{KL}(P\|M)
+\frac12\operatorname{KL}(Q\|M).
\]

It remains finite when the continuous modal output and integer exact output do
not share support.

Sliced Wasserstein projects full spatial samples onto \(N_\theta\) fixed
zero-sum unit directions \(\theta_j\):

\[
\operatorname{SW}_1(X,Y)=
\frac1{N_\theta}\sum_{j=1}^{N_\theta}
W_1(\theta_j^\top X,\theta_j^\top Y).
\]

Removing the constant direction prevents the exactly conserved mass component
from dominating the comparison. Each projected score is normalized by the
corresponding exact-sample standard deviation with a one-particle lower bound.

### Sampling-floor interpretation

A finite exact ensemble does not reproduce analytic moments exactly. Its moment
errors quantify the Monte Carlo floor at the same trial count. Distribution
metrics have no single analytic zero-sample target, so two independent exact
ensembles provide the corresponding exact-vs-exact floor. A modal value near
that floor is consistent with the resolution of the experiment; it is not proof
of identical probability laws.

## Appendix F: performance and memory metrics

Let \(t_s\) be reusable modal setup time, \(t_c\) optional compiled-readout
startup, \(t_m\) prepared modal trajectory time, \(t_x\) exact-walk trajectory
time, and \(K\) the number of independent trajectories on one configuration.
Then

\[
T_m(K)=t_s+Kt_m,
\]

\[
T_{m,\mathrm{startup}}(K)=t_s+t_c+Kt_m,
\]

and

\[
T_x(K)=Kt_x.
\]

When \(t_m<t_x\), the first trajectory count at which the modal total is no
greater is

\[
K_{\mathrm{break-even}}=
\left\lceil\frac{t_s}{t_x-t_m}\right\rceil.
\]

Compiled-readout startup is kept separate because a cached specialization can
be shared across configurations in one process and because empty-cache and
cached-process costs differ substantially.

Memory measures used in the report are:

- **Retained array bytes:** unique reachable NumPy backing storage held after
  preparation.
- **Post-setup RSS:** process resident memory after setup and garbage
  collection while the prepared object remains live.
- **Peak setup RSS:** process high-water resident memory observed during
  construction.
- **Output bytes:** exact logical bytes in a returned full-field array.
- **Logical setup-plus-output bytes:** retained prepared arrays plus selected
  returned arrays; this is not a process peak.

These quantities answer different questions and must not be added unless the
arrays are known to coexist. A complete paper benchmark should additionally
measure trajectory-phase high-water RSS in the same isolated process.

## Appendix G: source map

The principal implementation locations are:

- `src/eigendiffusion/baselines.py`: exact particle and aggregated count
  references.
- `src/eigendiffusion/operators.py`: reflecting diffusion operator and modal
  transforms.
- `src/eigendiffusion/eigenmarkov.py`: original independent modal model.
- `src/eigendiffusion/correlated_modal.py`: shared spatial Gaussian fluctuation,
  banked dynamics, and handoff models.
- `src/eigendiffusion/completion.py`: independent, persistent, and local analytic
  completions.
- `src/eigendiffusion/direct_modal.py`: direct modal Gaussian transition.
- `src/eigendiffusion/innovation_completion.py`: conditional completion, packed
  factors, compact runtime, and streaming constructor.
- `src/eigendiffusion/readouts.py`: continuous and integer readouts.
- `src/eigendiffusion/compiled_simplex.py`: compiled exact simplex bank.
- `src/eigendiffusion/metrics.py`, `temporal.py`, and
  `distribution_metrics.py`: validation metrics.
- `src/eigendiffusion/*_benchmark.py`: reproducible benchmark drivers and
  artifact schemas.

The historical development record and rejected-branch evidence remain in
`PROJECT_REPORT_2026-11-20.md`, `STRUCTURED_LOCAL_CLOSURE.md`,
`FITTED_CLOSURE_STUDY.md`, `MODE_SCALING.md`, and
`ADAPTIVE_RANK_SCALING.md`.
