# Adaptive full-to-reduced modal handoff

This update adds one new modal dynamics option:

```text
handoff_correlated_modal
```

The existing models and readouts are unchanged.

## Motivation

A spatial impulse contains high-frequency diffusion modes. Those modes are
important during the first few microseconds but decay rapidly. A fixed small
basis therefore performs poorly at the start even when it is accurate later.

The handoff model runs a high-rank correlated modal transient and then projects
the current state into a smaller nested eigenbasis:

\[
\mathbf m_H(t+\Delta t)
= (I-\Delta t\Lambda_H)\mathbf m_H(t)+\eta_H(t),
\qquad t < t_h,
\]

\[
\mathbf m_L(t_h)=V_L^\top V_H\mathbf m_H(t_h),
\]

\[
\mathbf m_L(t+\Delta t)
= (I-\Delta t\Lambda_L)\mathbf m_L(t)+\eta_L(t),
\qquad t \ge t_h.
\]

Here `initial_n_modes` defines the early high-rank basis, `final_n_modes`
defines the reduced basis, and `handoff_time` defines the projection time.
The correlated spatial noise construction is unchanged on either side of the
handoff.

The projection error is reported explicitly. No bank or Delta-Sigma readout is
used to hide omitted-mode error unless one is requested separately through the
existing `--readout` option.

## Files changed

- `src/eigendiffusion/correlated_modal.py`
  - adds `HandoffCorrelatedModalDiffusion`
  - adds `HandoffCorrelatedModalResult`
- `src/eigendiffusion/ensemble.py`
  - adds `handoff_correlated_modal` to the model registry
  - adds `run_handoff_correlated_modal_ensemble`
  - reports handoff projection error and an approximate modal-update work fraction
- `src/eigendiffusion/cli.py`
  - adds `--initial-modes` and `--handoff-time`
  - adds the `sweep-handoff` command
- `src/eigendiffusion/plotting.py`
  - adds the handoff sweep figure
- `src/eigendiffusion/__init__.py`
  - exports the new model and runner
- `tests/test_handoff_correlated_modal.py`
- `tests/test_cli_handoff.py`

## Single validation

```bash
python -m eigendiffusion validate \
  --modal-model handoff_correlated_modal \
  --initial-modes 101 \
  --modes 50 \
  --handoff-time 10 \
  --readout raw \
  --particles 5275 \
  --nodes 101 \
  --steps 101 \
  --impulse-index 59 \
  --runs 100 \
  --random-walk-method multinomial \
  --covariance-times 1 5 20 50 100 \
  --seed 0 \
  --output outputs/adaptive_handoff/handoff_t10_m50/validation.png \
  --data-output outputs/adaptive_handoff/handoff_t10_m50/validation.npz
```

## Recommended sweep

```bash
python -m eigendiffusion sweep-handoff \
  --initial-modes 101 \
  --final-modes 10 20 30 50 \
  --handoff-times 1 5 10 20 \
  --readout raw \
  --particles 5275 \
  --nodes 101 \
  --steps 101 \
  --impulse-index 59 \
  --runs 100 \
  --error-start-times 0 5 10 20 \
  --late-start-time 20 \
  --seed 0 \
  --output outputs/adaptive_handoff/sweeps/handoff_sweep.png \
  --csv-output outputs/adaptive_handoff/sweeps/handoff_sweep.csv
```

## Important interpretation

`handoff_modal_update_fraction_of_full` estimates only the relative number of
modal coordinates updated. The current correlated noise sampler still touches
the full spatial grid to construct the physical covariance, so this metric is
not a full runtime claim. Actual wall-clock benchmarking should be performed
before making an efficiency claim.
