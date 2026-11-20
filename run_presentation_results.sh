#!/usr/bin/env bash
#
# Generate presentation-ready EigenDiffusion figures and data.
#
# Before running:
#   source .venv/bin/activate
#
# Run everything:
#   ./run_presentation_results.sh
#
# For quicker rehearsal runs, reduce --runs, --repeats, and
# --amortization-runs below.

set -euo pipefail

mkdir -p outputs/presentation

echo "1/5 Comparing the core modal models..."
python -m eigendiffusion compare-modal-models \
  --models independent_modal correlated_modal banked_correlated_modal \
  --particles 5275 \
  --nodes 101 \
  --steps 101 \
  --impulse-index 59 \
  --runs 500 \
  --modes 101 \
  --random-walk-method multinomial \
  --covariance-times 1 5 20 50 100 \
  --profile-step 50 \
  --seed 0 \
  --output outputs/presentation/01_modal_models.png \
  --csv-output outputs/presentation/01_modal_models.csv \
  --data-output outputs/presentation/01_modal_models.npz

echo "2/5 Comparing reduced-model readouts and completion..."
python -m eigendiffusion compare-readouts \
  --modal-model handoff_correlated_modal \
  --initial-modes 101 \
  --handoff-time 10 \
  --modes 50 \
  --readouts raw simplex_bank unresolved_gaussian_completion persistent_unresolved_completion \
  --completion-start-time 10 \
  --completion-rank 10 \
  --particles 5275 \
  --nodes 101 \
  --steps 101 \
  --impulse-index 59 \
  --runs 500 \
  --random-walk-method multinomial \
  --covariance-times 1 5 20 50 100 \
  --profile-step 50 \
  --seed 1 \
  --output outputs/presentation/02_readouts.png \
  --csv-output outputs/presentation/02_readouts.csv \
  --data-output outputs/presentation/02_readouts.npz

echo "3/5 Sweeping handoff time and reduced mode count..."
python -m eigendiffusion sweep-handoff \
  --initial-modes 101 \
  --final-modes 20 35 50 75 \
  --handoff-times 1 5 10 20 50 \
  --particles 5275 \
  --nodes 101 \
  --steps 101 \
  --impulse-index 59 \
  --runs 250 \
  --error-start-times 0 10 20 50 \
  --late-start-time 20 \
  --seed 2 \
  --output outputs/presentation/03_handoff_sweep.png \
  --csv-output outputs/presentation/03_handoff_sweep.csv

echo "4/5 Comparing temporal correlations..."
python -m eigendiffusion compare-temporal-correlation \
  --initial-modes 101 \
  --handoff-time 10 \
  --modes 50 \
  --completion-start-time 10 \
  --completion-rank 10 \
  --particles 5275 \
  --nodes 101 \
  --steps 101 \
  --impulse-index 59 \
  --runs 500 \
  --lags 1 2 5 10 20 \
  --temporal-start-time 10 \
  --temporal-end-time 100 \
  --node-window-radius 10 \
  --profile-lag 10 \
  --random-walk-method multinomial \
  --seed 3 \
  --output outputs/presentation/04_temporal_correlation.png \
  --csv-output outputs/presentation/04_temporal_correlation.csv \
  --data-output outputs/presentation/04_temporal_correlation.npz

echo "5/5 Benchmarking model performance..."
python -m eigendiffusion benchmark-models \
  --methods multinomial_random_walk full_correlated_modal handoff_raw handoff_independent_completion handoff_persistent_completion \
  --node-counts 31 51 101 151 \
  --particle-counts 1000 5275 10000 \
  --base-nodes 101 \
  --base-particles 5275 \
  --retained-fraction 0.5 \
  --handoff-time 10 \
  --completion-rank 10 \
  --repeats 5 \
  --amortization-runs 500 \
  --seed 4 \
  --output outputs/presentation/05_performance.png \
  --csv-output outputs/presentation/05_performance.csv

echo "Presentation results are ready in outputs/presentation/."
