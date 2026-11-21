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

echo "1/10 Comparing the core modal models..."
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

echo "2/10 Comparing reduced-model readouts and completion..."
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

echo "3/10 Sweeping handoff time and reduced mode count..."
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

echo "4/10 Comparing temporal correlations..."
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

echo "5/10 Benchmarking model performance..."
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

echo "6/10 Measuring trial-count convergence..."
python -m eigendiffusion study-trial-convergence \
  --models independent_modal correlated_modal banked_correlated_modal \
  --particles 5275 \
  --nodes 101 \
  --steps 101 \
  --impulse-index 59 \
  --modes 101 \
  --trial-counts 100 250 500 1000 2000 5000 \
  --bootstrap-resamples 200 \
  --bootstrap-blocks 20 \
  --confidence-level 0.95 \
  --covariance-times 1 5 20 50 100 \
  --seed 20 \
  --output outputs/presentation/06_trial_convergence.png \
  --csv-output outputs/presentation/06_trial_convergence.csv \
  --data-output outputs/presentation/06_trial_convergence.npz

echo "7/10 Validating predictive handoff selection..."
python -m eigendiffusion study-handoff-selection \
  --particles 5275 \
  --nodes 101 \
  --steps 101 \
  --impulse-index 59 \
  --initial-modes 101 \
  --final-modes 35 50 60 75 \
  --handoff-times 10 15 20 30 \
  --runs 250 \
  --covariance-times 1 5 20 50 100 \
  --mean-tolerance 0.005 \
  --variance-tolerance 0.25 \
  --covariance-tolerance 0.30 \
  --seed 30 \
  --output outputs/presentation/07_handoff_selection.png \
  --csv-output outputs/presentation/07_handoff_selection.csv

echo "8/10 Sweeping completion rank and physical constraints..."
python -m eigendiffusion sweep-completion-rank \
  --particles 5275 \
  --nodes 101 \
  --steps 101 \
  --impulse-index 59 \
  --initial-modes 101 \
  --modes 50 \
  --handoff-time 10 \
  --completion-start-time 10 \
  --completion-ranks 0 5 10 20 30 51 \
  --completion-readouts unresolved_gaussian_completion persistent_unresolved_completion unresolved_gaussian_completion_simplex_bank persistent_unresolved_completion_simplex_bank \
  --completion-ridge 0.01 \
  --temporal-lag 1 \
  --runs 500 \
  --covariance-times 1 5 10 20 50 100 \
  --seed 40 \
  --output outputs/presentation/08_completion_selection.png \
  --csv-output outputs/presentation/08_completion_selection.csv

echo "9/10 Benchmarking selected pipelines and distribution accuracy..."
python -m eigendiffusion study-operating-points \
  --node-counts 51 101 151 201 \
  --accuracy-nodes 101 \
  --particles 5275 \
  --total-time 100 \
  --dt 1 \
  --completion-ridge 0.01 \
  --repeats 3 \
  --amortization-runs 100 \
  --accuracy-runs 500 \
  --distribution-times 20 50 100 \
  --distribution-node-offsets -10 0 10 \
  --projections 64 \
  --seed 5 \
  --output outputs/presentation/09_operating_points.png \
  --csv-output outputs/presentation/09_operating_point_scaling.csv \
  --distribution-csv-output outputs/presentation/09_distribution_accuracy.csv \
  --summary-output outputs/presentation/09_operating_point_summary.json

echo "10/10 Selecting and validating the structured local closure..."
python -m eigendiffusion study-structured-closure \
  --node-counts 51 101 151 201 \
  --accuracy-nodes 101 \
  --particles 5275 \
  --accuracy-runs 500 \
  --heldout-runs 250 \
  --bandwidths 4 8 12 20 \
  --repeats 3 \
  --amortization-runs 100 \
  --distribution-times 20 50 100 \
  --distribution-node-offsets -10 0 10 \
  --projections 64 \
  --seed 6 \
  --output-directory outputs/presentation/10_structured_closure

echo "Presentation results are ready in outputs/presentation/."
