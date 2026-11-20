# Predictive handoff selection

The `study-handoff-selection` command predicts the error introduced by handing
off from a full correlated-modal transient to a reduced basis. For retained
projection `Pi_M`, it evaluates the hybrid trajectory that is full rank before
the handoff and projected afterward:

```text
mean error = ||mu - Pi_M mu||_2 / ||mu||_2
variance error = ||diag(C) - diag(Pi_M C Pi_M)||_2 / ||diag(C)||_2
covariance error = ||C - Pi_M C Pi_M||_F / ||C||_F
```

Covariance error uses the requested diagnostic times. Because the eigenbasis is
orthonormal, omitted covariance energy is computed in modal coordinates without
forming every projected spatial covariance. The command also reports the exact
fraction of modal coordinates updated relative to a full-rank trajectory and
marks the nondominated accuracy/work Pareto frontier.

Optional `--runs` validate each prediction against a stochastic handoff
ensemble. A full correlated ensemble at the same run count supplies the Monte
Carlo sampling floor, which is combined in quadrature with the analytic
truncation prediction.

## Validation result

A held-out grid used final ranks `35, 50, 60, 75`, handoff times
`10, 15, 20, 30`, and 250 trials per point. Prediction-versus-observation
results were:

| Metric | Correlation | Relative prediction RMSE | Median predicted/observed |
| --- | ---: | ---: | ---: |
| Mean | 0.9992 | 0.075 | 1.18 |
| Variance | 0.9995 | 0.010 | 1.00 |
| Covariance | 0.9982 | 0.044 | 1.04 |

The existing 101-to-50 handoff at step 10 is Pareto optimal, but its predicted
raw covariance loss is about 0.52. Requiring raw mean/variance/covariance errors
below `0.005/0.25/0.30` selects approximately 75 modes at step 30, which still
uses 82% of full modal-update work. This quantitatively motivates unresolved
completion rather than retaining enough raw modes to meet the covariance
tolerance directly.

Outputs:

- `outputs/handoff_selection/analytic_grid.csv`
- `outputs/handoff_selection/analytic_grid.png`
- `outputs/handoff_selection/validated_grid_r250.csv`
- `outputs/handoff_selection/validated_grid_r250.png`
