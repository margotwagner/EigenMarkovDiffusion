# Adaptive completion-rank scaling

This workflow selects completion rank from the exact conditional innovation
spectrum, then requires the selected policy to transfer across both the base
length-4 domain and a length-6 domain. Both cases use 151 and 201 nodes so fixed
75-mode dynamics are genuinely smaller than the proportional baseline.

The candidate rank at each saved transition is the smallest rank whose factor
leaves at most 10% of the conditional innovation variance unresolved. The
screen compares:

- proportional modes with dense rank-5 completion;
- fixed 75 modes with dense adaptive-rank completion;
- fixed 75 modes with local adaptive-rank completion and physically scaled
  bandwidths corresponding to 12 and 20 modes on the length-4 domain.

Run the 100-trial selection stage:

```bash
PYTHONPATH=src python3 examples/run_adaptive_rank_scaling.py \
  --stage selection --output outputs/adaptive_rank_scaling
```

The confirmation stage automatically carries forward only policies that pass
every case and resolution. It uses a disjoint master seed and 250 trials:

```bash
PYTHONPATH=src python3 examples/run_adaptive_rank_scaling.py \
  --stage confirmation --output outputs/adaptive_rank_scaling
```

Then benchmark confirmed policies:

```bash
PYTHONPATH=src python3 examples/run_adaptive_rank_scaling.py \
  --stage timing --output outputs/adaptive_rank_scaling
```

A candidate passes the cost gate only if, at the largest physical-refinement
grid in both domains, median setup time over three constructions and resident
arrays are below the proportional baseline and median trajectory time over
seven runs is no more than 5% slower. Only a policy that passes both the
accuracy and cost intersections should advance to a 2,000-trial headline
comparison.

## Results

The 100-trial selection stage retained two candidates across both domains and
both grids: dense adaptive completion and the wider local adaptive completion.
The narrower local rule (bandwidth 12 on length 4, scaled to 18 on length 6)
failed the length-6, 201-node temporal gate with lag-covariance error 0.1686.
The wider local rule uses bandwidth 20 on length 4 and 30 on length 6.

Both retained candidates then passed the independent 250-trial confirmation at
151 and 201 nodes in both domains. The 90%-energy rule selected:

| Domain length | Nodes | Rank min/median/max | Dense minimum capture | Wide-local minimum capture |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 151 | 7 / 10 / 13 | 0.900206 | 0.900117 |
| 4 | 201 | 12 / 17 / 21 | 0.900086 | 0.900079 |
| 6 | 151 | 5 / 7 / 9 | 0.900730 | 0.900503 |
| 6 | 201 | 8 / 11 / 14 | 0.900275 | 0.900016 |

These diagnostics are the ranks actually used. For adaptive policies, the
larger `completion_rank` field in the CSV is a hard cap rather than the selected
rank. The baseline has no energy tolerance, so its tolerance flag should not be
interpreted as an adaptive-rank result.

The isolated timing gate rejects both candidates:

| Domain length | Candidate | Setup ratio | Resident-array ratio | Trajectory ratio | Cost gate |
| ---: | --- | ---: | ---: | ---: | --- |
| 4 | dense adaptive | 1.2185 | 1.9263 | 1.0043 | fail |
| 4 | wide local adaptive | 4.2701 | 0.4354 | 1.0793 | fail |
| 6 | dense adaptive | 1.2275 | 1.8327 | 1.0054 | fail |
| 6 | wide local adaptive | 6.6065 | 0.5531 | 1.0647 | fail |

The dense candidate preserves trajectory speed but increases both setup and
resident storage. The wide local candidate reduces resident arrays by
44.7--56.5%, but its full eigendecomposition and local-gain construction make
setup much slower, and its trajectory time exceeds the 5% allowance. Thus no
candidate is promotion-eligible and the predeclared 2,000-trial run is not
performed. The next implementation target is an operator-based partial
eigensolver for the conditional innovation, followed by optimization of the
local gain construction; accuracy margins should remain unchanged.

The implementation also stores local gains with one shared CSR row topology
and packed time-varying values. This removes repeated padded column-index
storage, but it does not remove the dense eigendecomposition that dominates
setup.
