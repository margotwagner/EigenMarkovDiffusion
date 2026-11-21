"""Run the cross-physics adaptive completion-rank promotion workflow."""

from __future__ import annotations

import argparse
from dataclasses import replace

from eigendiffusion.rank_scaling import (
    CrossCaseRankScalingSpec,
    RankScalingPhysicalCase,
    run_cross_case_rank_scaling,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("selection", "confirmation", "timing"),
        default="selection",
    )
    parser.add_argument("--nodes", type=int, nargs="+", default=(151, 201))
    parser.add_argument("--runs", type=int)
    parser.add_argument("--residual-variance-tolerance", type=float, default=0.10)
    parser.add_argument("--selection-seed", type=int, default=20_261_130)
    parser.add_argument("--confirmation-seed", type=int, default=20_261_201)
    parser.add_argument("--timing-seed", type=int, default=20_261_202)
    parser.add_argument("--timing-repeats", type=int, default=7)
    parser.add_argument("--setup-repeats", type=int, default=3)
    parser.add_argument("--policies", nargs="+")
    parser.add_argument("--output", default="outputs/adaptive_rank_scaling")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    nodes = tuple(sorted(set(args.nodes)))
    spec = CrossCaseRankScalingSpec(
        cases=(
            RankScalingPhysicalCase("base_length4", 4.0, nodes),
            RankScalingPhysicalCase("long_length6", 6.0, nodes),
        ),
        residual_variance_tolerance=args.residual_variance_tolerance,
        selection_seed=args.selection_seed,
        confirmation_seed=args.confirmation_seed,
        timing_seed=args.timing_seed,
        timing_repeats=args.timing_repeats,
        setup_repeats=args.setup_repeats,
    )
    if args.runs is not None:
        if args.stage == "selection":
            spec = replace(spec, selection_runs=args.runs)
        elif args.stage == "confirmation":
            spec = replace(spec, confirmation_runs=args.runs)
    summary = run_cross_case_rank_scaling(
        spec,
        args.output,
        stage=args.stage,
        selected_policy_names=args.policies,
    )
    for key, value in summary.items():
        if key != "case_summaries":
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
