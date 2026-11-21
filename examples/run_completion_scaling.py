"""Screen completion rank and modal bandwidth after the mode-scaling study.

The default experiment holds the reduced dynamical state at 75 modes and
compares its completed output with the existing proportional-mode baseline at
151 and 201 spatial nodes. Raw trajectories are shared across every
rank/bandwidth variant with the same retained mode count.
"""

from __future__ import annotations

import argparse

from eigendiffusion.mode_scaling import (
    ModeSelectionPolicy,
    ResolutionModeStudySpec,
    run_resolution_mode_study,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for reproducible study configuration."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, nargs="+", default=(151, 201))
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--fixed-modes", type=int, default=75)
    parser.add_argument("--ranks", type=int, nargs="+", default=(5, 10, 20))
    parser.add_argument("--bandwidths", type=int, nargs="+", default=(12, 20))
    parser.add_argument("--bootstrap-resamples", type=int, default=100)
    parser.add_argument("--bootstrap-blocks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--timing-only", action="store_true")
    parser.add_argument("--timing-repeats", type=int, default=7)
    parser.add_argument(
        "--output",
        default="outputs/mode_scaling/completion_screen",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    variants = tuple(
        ModeSelectionPolicy(
            name=f"fixed_{args.fixed_modes}_rank_{rank}_bandwidth_{bandwidth}",
            kind="fixed",
            fixed_modes=args.fixed_modes,
            completion_rank=rank,
            modal_bandwidth=bandwidth,
        )
        for bandwidth in sorted(set(args.bandwidths))
        for rank in sorted(set(args.ranks))
    )
    policies = (
        ModeSelectionPolicy(
            name="proportional_75_of_101",
            kind="proportional",
            completion_rank=5,
            modal_bandwidth=12,
        ),
        *variants,
    )
    spec = ResolutionModeStudySpec(
        node_counts=tuple(sorted(set(args.nodes))),
        policies=policies,
        screening_runs=args.runs,
        comparison_bootstrap_resamples=args.bootstrap_resamples,
        comparison_bootstrap_blocks=args.bootstrap_blocks,
        timing_repeats=args.timing_repeats,
        seed=args.seed,
    )
    summary = run_resolution_mode_study(
        spec,
        args.output,
        stage="analytic" if args.timing_only else "screening",
        include_timing=args.timing_only,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
