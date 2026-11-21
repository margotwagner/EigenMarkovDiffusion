"""Run the staged resolution-aware retained-mode study.

Examples
--------
Fast analytic map::

    python examples/run_mode_scaling.py --stage analytic

Bounded Monte Carlo screen::

    python examples/run_mode_scaling.py --stage screening --runs 250

The final 2,000-run stage should normally be restricted to policies selected
from the screen with ``--policies``.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from eigendiffusion.mode_scaling import ResolutionModeStudySpec, run_resolution_mode_study


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for reproducible study configuration."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("analytic", "screening", "final"),
        default="analytic",
    )
    parser.add_argument("--nodes", type=int, nargs="+", default=(101, 151, 201))
    parser.add_argument("--runs", type=int)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--timing-repeats", type=int, default=7)
    parser.add_argument("--setup-repeats", type=int, default=3)
    parser.add_argument("--bootstrap-resamples", type=int, default=200)
    parser.add_argument("--bootstrap-blocks", type=int, default=20)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--policies", nargs="+")
    parser.add_argument("--output", default="outputs/mode_scaling")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    spec = ResolutionModeStudySpec(
        node_counts=tuple(args.nodes),
        timing_repeats=args.timing_repeats,
        setup_repeats=args.setup_repeats,
        comparison_bootstrap_resamples=args.bootstrap_resamples,
        comparison_bootstrap_blocks=args.bootstrap_blocks,
        comparison_confidence_level=args.confidence_level,
        seed=args.seed,
    )
    if args.runs is not None:
        spec = (
            replace(spec, screening_runs=args.runs)
            if args.stage == "screening"
            else replace(spec, final_runs=args.runs)
        )
    summary = run_resolution_mode_study(
        spec,
        args.output,
        stage=args.stage,
        selected_policy_names=args.policies,
        include_timing=args.timing,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

