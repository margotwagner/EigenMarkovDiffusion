from eigendiffusion.cli import build_parser


def test_validate_defaults_to_raw_readout() -> None:
    args = build_parser().parse_args(["validate"])
    assert args.modal_model == "independent_modal"
    assert args.readout == "raw"


def test_compare_readouts_parser() -> None:
    args = build_parser().parse_args(
        [
            "compare-readouts",
            "--modal-model",
            "correlated_modal",
            "--readouts",
            "raw",
            "delta_sigma_temporal",
        ]
    )
    assert args.modal_model == "correlated_modal"
    assert args.readouts == ["raw", "delta_sigma_temporal"]


def test_unresolved_completion_parser_options() -> None:
    args = build_parser().parse_args(
        [
            "validate",
            "--modal-model",
            "handoff_correlated_modal",
            "--modes",
            "10",
            "--readout",
            "unresolved_gaussian_completion",
            "--completion-rank",
            "5",
            "--completion-ridge",
            "0.01",
        ]
    )
    assert args.readout == "unresolved_gaussian_completion"
    assert args.completion_rank == 5
    assert args.completion_ridge == 0.01


def test_completion_rank_sweep_parser() -> None:
    args = build_parser().parse_args(
        [
            "sweep-completion-rank",
            "--modes",
            "10",
            "--completion-ranks",
            "0",
            "5",
            "11",
        ]
    )
    assert args.modes == 10
    assert args.completion_ranks == [0, 5, 11]
