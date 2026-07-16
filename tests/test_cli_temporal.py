from eigendiffusion.cli import build_parser


def test_temporal_correlation_command_parses_completion_options():
    parser = build_parser()
    args = parser.parse_args(
        [
            "compare-temporal-correlation",
            "--modes",
            "50",
            "--initial-modes",
            "101",
            "--handoff-time",
            "10",
            "--completion-rank",
            "10",
            "--completion-ridge",
            "0.01",
            "--lags",
            "1",
            "5",
            "10",
        ]
    )
    assert args.command == "compare-temporal-correlation"
    assert args.modes == 50
    assert args.completion_rank == 10
    assert args.lags == [1.0, 5.0, 10.0]


def test_compare_readouts_now_parses_completion_options():
    parser = build_parser()
    args = parser.parse_args(
        [
            "compare-readouts",
            "--completion-start-time",
            "10",
            "--completion-rank",
            "10",
            "--completion-ridge",
            "0.01",
        ]
    )
    assert args.completion_start_time == 10.0
    assert args.completion_rank == 10


def test_validate_parses_persistent_completion_readout():
    parser = build_parser()
    args = parser.parse_args(
        [
            "validate",
            "--modal-model",
            "handoff_correlated_modal",
            "--modes",
            "50",
            "--readout",
            "persistent_unresolved_completion",
            "--completion-rank",
            "10",
        ]
    )
    assert args.readout == "persistent_unresolved_completion"
    assert args.completion_rank == 10
