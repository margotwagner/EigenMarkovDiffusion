from eigendiffusion.cli import build_parser


def test_validate_accepts_handoff_model_arguments() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "validate",
            "--modal-model",
            "handoff_correlated_modal",
            "--initial-modes",
            "21",
            "--modes",
            "8",
            "--handoff-time",
            "5",
        ]
    )
    assert args.modal_model == "handoff_correlated_modal"
    assert args.initial_modes == 21
    assert args.modes == 8
    assert args.handoff_time == 5.0


def test_sweep_handoff_parser() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "sweep-handoff",
            "--initial-modes",
            "21",
            "--final-modes",
            "5",
            "8",
            "--handoff-times",
            "1",
            "5",
        ]
    )
    assert args.initial_modes == 21
    assert args.final_modes == [5, 8]
    assert args.handoff_times == [1.0, 5.0]
