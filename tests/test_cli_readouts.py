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
