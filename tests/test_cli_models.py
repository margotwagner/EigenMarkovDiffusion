from eigendiffusion.cli import build_parser


def test_validate_defaults_to_independent_modal():
    parser = build_parser()
    args = parser.parse_args(["validate"])
    assert args.modal_model == "independent_modal"


def test_validate_accepts_correlated_modal():
    parser = build_parser()
    args = parser.parse_args(["validate", "--modal-model", "correlated_modal"])
    assert args.modal_model == "correlated_modal"


def test_compare_defaults_to_all_three_modal_models():
    parser = build_parser()
    args = parser.parse_args(["compare-modal-models"])
    assert args.models == [
        "independent_modal",
        "correlated_modal",
        "banked_correlated_modal",
    ]
