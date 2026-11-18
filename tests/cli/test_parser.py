"""Unit tests for the CLI argument parser."""

from tensorsuggestlite.cli.parser import build_parser


def test_build_parser_defaults():
    """Verify default global arguments and version flag."""
    parser = build_parser()
    args = parser.parse_args(["inspect", "dataset.json"])
    assert args.command == "inspect"
    assert args.file == "dataset.json"
    assert args.json_mode is False
    assert args.quiet is False
    assert args.verbose is False


def test_build_parser_train_hyperparameters():
    """Verify train subcommand argument types and defaults."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "dataset.yaml",
            "--epochs",
            "25",
            "--learning-rate",
            "0.002",
            "--dropout",
            "0.3",
            "--batch-size",
            "32",
            "--skeleton",
            "in:out",
        ]
    )
    assert args.command == "train"
    assert args.epochs == 25
    assert args.learning_rate == 0.002
    assert args.dropout == 0.3
    assert args.batch_size == 32
    assert args.skeleton == "in:out"


def test_build_parser_convert_quantize_choices():
    """Verify convert subcommand quantization choices."""
    parser = build_parser()
    args = parser.parse_args(["convert", "/tmp/model", "--quantize", "float16"])
    assert args.command == "convert"
    assert args.quantize == "float16"


def test_build_parser_json_flag_precedence():
    """Verify that --json flag works both before and after subcommands."""
    parser = build_parser()

    # --json before subcommand
    args1 = parser.parse_args(["--json", "inspect", "dataset.json"])
    assert args1.command == "inspect"
    assert args1.json_mode is True

    # --json after subcommand
    args2 = parser.parse_args(["inspect", "dataset.json", "--json"])
    assert args2.command == "inspect"
    assert args2.json_mode is True

    # no --json
    args3 = parser.parse_args(["inspect", "dataset.json"])
    assert args3.command == "inspect"
    assert args3.json_mode is False

    # --json before train
    args4 = parser.parse_args(["--json", "train", "dataset.json"])
    assert args4.command == "train"
    assert args4.json_mode is True
