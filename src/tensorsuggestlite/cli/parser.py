"""Command-line interface argument parser for TensorSuggestLite.

Configures global flags (-V, -q, -v, --json, --no-color) and subcommands
strictly aligned with the Sxnnyside CLI Interface Contract.
"""

from __future__ import annotations

import argparse

from tensorsuggestlite import __version__

PROG_NAME = "tensorsuggestlite"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser with full Sxnnyside CLI surface.

    Returns:
        argparse.ArgumentParser: Fully configured parser instance.
    """
    epilog_text = """EXAMPLES:
  # Inspect and validate a configuration dataset
  tensorsuggestlite inspect examples/example.json

  # Inspect in JSON machine-readable format
  tensorsuggestlite inspect examples/example.json --json

  # Train model from configuration file
  tensorsuggestlite train examples/example.json --epochs 12 --output-dir models/run_01

  # Run real-time prediction using TFLite runtime
  tensorsuggestlite predict models/run_01 "The service was great" --tflite

  # Generate shell completions
  tensorsuggestlite completions zsh > ~/.zsh/completion/_tensorsuggestlite
"""

    parser = argparse.ArgumentParser(
        prog=PROG_NAME,
        description="Configuration-driven text classification and TFLite model generation.",
        epilog=epilog_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mandatory Global Flags (-V, -q, -v, --json, --no-color)
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"{PROG_NAME} {__version__}",
        help="Print version information and exit",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress non-error diagnostic output on stderr",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Emit extra diagnostic details to stderr",
    )
    parser.add_argument(
        "--json",
        dest="json_mode",
        action="store_true",
        help="Emit machine-readable JSON output to stdout",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colored output (also respects NO_COLOR env var)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # GUI command
    subparsers.add_parser("gui", help="Launch desktop graphical interface")

    # Version command
    subparsers.add_parser("version", help="Print version information and exit")

    # Help command
    help_parser = subparsers.add_parser("help", help="Print usage and help for commands")
    help_parser.add_argument("target", nargs="?", default=None, help="Command to display help for")

    # Man command (hidden for packaging)
    subparsers.add_parser("man", help=argparse.SUPPRESS)

    # Inspect command
    inspect_parser = subparsers.add_parser(
        "inspect", help="Inspect and validate dataset configuration"
    )
    inspect_parser.add_argument("file", help="Path to JSON, YAML, or TOML dataset file")
    inspect_parser.add_argument(
        "--skeleton",
        help="Custom fixed schema skeleton (JSON string or 'in_key:out_key' shorthand)",
    )
    inspect_parser.add_argument(
        "--json",
        dest="json_mode",
        action="store_true",
        default=argparse.SUPPRESS,
        help="JSON output",
    )

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a text classification model")
    train_parser.add_argument("file", help="Path to JSON, YAML, or TOML file")
    train_parser.add_argument("--output-dir", help="Output directory for artifacts")
    train_parser.add_argument(
        "--skeleton",
        help="Custom fixed schema skeleton (JSON string or 'in_key:out_key' shorthand)",
    )
    train_parser.add_argument(
        "--epochs", type=int, default=12, help="Number of epochs (default: 12)"
    )
    train_parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate (default: 0.001)"
    )
    train_parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout regularization rate (default: 0.2)",
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=16, help="Training batch size (default: 16)"
    )
    train_parser.add_argument(
        "--json",
        dest="json_mode",
        action="store_true",
        default=argparse.SUPPRESS,
        help="JSON output",
    )

    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert Keras model to TFLite binary")
    convert_parser.add_argument("model_dir", help="Directory containing text_classifier.keras")
    convert_parser.add_argument("--output", help="Destination path for .tflite file")
    convert_parser.add_argument(
        "--quantize",
        choices=["dynamic", "float16", "none"],
        default="dynamic",
        help="TFLite quantization mode: dynamic, float16, none (default: dynamic)",
    )
    convert_parser.add_argument(
        "--json",
        dest="json_mode",
        action="store_true",
        default=argparse.SUPPRESS,
        help="JSON output",
    )

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict text classification")
    predict_parser.add_argument("model_dir", help="Directory containing trained model artifacts")
    predict_parser.add_argument(
        "text",
        nargs="?",
        default=None,
        help="Text sentence to classify (or '-' to read from standard input)",
    )
    predict_parser.add_argument(
        "--file",
        dest="batch_file",
        help="Path to file containing sentences for batch prediction (one per line or JSON array)",
    )
    predict_parser.add_argument("--tflite", action="store_true", help="Use TFLite binary runtime")
    predict_parser.add_argument(
        "--compare", action="store_true", help="Compare Keras vs TFLite side-by-side"
    )
    predict_parser.add_argument(
        "--json",
        dest="json_mode",
        action="store_true",
        default=argparse.SUPPRESS,
        help="JSON output",
    )

    # History command
    history_parser = subparsers.add_parser(
        "history", help="List and inspect model metadata and training runs"
    )
    history_parser.add_argument("model_dir", help="Directory containing model artifacts or runs")
    history_parser.add_argument(
        "--json",
        dest="json_mode",
        action="store_true",
        default=argparse.SUPPRESS,
        help="JSON output",
    )

    # Completions command
    completions_parser = subparsers.add_parser("completions", help="Generate shell completions")
    completions_parser.add_argument(
        "shell",
        choices=["bash", "zsh", "fish", "powershell"],
        help="Target shell for completions",
    )

    return parser
