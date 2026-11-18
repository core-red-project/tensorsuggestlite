"""Command-line interface module for TensorSuggestLite."""

from __future__ import annotations

import difflib
import os
import sys
from collections.abc import Sequence

from tensorsuggestlite import __version__
from tensorsuggestlite.cli.commands import (
    run_convert,
    run_history,
    run_inspect,
    run_predict,
    run_train,
)
from tensorsuggestlite.cli.completions import generate_completions
from tensorsuggestlite.cli.manpage import generate_man_page
from tensorsuggestlite.cli.parser import PROG_NAME, build_parser
from tensorsuggestlite.core.logging import configure_logging

KNOWN_COMMANDS = [
    "gui",
    "version",
    "help",
    "man",
    "inspect",
    "train",
    "convert",
    "predict",
    "history",
    "completions",
]


def cli_main(argv: Sequence[str] | None = None) -> int:
    """CLI execution dispatcher.

    Args:
        argv: Optional argument sequence (defaults to sys.argv[1:]).

    Returns:
        int: Exit code (0 for success, 1 for runtime error, 2 for usage error).
    """
    parser = build_parser()
    args_list = list(argv) if argv is not None else sys.argv[1:]

    # Intercept unknown command candidates before argparse hard-exits
    first_cmd = next((arg for arg in args_list if not arg.startswith("-")), None)
    if first_cmd and first_cmd not in KNOWN_COMMANDS:
        matches = difflib.get_close_matches(first_cmd, KNOWN_COMMANDS, n=1, cutoff=0.5)
        sys.stderr.write(f"error: unknown command '{first_cmd}'.")
        if matches:
            sys.stderr.write(f" Did you mean '{matches[0]}'?")
        sys.stderr.write(f"\nSee '{PROG_NAME} --help' for usage.\n")
        sys.stderr.flush()
        return 2

    try:
        args = parser.parse_args(args_list)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2

    quiet = bool(getattr(args, "quiet", False))
    verbose = bool(getattr(args, "verbose", False))
    json_mode = bool(getattr(args, "json_mode", False))
    no_color = bool(getattr(args, "no_color", False)) or bool(os.environ.get("NO_COLOR"))

    if no_color:
        os.environ["NO_COLOR"] = "1"

    configure_logging(verbose=verbose, quiet=quiet)

    if args.command == "version":
        sys.stdout.write(f"{PROG_NAME} {__version__}\n")
        sys.stdout.flush()
        return 0
    elif args.command == "man":
        sys.stdout.write(generate_man_page())
        sys.stdout.flush()
        return 0
    elif args.command == "help":
        target = getattr(args, "target", None)
        if target:
            # Print help for target command
            try:
                sub_args = [target, "--help"]
                parser.parse_args(sub_args)
            except SystemExit:
                return 0
        parser.print_help(sys.stdout)
        return 0
    elif args.command is None or args.command == "gui":
        from tensorsuggestlite.ui.app import launch_gui

        return launch_gui()
    elif args.command == "inspect":
        return run_inspect(
            args.file,
            skeleton=getattr(args, "skeleton", None),
            json_mode=json_mode,
            quiet=quiet,
        )
    elif args.command == "train":
        return run_train(
            file_path=args.file,
            output_dir=args.output_dir,
            skeleton=getattr(args, "skeleton", None),
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            dropout=args.dropout,
            batch_size=args.batch_size,
            json_mode=json_mode,
            quiet=quiet,
        )
    elif args.command == "convert":
        return run_convert(
            args.model_dir,
            args.output,
            quantization=getattr(args, "quantize", "dynamic"),
            json_mode=json_mode,
            quiet=quiet,
        )
    elif args.command == "predict":
        return run_predict(
            args.model_dir,
            text=args.text,
            batch_file=getattr(args, "batch_file", None),
            prefer_tflite=args.tflite,
            compare=args.compare,
            json_mode=json_mode,
        )
    elif args.command == "history":
        return run_history(args.model_dir, json_mode=json_mode, quiet=quiet)
    elif args.command == "completions":
        return generate_completions(args.shell)
    else:
        parser.print_help(sys.stderr)
        return 2


__all__ = [
    "KNOWN_COMMANDS",
    "PROG_NAME",
    "build_parser",
    "cli_main",
    "generate_completions",
    "generate_man_page",
    "run_convert",
    "run_history",
    "run_inspect",
    "run_predict",
    "run_train",
]
