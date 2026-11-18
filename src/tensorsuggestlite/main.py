"""Application entry point for TensorSuggestLite.

Routes CLI commands to tensorsuggestlite.cli and graphical desktop presentation
to tensorsuggestlite.ui, adhering strictly to single-responsibility DX architecture.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence

# Prevent thread contention and stack pressure in native math libraries
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from tensorsuggestlite.cli import (
    build_parser,
    cli_main,
    generate_completions,
    run_convert,
    run_history,
    run_inspect,
    run_predict,
    run_train,
)
from tensorsuggestlite.ui import launch_gui


def run_gui() -> int:
    """Launch the graphical user interface.

    Returns:
        int: Exit status code (0 for clean execution, 1 if GUI dependencies are missing).
    """
    return launch_gui()


def main(argv: Sequence[str] | None = None) -> int:
    """Unified application entrypoint.

    Args:
        argv: Optional sequence of command-line arguments.

    Returns:
        int: Process exit code.
    """
    return cli_main(argv)


if __name__ == "__main__":
    sys.exit(main())

__all__ = [
    "build_parser",
    "cli_main",
    "generate_completions",
    "launch_gui",
    "main",
    "run_convert",
    "run_history",
    "run_inspect",
    "run_predict",
    "run_train",
]
