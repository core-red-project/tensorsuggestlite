"""Structured logging subsystem for TensorSuggestLite.

Provides unified logger configuration conforming to standard library conventions,
supporting diagnostic filtering and machine-readable or formatted output.
"""

from __future__ import annotations

import logging
import sys

LOGGER_NAME = "tensorsuggestlite"


def get_logger() -> logging.Logger:
    """Retrieve the core TensorSuggestLite logger instance.

    Returns:
        logging.Logger: Configured application logger.
    """
    return logging.getLogger(LOGGER_NAME)


def configure_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Configure handler and log level for the application logger.

    Args:
        verbose: If True, set level to DEBUG for granular diagnostics.
        quiet: If True, suppress all logs below ERROR level.
    """
    logger = get_logger()
    logger.handlers.clear()

    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


__all__ = ["LOGGER_NAME", "configure_logging", "get_logger"]
