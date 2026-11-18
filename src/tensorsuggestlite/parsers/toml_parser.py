"""TOML configuration parser using Python 3.11+ standard library tomllib."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from tensorsuggestlite.parsers.base import BaseConfigParser


class TomlConfigParser(BaseConfigParser):
    """Parser implementation for TOML dataset configuration files."""

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        """Return supported TOML file extension."""
        return (".toml",)

    def load_raw_data(self, file_path: Path) -> Any:
        """Decode binary TOML content into a Python dictionary using tomllib.

        Args:
            file_path: Path to the target TOML file.

        Returns:
            Any: Parsed TOML data.
        """
        with open(file_path, "rb") as f:
            return tomllib.load(f)


__all__ = ["TomlConfigParser"]
