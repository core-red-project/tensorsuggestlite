"""JSON configuration parser for TensorSuggestLite datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tensorsuggestlite.parsers.base import BaseConfigParser


class JsonConfigParser(BaseConfigParser):
    """Parser implementation for JSON dataset configuration files."""

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        """Return supported JSON file extension."""
        return (".json",)

    def load_raw_data(self, file_path: Path) -> Any:
        """Load and decode raw JSON structure into a Python dictionary or list.

        Args:
            file_path: Path to the target JSON file.

        Returns:
            Any: Parsed JSON data.
        """
        with open(file_path, encoding="utf-8") as f:
            return json.load(f)


__all__ = ["JsonConfigParser"]
