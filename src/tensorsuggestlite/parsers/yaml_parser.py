"""YAML configuration parser for TensorSuggestLite datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from tensorsuggestlite.parsers.base import BaseConfigParser


class YamlConfigParser(BaseConfigParser):
    """Parser implementation for YAML and YML dataset configuration files."""

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        """Return supported YAML file extensions."""
        return (".yaml", ".yml")

    def load_raw_data(self, file_path: Path) -> Any:
        """Safely decode YAML structure into a Python dictionary or list.

        Args:
            file_path: Path to the target YAML file.

        Returns:
            Any: Parsed YAML data.
        """
        with open(file_path, encoding="utf-8") as f:
            return yaml.safe_load(f)


__all__ = ["YamlConfigParser"]
