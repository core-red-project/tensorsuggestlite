"""TOML configuration parser."""

from pathlib import Path
from typing import Any, cast

import toml

from tensorsuggestlite.parsers.base import BaseConfigParser


class TomlConfigParser(BaseConfigParser):
    """Parser for TOML training files."""

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return (".toml",)

    def load_raw_data(self, file_path: Path) -> dict[str, Any]:
        with open(file_path, encoding="utf-8") as f:
            data = toml.load(f)
            return cast(dict[str, Any], data)
