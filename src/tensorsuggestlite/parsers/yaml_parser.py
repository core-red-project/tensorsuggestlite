"""YAML configuration parser."""

from pathlib import Path
from typing import Any, cast

import yaml

from tensorsuggestlite.parsers.base import BaseConfigParser


class YamlConfigParser(BaseConfigParser):
    """Parser for YAML training files."""

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return (".yaml", ".yml")

    def load_raw_data(self, file_path: Path) -> dict[str, Any]:
        with open(file_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return cast(dict[str, Any], data)
