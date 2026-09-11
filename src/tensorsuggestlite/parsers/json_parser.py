"""JSON configuration parser."""

import json
from pathlib import Path
from typing import Any, cast

from tensorsuggestlite.parsers.base import BaseConfigParser


class JsonConfigParser(BaseConfigParser):
    """Parser for JSON training files."""

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return (".json",)

    def load_raw_data(self, file_path: Path) -> dict[str, Any]:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
            return cast(dict[str, Any], data)
