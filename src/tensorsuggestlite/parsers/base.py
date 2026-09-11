"""Base parser and registry for configuration files."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from tensorsuggestlite.core.entities import TrainingDataset, TrainingSample
from tensorsuggestlite.core.exceptions import ConfigParseError, EmptyDatasetError
from tensorsuggestlite.core.interfaces import ConfigParserProtocol


class BaseConfigParser(ABC, ConfigParserProtocol):
    """Abstract base parser that standardizes sample extraction."""

    @property
    @abstractmethod
    def supported_extensions(self) -> tuple[str, ...]:
        """Tuple of supported file extensions."""
        pass

    @abstractmethod
    def load_raw_data(self, file_path: Path) -> dict[str, Any]:
        """Load and parse the file into a Python dictionary."""
        pass

    def parse(self, file_path: Path | str) -> TrainingDataset:
        """Parse configuration file into a TrainingDataset."""
        path = Path(file_path).resolve()
        if not path.is_file():
            raise ConfigParseError(f"Configuration file not found: {path}")

        try:
            data = self.load_raw_data(path)
        except Exception as err:
            raise ConfigParseError(f"Failed to parse {path.name}: {err}") from err

        if not isinstance(data, dict):
            raise ConfigParseError(f"Root structure must be a dictionary in {path.name}")

        samples = self.extract_samples(data)
        if not samples:
            raise EmptyDatasetError(f"No training samples found in {path.name}")

        return TrainingDataset(samples=samples)

    @staticmethod
    def extract_samples(data: dict[str, Any]) -> list[TrainingSample]:
        """Extract training samples supporting both English and Spanish schemas."""
        samples: list[TrainingSample] = []

        for category, content in data.items():
            if not isinstance(content, dict):
                continue

            category_name = str(category).strip()

            # Support both 'responses' (English) and 'respuestas' (legacy/Spanish)
            responses = content.get("responses") or content.get("respuestas") or []
            if isinstance(responses, list):
                for item in responses:
                    if isinstance(item, dict):
                        text = item.get("response") or item.get("respuesta")
                        if text and isinstance(text, str) and text.strip():
                            samples.append(
                                TrainingSample(text=text.strip(), category=category_name)
                            )
                    elif isinstance(item, str) and item.strip():
                        samples.append(TrainingSample(text=item.strip(), category=category_name))

            # Support both 'synonyms' (English) and 'sinonimos' (legacy/Spanish)
            synonyms = content.get("synonyms") or content.get("sinonimos") or []
            if isinstance(synonyms, list):
                for syn in synonyms:
                    if isinstance(syn, str) and syn.strip():
                        samples.append(TrainingSample(text=syn.strip(), category=category_name))

        return samples


class ParserRegistry:
    """Registry to resolve the appropriate parser for a given file extension."""

    def __init__(self) -> None:
        self._parsers: dict[str, ConfigParserProtocol] = {}

    def register(self, parser: ConfigParserProtocol) -> None:
        """Register a parser for its supported extensions."""
        for ext in parser.supported_extensions:
            self._parsers[ext.lower()] = parser

    def get_parser_for_path(self, file_path: Path | str) -> ConfigParserProtocol:
        """Retrieve parser matching the file extension."""
        path = Path(file_path)
        ext = path.suffix.lower()
        if ext not in self._parsers:
            supported = ", ".join(sorted(self._parsers.keys()))
            raise ConfigParseError(
                f"Unsupported file format '{ext}'. Supported formats: {supported}"
            )
        return self._parsers[ext]
