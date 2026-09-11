"""Interfaces and protocols for TensorSuggestLite."""

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from tensorsuggestlite.core.entities import TrainingConfig, TrainingDataset, TrainingResult

ProgressCallback = Callable[[int], None]
LogCallback = Callable[[str], None]


class ConfigParserProtocol(Protocol):
    """Protocol for configuration file parsers."""

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        """Tuple of supported file extensions (e.g. ('.json',))."""
        ...

    def parse(self, file_path: Path | str) -> TrainingDataset:
        """Parse configuration file into a TrainingDataset."""
        ...


class ModelTrainerProtocol(Protocol):
    """Protocol for training text classification models."""

    def train(
        self,
        dataset: TrainingDataset,
        output_dir: Path | str,
        config: TrainingConfig | None = None,
        progress_cb: ProgressCallback | None = None,
        log_cb: LogCallback | None = None,
    ) -> TrainingResult:
        """Train model from dataset and save artifacts."""
        ...


class ModelExporterProtocol(Protocol):
    """Protocol for exporting models to TFLite format."""

    def export_tflite(
        self,
        model_dir: Path | str,
        output_path: Path | str | None = None,
        progress_cb: ProgressCallback | None = None,
        log_cb: LogCallback | None = None,
    ) -> Path:
        """Convert Keras model to TFLite."""
        ...
