"""Interfaces and runtime protocols defining architectural contracts for TensorSuggestLite.

Defines protocols for configuration parsers, model trainers, and TFLite
model exporters conforming to Clean Architecture and Interface Segregation principles.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from tensorsuggestlite.core.entities import TrainingConfig, TrainingDataset, TrainingResult

ProgressCallback = Callable[[int], None]
LogCallback = Callable[[str], None]


class ConfigParserProtocol(Protocol):
    """Protocol defining the contract for structured file dataset parsers."""

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        """Tuple of file extensions supported by this parser (e.g. ('.json',))."""
        ...

    def parse(
        self,
        file_path: Path | str,
        skeleton: dict[str, str] | None = None,
    ) -> TrainingDataset:
        """Parse an input dataset configuration into a TrainingDataset.

        Args:
            file_path: Filesystem path to the input dataset file.

        Returns:
            TrainingDataset: Validated dataset containing extracted samples.

        Raises:
            ConfigParseError: If file structure cannot be parsed or decoded.
            EmptyDatasetError: If no valid classification samples are found.
        """
        ...


class ModelTrainerProtocol(Protocol):
    """Protocol for training text classification neural models."""

    def train(
        self,
        dataset: TrainingDataset,
        output_dir: Path | str,
        config: TrainingConfig | None = None,
        progress_cb: ProgressCallback | None = None,
        log_cb: LogCallback | None = None,
    ) -> TrainingResult:
        """Train a neural classification model and persist serialized artifacts.

        Args:
            dataset: Ingested training samples and labels.
            output_dir: Directory where trained model artifacts will be saved.
            config: Optional hyperparameters configuration.
            progress_cb: Optional callback emitting progress percentages [0-100].
            log_cb: Optional callback emitting diagnostic log messages.

        Returns:
            TrainingResult: Summary containing artifact paths and evaluation metrics.

        Raises:
            ModelTrainingError: If training or serialization encounters an error.
        """
        ...


class ModelExporterProtocol(Protocol):
    """Protocol for converting and exporting models to TFLite format."""

    def export_tflite(
        self,
        model_dir: Path | str,
        output_path: Path | str | None = None,
        quantization: str = "dynamic",
        progress_cb: ProgressCallback | None = None,
        log_cb: LogCallback | None = None,
    ) -> Path:
        """Convert a trained model into a portable TFLite binary.

        Args:
            model_dir: Directory containing the trained Keras model.
            output_path: Optional explicit output path for the .tflite binary.
            progress_cb: Optional callback reporting conversion progress [0-100].
            log_cb: Optional callback reporting conversion events.

        Returns:
            Path: Path to the generated .tflite model file.

        Raises:
            ModelExportError: If conversion or binary serialization fails.
        """
        ...


__all__ = [
    "ConfigParserProtocol",
    "LogCallback",
    "ModelExporterProtocol",
    "ModelTrainerProtocol",
    "ProgressCallback",
]
