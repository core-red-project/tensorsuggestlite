"""Use case for exporting trained Keras models to TFLite format."""

from __future__ import annotations

from pathlib import Path

from tensorsuggestlite.core.interfaces import (
    LogCallback,
    ModelExporterProtocol,
    ProgressCallback,
)
from tensorsuggestlite.ml.exporter import TFLiteExporter


class ExportTFLiteUseCase:
    """Orchestrates loading a trained neural model and exporting it to an optimized TFLite binary."""

    def __init__(self, exporter: ModelExporterProtocol | None = None) -> None:
        """Initialize the export use case.

        Args:
            exporter: Custom exporter implementing ModelExporterProtocol.
        """
        self.exporter = exporter or TFLiteExporter()

    def execute(
        self,
        model_dir: Path | str,
        output_path: Path | str | None = None,
        quantization: str = "dynamic",
        progress_cb: ProgressCallback | None = None,
        log_cb: LogCallback | None = None,
    ) -> Path:
        """Execute TFLite model conversion and export with configurable quantization.

        Args:
            model_dir: Directory containing text_classifier.keras.
            output_path: Optional explicit destination path for the exported .tflite binary.
            quantization: Quantization mode ('dynamic', 'float16', 'none').
            progress_cb: Optional callback emitting progress percentages.
            log_cb: Optional callback emitting informational log messages.

        Returns:
            Path: Path to the generated .tflite model file.

        Raises:
            FileNotFoundError: If model_dir does not exist or is not a directory.
            ModelExportError: If TFLite conversion or disk writing fails.
        """
        resolved_dir = Path(model_dir).resolve()
        if not resolved_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {resolved_dir}")

        return self.exporter.export_tflite(
            model_dir=resolved_dir,
            output_path=output_path,
            quantization=quantization,
            progress_cb=progress_cb,
            log_cb=log_cb,
        )


__all__ = ["ExportTFLiteUseCase"]
