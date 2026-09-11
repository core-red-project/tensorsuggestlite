"""Use case for exporting Keras model to TFLite format."""

from pathlib import Path

from tensorsuggestlite.core.interfaces import (
    LogCallback,
    ModelExporterProtocol,
    ProgressCallback,
)
from tensorsuggestlite.ml.exporter import TFLiteExporter


class ExportTFLiteUseCase:
    """Orchestrates loading a trained model and exporting it as TFLite."""

    def __init__(self, exporter: ModelExporterProtocol | None = None) -> None:
        self.exporter = exporter or TFLiteExporter()

    def execute(
        self,
        model_dir: Path | str,
        output_path: Path | str | None = None,
        progress_cb: ProgressCallback | None = None,
        log_cb: LogCallback | None = None,
    ) -> Path:
        return self.exporter.export_tflite(
            model_dir=model_dir,
            output_path=output_path,
            progress_cb=progress_cb,
            log_cb=log_cb,
        )
