"""TFLite model exporter with optimization and quantization support."""

from __future__ import annotations

import contextlib
from pathlib import Path

from tensorsuggestlite.core.exceptions import ModelExportError
from tensorsuggestlite.core.interfaces import (
    LogCallback,
    ModelExporterProtocol,
    ProgressCallback,
)


class TFLiteExporter(ModelExporterProtocol):
    """Converts trained Keras neural network models into optimized, portable TFLite binaries."""

    def export_tflite(
        self,
        model_dir: Path | str,
        output_path: Path | str | None = None,
        quantization: str = "dynamic",
        progress_cb: ProgressCallback | None = None,
        log_cb: LogCallback | None = None,
    ) -> Path:
        """Convert a saved Keras model into an optimized TFLite binary.

        Loads text_classifier.keras from model_dir, configures requested quantization
        ('dynamic', 'float16', 'none'), and serializes the resulting flatbuffer binary.

        Args:
            model_dir: Directory containing text_classifier.keras.
            output_path: Optional destination path for the exported .tflite binary.
            quantization: Quantization mode ('dynamic', 'float16', 'none').
            progress_cb: Optional callback emitting progress percentages [0-100].
            log_cb: Optional callback emitting diagnostic log messages.

        Returns:
            Path: Resolved path to the exported TFLite binary file.

        Raises:
            ModelExportError: If TensorFlow is unavailable, model loading fails, or conversion aborts.
        """

        def report_log(msg: str) -> None:
            if log_cb:
                with contextlib.suppress(Exception):
                    log_cb(msg)

        def report_prog(val: int) -> None:
            if progress_cb:
                with contextlib.suppress(Exception):
                    progress_cb(max(0, min(100, val)))

        valid_quantizations = {"dynamic", "float16", "none"}
        if quantization not in valid_quantizations:
            raise ModelExportError(
                f"Unsupported quantization: '{quantization}'. Choose 'dynamic', 'float16', or 'none'."
            )

        report_prog(0)
        report_log("Starting TFLite conversion...")

        try:
            import tensorflow as tf
        except ImportError as err:
            raise ModelExportError(f"TensorFlow is not available: {err}") from err

        dir_path = Path(model_dir).resolve()
        keras_path = dir_path / "text_classifier.keras"

        if not keras_path.exists():
            raise ModelExportError(f"Keras model not found at: {keras_path}")

        report_prog(20)
        report_log(f"Loading Keras model from {keras_path.name}...")

        try:
            model = tf.keras.models.load_model(keras_path)
        except Exception as err:
            raise ModelExportError(f"Failed to load Keras model: {err}") from err

        report_prog(50)
        report_log(f"Configuring TFLite converter with '{quantization}' quantization...")

        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            if quantization == "dynamic":
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            elif quantization == "float16":
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
            elif quantization == "none":
                converter.optimizations = []
            tflite_model = converter.convert()
        except Exception as err:
            raise ModelExportError(f"TFLite conversion failed: {err}") from err

        report_prog(80)
        dest_path = (
            Path(output_path).resolve() if output_path else dir_path / "text_classifier.tflite"
        )
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        report_log(f"Writing TFLite binary to {dest_path}...")
        with open(dest_path, "wb") as f:
            f.write(tflite_model)

        # Update model_metadata.json telemetry if present
        import json

        metadata_file = dir_path / "model_metadata.json"
        if metadata_file.is_file():
            with contextlib.suppress(Exception):
                with open(metadata_file, encoding="utf-8") as mf:
                    meta = json.load(mf)
                tflite_size_kb = len(tflite_model) / 1024.0
                keras_size_kb = meta.get("tensors", {}).get("keras_model_size_kb", 0.0)
                meta["tflite"] = {
                    "quantization": quantization,
                    "tflite_model_size_kb": round(tflite_size_kb, 2),
                    "compression_ratio": round(keras_size_kb / max(tflite_size_kb, 0.001), 2),
                }
                with open(metadata_file, "w", encoding="utf-8") as mf:
                    json.dump(meta, mf, indent=2)

        report_prog(100)
        report_log("TFLite model successfully exported.")
        return dest_path


__all__ = ["TFLiteExporter"]
