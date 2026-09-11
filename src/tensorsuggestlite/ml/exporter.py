"""TFLite model exporter with quantization support."""

from pathlib import Path

from tensorsuggestlite.core.exceptions import ModelExportError
from tensorsuggestlite.core.interfaces import LogCallback, ModelExporterProtocol, ProgressCallback


class TFLiteExporter(ModelExporterProtocol):
    """Converts trained Keras models to optimized TFLite artifacts."""

    def export_tflite(
        self,
        model_dir: Path | str,
        output_path: Path | str | None = None,
        progress_cb: ProgressCallback | None = None,
        log_cb: LogCallback | None = None,
    ) -> Path:
        def report_log(msg: str) -> None:
            if log_cb:
                try:
                    log_cb(msg)
                except Exception:
                    pass

        def report_prog(val: int) -> None:
            if progress_cb:
                try:
                    progress_cb(max(0, min(100, val)))
                except Exception:
                    pass

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
        report_log("Configuring TFLite converter with default optimizations...")

        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
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

        report_prog(100)
        report_log("TFLite model successfully exported.")
        return dest_path
