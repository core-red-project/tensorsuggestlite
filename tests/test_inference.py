"""Tests verifying text classification inference with Keras and TFLite."""

from pathlib import Path

from tensorsuggestlite.core.entities import TrainingConfig, TrainingDataset
from tensorsuggestlite.ml.trainer import KerasTextTrainer
from tensorsuggestlite.use_cases import ExportTFLiteUseCase, PredictTextUseCase


def test_inference_keras_and_tflite(sample_dataset: TrainingDataset, tmp_path: Path):
    """Verify inference pipeline on both Keras and TFLite runtimes."""
    # 1. Train model
    trainer = KerasTextTrainer()
    trainer.train(sample_dataset, output_dir=tmp_path, config=TrainingConfig(epochs=2))

    # 2. Export TFLite
    exporter = ExportTFLiteUseCase()
    exporter.execute(model_dir=tmp_path)

    # 3. Predict with Keras
    use_case = PredictTextUseCase(tmp_path)
    res_keras = use_case.predict_keras("This product is really amazing and works well")

    assert res_keras.runtime == "keras"
    assert res_keras.top_category in {"POSITIVE", "NEGATIVE"}
    assert 0.0 <= res_keras.top_confidence <= 1.0
    assert len(res_keras.scores) == 2
    assert res_keras.latency_ms > 0.0

    # 4. Predict with TFLite
    res_tflite = use_case.predict_tflite("This product is really amazing and works well")

    assert res_tflite.runtime == "tflite"
    assert res_tflite.top_category in {"POSITIVE", "NEGATIVE"}
    assert 0.0 <= res_tflite.top_confidence <= 1.0
    assert len(res_tflite.scores) == 2
    assert res_tflite.latency_ms > 0.0

    # Verify classification agreement between Keras and TFLite
    assert res_keras.top_category == res_tflite.top_category
