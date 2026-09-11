"""Tests for cross-validation between Keras and TFLite runtimes."""

from pathlib import Path

from tensorsuggestlite.core.entities import CrossValidationResult, TrainingConfig, TrainingDataset
from tensorsuggestlite.use_cases import ExportTFLiteUseCase, PredictTextUseCase, TrainModelUseCase


def test_cross_validation_parity(tmp_path: Path, sample_dataset: TrainingDataset) -> None:
    train_use_case = TrainModelUseCase()
    export_use_case = ExportTFLiteUseCase()

    cfg = TrainingConfig(epochs=6, batch_size=4)
    result = train_use_case.trainer.train(sample_dataset, output_dir=tmp_path, config=cfg)
    assert Path(result.model_path).is_file()

    # Export to TFLite
    tflite_path = export_use_case.execute(tmp_path)
    assert tflite_path.is_file()

    # Predict & Compare
    predictor = PredictTextUseCase(tmp_path)
    comparison: CrossValidationResult = predictor.compare("This is fantastic and great")

    assert comparison.keras_result.top_category in {"POSITIVE", "NEGATIVE"}
    assert comparison.tflite_result.top_category in {"POSITIVE", "NEGATIVE"}
    assert comparison.is_consistent is True
    assert 0.0 <= comparison.delta_confidence < 0.15
    assert 0.0 <= comparison.max_class_delta < 0.20
    assert comparison.keras_result.latency_ms >= 0.0
    assert comparison.tflite_result.latency_ms >= 0.0
