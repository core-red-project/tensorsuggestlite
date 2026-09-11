"""End-to-end integration tests verifying model training and TFLite export."""

import json
from pathlib import Path

from tensorsuggestlite.core.entities import TrainingConfig, TrainingDataset
from tensorsuggestlite.ml.trainer import KerasTextTrainer
from tensorsuggestlite.use_cases import ExportTFLiteUseCase


def test_e2e_train_and_export_tflite(sample_dataset: TrainingDataset, tmp_path: Path):
    """Verify full end-to-end flow: Training -> Artifacts -> TFLite export."""
    trainer = KerasTextTrainer()
    config = TrainingConfig(epochs=2, test_size=0.0)

    # 1. Train model
    result = trainer.train(
        dataset=sample_dataset,
        output_dir=tmp_path,
        config=config,
    )

    assert result.items_serialized == len(sample_dataset)
    assert result.labels_count == len(sample_dataset.unique_categories)

    # Check generated files
    assert result.artifacts.keras_model_path.is_file()
    assert result.artifacts.tokenizer_path.is_file()
    assert result.artifacts.label_encoder_path.is_file()

    with open(result.artifacts.label_encoder_path, encoding="utf-8") as f:
        encoder_data = json.load(f)
    assert "POSITIVE" in encoder_data
    assert "NEGATIVE" in encoder_data

    # 2. Export to TFLite
    exporter = ExportTFLiteUseCase()
    tflite_path = exporter.execute(model_dir=tmp_path)

    assert tflite_path.is_file()
    assert tflite_path.stat().st_size > 1000  # Non-trivial model file
