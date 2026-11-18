"""Tests verifying model metadata and quality report generation."""

import json
from pathlib import Path

from tensorsuggestlite.core.entities import TrainingConfig, TrainingDataset
from tensorsuggestlite.ml.trainer import KerasTextTrainer


def test_metadata_generation(sample_dataset: TrainingDataset, tmp_path: Path):
    """Verify that model_metadata.json is generated with required telemetry."""
    trainer = KerasTextTrainer()
    config = TrainingConfig(epochs=2, dropout_rate=0.3, learning_rate=0.005)

    result = trainer.train(sample_dataset, output_dir=tmp_path, config=config)

    metadata_path = result.artifacts.metadata_path
    assert metadata_path is not None
    assert metadata_path.is_file()

    with open(metadata_path, encoding="utf-8") as f:
        meta = json.load(f)

    assert "created_at" in meta
    assert "dataset" in meta
    assert meta["dataset"]["total_samples"] == len(sample_dataset)
    assert "sha256" in meta["dataset"]
    assert meta["hyperparameters"]["dropout_rate"] == 0.3
    assert meta["hyperparameters"]["learning_rate"] == 0.005
    assert "classification_report" in meta
    assert "final_accuracy" in meta["metrics"]
    assert "confusion_matrix" in meta
    assert "matrix" in meta["confusion_matrix"]
    assert "labels" in meta["confusion_matrix"]
    assert "profiling" in meta
    assert "imbalance_ratio" in meta["profiling"]
    assert "tensors" in meta
    assert "sequence_length" in meta["tensors"]
