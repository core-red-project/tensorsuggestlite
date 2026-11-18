"""Unit tests for core entities and validation."""

from pathlib import Path

from tensorsuggestlite.core.entities import (
    ArtifactBundle,
    PredictionResult,
    PredictionScore,
    TrainingConfig,
    TrainingDataset,
    TrainingResult,
    TrainingSample,
)


def test_training_config_defaults_and_custom():
    """Verify TrainingConfig values and serialization."""
    cfg = TrainingConfig()
    assert cfg.epochs == 12
    assert cfg.learning_rate == 0.001
    assert cfg.dropout_rate == 0.20
    assert cfg.batch_size == 16


def test_training_dataset_properties():
    """Verify TrainingDataset properties and unique categories calculation."""
    ds = TrainingDataset(
        samples=[
            TrainingSample(text="hello", category="GREETING"),
            TrainingSample(text="world", category="GENERAL"),
            TrainingSample(text="greetings", category="GREETING"),
        ]
    )
    assert len(ds) == 3
    assert set(ds.unique_categories) == {"GENERAL", "GREETING"}


def test_prediction_result_structure():
    """Verify PredictionResult fields and confidence scores."""
    scores = [
        PredictionScore(category="TECH", confidence=0.85),
        PredictionScore(category="BILLING", confidence=0.15),
    ]
    res = PredictionResult(
        text="The app crashes on boot",
        top_category="TECH",
        top_confidence=0.85,
        scores=scores,
        latency_ms=1.45,
        runtime="tflite",
    )
    assert res.top_category == "TECH"
    assert res.top_confidence == 0.85
    assert len(res.scores) == 2
    assert res.runtime == "tflite"


def test_artifact_bundle_and_training_result(tmp_path: Path):
    """Verify ArtifactBundle and TrainingResult container behavior."""
    bundle = ArtifactBundle(
        output_dir=tmp_path,
        tokenizer_path=tmp_path / "tokenizer.json",
        keras_model_path=tmp_path / "text_classifier.keras",
        label_encoder_path=tmp_path / "label_encoder.json",
        tflite_model_path=tmp_path / "text_classifier.tflite",
    )
    assert bundle.output_dir == tmp_path
    assert bundle.keras_model_path.name == "text_classifier.keras"

    result = TrainingResult(
        model_path=str(tmp_path / "text_classifier.keras"),
        items_serialized=10,
        labels_count=2,
        artifacts=bundle,
        metrics={"final_accuracy": 0.98, "final_loss": 0.05},
    )
    assert result.items_serialized == 10
    assert result.metrics["final_accuracy"] == 0.98


def test_training_config_validation():
    """Verify TrainingConfig post-init hyperparameter validation."""
    import pytest

    # Valid config
    cfg = TrainingConfig(
        epochs=1, batch_size=1, learning_rate=0.001, dropout_rate=0.0, test_size=0.0
    )
    assert cfg.epochs == 1
    assert cfg.test_size == 0.0

    # Invalid epochs
    with pytest.raises(ValueError, match="epochs must be >= 1"):
        TrainingConfig(epochs=0)

    # Invalid batch size
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        TrainingConfig(batch_size=0)

    # Invalid dropout rate
    with pytest.raises(ValueError, match="dropout_rate must be in"):
        TrainingConfig(dropout_rate=1.0)
    with pytest.raises(ValueError, match="dropout_rate must be in"):
        TrainingConfig(dropout_rate=-0.1)

    # Invalid learning rate
    with pytest.raises(ValueError, match=r"learning_rate must be > 0\.0"):
        TrainingConfig(learning_rate=0.0)

    # Invalid test size
    with pytest.raises(ValueError, match="test_size must be in"):
        TrainingConfig(test_size=-0.1)
    with pytest.raises(ValueError, match="test_size must be in"):
        TrainingConfig(test_size=1.0)
