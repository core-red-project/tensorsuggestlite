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


def test_batch_inference_keras_and_tflite(sample_dataset: TrainingDataset, tmp_path: Path):
    """Verify batch inference execution across both Keras and TFLite runtimes."""
    trainer = KerasTextTrainer()
    trainer.train(sample_dataset, output_dir=tmp_path, config=TrainingConfig(epochs=2))

    exporter = ExportTFLiteUseCase()
    exporter.execute(model_dir=tmp_path, quantization="dynamic")

    use_case = PredictTextUseCase(tmp_path)
    test_batch = [
        "This product is really amazing and works well",
        "Terrible experience and broken customer support",
        "Absolutely love using this tool everyday",
    ]

    # Keras batch inference
    keras_batch_res = use_case.predict_batch(test_batch, prefer_tflite=False)
    assert len(keras_batch_res) == len(test_batch)
    for res in keras_batch_res:
        assert res.runtime == "keras"
        assert res.top_category in {"POSITIVE", "NEGATIVE"}
        assert 0.0 <= res.top_confidence <= 1.0

    # TFLite batch inference
    tflite_batch_res = use_case.predict_batch(test_batch, prefer_tflite=True)
    assert len(tflite_batch_res) == len(test_batch)
    for res in tflite_batch_res:
        assert res.runtime == "tflite"
        assert res.top_category in {"POSITIVE", "NEGATIVE"}
        assert 0.0 <= res.top_confidence <= 1.0


def test_configurable_quantization_modes(sample_dataset: TrainingDataset, tmp_path: Path):
    """Verify that dynamic, float16, and none quantization export valid TFLite binaries."""
    trainer = KerasTextTrainer()
    trainer.train(sample_dataset, output_dir=tmp_path, config=TrainingConfig(epochs=2))

    exporter = ExportTFLiteUseCase()

    for q_mode in ["dynamic", "float16", "none"]:
        target_path = tmp_path / f"model_{q_mode}.tflite"
        dest = exporter.execute(model_dir=tmp_path, output_path=target_path, quantization=q_mode)
        assert dest.is_file()
        assert dest.stat().st_size > 0

    # Verify invalid quantization fails fast
    import pytest

    from tensorsuggestlite.core.exceptions import ModelExportError

    with pytest.raises(ModelExportError, match="Unsupported quantization"):
        exporter.execute(model_dir=tmp_path, quantization="unsupported_mode")


def test_extended_metrics_and_metadata(sample_dataset: TrainingDataset, tmp_path: Path):
    """Verify extended metrics (training time, macro f1, per-class metrics, confusion matrix) in metadata."""
    import json

    trainer = KerasTextTrainer()
    trainer.train(sample_dataset, output_dir=tmp_path, config=TrainingConfig(epochs=2))

    metadata_path = tmp_path / "model_metadata.json"
    assert metadata_path.is_file()

    with open(metadata_path, encoding="utf-8") as f:
        meta = json.load(f)

    assert "version" in meta
    metrics = meta.get("metrics", {})
    assert "training_duration_seconds" in metrics
    assert metrics["training_duration_seconds"] >= 0.0
    assert "macro_avg_f1" in metrics
    assert "weighted_avg_f1" in metrics
    assert "total_parameters" in metrics

    assert "confusion_matrix" in meta
    assert "matrix" in meta["confusion_matrix"]
    assert len(meta["confusion_matrix"]["matrix"]) == 2

    assert "per_class_metrics" in meta
    for cat in ["POSITIVE", "NEGATIVE"]:
        assert cat in meta["per_class_metrics"]
        assert "precision" in meta["per_class_metrics"][cat]
        assert "recall" in meta["per_class_metrics"][cat]
        assert "f1_score" in meta["per_class_metrics"][cat]
        assert "support" in meta["per_class_metrics"][cat]


def test_training_requires_at_least_two_classes(tmp_path: Path):
    """Verify that training fails gracefully when dataset has fewer than 2 distinct classes."""
    import pytest

    from tensorsuggestlite.core.entities import TrainingSample
    from tensorsuggestlite.core.exceptions import InvalidDatasetError

    single_class_dataset = TrainingDataset(
        samples=[
            TrainingSample(text="first positive sentence", category="ONLY_ONE"),
            TrainingSample(text="second positive sentence", category="ONLY_ONE"),
        ]
    )

    trainer = KerasTextTrainer()
    with pytest.raises(InvalidDatasetError, match="requires at least 2 distinct categories"):
        trainer.train(single_class_dataset, output_dir=tmp_path)


def test_model_actually_learns_distinct_classes(tmp_path: Path):
    """Verify that model actually learns distinct classes with high accuracy and confidence."""
    from tensorsuggestlite.core.entities import TrainingSample

    greetings = [
        "hello there",
        "hi friend",
        "good morning to you",
        "hey you",
        "greetings to everyone",
        "hello world",
        "good day everyone",
        "hi there friend",
        "hey buddy",
        "welcome back friend",
    ]
    farewells = [
        "goodbye my friend",
        "bye see you later",
        "farewell for now",
        "have a nice day",
        "bye bye take care",
        "see you tomorrow",
        "take care and goodbye",
        "so long friend",
        "farewell everyone",
        "bye take care",
    ]

    samples = [TrainingSample(text=t, category="GREETING") for t in greetings] + [
        TrainingSample(text=t, category="FAREWELL") for t in farewells
    ]
    dataset = TrainingDataset(samples=samples)

    trainer = KerasTextTrainer()
    trainer.train(
        dataset,
        output_dir=tmp_path,
        config=TrainingConfig(epochs=25, batch_size=4, learning_rate=0.01),
    )

    exporter = ExportTFLiteUseCase()
    exporter.execute(model_dir=tmp_path)

    use_case = PredictTextUseCase(tmp_path)

    # 1. Test Keras prediction on representative utterances
    res_greeting = use_case.predict_keras("hello there")
    assert res_greeting.top_category == "GREETING"
    assert res_greeting.top_confidence > 0.6

    res_farewell = use_case.predict_keras("goodbye my friend")
    assert res_farewell.top_category == "FAREWELL"
    assert res_farewell.top_confidence > 0.6

    # 2. Test TFLite prediction on representative utterances
    res_greeting_tf = use_case.predict_tflite("hello there")
    assert res_greeting_tf.top_category == "GREETING"
    assert res_greeting_tf.top_confidence > 0.6

    res_farewell_tf = use_case.predict_tflite("goodbye my friend")
    assert res_farewell_tf.top_category == "FAREWELL"
    assert res_farewell_tf.top_confidence > 0.6
