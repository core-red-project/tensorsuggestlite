"""Tests for Zero-TensorFlow lightweight inference runner."""

from pathlib import Path

from tensorsuggestlite.core.entities import TrainingConfig, TrainingDataset
from tensorsuggestlite.ml.lite_runtime import LiteInferenceRunner, PureTextTokenizer
from tensorsuggestlite.use_cases import ExportTFLiteUseCase, TrainModelUseCase


def test_pure_text_tokenizer(tmp_path: Path, sample_dataset: TrainingDataset) -> None:
    train_use_case = TrainModelUseCase()
    cfg = TrainingConfig(epochs=4, batch_size=4)
    result = train_use_case.trainer.train(sample_dataset, output_dir=tmp_path, config=cfg)

    tok_path = result.artifacts.tokenizer_path
    tokenizer = PureTextTokenizer(tok_path)

    seq = tokenizer.text_to_sequence("great awesome unknownword")
    assert len(seq) == 3
    padded = tokenizer.pad_sequence(seq, max_len=10)
    assert padded.shape == (1, 10)
    assert padded.max() >= 1.0
    assert padded[0, 0] > 0


def test_lite_inference_runner(tmp_path: Path, sample_dataset: TrainingDataset) -> None:
    train_use_case = TrainModelUseCase()
    export_use_case = ExportTFLiteUseCase()

    cfg = TrainingConfig(epochs=4, batch_size=4)
    train_use_case.trainer.train(sample_dataset, output_dir=tmp_path, config=cfg)
    export_use_case.execute(tmp_path)

    runner = LiteInferenceRunner(tmp_path)
    res = runner.predict("I really love this product!")

    assert res.top_category in {"POSITIVE", "NEGATIVE"}
    assert 0.0 <= res.top_confidence <= 1.0
    assert len(res.scores) == 2
    assert res.runtime == "tflite"
    assert res.latency_ms > 0.0
