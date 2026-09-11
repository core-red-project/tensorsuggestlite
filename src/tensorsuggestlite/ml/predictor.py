"""Inference engines for Keras and TFLite models."""

import json
import time
from pathlib import Path

from tensorsuggestlite.core.entities import PredictionResult, PredictionScore
from tensorsuggestlite.core.exceptions import ModelTrainingError


class KerasPredictor:
    """Performs real-time text classification using the full Keras model."""

    def __init__(self, model_dir: Path | str) -> None:
        self.dir_path = Path(model_dir).resolve()
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        keras_path = self.dir_path / "text_classifier.keras"
        tokenizer_path = self.dir_path / "tokenizer.json"
        encoder_path = self.dir_path / "label_encoder.json"

        if not keras_path.is_file():
            raise ModelTrainingError(f"Keras model not found at {keras_path}")
        if not tokenizer_path.is_file():
            raise ModelTrainingError(f"Tokenizer not found at {tokenizer_path}")
        if not encoder_path.is_file():
            raise ModelTrainingError(f"Label encoder not found at {encoder_path}")

        try:
            import tensorflow as tf
        except ImportError as err:
            raise ModelTrainingError(f"TensorFlow is required for inference: {err}") from err

        with open(tokenizer_path, encoding="utf-8") as f:
            self.tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(f.read())

        with open(encoder_path, encoding="utf-8") as f:
            encoder_dict = json.load(f)
            # Invert mapping: {idx: label}
            self.idx_to_label = {v: k for k, v in encoder_dict.items()}

        self.model = tf.keras.models.load_model(keras_path)

    def predict(self, text: str) -> PredictionResult:
        """Classify input text and return probabilities."""
        import tensorflow as tf

        start_time = time.perf_counter()

        sequences = self.tokenizer.texts_to_sequences([text])
        padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding="post")

        if len(self.tokenizer.word_index) > 0:
            padded = padded / float(len(self.tokenizer.word_index))

        raw_preds = self.model.predict(padded, verbose=0)[0]
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        scores: list[PredictionScore] = []
        for idx, prob in enumerate(raw_preds):
            label = self.idx_to_label.get(idx, f"Class_{idx}")
            scores.append(PredictionScore(category=label, confidence=float(prob)))

        # Sort descending by confidence
        scores.sort(key=lambda s: s.confidence, reverse=True)
        top = scores[0]

        return PredictionResult(
            text=text,
            top_category=top.category,
            top_confidence=top.confidence,
            scores=scores,
            latency_ms=latency_ms,
            runtime="keras",
        )


class TFLitePredictor:
    """Performs lightweight text classification using the quantized TFLite binary."""

    def __init__(self, model_dir: Path | str) -> None:
        self.dir_path = Path(model_dir).resolve()
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        tflite_path = self.dir_path / "text_classifier.tflite"
        tokenizer_path = self.dir_path / "tokenizer.json"
        encoder_path = self.dir_path / "label_encoder.json"

        if not tflite_path.is_file():
            raise ModelTrainingError(f"TFLite binary not found at {tflite_path}")
        if not tokenizer_path.is_file():
            raise ModelTrainingError(f"Tokenizer not found at {tokenizer_path}")
        if not encoder_path.is_file():
            raise ModelTrainingError(f"Label encoder not found at {encoder_path}")

        try:
            import tensorflow as tf
        except ImportError as err:
            raise ModelTrainingError(f"TensorFlow runtime is required: {err}") from err

        with open(tokenizer_path, encoding="utf-8") as f:
            self.tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(f.read())

        with open(encoder_path, encoding="utf-8") as f:
            encoder_dict = json.load(f)
            self.idx_to_label = {v: k for k, v in encoder_dict.items()}

        self.interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, text: str) -> PredictionResult:
        """Classify text using TFLite interpreter."""
        import numpy as np
        import tensorflow as tf

        start_time = time.perf_counter()

        sequences = self.tokenizer.texts_to_sequences([text])
        padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding="post")

        if len(self.tokenizer.word_index) > 0:
            padded = padded / float(len(self.tokenizer.word_index))

        input_data = np.array(padded, dtype=self.input_details[0]["dtype"])

        # Resize input tensor if sequence length differs
        if list(input_data.shape) != list(self.input_details[0]["shape"]):
            self.interpreter.resize_tensor_input(self.input_details[0]["index"], input_data.shape)
            self.interpreter.allocate_tensors()

        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        scores: list[PredictionScore] = []
        for idx, prob in enumerate(output_data):
            label = self.idx_to_label.get(idx, f"Class_{idx}")
            scores.append(PredictionScore(category=label, confidence=float(prob)))

        scores.sort(key=lambda s: s.confidence, reverse=True)
        top = scores[0]

        return PredictionResult(
            text=text,
            top_category=top.category,
            top_confidence=top.confidence,
            scores=scores,
            latency_ms=latency_ms,
            runtime="tflite",
        )
