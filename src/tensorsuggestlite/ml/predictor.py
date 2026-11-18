"""Inference engine for trained Keras neural classification models.

Provides KerasPredictor for full TensorFlow execution and batch evaluation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from tensorsuggestlite.core.entities import PredictionResult, PredictionScore
from tensorsuggestlite.core.exceptions import ModelTrainingError


class KerasPredictor:
    """Performs real-time text classification using the full Keras neural model."""

    def __init__(self, model_dir: Path | str) -> None:
        """Initialize predictor and load serialized model and tokenizers.

        Args:
            model_dir: Directory containing text_classifier.keras and tokenizer.json.

        Raises:
            ModelTrainingError: If required model files or dependencies are missing.
        """
        self.dir_path = Path(model_dir).resolve()
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load and deserialize Keras model, tokenizer, and label encoder."""
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
        """Classify input text and compute confidence probabilities across all classes.

        Args:
            text: Input sentence or utterance to classify.

        Returns:
            PredictionResult: Complete inference report with latency and class probabilities.
        """
        import tensorflow as tf

        start_time = time.perf_counter()

        sequences = self.tokenizer.texts_to_sequences([text])
        padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding="post")

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

    def predict_batch(self, texts: list[str]) -> list[PredictionResult]:
        """Classify a batch of texts using vectorized Keras inference for high throughput.

        Args:
            texts: List of text sentences to classify.

        Returns:
            list[PredictionResult]: List of inference reports in input order.
        """
        if not texts:
            return []

        import tensorflow as tf

        start_time = time.perf_counter()

        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding="post")

        raw_preds = self.model.predict(padded, verbose=0)
        total_latency_ms = (time.perf_counter() - start_time) * 1000.0
        avg_latency_ms = total_latency_ms / len(texts)

        results: list[PredictionResult] = []
        for text, preds in zip(texts, raw_preds, strict=False):
            scores: list[PredictionScore] = []
            for idx, prob in enumerate(preds):
                label = self.idx_to_label.get(idx, f"Class_{idx}")
                scores.append(PredictionScore(category=label, confidence=float(prob)))

            scores.sort(key=lambda s: s.confidence, reverse=True)
            top = scores[0] if scores else PredictionScore(category="Unknown", confidence=0.0)

            results.append(
                PredictionResult(
                    text=text,
                    top_category=top.category,
                    top_confidence=top.confidence,
                    scores=scores,
                    latency_ms=avg_latency_ms,
                    runtime="keras",
                )
            )

        return results


__all__ = ["KerasPredictor"]
