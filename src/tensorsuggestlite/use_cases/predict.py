"""Use case for text classification prediction and cross-runtime validation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tensorsuggestlite.core.entities import CrossValidationResult, PredictionResult
from tensorsuggestlite.ml.lite_runtime import LiteInferenceRunner

if TYPE_CHECKING:
    from tensorsuggestlite.ml.predictor import KerasPredictor


class PredictTextUseCase:
    """Orchestrates model loading, real-time inference, and cross-runtime comparison."""

    def __init__(self, model_dir: Path | str) -> None:
        """Initialize the prediction use case with a model directory.

        Args:
            model_dir: Directory containing trained model artifacts and tokenizers.

        Raises:
            FileNotFoundError: If model_dir does not exist or is not a directory.
        """
        self.model_dir = Path(model_dir).resolve()
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        self._keras_predictor: KerasPredictor | None = None
        self._lite_runner: LiteInferenceRunner | None = None

    def predict_keras(self, text: str) -> PredictionResult:
        """Predict class probabilities using the full Keras neural model.

        Args:
            text: Utterance to classify.

        Returns:
            PredictionResult: Classification result containing winning class and confidence scores.
        """
        if self._keras_predictor is None:
            from tensorsuggestlite.ml.predictor import KerasPredictor

            self._keras_predictor = KerasPredictor(self.model_dir)
        return self._keras_predictor.predict(text)

    def predict_tflite(self, text: str) -> PredictionResult:
        """Predict class probabilities using the optimized TFLite binary with zero-TensorFlow runtime.

        Args:
            text: Utterance to classify.

        Returns:
            PredictionResult: Classification result containing winning class and confidence scores.
        """
        if self._lite_runner is None:
            self._lite_runner = LiteInferenceRunner(self.model_dir)
        return self._lite_runner.predict(text)

    def predict(self, text: str, prefer_tflite: bool = False) -> PredictionResult:
        """Classify input text using the preferred runtime with automatic fallback.

        Args:
            text: Input utterance to classify.
            prefer_tflite: If True and .tflite model exists, use TFLite; otherwise use Keras.

        Returns:
            PredictionResult: Winning class and full probability distribution.
        """
        tflite_file = self.model_dir / "text_classifier.tflite"
        keras_file = self.model_dir / "text_classifier.keras"

        if (prefer_tflite or not keras_file.is_file()) and tflite_file.is_file():
            return self.predict_tflite(text)
        return self.predict_keras(text)

    def predict_batch(
        self, texts: list[str], prefer_tflite: bool = False
    ) -> list[PredictionResult]:
        """Classify a batch of texts using the preferred runtime with automatic fallback.

        Args:
            texts: List of input sentences to classify.
            prefer_tflite: If True and .tflite model exists, use TFLite; otherwise use Keras.

        Returns:
            list[PredictionResult]: Inferences corresponding to each input text.
        """
        if not texts:
            return []

        tflite_file = self.model_dir / "text_classifier.tflite"
        if prefer_tflite and tflite_file.is_file():
            if self._lite_runner is None:
                self._lite_runner = LiteInferenceRunner(self.model_dir)
            return [self._lite_runner.predict(t) for t in texts]

        if self._keras_predictor is None:
            from tensorsuggestlite.ml.predictor import KerasPredictor

            self._keras_predictor = KerasPredictor(self.model_dir)
        return self._keras_predictor.predict_batch(texts)

    def compare(self, text: str) -> CrossValidationResult:
        """Execute side-by-side inference on both Keras and TFLite to evaluate numerical parity.

        Args:
            text: Test utterance to evaluate across runtimes.

        Returns:
            CrossValidationResult: Comparison report measuring decision consistency and confidence deltas.
        """
        k_res = self.predict_keras(text)
        t_res = self.predict_tflite(text)

        k_map = {s.category: s.confidence for s in k_res.scores}
        t_map = {s.category: s.confidence for s in t_res.scores}
        all_cats = set(k_map.keys()) | set(t_map.keys())

        max_delta = 0.0
        if all_cats:
            max_delta = max(abs(k_map.get(c, 0.0) - t_map.get(c, 0.0)) for c in all_cats)

        return CrossValidationResult(
            text=text,
            keras_result=k_res,
            tflite_result=t_res,
            is_consistent=(k_res.top_category == t_res.top_category),
            delta_confidence=abs(k_res.top_confidence - t_res.top_confidence),
            max_class_delta=max_delta,
        )


__all__ = ["PredictTextUseCase"]
