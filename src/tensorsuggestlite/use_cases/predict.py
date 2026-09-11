"""Use case for text classification prediction."""

from pathlib import Path

from tensorsuggestlite.core.entities import CrossValidationResult, PredictionResult
from tensorsuggestlite.ml.predictor import KerasPredictor, TFLitePredictor


class PredictTextUseCase:
    """Orchestrates model loading, text inference, and cross-runtime comparison."""

    def __init__(self, model_dir: Path | str) -> None:
        self.model_dir = Path(model_dir).resolve()
        self._keras_predictor: KerasPredictor | None = None
        self._tflite_predictor: TFLitePredictor | None = None

    def predict_keras(self, text: str) -> PredictionResult:
        """Predict using full Keras model."""
        if self._keras_predictor is None:
            self._keras_predictor = KerasPredictor(self.model_dir)
        return self._keras_predictor.predict(text)

    def predict_tflite(self, text: str) -> PredictionResult:
        """Predict using quantized TFLite binary."""
        if self._tflite_predictor is None:
            self._tflite_predictor = TFLitePredictor(self.model_dir)
        return self._tflite_predictor.predict(text)

    def predict(self, text: str, prefer_tflite: bool = False) -> PredictionResult:
        """Predict using preferred runtime, falling back to Keras if TFLite not yet generated."""
        tflite_file = self.model_dir / "text_classifier.tflite"
        if prefer_tflite and tflite_file.is_file():
            return self.predict_tflite(text)
        return self.predict_keras(text)

    def compare(self, text: str) -> CrossValidationResult:
        """Execute side-by-side inference on both Keras and TFLite to evaluate parity."""
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
