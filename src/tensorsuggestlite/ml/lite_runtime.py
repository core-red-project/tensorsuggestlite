"""Zero-TensorFlow lightweight inference runtime.

Provides sub-2ms text classification inference using only numpy and a lightweight
TFLite interpreter (ai-edge-litert or tflite-runtime), without requiring the ~500 MB
TensorFlow package.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from tensorsuggestlite.core.entities import PredictionResult, PredictionScore
from tensorsuggestlite.core.exceptions import InferenceError


def load_interpreter(model_path: Path | str) -> Any:
    """Load TFLite interpreter using the lightest available runtime."""
    # 1. Try Google's modern ai_edge_litert
    try:
        from ai_edge_litert.interpreter import Interpreter

        return Interpreter(model_path=str(model_path))
    except ImportError:
        pass

    # 2. Try lightweight tflite_runtime (~5 MB)
    try:
        from tflite_runtime.interpreter import Interpreter

        return Interpreter(model_path=str(model_path))
    except ImportError:
        pass

    # 3. Fallback to tensorflow lite interpreter with deprecation suppression
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=".*Interpreter is deprecated.*",
            )
            from tensorflow.lite.python.interpreter import Interpreter

            return Interpreter(model_path=str(model_path))
    except ImportError as err:
        raise InferenceError(
            "No compatible TFLite runtime found. Please install 'tflite-runtime', 'ai-edge-litert', or 'tensorflow'."
        ) from err


class PureTextTokenizer:
    """Pure Python text tokenizer matching Keras Tokenizer behavior without TensorFlow."""

    def __init__(self, tokenizer_config_path: Path | str) -> None:
        with open(tokenizer_config_path, encoding="utf-8") as f:
            raw = json.load(f)

        cfg = raw.get("config", {})
        wi = cfg.get("word_index", {})
        if isinstance(wi, str):
            wi = json.loads(wi)
        self.word_index: dict[str, int] = {str(k): int(v) for k, v in wi.items()}
        self.oov_token: str = cfg.get("oov_token", "<OOV>")
        self.oov_index: int = self.word_index.get(self.oov_token, 1)
        self.filters: str = cfg.get("filters", '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
        self.lower: bool = cfg.get("lower", True)
        self._filter_regex = re.compile(f"[{re.escape(self.filters)}]")

    def text_to_sequence(self, text: str) -> list[int]:
        """Convert raw text into list of integer token IDs."""
        if self.lower:
            text = text.lower()
        cleaned = self._filter_regex.sub(" ", text)
        words = cleaned.split()
        return [self.word_index.get(w, self.oov_index) for w in words]

    def pad_sequence(self, sequence: list[int], max_len: int) -> np.ndarray:
        """Pad sequence with post-zeros."""
        arr = np.zeros((1, max_len), dtype=np.float32)
        take = min(len(sequence), max_len)
        if take > 0:
            arr[0, :take] = sequence[:take]
        return arr


class LiteInferenceRunner:
    """Zero-TensorFlow lightweight inference engine."""

    def __init__(self, model_dir: Path | str) -> None:
        self.model_dir = Path(model_dir).resolve()
        tflite_path = self.model_dir / "text_classifier.tflite"
        tok_path = self.model_dir / "tokenizer.json"
        le_path = self.model_dir / "label_encoder.json"

        if not tflite_path.is_file():
            raise InferenceError(f"TFLite model not found at {tflite_path}")
        if not tok_path.is_file():
            raise InferenceError(f"Tokenizer not found at {tok_path}")
        if not le_path.is_file():
            raise InferenceError(f"Label encoder not found at {le_path}")

        self.tokenizer = PureTextTokenizer(tok_path)
        with open(le_path, encoding="utf-8") as f:
            self.label_encoder: dict[str, int] = json.load(f)
        self.inv_encoder = {idx: label for label, idx in self.label_encoder.items()}

        self.interpreter = load_interpreter(tflite_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]["shape"]
        self.max_len = int(self.input_shape[1])

    def predict(self, text: str) -> PredictionResult:
        """Run sub-millisecond classification inference."""
        t0 = time.perf_counter()
        seq = self.tokenizer.text_to_sequence(text)
        padded = self.tokenizer.pad_sequence(seq, self.max_len)

        self.interpreter.set_tensor(self.input_details[0]["index"], padded)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])[0]

        latency_ms = (time.perf_counter() - t0) * 1000.0
        scores: list[PredictionScore] = []
        for idx, conf in enumerate(output_data):
            label = self.inv_encoder.get(idx, f"CLASS_{idx}")
            scores.append(PredictionScore(category=label, confidence=float(conf)))

        scores.sort(key=lambda s: s.confidence, reverse=True)
        top = scores[0] if scores else PredictionScore("UNKNOWN", 0.0)

        return PredictionResult(
            text=text,
            top_category=top.category,
            top_confidence=top.confidence,
            scores=scores,
            latency_ms=latency_ms,
            runtime="tflite",
        )


__all__ = [
    "LiteInferenceRunner",
    "PureTextTokenizer",
    "load_interpreter",
]
