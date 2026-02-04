import os
import json
from typing import Callable, Dict, Any, Optional


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def train_from_json(json_path: str, progress_cb: Optional[Callable[[int], None]] = None, log_cb: Optional[Callable[[str], None]] = None, epochs: int = 12) -> Dict[str, Any]:
    """Train a model from JSON with expected structure.

    Imports TensorFlow and scikit-learn at runtime to allow module import
    even if TensorFlow is not available.
    """
    try:
        import tensorflow as tf
        from sklearn.model_selection import train_test_split
        from sklearn.utils import class_weight
        import numpy as np
    except Exception as e:
        raise RuntimeError("TensorFlow and required dependencies are not available: %s" % e)

    class _EpochProgressCallback(tf.keras.callbacks.Callback):
        def __init__(self, epochs: int, progress_cb: Optional[Callable[[int], None]] = None, log_cb: Optional[Callable[[str], None]] = None, start: int = 20, end: int = 85):
            super().__init__()
            self.epochs = epochs
            self.progress_cb = progress_cb
            self.log_cb = log_cb
            # Range where epoch progress is mapped
            self.start = start
            self.end = end

        def on_epoch_end(self, epoch, logs=None):
            if not self.progress_cb:
                return
            try:
                pct = self.start + int(((epoch + 1) / float(self.epochs)) * (self.end - self.start))
                # clamp
                pct = max(0, min(100, pct))
                self.progress_cb(pct)
                # Descriptive epoch message
                if self.log_cb:
                    try:
                        self.log_cb(f'Epoch {epoch+1}/{self.epochs} completed ({pct}%).')
                    except Exception:
                        pass
            except Exception:
                pass

    # helpers for logging & progress
    def _safe_log(msg: str):
        if log_cb:
            try:
                log_cb(msg)
            except Exception:
                pass

    def _safe_progress(p: int):
        if progress_cb:
            try:
                progress_cb(p)
            except Exception:
                pass

    # Report start
    _safe_progress(0)
    _safe_log('Starting training process')

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    texts = []
    categories = []

    # Parse expected structure with English keys
    for category, content in data.items():
        # 'responses' expected as list of dicts with 'response' key
        for response in content.get('responses', []):
            if isinstance(response, dict) and 'response' in response:
                texts.append(response['response'])
                categories.append(category)
            elif isinstance(response, str):
                texts.append(response)
                categories.append(category)

        # 'synonyms' as list of strings
        for synonym in content.get('synonyms', []):
            texts.append(synonym)
            categories.append(category)

    if not texts:
        raise ValueError('No training data found in JSON file.')

    # Report progress after parsing
    _safe_progress(5)
    _safe_log(f'Parsed JSON: {len(texts)} entries, {len(set(categories))} categories detected')

    # Tokenizer and sequences
    tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)

    _safe_progress(10)
    _safe_log('Tokenizer trained on input texts')

    sequences = tokenizer.texts_to_sequences(texts)
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')

    # Normalize (optional)
    if len(tokenizer.word_index) > 0:
        padded = padded / float(len(tokenizer.word_index))

    # Encode categories
    unique_labels = list(sorted(set(categories)))
    label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
    encoded = [label_encoder[c] for c in categories]
    encoded = np.array(encoded, dtype=np.int32)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(padded, encoded, test_size=0.1, random_state=42)

    _safe_progress(15)
    _safe_log('Preprocessing complete. Preparing dataset and defining model.')

    # Simple model
    vocab_size = len(tokenizer.word_index) + 1
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=64, mask_zero=True),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(unique_labels), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Class weights (optional, for robustness)
    try:
        classes = np.unique(categories)
        cw = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=categories)
        class_weights_dict = {label_encoder[label]: float(w) for label, w in zip(classes, cw)}
    except Exception:
        class_weights_dict = None

    # Callbacks for progress
    cb = []
    if progress_cb or log_cb:
        cb.append(_EpochProgressCallback(epochs, progress_cb, log_cb=log_cb, start=20, end=85))
        try:
            # LambdaCallback as fallback to ensure calls per epoch and log
            def _lambda_on_epoch_end(epoch, logs, *, _epochs=epochs, _start=20, _end=85, _p=progress_cb, _l=log_cb):
                try:
                    pct = _start + int(((epoch + 1) / float(_epochs)) * (_end - _start))
                    pct = max(0, min(100, pct))
                    if _p:
                        try:
                            _p(pct)
                        except Exception:
                            pass
                    if _l:
                        try:
                            _l(f'Epoch {epoch+1}/{_epochs} completed ({pct}%).')
                        except Exception:
                            pass
                except Exception:
                    pass

            cb.append(tf.keras.callbacks.LambdaCallback(on_epoch_end=_lambda_on_epoch_end))
        except Exception:
            pass

    # Fit
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=32,
        class_weight=class_weights_dict,
        callbacks=cb
    )

    # Preparar carpeta de salida: generated/json
    root = _project_root()
    gen_dir = os.path.join(root, 'generated', 'json')
    os.makedirs(gen_dir, exist_ok=True)

    tokenizer_path = os.path.join(gen_dir, 'tokenizer.json')
    model_path = os.path.join(gen_dir, 'text_classifier.keras')
    label_path = os.path.join(gen_dir, 'label_encoder.json')

    # tokenizer
    tok_json = tokenizer.to_json()
    with open(tokenizer_path, 'w', encoding='utf-8') as f:
        f.write(tok_json)
    _safe_log(f'Tokenizer saved: {tokenizer_path}')

    # model
    model.save(model_path)
    _safe_log(f'Keras model saved: {model_path}')

    # label encoder
    with open(label_path, 'w', encoding='utf-8') as f:
        json.dump(label_encoder, f, ensure_ascii=False, indent=2)
    _safe_log(f'Label encoder saved: {label_path}')

    # Final progress report
    _safe_progress(95)

    items_serialized = len(texts)

    _safe_progress(100)

    return {
        'items_serialized': items_serialized,
        'tokenizer_path': tokenizer_path,
        'model_path': model_path,
        'label_encoder_path': label_path,
    }


def convert_to_tflite(model_dir: Optional[str] = None, progress_cb: Optional[Callable[[int], None]] = None, log_cb: Optional[Callable[[str], None]] = None) -> str:
    """Convert saved Keras model to TFLite.

    By default looks for `generated/json/text_classifier.keras`. If `model_dir`
    is provided, will look inside that folder.
    Returns path to TFLite file.
    """
    try:
        import tensorflow as tf
    except Exception as e:
        raise RuntimeError("TensorFlow is not available: %s" % e)

    def _safe_log(msg: str):
        if log_cb:
            try:
                log_cb(msg)
            except Exception:
                pass

    def _safe_progress(p: int):
        if progress_cb:
            try:
                progress_cb(int(max(0, min(100, int(p)))))
            except Exception:
                pass

    _safe_progress(5)
    _safe_log('Starting TFLite conversion')

    root = _project_root()
    if model_dir:
        model_path = os.path.join(model_dir, 'text_classifier.keras')
        out_dir = model_dir
    else:
        model_path = os.path.join(root, 'generated', 'json', 'text_classifier.keras')
        out_dir = os.path.join(root, 'generated', 'json')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Keras model not found: {model_path}")

    model = tf.keras.models.load_model(model_path)
    _safe_progress(25)
    _safe_log('Model loaded. Converting with TFLiteConverter')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    try:
        tflite_model = converter.convert()
    except Exception as e:
        _safe_log(f'Conversion error: {e}')
        raise RuntimeError(f"TFLite conversion failed: {e}") from e

    _safe_progress(90)
    _safe_log('Conversion complete. Saving TFLite file...')

    out_path = os.path.join(out_dir, 'text_classifier.tflite')
    try:
        with open(out_path, 'wb') as f:
            f.write(tflite_model)
    except OSError as e:
        raise OSError(f"Failed to write TFLite file to {out_path}: {e}") from e

    _safe_progress(100)
    _safe_log(f'TFLite saved: {out_path}')

    return out_path
