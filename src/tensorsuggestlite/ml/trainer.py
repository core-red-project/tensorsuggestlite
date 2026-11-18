"""Unified Keras text classification trainer with metadata and diagnostics."""

import contextlib
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from tensorsuggestlite import __version__
from tensorsuggestlite.core.entities import (
    ArtifactBundle,
    TrainingConfig,
    TrainingDataset,
    TrainingResult,
)
from tensorsuggestlite.core.exceptions import (
    EmptyDatasetError,
    InvalidDatasetError,
    ModelTrainingError,
)
from tensorsuggestlite.core.interfaces import LogCallback, ModelTrainerProtocol, ProgressCallback
from tensorsuggestlite.core.profiler import DatasetProfiler
from tensorsuggestlite.ml.callbacks import create_keras_epoch_callback


class KerasTextTrainer(ModelTrainerProtocol):
    """Encapsulates TensorFlow/Keras training, metrics computation, and metadata generation."""

    def train(
        self,
        dataset: TrainingDataset,
        output_dir: Path | str,
        config: TrainingConfig | None = None,
        progress_cb: ProgressCallback | None = None,
        log_cb: LogCallback | None = None,
    ) -> TrainingResult:
        """Train a Keras text classifier, evaluate metrics, and save serialized artifacts.

        Args:
            dataset: Training dataset containing text samples and category labels.
            output_dir: Destination directory where model, tokenizer, and metadata are saved.
            config: Hyperparameters governing training epochs, learning rate, and batch size.
            progress_cb: Optional callback receiving completion percentage (0-100).
            log_cb: Optional callback receiving real-time log messages.

        Returns:
            TrainingResult containing artifact paths, metrics, and dataset summaries.

        Raises:
            EmptyDatasetError: If the provided dataset contains zero samples.
            ModelTrainingError: If underlying ML dependencies cannot be loaded.
        """
        if len(dataset) == 0:
            raise EmptyDatasetError("Dataset contains no samples to train.")

        unique_labels = dataset.unique_categories
        if len(unique_labels) < 2:
            raise InvalidDatasetError(
                f"A text classification model requires at least 2 distinct categories, found {len(unique_labels)}: {unique_labels}"
            )

        cfg = config or TrainingConfig()
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        def report_log(msg: str) -> None:
            if log_cb:
                with contextlib.suppress(Exception):
                    log_cb(msg)

        def report_prog(val: int) -> None:
            if progress_cb:
                with contextlib.suppress(Exception):
                    progress_cb(max(0, min(100, val)))

        report_prog(0)
        report_log("Initializing TensorFlow training engine...")
        import time

        train_start_time = time.perf_counter()

        try:
            import numpy as np
            import tensorflow as tf
            from sklearn.metrics import classification_report, confusion_matrix
            from sklearn.model_selection import train_test_split
            from sklearn.utils import class_weight
        except ImportError as err:
            raise ModelTrainingError(f"Required ML libraries not available: {err}") from err

        texts = dataset.texts
        categories = dataset.categories
        unique_labels = dataset.unique_categories

        # Profile dataset
        profile = DatasetProfiler.profile(dataset)
        if profile.imbalance_warning:
            report_log(f"[WARN] {profile.imbalance_warning}")
        if profile.contradictory_samples:
            report_log(
                f"[WARN] Contradictory samples detected: {len(profile.contradictory_samples)} cases."
            )
        if profile.duplicate_samples:
            report_log(
                f"[INFO] Duplicate samples detected: {len(profile.duplicate_samples)} cases."
            )

        # Dataset sanity & hash
        dataset_content = "\n".join(f"{s.category}:{s.text}" for s in dataset.samples)
        dataset_sha256 = hashlib.sha256(dataset_content.encode("utf-8")).hexdigest()

        report_prog(5)
        report_log(f"Dataset ready: {len(texts)} samples across {len(unique_labels)} categories.")

        # Check for class counts
        counts = {cat: categories.count(cat) for cat in unique_labels}

        # 1. Fit Tokenizer
        tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<OOV>")
        tokenizer.fit_on_texts(texts)

        report_prog(10)
        report_log(f"Vocabulary index created ({len(tokenizer.word_index)} unique tokens).")

        sequences = tokenizer.texts_to_sequences(texts)
        padded = tf.keras.preprocessing.sequence.pad_sequences(
            sequences, maxlen=cfg.max_sequence_length, padding="post"
        )

        # 2. Encode Labels
        label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        encoded = np.array([label_encoder[c] for c in categories], dtype=np.int32)

        # 3. Train/Test Split
        test_size = cfg.test_size if len(texts) >= 10 else 0.0
        if test_size > 0:
            counts_per_class = np.bincount(encoded)
            can_stratify = bool(len(counts_per_class) >= 2 and np.min(counts_per_class) >= 2)
            strat = encoded if can_stratify else None
            X_train, X_test, y_train, y_test = train_test_split(
                padded, encoded, test_size=test_size, random_state=cfg.random_state, stratify=strat
            )
        else:
            X_train, y_train = padded, encoded
            X_test, y_test = padded, encoded

        report_prog(15)
        report_log(
            f"Building neural network with Dropout={cfg.dropout_rate}, LR={cfg.learning_rate}..."
        )

        # 4. Model Architecture with Regularization
        vocab_size = len(tokenizer.word_index) + 1
        num_classes = len(unique_labels)

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(int(padded.shape[1]),)),
                tf.keras.layers.Embedding(
                    input_dim=vocab_size, output_dim=cfg.embedding_dim, mask_zero=True
                ),
                tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Dense(cfg.hidden_dim, activation="relu"),
                tf.keras.layers.Dropout(cfg.dropout_rate),
                tf.keras.layers.Dense(num_classes, activation="softmax"),
            ]
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
        model.compile(
            optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )

        # Class weights
        cw_dict = None
        try:
            classes = np.unique(categories)
            cw = class_weight.compute_class_weight(
                class_weight="balanced", classes=classes, y=categories
            )
            cw_dict = {label_encoder[label]: float(w) for label, w in zip(classes, cw, strict=True)}
        except Exception:
            cw_dict = None

        epoch_cb = create_keras_epoch_callback(
            epochs=cfg.epochs,
            progress_cb=progress_cb,
            log_cb=log_cb,
            start_pct=20,
            end_pct=85,
        )

        report_log(f"Starting training for {cfg.epochs} epochs (batch_size={cfg.batch_size})...")
        history = model.fit(
            X_train,
            y_train,
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            validation_data=(X_test, y_test) if test_size > 0 else None,
            class_weight=cw_dict,
            callbacks=[epoch_cb],
            verbose=0,
        )

        report_prog(90)
        report_log("Computing evaluation metrics & classification diagnostics...")

        # 5. Evaluate Metrics
        eval_metrics: dict[str, float] = {}
        if history and history.history:
            eval_metrics["final_loss"] = float(history.history.get("loss", [0.0])[-1])
            eval_metrics["final_accuracy"] = float(history.history.get("accuracy", [0.0])[-1])

        # Predictions on evaluation set
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)

        eval_target_indices = sorted(set(y_test) | set(y_pred_classes))
        cls_report = classification_report(
            y_test,
            y_pred_classes,
            target_names=[unique_labels[i] for i in eval_target_indices],
            output_dict=True,
            zero_division=0,
        )

        acc_metric_name = "eval_accuracy" if test_size > 0 else "train_accuracy"
        if "accuracy" in cls_report:
            eval_metrics[acc_metric_name] = float(cls_report["accuracy"])

        # Extended metrics (Bloque 2.6)
        train_duration = time.perf_counter() - train_start_time
        eval_metrics["training_duration_seconds"] = round(train_duration, 2)
        eval_metrics["total_parameters"] = float(model.count_params())

        macro_avg = cls_report.get("macro avg", {})
        weighted_avg = cls_report.get("weighted avg", {})
        eval_metrics["macro_avg_f1"] = round(float(macro_avg.get("f1-score", 0.0)), 4)
        eval_metrics["macro_avg_precision"] = round(float(macro_avg.get("precision", 0.0)), 4)
        eval_metrics["macro_avg_recall"] = round(float(macro_avg.get("recall", 0.0)), 4)
        eval_metrics["weighted_avg_f1"] = round(float(weighted_avg.get("f1-score", 0.0)), 4)
        eval_metrics["weighted_avg_precision"] = round(float(weighted_avg.get("precision", 0.0)), 4)
        eval_metrics["weighted_avg_recall"] = round(float(weighted_avg.get("recall", 0.0)), 4)

        # Per-class metrics summary
        per_class_metrics: dict[str, dict[str, float]] = {}
        for cat in unique_labels:
            if cat in cls_report and isinstance(cls_report[cat], dict):
                per_class_metrics[cat] = {
                    "precision": round(float(cls_report[cat].get("precision", 0.0)), 4),
                    "recall": round(float(cls_report[cat].get("recall", 0.0)), 4),
                    "f1_score": round(float(cls_report[cat].get("f1-score", 0.0)), 4),
                    "support": float(cls_report[cat].get("support", 0)),
                }

        # Confusion Matrix
        cm = confusion_matrix(
            y_test,
            y_pred_classes,
            labels=[label_encoder[label] for label in unique_labels],
        )
        cm_dict = {
            "labels": unique_labels,
            "matrix": cm.tolist(),
        }

        # Log Confusion Matrix table
        report_log("Confusion Matrix (Rows=Actual, Cols=Pred):")
        col_header = " | ".join(f"{lbl[:8]:>8}" for lbl in unique_labels)
        report_log(f"  {'':>10} | {col_header}")
        for i, row in enumerate(cm):
            row_str = " | ".join(f"{val:>8}" for val in row)
            report_log(f"  {unique_labels[i][:10]:>10} | {row_str}")

        # 6. Save Artifacts & Metadata
        tokenizer_path = out_dir / "tokenizer.json"
        keras_model_path = out_dir / "text_classifier.keras"
        label_encoder_path = out_dir / "label_encoder.json"
        metadata_path = out_dir / "model_metadata.json"

        model.save(keras_model_path)

        with open(tokenizer_path, "w", encoding="utf-8") as f:
            f.write(tokenizer.to_json())

        # Invalidate any stale TFLite binary from previous training run
        stale_tflite = out_dir / "text_classifier.tflite"
        if stale_tflite.is_file():
            with contextlib.suppress(OSError):
                stale_tflite.unlink()

        with open(label_encoder_path, "w", encoding="utf-8") as f:
            json.dump(label_encoder, f, indent=2)

        # Telemetry
        keras_size_kb = (
            keras_model_path.stat().st_size / 1024.0 if keras_model_path.is_file() else 0.0
        )
        seq_len = int(padded.shape[1])
        tensor_telemetry = {
            "input_shape": [None, seq_len],
            "input_dtype": "float32",
            "output_shape": [None, num_classes],
            "output_dtype": "float32",
            "sequence_length": seq_len,
            "vocab_size": vocab_size,
            "classes": unique_labels,
            "keras_model_size_kb": round(keras_size_kb, 2),
        }

        metadata_dict = {
            "created_at": datetime.now(UTC).isoformat(),
            "version": __version__,
            "dataset": {
                "sha256": dataset_sha256,
                "total_samples": len(texts),
                "categories": counts,
            },
            "profiling": {
                "imbalance_ratio": round(profile.imbalance_ratio, 2),
                "imbalance_warning": profile.imbalance_warning,
                "duplicate_count": len(profile.duplicate_samples),
                "contradiction_count": len(profile.contradictory_samples),
            },
            "hyperparameters": {
                "epochs": cfg.epochs,
                "learning_rate": cfg.learning_rate,
                "dropout_rate": cfg.dropout_rate,
                "embedding_dim": cfg.embedding_dim,
                "hidden_dim": cfg.hidden_dim,
                "batch_size": cfg.batch_size,
                "max_sequence_length": cfg.max_sequence_length,
            },
            "tensors": tensor_telemetry,
            "metrics": eval_metrics,
            "per_class_metrics": per_class_metrics,
            "classification_report": cls_report,
            "confusion_matrix": cm_dict,
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, indent=2)

        report_prog(100)
        report_log(f"Training completed successfully. Artifacts saved to: {out_dir}")

        artifacts = ArtifactBundle(
            output_dir=out_dir,
            tokenizer_path=tokenizer_path,
            keras_model_path=keras_model_path,
            label_encoder_path=label_encoder_path,
            metadata_path=metadata_path,
        )

        return TrainingResult(
            model_path=str(keras_model_path),
            items_serialized=len(texts),
            labels_count=len(unique_labels),
            artifacts=artifacts,
            metrics=eval_metrics,
        )
