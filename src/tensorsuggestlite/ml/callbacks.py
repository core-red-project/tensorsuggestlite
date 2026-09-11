"""Callbacks for ML training progress and logging."""

from typing import Any

from tensorsuggestlite.core.interfaces import LogCallback, ProgressCallback


def create_keras_epoch_callback(
    epochs: int,
    progress_cb: ProgressCallback | None = None,
    log_cb: LogCallback | None = None,
    start_pct: int = 20,
    end_pct: int = 85,
) -> Any:
    """Create a Keras callback that updates progress and logs safely."""
    import tensorflow as tf

    class EpochProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
            if progress_cb:
                try:
                    pct = start_pct + int(((epoch + 1) / float(epochs)) * (end_pct - start_pct))
                    progress_cb(max(0, min(100, pct)))
                except Exception:
                    pass

            if log_cb:
                try:
                    loss = logs.get("loss", 0.0) if logs else 0.0
                    acc = logs.get("accuracy", 0.0) if logs else 0.0
                    log_cb(f"Epoch {epoch + 1}/{epochs} - loss: {loss:.4f} - accuracy: {acc:.4f}")
                except Exception:
                    pass

    return EpochProgressCallback()
