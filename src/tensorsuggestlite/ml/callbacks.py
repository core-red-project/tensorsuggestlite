"""Keras training callbacks reporting epoch progress and loss metrics."""

from __future__ import annotations

import contextlib
from typing import Any

from tensorsuggestlite.core.interfaces import LogCallback, ProgressCallback


def create_keras_epoch_callback(
    epochs: int,
    progress_cb: ProgressCallback | None = None,
    log_cb: LogCallback | None = None,
    start_pct: int = 20,
    end_pct: int = 85,
) -> Any:
    """Create a Keras training callback that safely emits progress percentages and epoch metrics.

    Args:
        epochs: Total number of planned training epochs.
        progress_cb: Optional callback receiving integer progress percentage [0-100].
        log_cb: Optional callback receiving formatted epoch summary strings.
        start_pct: Base progress percentage at which epoch training starts.
        end_pct: Ceiling progress percentage when all epochs complete.

    Returns:
        Any: Configured tf.keras.callbacks.Callback instance.
    """
    import tensorflow as tf

    class EpochProgressCallback(tf.keras.callbacks.Callback):
        """Internal callback intercepting epoch completions."""

        def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
            """Emit progress and loss metrics at the conclusion of each epoch.

            Args:
                epoch: Zero-indexed current epoch number.
                logs: Optional dictionary containing evaluation loss and accuracy metrics.
            """
            if progress_cb:
                with contextlib.suppress(Exception):
                    pct = start_pct + int(((epoch + 1) / float(epochs)) * (end_pct - start_pct))
                    progress_cb(max(0, min(100, pct)))

            if log_cb:
                with contextlib.suppress(Exception):
                    loss = logs.get("loss", 0.0) if logs else 0.0
                    acc = logs.get("accuracy", 0.0) if logs else 0.0
                    log_cb(f"Epoch {epoch + 1}/{epochs} - loss: {loss:.4f} - accuracy: {acc:.4f}")

    return EpochProgressCallback()


__all__ = ["create_keras_epoch_callback"]
