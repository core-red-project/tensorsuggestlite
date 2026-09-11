"""Asynchronous background worker for PyQt6 using QThread."""

from collections.abc import Callable
from typing import Any

from PyQt6.QtCore import QThread, pyqtSignal


class TaskWorker(QThread):
    """Background worker that runs tasks asynchronously and emits Qt signals."""

    progress = pyqtSignal(int)
    log_message = pyqtSignal(str)
    finished_success = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, task_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        # Set 8MB stack size to prevent OpenBLAS/NumPy/TensorFlow stack overflow (SIGBUS 138)
        # on secondary threads on macOS/Linux.
        self.setStackSize(8 * 1024 * 1024)
        self._task_fn = task_fn
        self._args = args
        self._kwargs = kwargs

    def run(self) -> None:
        try:
            # Inject progress and log callbacks into kwargs
            kwargs = dict(self._kwargs)
            kwargs["progress_cb"] = self.progress.emit
            kwargs["log_cb"] = self.log_message.emit

            result = self._task_fn(*self._args, **kwargs)
            self.finished_success.emit(result)
        except Exception as err:
            self.failed.emit(str(err))
