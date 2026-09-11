"""Tests for Qt TaskWorker thread stack configuration and task execution."""

import pytest


def test_task_worker_stack_size():
    """Verify that TaskWorker configures sufficient stack size for ML libraries."""
    try:
        from PyQt6.QtWidgets import QApplication

        from tensorsuggestlite.ui.worker import TaskWorker
    except ImportError:
        pytest.skip("PyQt6 not installed in environment")

    # Ensure QApplication exists in test context
    _app = QApplication.instance() or QApplication([])

    def dummy_task(progress_cb=None, log_cb=None):
        if progress_cb:
            progress_cb(50)
        if log_cb:
            log_cb("test")
        return "done"

    worker = TaskWorker(dummy_task)
    # Stack size must be explicitly set >= 8MB (8388608 bytes)
    assert worker.stackSize() >= 8 * 1024 * 1024
