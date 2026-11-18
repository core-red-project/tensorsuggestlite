"""Tests for TaskWorker thread execution and stack configuration."""

from tensorsuggestlite.ui.worker import TaskWorker


def test_task_worker_execution_and_stack_size():
    """Verify that TaskWorker configures sufficient stack size and executes callbacks."""
    progress_recorded: list[int] = []
    logs_recorded: list[str] = []
    success_recorded: list[object] = []

    def dummy_task(progress_cb=None, log_cb=None):
        if progress_cb:
            progress_cb(50)
        if log_cb:
            log_cb("test-log")
        return "result-value"

    worker = TaskWorker(
        dummy_task,
        on_progress=progress_recorded.append,
        on_log=logs_recorded.append,
        on_success=success_recorded.append,
    )
    assert worker.stack_size() >= 8 * 1024 * 1024

    worker.run()
    assert progress_recorded == [50]
    assert logs_recorded == ["test-log"]
    assert success_recorded == ["result-value"]


def test_task_worker_failure_callback():
    """Verify that exceptions in task_fn are forwarded to on_failed."""
    errors_recorded: list[str] = []

    def failing_task(progress_cb=None, log_cb=None):
        raise ValueError("simulated background error")

    worker = TaskWorker(failing_task, on_failed=errors_recorded.append)
    worker.run()
    assert len(errors_recorded) == 1
    assert "simulated background error" in errors_recorded[0]
