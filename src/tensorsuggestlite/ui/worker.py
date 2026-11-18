# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Asynchronous background worker executing ML tasks off the main GUI thread."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable
from typing import Any

from tensorsuggestlite.core.logging import get_logger

logger = get_logger()


class TaskWorker(threading.Thread):
    """Executes long-running machine learning or export tasks in a dedicated thread.

    Configures an expanded stack size (8MB) to safeguard native math libraries
    (OpenBLAS, TensorFlow C-API) against thread stack exhaustion on POSIX environments.
    """

    def __init__(
        self,
        task_fn: Callable[..., Any],
        *args: Any,
        on_progress: Callable[[int], None] | None = None,
        on_log: Callable[[str], None] | None = None,
        on_success: Callable[[Any], None] | None = None,
        on_failed: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the background worker.

        Args:
            task_fn: Callable function or method to execute off the GUI thread.
            *args: Positional arguments forwarded to task_fn.
            on_progress: Optional callback invoked when progress percentage updates.
            on_log: Optional callback invoked when diagnostic logs are emitted.
            on_success: Optional callback invoked when task completes successfully.
            on_failed: Optional callback invoked when task raises an exception.
            **kwargs: Keyword arguments forwarded to task_fn.
        """
        super().__init__(daemon=True)
        self._task_fn = task_fn
        self._args = args
        self._kwargs = kwargs
        self.on_progress = on_progress
        self.on_log = on_log
        self.on_success = on_success
        self.on_failed = on_failed
        self._stack_size = 8 * 1024 * 1024
        with contextlib.suppress(ValueError, OSError):
            threading.stack_size(self._stack_size)

    def stack_size(self) -> int:
        """Return configured thread stack size in bytes."""
        return self._stack_size

    def run(self) -> None:
        """Execute the target task, forwarding callbacks and catching runtime errors."""
        try:
            task_kwargs = dict(self._kwargs)
            if self.on_progress and "progress_cb" not in task_kwargs:
                task_kwargs["progress_cb"] = self.on_progress
            if self.on_log and "log_cb" not in task_kwargs:
                task_kwargs["log_cb"] = self.on_log

            result = self._task_fn(*self._args, **task_kwargs)
            if self.on_success:
                self.on_success(result)
        except Exception as err:
            logger.exception("TaskWorker background execution failed: %s", err)
            if self.on_failed:
                self.on_failed(str(err))
