"""Filesystem and XDG Base Directory path resolution for TensorSuggestLite.

Ensures standards-compliant storage of application data, models, and caches
aligned with XDG Base Directory Specification on Linux, AppData on Windows,
and Application Support on macOS.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path


def get_default_data_dir() -> Path:
    """Resolve the default data storage directory according to OS conventions.

    Falls back to user home or current directory if OS system directories
    are restricted by sandboxes or permission controls.

    Returns:
        Path: Dedicated application data directory for TensorSuggestLite.
    """
    candidates: list[Path] = []
    if sys.platform == "win32":
        base_str = os.environ.get("LOCALAPPDATA")
        if base_str:
            candidates.append(Path(base_str) / "tensorsuggestlite")
        candidates.append(Path.home() / "AppData" / "Local" / "tensorsuggestlite")
    elif sys.platform == "darwin":
        base_str = os.environ.get("XDG_DATA_HOME")
        if base_str:
            candidates.append(Path(base_str) / "tensorsuggestlite")
        candidates.append(Path.home() / "Library" / "Application Support" / "tensorsuggestlite")
    else:
        # POSIX / Linux XDG Base Directory Specification
        base_str = os.environ.get("XDG_DATA_HOME")
        if base_str:
            candidates.append(Path(base_str) / "tensorsuggestlite")
        candidates.append(Path.home() / ".local" / "share" / "tensorsuggestlite")

    # Safe fallbacks if OS directory is restricted
    candidates.append(Path.home() / ".tensorsuggestlite")
    candidates.append(Path.cwd() / ".tensorsuggestlite")

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            test_file = candidate / ".perm_check"
            test_file.touch(exist_ok=True)
            test_file.unlink(missing_ok=True)
            return candidate
        except (OSError, PermissionError):
            continue

    fallback = Path.cwd() / ".tensorsuggestlite"
    with contextlib.suppress(OSError):
        fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def get_default_models_dir() -> Path:
    """Resolve the directory designated for generated models and artifacts.

    Returns:
        Path: Directory housing trained models and TFLite binaries.
    """
    models_dir = get_default_data_dir() / "models"
    with contextlib.suppress(OSError):
        models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


__all__ = ["get_default_data_dir", "get_default_models_dir"]
