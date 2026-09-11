"""Theme loader and font management for PyQt6 interface."""

import sys
from pathlib import Path

from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtWidgets import QApplication, QWidget


def get_ui_font(size: int = 10, bold: bool = False) -> QFont:
    """Resolve cross-platform native UI font to eliminate font aliasing overhead."""
    family = (
        ".AppleSystemUIFont"
        if sys.platform == "darwin"
        else ("Segoe UI" if sys.platform == "win32" else "Helvetica")
    )
    font = QFont(family, size)
    if bold:
        font.setWeight(QFont.Weight.Bold)
    return font


def get_mono_font(size: int = 10) -> QFont:
    """Resolve cross-platform native monospace font."""
    family = (
        "Menlo"
        if sys.platform == "darwin"
        else ("Consolas" if sys.platform == "win32" else "Courier")
    )
    return QFont(family, size)


def get_styles_dir() -> Path:
    """Resolve styles directory location."""
    root_candidate = Path(__file__).resolve().parent.parent.parent.parent / "styles"
    if root_candidate.is_dir():
        return root_candidate

    pkg_candidate = Path(__file__).resolve().parent.parent / "styles"
    if pkg_candidate.is_dir():
        return pkg_candidate

    return root_candidate


def get_icons_dir() -> Path:
    """Resolve icons directory location."""
    if hasattr(sys, "_MEIPASS"):
        meipass_candidate = Path(sys._MEIPASS) / "icons"
        if meipass_candidate.is_dir():
            return meipass_candidate

    root_candidate = Path(__file__).resolve().parent.parent.parent.parent / "icons"
    if root_candidate.is_dir():
        return root_candidate

    pkg_candidate = Path(__file__).resolve().parent.parent / "icons"
    if pkg_candidate.is_dir():
        return pkg_candidate

    return root_candidate


def get_icon(name: str) -> QIcon:
    """Load QIcon by filename from icons directory."""
    path = get_icons_dir() / name
    if path.is_file():
        return QIcon(str(path))
    return QIcon()


def apply_theme(target: QWidget | QApplication, theme_name: str = "dark") -> bool:
    """Apply QSS stylesheet (dark or light) to a widget or application."""
    styles_dir = get_styles_dir()
    qss_file = styles_dir / f"{theme_name}.qss"

    if not qss_file.is_file():
        return False

    try:
        with open(qss_file, encoding="utf-8") as f:
            stylesheet = f.read()
        target.setStyleSheet(stylesheet)
        return True
    except Exception:
        return False
