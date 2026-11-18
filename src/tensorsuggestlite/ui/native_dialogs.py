# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Cross-platform native OS file dialog bridge using ctypes and platform dispatchers.

Provides native file selection and directory picker dialogs (macOS Finder,
Windows Explorer, Linux portals) without external C++ compilers or Qt dependencies.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


class NativeDialogBridge:
    """Invokes host OS native file selection dialogs without Qt."""

    @staticmethod
    def open_file_dialog(
        title: str = "Select Dataset File",
        file_types: list[str] | None = None,
        initial_dir: Path | str | None = None,
    ) -> Path | None:
        """Open a native OS file dialog and return the selected path, or None if cancelled."""
        types = file_types or ["json", "yaml", "yml", "toml"]

        if sys.platform == "darwin":
            return NativeDialogBridge._open_file_macos(title, types, initial_dir)
        elif sys.platform == "win32":
            return NativeDialogBridge._open_file_windows(title, types, initial_dir)
        else:
            return NativeDialogBridge._open_file_linux(title, types, initial_dir)

    @staticmethod
    def open_directory_dialog(
        title: str = "Select Model Directory",
        initial_dir: Path | str | None = None,
    ) -> Path | None:
        """Open a native OS directory picker dialog and return the selected path, or None."""
        if sys.platform == "darwin":
            return NativeDialogBridge._open_dir_macos(title, initial_dir)
        elif sys.platform == "win32":
            return NativeDialogBridge._open_dir_windows(title, initial_dir)
        else:
            return NativeDialogBridge._open_dir_linux(title, initial_dir)

    @staticmethod
    def save_file_dialog(
        title: str = "Export TFLite Model",
        default_name: str = "text_classifier.tflite",
        initial_dir: Path | str | None = None,
    ) -> Path | None:
        """Open a native OS save file dialog and return the destination path, or None."""
        if sys.platform == "darwin":
            return NativeDialogBridge._save_file_macos(title, default_name, initial_dir)
        elif sys.platform == "win32":
            return NativeDialogBridge._save_file_windows(title, default_name, initial_dir)
        else:
            return NativeDialogBridge._save_file_linux(title, default_name, initial_dir)

    @staticmethod
    def _open_file_macos(
        title: str, file_types: list[str], initial_dir: Path | str | None
    ) -> Path | None:
        type_str = ", ".join(f'"{ext.lstrip(".")}"' for ext in file_types)
        init_clause = (
            f'default location POSIX file "{Path(initial_dir).resolve()}"' if initial_dir else ""
        )
        script = f'set f to POSIX path of (choose file with prompt "{title}" of type {{{type_str}}} {init_clause})'
        try:
            res = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                check=False,
            )
            out = res.stdout.strip()
            return Path(out) if out and res.returncode == 0 else None
        except Exception:
            return None

    @staticmethod
    def _open_dir_macos(title: str, initial_dir: Path | str | None) -> Path | None:
        init_clause = (
            f'default location POSIX file "{Path(initial_dir).resolve()}"' if initial_dir else ""
        )
        script = f'set f to POSIX path of (choose folder with prompt "{title}" {init_clause})'
        try:
            res = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                check=False,
            )
            out = res.stdout.strip()
            return Path(out) if out and res.returncode == 0 else None
        except Exception:
            return None

    @staticmethod
    def _save_file_macos(
        title: str, default_name: str, initial_dir: Path | str | None
    ) -> Path | None:
        init_clause = (
            f'default location POSIX file "{Path(initial_dir).resolve()}"' if initial_dir else ""
        )
        script = f'set f to POSIX path of (choose file name with prompt "{title}" default name "{default_name}" {init_clause})'
        try:
            res = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                check=False,
            )
            out = res.stdout.strip()
            return Path(out) if out and res.returncode == 0 else None
        except Exception:
            return None

    @staticmethod
    def _open_file_windows(
        title: str, file_types: list[str], initial_dir: Path | str | None
    ) -> Path | None:
        try:
            import ctypes
            from ctypes import wintypes

            class OPENFILENAME(ctypes.Structure):
                _fields_ = [
                    ("lStructSize", wintypes.DWORD),
                    ("hwndOwner", wintypes.HWND),
                    ("hInstance", wintypes.HINSTANCE),
                    ("lpstrFilter", wintypes.LPCWSTR),
                    ("lpstrCustomFilter", wintypes.LPWSTR),
                    ("nMaxCustFilter", wintypes.DWORD),
                    ("nFilterIndex", wintypes.DWORD),
                    ("lpstrFile", wintypes.LPWSTR),
                    ("nMaxFile", wintypes.DWORD),
                    ("lpstrFileTitle", wintypes.LPWSTR),
                    ("nMaxFileTitle", wintypes.DWORD),
                    ("lpstrInitialDir", wintypes.LPCWSTR),
                    ("lpstrTitle", wintypes.LPCWSTR),
                    ("Flags", wintypes.DWORD),
                    ("nFileOffset", wintypes.WORD),
                    ("nFileExtension", wintypes.WORD),
                    ("lpstrDefExt", wintypes.LPCWSTR),
                    ("lCustData", wintypes.LPARAM),
                    ("lpfnHook", ctypes.c_void_p),
                    ("lpTemplateName", wintypes.LPCWSTR),
                    ("pvReserved", ctypes.c_void_p),
                    ("dwReserved", wintypes.DWORD),
                    ("FlagsEx", wintypes.DWORD),
                ]

            filter_str = (
                f"Dataset Files\0{';'.join(f'*.{e}' for e in file_types)}\0All Files\0*.*\0\0"
            )
            buffer = ctypes.create_unicode_buffer(1024)
            ofn = OPENFILENAME()
            ofn.lStructSize = ctypes.sizeof(OPENFILENAME)
            ofn.lpstrFilter = filter_str
            ofn.lpstrFile = ctypes.cast(buffer, wintypes.LPWSTR)
            ofn.nMaxFile = 1024
            ofn.lpstrTitle = title
            ofn.Flags = 0x00080000 | 0x00001000
            if initial_dir:
                ofn.lpstrInitialDir = str(initial_dir)

            windll = getattr(ctypes, "windll", None)
            if windll and windll.comdlg32.GetOpenFileNameW(ctypes.byref(ofn)):
                return Path(buffer.value)
        except Exception:
            pass
        return None

    @staticmethod
    def _open_dir_windows(title: str, initial_dir: Path | str | None) -> Path | None:
        try:
            init_line = f"$dialog.SelectedPath = '{initial_dir}'; " if initial_dir else ""
            ps_cmd = (
                "[System.Reflection.Assembly]::LoadWithPartialName('System.windows.forms') | Out-Null; "
                "$dialog = New-Object System.Windows.Forms.FolderBrowserDialog; "
                f"$dialog.Description = '{title}'; "
                f"{init_line}"
                "if($dialog.ShowDialog() -eq 'OK'){ $dialog.SelectedPath }"
            )
            res = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_cmd],
                capture_output=True,
                text=True,
                check=False,
            )
            out = res.stdout.strip()
            return Path(out) if out and res.returncode == 0 else None
        except Exception:
            return None

    @staticmethod
    def _save_file_windows(
        title: str, default_name: str, initial_dir: Path | str | None
    ) -> Path | None:
        try:
            init_line = f"$dialog.InitialDirectory = '{initial_dir}'; " if initial_dir else ""
            ps_cmd = (
                "[System.Reflection.Assembly]::LoadWithPartialName('System.windows.forms') | Out-Null; "
                "$dialog = New-Object System.Windows.Forms.SaveFileDialog; "
                f"$dialog.Title = '{title}'; "
                f"$dialog.FileName = '{default_name}'; "
                f"{init_line}"
                "if($dialog.ShowDialog() -eq 'OK'){ $dialog.FileName }"
            )
            res = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_cmd],
                capture_output=True,
                text=True,
                check=False,
            )
            out = res.stdout.strip()
            return Path(out) if out and res.returncode == 0 else None
        except Exception:
            return None

    @staticmethod
    def _open_file_linux(
        title: str, file_types: list[str], initial_dir: Path | str | None
    ) -> Path | None:
        if shutil.which("zenity"):
            cmd = ["zenity", "--file-selection", f"--title={title}"]
            if initial_dir:
                init_path = Path(initial_dir)
                init_str = str(init_path) if init_path.is_dir() else str(init_path.parent)
                cmd.append(f"--filename={init_str.rstrip('/')}/")
            for ext in file_types:
                cmd.append(f"--file-filter=*.{ext.lstrip('.')}")
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            out = res.stdout.strip()
            return Path(out) if out and res.returncode == 0 else None
        return None

    @staticmethod
    def _open_dir_linux(title: str, initial_dir: Path | str | None) -> Path | None:
        if shutil.which("zenity"):
            cmd = ["zenity", "--file-selection", "--directory", f"--title={title}"]
            if initial_dir:
                cmd.append(f"--filename={str(initial_dir).rstrip('/')}/")
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            out = res.stdout.strip()
            return Path(out) if out and res.returncode == 0 else None
        return None

    @staticmethod
    def _save_file_linux(
        title: str, default_name: str, initial_dir: Path | str | None
    ) -> Path | None:
        if shutil.which("zenity"):
            filename = default_name
            if initial_dir:
                filename = str(Path(initial_dir) / default_name)
            cmd = [
                "zenity",
                "--file-selection",
                "--save",
                "--confirm-overwrite",
                f"--filename={filename}",
                f"--title={title}",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            out = res.stdout.strip()
            return Path(out) if out and res.returncode == 0 else None
        return None
