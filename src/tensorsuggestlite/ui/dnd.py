# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""[EXPERIMENTAL] Native OS drag-and-drop integration for Dear PyGui.

Enables dropping files or directories directly from Finder (macOS) onto the running
application window via native Cocoa/Metal hooks.

STATUS: EXPERIMENTAL.
- macOS (Cocoa/AppKit): Verified and functional.
- Windows / Linux: Untested and experimental (GLFW callback hook provided as unverified fallback).
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import sys
from collections.abc import Callable
from typing import Any

_CALLBACK_REFS: list[object] = []


def _hook_macos_cocoa_dnd(on_drop_callback: Callable[[list[str]], None]) -> bool:
    """Register native Cocoa drag-and-drop handlers on macOS (NSWindow & GLFWContentView)."""
    try:
        objc_lib = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc") or "libobjc.dylib")
    except Exception:
        return False

    def _call(ret_type: type, name: str, *arg_types: type) -> Any:
        func = getattr(objc_lib, name)
        addr = ctypes.cast(func, ctypes.c_void_p).value
        assert addr is not None
        return ctypes.CFUNCTYPE(ret_type, *arg_types)(addr)

    try:
        get_class = _call(ctypes.c_void_p, "objc_getClass", ctypes.c_char_p)
        sel_register = _call(ctypes.c_void_p, "sel_registerName", ctypes.c_char_p)
        msg_send = _call(ctypes.c_void_p, "objc_msgSend", ctypes.c_void_p, ctypes.c_void_p)
        msg_send_obj = _call(
            ctypes.c_void_p, "objc_msgSend", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulonglong
        )
        msg_send_1arg = _call(
            ctypes.c_void_p, "objc_msgSend", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
        )
        msg_send_str = _call(
            ctypes.c_void_p, "objc_msgSend", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p
        )
        msg_send_int = _call(ctypes.c_ulonglong, "objc_msgSend", ctypes.c_void_p, ctypes.c_void_p)
        msg_send_cstr = _call(ctypes.c_char_p, "objc_msgSend", ctypes.c_void_p, ctypes.c_void_p)
        object_get_class = _call(ctypes.c_void_p, "object_getClass", ctypes.c_void_p)
        class_replace_method = _call(
            ctypes.c_void_p,
            "class_replaceMethod",
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_char_p,
        )

        ns_app_cls = get_class(b"NSApplication")
        app = msg_send(ns_app_cls, sel_register(b"sharedApplication"))
        windows = msg_send(app, sel_register(b"windows"))
        count = msg_send_int(windows, sel_register(b"count"))
        if count == 0:
            return False

        win = msg_send_obj(windows, sel_register(b"objectAtIndex:"), 0)
        content_view = msg_send(win, sel_register(b"contentView"))
        if not content_view:
            return False

        ns_string_cls = get_class(b"NSString")
        ns_array_cls = get_class(b"NSArray")
        pboard_type = msg_send_str(
            ns_string_cls, sel_register(b"stringWithUTF8String:"), b"NSFilenamesPboardType"
        )
        types_arr = msg_send_1arg(ns_array_cls, sel_register(b"arrayWithObject:"), pboard_type)

        msg_send_1arg(content_view, sel_register(b"registerForDraggedTypes:"), types_arr)
        msg_send_1arg(win, sel_register(b"registerForDraggedTypes:"), types_arr)

        def _dragging_entered(_self: int, _cmd: int, _sender: int) -> int:
            return 1  # NSDragOperationCopy

        def _perform_drag(_self: int, _cmd: int, sender: int) -> bool:
            try:
                pboard = msg_send(sender, sel_register(b"draggingPasteboard"))
                files_arr = msg_send_1arg(
                    pboard, sel_register(b"propertyListForType:"), pboard_type
                )
                if not files_arr:
                    return False

                f_count = msg_send_int(files_arr, sel_register(b"count"))
                paths: list[str] = []
                for i in range(f_count):
                    item = msg_send_obj(files_arr, sel_register(b"objectAtIndex:"), i)
                    cstr = msg_send_cstr(item, sel_register(b"UTF8String"))
                    if cstr:
                        paths.append(cstr.decode("utf-8"))

                if paths:
                    on_drop_callback(paths)
                return True
            except Exception:
                return False

        cb_enter_t = ctypes.CFUNCTYPE(
            ctypes.c_ulonglong, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
        )
        cb_perform_t = ctypes.CFUNCTYPE(
            ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
        )

        c_enter = cb_enter_t(_dragging_entered)
        c_perform = cb_perform_t(_perform_drag)

        _CALLBACK_REFS.extend([c_enter, c_perform])

        view_cls = object_get_class(content_view)
        win_cls = object_get_class(win)

        for target_cls in [view_cls, win_cls]:
            class_replace_method(
                target_cls,
                sel_register(b"draggingEntered:"),
                ctypes.cast(c_enter, ctypes.c_void_p),
                b"Q@:@",
            )
            class_replace_method(
                target_cls,
                sel_register(b"performDragOperation:"),
                ctypes.cast(c_perform, ctypes.c_void_p),
                b"B@:@",
            )

        return True
    except Exception:
        return False


def _hook_glfw_dnd(on_drop_callback: Callable[[list[str]], None]) -> bool:
    """Register GLFW drop callback for Windows and Linux Dear PyGui runtimes."""
    try:
        import dearpygui._dearpygui as dpg_core
    except ImportError:
        return False

    try:
        dpg_lib = ctypes.CDLL(dpg_core.__file__)
        if not hasattr(dpg_lib, "glfwGetCurrentContext") or not hasattr(
            dpg_lib, "glfwSetDropCallback"
        ):
            return False

        dpg_lib.glfwGetCurrentContext.restype = ctypes.c_void_p
        dpg_lib.glfwGetCurrentContext.argtypes = []

        window = dpg_lib.glfwGetCurrentContext()
        if not window:
            return False

        glfw_drop_fun = ctypes.CFUNCTYPE(
            None, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)
        )

        def _c_drop_handler(_win: Any, count: int, paths_ptr: Any) -> None:
            py_paths: list[str] = []
            ptr_typed = ctypes.cast(paths_ptr, ctypes.POINTER(ctypes.c_char_p))
            for i in range(count):
                with contextlib.suppress(Exception):
                    p_bytes = ptr_typed[i]
                    if p_bytes:
                        py_paths.append(p_bytes.decode(sys.getfilesystemencoding()))
            if py_paths:
                on_drop_callback(py_paths)

        c_callback = glfw_drop_fun(_c_drop_handler)
        _CALLBACK_REFS.append(c_callback)

        dpg_lib.glfwSetDropCallback.restype = ctypes.c_void_p
        dpg_lib.glfwSetDropCallback.argtypes = [ctypes.c_void_p, glfw_drop_fun]
        dpg_lib.glfwSetDropCallback(window, c_callback)
        return True
    except Exception:
        return False


def enable_os_drag_and_drop(on_drop_callback: Callable[[list[str]], None]) -> bool:
    """Hook native OS drag-and-drop into Dear PyGui across platforms.

    Args:
        on_drop_callback: Python callable receiving a list of absolute path strings.

    Returns:
        bool: True if drop handler was successfully bound, False otherwise.
    """
    if sys.platform != "darwin":
        return _hook_glfw_dnd(on_drop_callback)
    return _hook_macos_cocoa_dnd(on_drop_callback) or _hook_glfw_dnd(on_drop_callback)


__all__ = ["enable_os_drag_and_drop"]
