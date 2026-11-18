# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""GUI application launcher powered by Dear PyGui and Core Red design."""

from __future__ import annotations

import sys


def is_gui_available() -> bool:
    """Check whether Dear PyGui is installed and importable."""
    try:
        import dearpygui.dearpygui  # noqa: F401

        return True
    except ImportError:
        return False


def launch_gui() -> int:
    """Launch the graphical desktop interface powered by Dear PyGui.

    Returns:
        int: Process exit code (0 for clean execution, 1 if Dear PyGui is unavailable).
    """
    if not is_gui_available() or sys.modules.get("dearpygui") is None:
        sys.stderr.write("error: Dear PyGui is not installed.\n")
        sys.stderr.write("To run the desktop GUI, install it with:\n")
        sys.stderr.write("    pip install 'tensorsuggestlite[gui]'\n")
        sys.stderr.write("    or: pip install dearpygui\n")
        sys.stderr.flush()
        return 1

    import dearpygui.dearpygui as dpg

    from tensorsuggestlite.ui.dnd import enable_os_drag_and_drop
    from tensorsuggestlite.ui.main_window import TensorSuggestLiteUI
    from tensorsuggestlite.ui.theme import apply_theme, get_icon_path

    dpg.create_context()
    ui = TensorSuggestLiteUI()
    ui.build_ui()
    apply_theme("dark")

    icon_path = get_icon_path("app_icon.png")
    icon_str = str(icon_path) if icon_path else ""

    dpg.create_viewport(
        title="TensorSuggestLite",
        width=980,
        height=640,
        min_width=880,
        min_height=580,
        small_icon=icon_str,
        large_icon=icon_str,
    )
    dpg.setup_dearpygui()
    dpg.set_viewport_resize_callback(ui.on_viewport_resize)
    dpg.set_primary_window("primary_window", True)
    dpg.show_viewport()
    ui.on_viewport_resize()

    # Global keyboard shortcuts
    with dpg.handler_registry():
        dpg.add_key_press_handler(
            key=dpg.mvKey_Escape,
            callback=lambda: ui._clear_console(),
        )

    # Enable native OS Drag and Drop from Finder / File Explorer on active window frame
    dpg.set_frame_callback(1, lambda: enable_os_drag_and_drop(ui.handle_dropped_paths))

    dpg.start_dearpygui()
    dpg.destroy_context()
    return 0
