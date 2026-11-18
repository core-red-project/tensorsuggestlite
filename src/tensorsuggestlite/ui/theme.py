# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Theme manager and styling engine faithful to Core Red Dark Obsidian & Clean Light QSS.

Translates color palettes, typography, button gradients, console fonts, rounded
geometry, and button hierarchy from styles/dark.qss and styles/light.qss into native
Dear PyGui themes with high contrast across light and dark modes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import dearpygui.dearpygui as dpg

_THEMES: dict[str, Any] = {}


def get_icon_path(name: str = "app_icon.png") -> Path | None:
    """Resolve icon file location across source checkout and frozen bundles."""
    candidates = [
        Path.cwd() / "icons" / name,
        Path(__file__).resolve().parent.parent.parent.parent / "icons" / name,
    ]
    if hasattr(sys, "_MEIPASS"):
        candidates.insert(0, Path(sys._MEIPASS) / "icons" / name)

    for p in candidates:
        if p.is_file():
            return p
    return None


def create_core_red_dark_theme() -> Any:
    """Create Dear PyGui theme matching styles/dark.qss (Core Red Dark Obsidian)."""
    with dpg.theme() as theme_id, dpg.theme_component(dpg.mvAll):
        # Window & Panels (from styles/dark.qss: #0d0f14 and #111318)
        dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (13, 15, 20, 255))
        dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (17, 19, 24, 255))
        dpg.add_theme_color(dpg.mvThemeCol_PopupBg, (17, 19, 24, 250))
        dpg.add_theme_color(dpg.mvThemeCol_Border, (220, 38, 38, 70))

        # Typography (from styles/dark.qss: #e5e7eb and #6b7280)
        dpg.add_theme_color(dpg.mvThemeCol_Text, (229, 231, 235, 255))
        dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, (107, 114, 128, 255))

        # Inputs & Frames (from styles/dark.qss #pathField: #181b22, border rgba(255,255,255,0.1))
        dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (24, 27, 34, 255))
        dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (34, 38, 48, 255))
        dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (42, 48, 60, 255))

        # Frame border definition
        dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1)
        dpg.add_theme_color(dpg.mvThemeCol_Border, (220, 38, 38, 80))

        # Core Red Neumorphic Buttons (from styles/dark.qss: #991b1b, hover: #b91c1c, active: #581414)
        dpg.add_theme_color(dpg.mvThemeCol_Button, (153, 27, 27, 255))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (185, 28, 28, 255))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (88, 20, 20, 255))

        # Core Red Progress Bar & Sliders (from styles/dark.qss: chunk #b91c1c to #ef4444)
        dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, (239, 68, 68, 255))
        dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (239, 68, 68, 255))
        dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (220, 38, 38, 255))

        # Geometry & 8px Border-Radius from styles/dark.qss
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8)
        dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 8)
        dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 8)
        dpg.add_theme_style(dpg.mvStyleVar_PopupRounding, 8)
        dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 6)
        dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 5)

    return theme_id


def create_core_red_light_theme() -> Any:
    """Create Dear PyGui theme matching styles/light.qss with high contrast and defined borders."""
    with dpg.theme() as theme_id, dpg.theme_component(dpg.mvAll):
        # Window & Panels (from styles/light.qss: #f8fafc and #ffffff)
        dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (248, 250, 252, 255))
        dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (255, 255, 255, 255))
        dpg.add_theme_color(dpg.mvThemeCol_PopupBg, (255, 255, 255, 255))
        dpg.add_theme_color(dpg.mvThemeCol_Border, (203, 213, 225, 255))  # Crisp slate 300 border

        # Typography (from styles/light.qss: high contrast #0f172a and #64748b)
        dpg.add_theme_color(dpg.mvThemeCol_Text, (15, 23, 42, 255))
        dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, (100, 116, 139, 255))

        # Inputs & Frames (from styles/light.qss: #f8fafc with #cbd5e1 border)
        dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (248, 250, 252, 255))
        dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (241, 245, 249, 255))
        dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (226, 232, 240, 255))

        # 1px defined frame border for high contrast inputs and controls
        dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1)

        # Buttons (from styles/light.qss: #ffffff / #f1f5f9, hover #fee2e2 soft red)
        dpg.add_theme_color(dpg.mvThemeCol_Button, (241, 245, 249, 255))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (254, 226, 226, 255))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (254, 202, 202, 255))

        # Core Red Progress Bar & Sliders (from styles/light.qss: #b91c1c to #dc2626)
        dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, (185, 28, 28, 255))
        dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (185, 28, 28, 255))
        dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (220, 38, 38, 255))

        # Geometry & 8px Border-Radius from styles/light.qss
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8)
        dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 8)
        dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 8)
        dpg.add_theme_style(dpg.mvStyleVar_PopupRounding, 8)
        dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 6)
        dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 5)

    return theme_id


def create_console_theme(dark: bool = True) -> Any:
    """Create dedicated console theme matching #consoleBox (#090a0d, #38bdf8 cyan text)."""
    with dpg.theme() as theme_id, dpg.theme_component(dpg.mvAll):
        if dark:
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (9, 10, 13, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (56, 189, 248, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Border, (220, 38, 38, 40))
        else:
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (15, 23, 42, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (56, 189, 248, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Border, (203, 213, 225, 255))
        dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 8)
    return theme_id


def create_primary_button_theme(dark: bool = True) -> Any:
    """Create high-contrast prominent primary CTA theme for 'Train Model' button."""
    with dpg.theme() as theme_id, dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button, (220, 38, 38, 255))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (239, 68, 68, 255))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (185, 28, 28, 255))
        dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255, 255))
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8)
    return theme_id


def create_secondary_button_theme(dark: bool = True) -> Any:
    """Create secondary action theme for 'Generate TFLite' button."""
    with dpg.theme() as theme_id, dpg.theme_component(dpg.mvButton):
        if dark:
            dpg.add_theme_color(dpg.mvThemeCol_Button, (153, 27, 27, 220))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (185, 28, 28, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (127, 29, 29, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255, 255))
        else:
            dpg.add_theme_color(dpg.mvThemeCol_Button, (185, 28, 28, 235))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (220, 38, 38, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (153, 27, 27, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255, 255))
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8)
    return theme_id


def create_export_button_theme() -> Any:
    """Create success emerald CTA theme for 'Export TFLite' button."""
    with dpg.theme() as theme_id, dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button, (16, 185, 129, 230))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (5, 150, 105, 255))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (4, 120, 87, 255))
        dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255, 255))
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8)
    return theme_id


def create_utility_button_theme(dark: bool = True) -> Any:
    """Create muted utility theme for 'Clear Folders' and 'Clear Screen' buttons."""
    with dpg.theme() as theme_id, dpg.theme_component(dpg.mvButton):
        if dark:
            dpg.add_theme_color(dpg.mvThemeCol_Button, (30, 34, 46, 200))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 65, 85, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (24, 27, 36, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (148, 163, 184, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Border, (51, 65, 85, 150))
        else:
            dpg.add_theme_color(dpg.mvThemeCol_Button, (241, 245, 249, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (226, 232, 240, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (203, 213, 225, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (51, 65, 85, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Border, (203, 213, 225, 255))
        dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1)
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6)
    return theme_id


def create_import_button_theme(dark: bool = True) -> Any:
    """Create clean symmetrical theme for import and playground buttons with high contrast."""
    with dpg.theme() as theme_id, dpg.theme_component(dpg.mvButton):
        if dark:
            dpg.add_theme_color(dpg.mvThemeCol_Button, (34, 40, 54, 240))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (153, 27, 27, 220))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (127, 29, 29, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (241, 245, 249, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Border, (220, 38, 38, 80))
        else:
            dpg.add_theme_color(dpg.mvThemeCol_Button, (255, 255, 255, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (254, 226, 226, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (254, 202, 202, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (15, 23, 42, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Border, (203, 213, 225, 255))
        dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1)
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8)
    return theme_id


def create_app_bar_dark_theme() -> Any:
    """Create app bar sleek obsidian background matching Core Red styling."""
    with dpg.theme() as theme_id, dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (18, 20, 26, 255))
        dpg.add_theme_color(dpg.mvThemeCol_Border, (220, 38, 38, 80))
        dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 6)
        dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 8, 4)
        dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 4, 2)
    return theme_id


def create_app_bar_light_theme() -> Any:
    """Create app bar banner matching Core Red Clean Light (#b91c1c banner with crisp white text)."""
    with dpg.theme() as theme_id, dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (185, 28, 28, 255))  # Rich Core Red
        dpg.add_theme_color(dpg.mvThemeCol_Border, (153, 27, 27, 255))
        dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 6)
        dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 8, 4)
        dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 4, 2)
    return theme_id


def create_theme_toggle_button_theme(dark: bool = True) -> Any:
    """Create clean square theme for theme toggle button with subtle translucent hover."""
    with dpg.theme() as theme_id, dpg.theme_component(dpg.mvImageButton):
        if dark:
            dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 0, 0, 0))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (220, 38, 38, 120))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (239, 68, 68, 180))
        else:
            dpg.add_theme_color(dpg.mvThemeCol_Button, (255, 255, 255, 40))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (255, 255, 255, 80))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (255, 255, 255, 120))
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6)
        dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)
    return theme_id


def apply_theme(name: str = "dark") -> None:
    """Apply designated theme and item-level UX hierarchies to the active Dear PyGui session."""
    global _THEMES
    dark = name == "dark"

    if "dark" not in _THEMES:
        _THEMES["dark"] = create_core_red_dark_theme()
    if "light" not in _THEMES:
        _THEMES["light"] = create_core_red_light_theme()

    active_theme = _THEMES["dark"] if dark else _THEMES["light"]
    dpg.bind_theme(active_theme)

    # Bind App Bar
    if dpg.does_item_exist("app_bar_container"):
        bar_theme = create_app_bar_dark_theme() if dark else create_app_bar_light_theme()
        dpg.bind_item_theme("app_bar_container", bar_theme)

    # Topbar Typography Contrast
    if dpg.does_item_exist("app_title_label"):
        dpg.configure_item("app_title_label", color=(255, 255, 255, 255))
    if dpg.does_item_exist("app_subtitle_label"):
        sub_col = (248, 113, 113, 220) if dark else (254, 202, 202, 240)
        dpg.configure_item("app_subtitle_label", color=sub_col)

    if dpg.does_item_exist("btn_theme_toggle"):
        dpg.bind_item_theme("btn_theme_toggle", create_theme_toggle_button_theme(dark))

    # Console Theme
    if dpg.does_item_exist("console_window"):
        dpg.bind_item_theme("console_window", create_console_theme(dark))

    # Bind Button UX Hierarchy
    if dpg.does_item_exist("btn_train"):
        dpg.bind_item_theme("btn_train", create_primary_button_theme(dark))

    if dpg.does_item_exist("btn_convert"):
        dpg.bind_item_theme("btn_convert", create_secondary_button_theme(dark))

    if dpg.does_item_exist("btn_export"):
        dpg.bind_item_theme("btn_export", create_export_button_theme())

    for util_btn in ["btn_clear_folders", "btn_clear_screen"]:
        if dpg.does_item_exist(util_btn):
            dpg.bind_item_theme(util_btn, create_utility_button_theme(dark))

    for imp_btn in [
        "btn_import_json",
        "btn_import_toml",
        "btn_import_yaml",
        "btn_predict",
        "btn_compare",
        "btn_load_dir",
    ]:
        if dpg.does_item_exist(imp_btn):
            dpg.bind_item_theme(imp_btn, create_import_button_theme(dark))

    # Text Elements Contrast Updates
    if dpg.does_item_exist("label_hp_header"):
        hp_col = (248, 113, 113, 255) if dark else (185, 28, 28, 255)
        dpg.configure_item("label_hp_header", color=hp_col)

    if dpg.does_item_exist("status_label"):
        stat_col = (248, 113, 113, 255) if dark else (185, 28, 28, 255)
        dpg.configure_item("status_label", color=stat_col)

    if dpg.does_item_exist("label_footer"):
        foot_col = (100, 116, 139, 255) if dark else (71, 85, 105, 255)
        dpg.configure_item("label_footer", color=foot_col)

    if dpg.does_item_exist("playground_result"):
        pg_col = (226, 232, 240, 255) if dark else (15, 23, 42, 255)
        dpg.configure_item("playground_result", color=pg_col)
