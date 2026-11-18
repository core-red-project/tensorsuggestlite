# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""UI layout builder for TensorSuggestLite Dear PyGui desktop presentation.

Decouples widget tree construction and layout geometry from state management
and event handling (adhering to SOLID Single Responsibility Principle).
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import dearpygui.dearpygui as dpg

from tensorsuggestlite.ui.theme import create_console_theme, get_icon_path

if TYPE_CHECKING:
    from tensorsuggestlite.ui.main_window import TensorSuggestLiteUI


class UILayoutBuilder:
    """Constructs the responsive Dear PyGui visual widget tree."""

    @staticmethod
    def register_textures() -> None:
        """Load and register application icons into the Dear PyGui texture registry."""
        icons_to_load = [
            ("app_icon.png", "app_logo_texture"),
            ("sun.png", "sun_icon_texture"),
            ("moon.png", "moon_icon_texture"),
        ]
        with dpg.texture_registry(show=False):
            for icon_file, tex_tag in icons_to_load:
                icon_path = get_icon_path(icon_file)
                if icon_path and icon_path.is_file():
                    try:
                        width, height, _channels, data = dpg.load_image(str(icon_path))
                        dpg.add_static_texture(
                            width=width, height=height, default_value=data, tag=tex_tag
                        )
                    except Exception:
                        pass

    @classmethod
    def build(cls, ui: TensorSuggestLiteUI) -> None:
        """Construct the complete widget tree fitting entirely without outer scrollbars."""
        cls.register_textures()

        with dpg.window(
            tag="primary_window",
            label="TensorSuggestLite",
            no_title_bar=True,
            no_move=True,
            no_resize=True,
            no_collapse=True,
            no_scrollbar=True,
            no_scroll_with_mouse=True,
        ):
            cls._build_top_bar(ui)
            dpg.add_spacer(height=3)

            with dpg.table(
                tag="main_split_table", header_row=False, policy=dpg.mvTable_SizingStretchProp
            ):
                dpg.add_table_column(init_width_or_weight=0.66)
                dpg.add_table_column(init_width_or_weight=0.34)

                with dpg.table_row():
                    cls._build_left_panel(ui)
                    cls._build_right_panel(ui)

        cls._build_confirmation_modal(ui)

    @staticmethod
    def _build_confirmation_modal(ui: TensorSuggestLiteUI) -> None:
        """Construct confirmation dialog modal for destructive actions."""
        with dpg.window(
            tag="modal_confirm_clear",
            label="Confirm Deletion",
            modal=True,
            show=False,
            no_resize=True,
            width=420,
            height=140,
            pos=[280, 220],
        ):
            dpg.add_text("Are you sure you want to delete all generated artifacts?")
            dpg.add_text("This operation cannot be undone.", color=[235, 90, 90])
            dpg.add_spacer(height=8)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    tag="btn_modal_confirm_delete",
                    label="Yes, Delete Artifacts",
                    callback=ui._perform_clear_folders,
                    width=190,
                    height=28,
                )
                dpg.add_button(
                    tag="btn_modal_cancel_delete",
                    label="Cancel",
                    callback=lambda: dpg.configure_item("modal_confirm_clear", show=False),
                    width=190,
                    height=28,
                )

    @staticmethod
    def _build_top_bar(ui: TensorSuggestLiteUI) -> None:
        """Construct compact pixel-perfect 36px top application bar."""
        with (
            dpg.child_window(
                tag="app_bar_container",
                width=-1,
                height=36,
                border=True,
                no_scrollbar=True,
                no_scroll_with_mouse=True,
            ),
            dpg.table(header_row=False, policy=dpg.mvTable_SizingStretchProp),
        ):
            dpg.add_table_column(init_width_or_weight=1.0)
            dpg.add_table_column(width_fixed=True, init_width_or_weight=32)

            with dpg.table_row():
                with dpg.group(horizontal=True):
                    if dpg.does_item_exist("app_logo_texture"):
                        dpg.add_image("app_logo_texture", width=20, height=20)
                        dpg.add_spacer(width=6)

                    dpg.add_text(
                        "TensorSuggestLite",
                        tag="app_title_label",
                        color=(255, 255, 255, 255),
                    )
                    dpg.add_text(
                        " | Core Red Machine Learning",
                        tag="app_subtitle_label",
                        color=(248, 113, 113, 220),
                    )

                if dpg.does_item_exist("sun_icon_texture"):
                    dpg.add_image_button(
                        "sun_icon_texture",
                        tag="btn_theme_toggle",
                        callback=ui.toggle_theme,
                        width=18,
                        height=18,
                    )

    @staticmethod
    def _build_left_panel(ui: TensorSuggestLiteUI) -> None:
        """Construct left column containing imports, dataset path, console, and playground."""
        with dpg.child_window(
            tag="left_panel_child",
            width=-1,
            height=480,
            border=True,
            no_scrollbar=True,
            no_scroll_with_mouse=True,
        ):
            # Symmetrical Import Row (3 equal columns)
            with dpg.table(header_row=False, policy=dpg.mvTable_SizingStretchProp):
                dpg.add_table_column(init_width_or_weight=1.0)
                dpg.add_table_column(init_width_or_weight=1.0)
                dpg.add_table_column(init_width_or_weight=1.0)
                with dpg.table_row():
                    dpg.add_button(
                        tag="btn_import_json",
                        label="Import JSON",
                        callback=lambda: ui._on_import_clicked("json"),
                        width=-1,
                        height=30,
                    )
                    dpg.add_button(
                        tag="btn_import_toml",
                        label="Import TOML",
                        callback=lambda: ui._on_import_clicked("toml"),
                        width=-1,
                        height=30,
                    )
                    dpg.add_button(
                        tag="btn_import_yaml",
                        label="Import YAML",
                        callback=lambda: ui._on_import_clicked("yaml"),
                        width=-1,
                        height=30,
                    )

            dpg.add_spacer(height=2)
            dnd_hint = (
                "Drop dataset file or click Import above [macOS Drag & Drop active]..."
                if sys.platform == "darwin"
                else "Click Import buttons above to select dataset [Windows/Linux]..."
            )
            dpg.add_input_text(
                tag="path_display",
                default_value=dnd_hint,
                readonly=True,
                width=-1,
            )

            dpg.add_spacer(height=2)
            dpg.add_input_text(
                tag="input_skeleton",
                hint="Fixed Skeleton (optional, e.g. in:out or prompt:intent)...",
                width=-1,
            )

            dpg.add_spacer(height=2)
            dpg.add_text("Console Output:")
            # Console window: ONLY element in the app that scrolls
            with dpg.child_window(
                tag="console_window",
                width=-1,
                height=130,
                border=True,
                horizontal_scrollbar=False,
            ):
                dpg.add_text(
                    tag="console_text",
                    default_value="[System] TensorSuggestLite Core Red initialized.\n",
                )
            dpg.bind_item_theme("console_window", create_console_theme())

            dpg.add_spacer(height=3)
            # Model Playground Section
            dpg.add_text("Model Playground:")
            dpg.add_input_text(
                tag="input_playground",
                hint="Test text (e.g. 'This product is excellent!')...",
                width=-1,
            )

            dpg.add_spacer(height=2)
            # Symmetrical Playground Action Row (3 equal columns)
            with dpg.table(header_row=False, policy=dpg.mvTable_SizingStretchProp):
                dpg.add_table_column(init_width_or_weight=1.0)
                dpg.add_table_column(init_width_or_weight=1.0)
                dpg.add_table_column(init_width_or_weight=1.0)
                with dpg.table_row():
                    dpg.add_button(
                        tag="btn_predict",
                        label="Test Inference",
                        callback=ui._on_predict_clicked,
                        enabled=False,
                        width=-1,
                        height=28,
                    )
                    dpg.add_button(
                        tag="btn_compare",
                        label="Compare Keras vs TFLite",
                        callback=ui._on_compare_clicked,
                        enabled=False,
                        width=-1,
                        height=28,
                    )
                    dpg.add_button(
                        tag="btn_load_dir",
                        label="Load Model Dir",
                        callback=ui._on_load_model_clicked,
                        width=-1,
                        height=28,
                    )

            dpg.add_spacer(height=2)
            dpg.add_text(
                tag="playground_result",
                default_value="Playground: Train or load a model to test inference",
                color=(148, 163, 184),
            )

            dpg.add_spacer(height=4)
            dpg.add_text(
                "TensorSuggestLite © Core Red Project 2026",
                tag="label_footer",
                color=(100, 116, 139, 255),
            )

    @staticmethod
    def _build_right_panel(ui: TensorSuggestLiteUI) -> None:
        """Construct right column containing hyperparameters and primary/secondary CTAs."""
        with dpg.child_window(
            tag="right_panel_child",
            width=-1,
            height=480,
            border=True,
            no_scrollbar=True,
            no_scroll_with_mouse=True,
        ):
            dpg.add_text("Hyperparameters:", tag="label_hp_header", color=(248, 113, 113, 255))
            dpg.add_text("Epochs:")
            dpg.add_slider_int(
                tag="hp_epochs", width=-1, default_value=12, min_value=1, max_value=100
            )
            dpg.add_text("Learning Rate:")
            dpg.add_input_float(
                tag="hp_lr",
                width=-1,
                default_value=0.001,
                step=0.0005,
                format="%.4f",
            )
            dpg.add_text("Dropout Rate:")
            dpg.add_slider_float(
                tag="hp_dropout",
                width=-1,
                default_value=0.20,
                min_value=0.0,
                max_value=0.8,
                format="%.2f",
            )
            dpg.add_text("Batch Size:")
            dpg.add_slider_int(
                tag="hp_batch",
                width=-1,
                default_value=16,
                min_value=4,
                max_value=128,
            )

            dpg.add_spacer(height=4)
            dpg.add_separator()
            dpg.add_spacer(height=4)

            # Primary CTA (prominent Core Red action button)
            dpg.add_button(
                tag="btn_train",
                label="Train Model",
                callback=ui._on_train_clicked,
                enabled=False,
                width=-1,
                height=34,
            )

            dpg.add_spacer(height=2)
            # Quantization configuration (Bloque 2.6)
            dpg.add_text("TFLite Quantization:")
            dpg.add_combo(
                items=["dynamic", "float16", "int8", "none"],
                default_value="dynamic",
                tag="combo_quantization",
                width=-1,
            )

            # Secondary CTA (convert to TFLite)
            dpg.add_button(
                tag="btn_convert",
                label="Generate TFLite",
                callback=ui._on_convert_clicked,
                enabled=False,
                width=-1,
                height=30,
            )

            # Success CTA (export artifact)
            dpg.add_button(
                tag="btn_export",
                label="Export TFLite",
                callback=ui._on_export_clicked,
                enabled=False,
                show=False,
                width=-1,
                height=30,
            )

            dpg.add_spacer(height=2)

            # Utility Actions Row (symmetrical 2-column side-by-side)
            with dpg.table(header_row=False, policy=dpg.mvTable_SizingStretchProp):
                dpg.add_table_column(init_width_or_weight=1.0)
                dpg.add_table_column(init_width_or_weight=1.0)
                with dpg.table_row():
                    dpg.add_button(
                        tag="btn_clear_folders",
                        label="Clear Folders",
                        callback=ui._on_clear_generated_clicked,
                        width=-1,
                        height=28,
                    )
                    dpg.add_button(
                        tag="btn_clear_screen",
                        label="Clear Screen",
                        callback=ui._clear_console,
                        width=-1,
                        height=28,
                    )

            dpg.add_spacer(height=6)
            dpg.add_text(tag="status_label", default_value="Status: Idle (No dataset loaded)")
            dpg.add_progress_bar(tag="progress_bar", default_value=0.0, width=-1, height=14)
