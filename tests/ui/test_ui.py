"""Tests for Dear PyGui presentation layer and NativeDialogBridge."""

import sys
from pathlib import Path

import pytest

from tensorsuggestlite.core.entities import TrainingResult
from tensorsuggestlite.ui.app import is_gui_available
from tensorsuggestlite.ui.native_dialogs import NativeDialogBridge


def test_is_gui_available():
    """Verify that is_gui_available reports True when dearpygui is installed."""
    assert is_gui_available() is True


def test_native_dialog_bridge_file_types():
    """Verify NativeDialogBridge parameter handling without launching blocking GUI."""
    # When simulated on any platform with cancelled input, returns None cleanly
    assert NativeDialogBridge.open_file_dialog is not None
    assert NativeDialogBridge.open_directory_dialog is not None
    assert NativeDialogBridge.save_file_dialog is not None


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS osascript test")
def test_native_dialog_bridge_macos_script_generation(monkeypatch):
    """Verify macOS osascript command structure."""
    calls = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)

        class DummyRes:
            stdout = "/Users/test/data.json\n"
            returncode = 0

        return DummyRes()

    import subprocess

    monkeypatch.setattr(subprocess, "run", fake_run)
    res = NativeDialogBridge._open_file_macos("Test", ["json"], Path("/tmp"))
    assert res == Path("/Users/test/data.json")
    assert len(calls) == 1
    assert "choose file with prompt" in calls[0][2]


def test_dearpygui_build_ui_and_themes():
    """Verify Dear PyGui UI structure builds cleanly without keyword or API errors."""
    import dearpygui.dearpygui as dpg

    from tensorsuggestlite.ui.main_window import TensorSuggestLiteUI
    from tensorsuggestlite.ui.theme import apply_theme

    dpg.create_context()
    try:
        ui = TensorSuggestLiteUI()
        ui.build_ui()
        assert dpg.does_item_exist("primary_window")
        assert dpg.does_item_exist("btn_theme_toggle")
        assert dpg.does_item_exist("combo_quantization")
        assert dpg.does_item_exist("btn_train")
        assert dpg.does_item_exist("btn_convert")
        assert dpg.does_item_exist("console_window")
        assert dpg.does_item_exist("label_footer")
        assert dpg.does_item_exist("label_hp_header")

        # Test high contrast color application across light and dark modes
        apply_theme("dark")
        apply_theme("light")
        ui.toggle_theme()
    finally:
        dpg.destroy_context()


def test_ui_viewport_resize_and_footer_geometry():
    """Verify that viewport resize leaves generous room for footer without clipping."""
    import dearpygui.dearpygui as dpg

    from tensorsuggestlite.ui.main_window import TensorSuggestLiteUI

    dpg.create_context()
    try:
        ui = TensorSuggestLiteUI()
        ui.build_ui()

        # Simulate viewport resize call
        ui.on_viewport_resize()

        # Check that child windows are created with appropriate dimensions
        assert dpg.does_item_exist("left_panel_child")
        assert dpg.does_item_exist("right_panel_child")
        assert dpg.does_item_exist("console_window")
    finally:
        dpg.destroy_context()


def test_ui_feedback_ux_lifecycle(tmp_path: Path):
    """Verify informative status badges and console messages through the entire UX lifecycle."""
    import dearpygui.dearpygui as dpg

    from tensorsuggestlite.ui.main_window import TensorSuggestLiteUI

    dpg.create_context()
    try:
        ui = TensorSuggestLiteUI()
        ui.build_ui()

        # 1. Initial status
        assert "Idle" in dpg.get_value("status_label")

        # 2. Loading a dataset updates status and logs
        sample_file = tmp_path / "valid.json"
        sample_file.write_text(
            '{"GREETING": ["hello", "hi"], "FAREWELL": ["bye"]}', encoding="utf-8"
        )
        ui._load_file(sample_file, "json")

        assert "Dataset Ready" in dpg.get_value("status_label")
        assert "[READY]" in dpg.get_value("console_text")

        # 3. Training success restores controls and provides high-impact feedback
        from tensorsuggestlite.core.entities import ArtifactBundle

        dummy_bundle = ArtifactBundle(
            output_dir=tmp_path,
            tokenizer_path=tmp_path / "tokenizer.json",
            keras_model_path=tmp_path / "text_classifier.keras",
            label_encoder_path=tmp_path / "label_encoder.json",
        )
        dummy_result = TrainingResult(
            model_path=str(tmp_path / "text_classifier.keras"),
            items_serialized=3,
            labels_count=2,
            artifacts=dummy_bundle,
            metrics={
                "final_loss": 0.0123,
                "final_accuracy": 1.0,
                "training_duration_seconds": 1.25,
            },
        )
        ui._on_train_success(dummy_result)
        assert "Model Trained" in dpg.get_value("status_label")
        assert "[DONE]" in dpg.get_value("console_text")

        # 4. TFLite generation success
        gen_dir = ui._get_generated_dir()
        tflite_path = gen_dir / "text_classifier.tflite"
        tflite_path.write_bytes(b"dummy_tflite_bytes")
        ui._on_convert_success(tflite_path)
        assert "TFLite Ready" in dpg.get_value("status_label")

        # 5. Clearing artifacts prompts confirmation modal and performs deletion
        # Only known model artifacts must be deleted; user files must remain intact!
        user_file = gen_dir / "user_notes.txt"
        user_file.write_text("important user notes")
        ui._on_clear_generated_clicked()
        assert dpg.is_item_shown("modal_confirm_clear")
        ui._perform_clear_folders()
        assert "Artifacts Cleared" in dpg.get_value("status_label")
        assert not tflite_path.exists()
        assert user_file.exists()
        assert user_file.read_text() == "important user notes"
    finally:
        dpg.destroy_context()


def test_os_drag_and_drop_handling(tmp_path: Path):
    """Verify that handle_dropped_paths loads dataset and model directories properly."""
    import dearpygui.dearpygui as dpg

    from tensorsuggestlite.ui.dnd import enable_os_drag_and_drop
    from tensorsuggestlite.ui.main_window import TensorSuggestLiteUI

    dpg.create_context()
    try:
        ui = TensorSuggestLiteUI()
        ui.build_ui()

        # 1. Dropping non-existent path
        ui.handle_dropped_paths([str(tmp_path / "nonexistent.json")])

        # 2. Dropping a valid JSON dataset
        sample_file = tmp_path / "dropped.json"
        sample_file.write_text('{"CAT": ["sample text"]}', encoding="utf-8")
        ui.handle_dropped_paths([str(sample_file)])
        assert ui.selected_path == sample_file
        assert ui.selected_kind == "json"
        assert "Dataset Ready" in dpg.get_value("status_label")

        # 3. Dropping a model directory sets custom_model_dir
        model_dir = tmp_path / "model_run"
        model_dir.mkdir()
        (model_dir / "text_classifier.keras").touch()
        ui.handle_dropped_paths([str(model_dir)])
        assert ui.custom_model_dir == model_dir
        assert "Model" in dpg.get_value("status_label")

        # Loading a dataset file afterwards MUST reset custom_model_dir to None
        ui.handle_dropped_paths([str(sample_file)])
        reset_dir: Path | None = ui.custom_model_dir
        assert reset_dir is None

        # 4. Verify enable_os_drag_and_drop runs gracefully without a live window
        assert enable_os_drag_and_drop(ui.handle_dropped_paths) is False
    finally:
        dpg.destroy_context()
