# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Main application controller using Dear PyGui, responsive layout, and zero outer scrollbars."""

from __future__ import annotations

import contextlib
import shutil
from datetime import datetime
from pathlib import Path

import dearpygui.dearpygui as dpg

from tensorsuggestlite.core.entities import TrainingConfig, TrainingResult
from tensorsuggestlite.core.profiler import DatasetProfiler
from tensorsuggestlite.ui.layout import UILayoutBuilder
from tensorsuggestlite.ui.native_dialogs import NativeDialogBridge
from tensorsuggestlite.ui.theme import apply_theme
from tensorsuggestlite.ui.worker import TaskWorker
from tensorsuggestlite.use_cases import ExportTFLiteUseCase, TrainModelUseCase

KNOWN_MODEL_ARTIFACTS: tuple[str, ...] = (
    "text_classifier.keras",
    "text_classifier.tflite",
    "tokenizer.json",
    "label_encoder.json",
    "model_metadata.json",
)


class TensorSuggestLiteUI:
    """Dear PyGui desktop controller with clean architecture and zero outer scrollbars."""

    def __init__(
        self,
        train_use_case: TrainModelUseCase | None = None,
        export_use_case: ExportTFLiteUseCase | None = None,
    ) -> None:
        """Initialize use cases, state, and UI layout."""
        self.train_use_case = train_use_case or TrainModelUseCase()
        self.export_use_case = export_use_case or ExportTFLiteUseCase()

        # State
        self.selected_path: Path | None = None
        self.selected_kind: str | None = None
        self.custom_model_dir: Path | None = None
        self.current_theme: str = "dark"
        self._active_worker: TaskWorker | None = None
        self._console_lines: list[str] = []
        self._cached_use_case: object | None = None
        self._cached_use_case_dir: Path | None = None

    def _get_predict_use_case(self, target_dir: Path) -> object:
        """Obtain or reuse cached PredictTextUseCase to prevent freezing UI on every inference."""
        from tensorsuggestlite.use_cases import PredictTextUseCase

        if self._cached_use_case is None or self._cached_use_case_dir != target_dir:
            self._cached_use_case = PredictTextUseCase(target_dir)
            self._cached_use_case_dir = target_dir
        return self._cached_use_case

    def _invalidate_predict_cache(self) -> None:
        """Reset cached inference use case upon model retraining, conversion, or artifact clearing."""
        self._cached_use_case = None
        self._cached_use_case_dir = None

    def _get_generated_dir(self) -> Path:
        """Return the target directory for trained model artifacts matching CLI standard."""
        if self.custom_model_dir and self.custom_model_dir.is_dir():
            return self.custom_model_dir

        from tensorsuggestlite.core.paths import get_default_models_dir

        if self.selected_path:
            target_dir = get_default_models_dir() / self.selected_path.stem
        else:
            kind = self.selected_kind or "json"
            target_dir = get_default_models_dir() / kind

        with contextlib.suppress(OSError):
            target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

    def toggle_theme(self) -> None:
        """Toggle between light and dark theme modes."""
        self.current_theme = "light" if self.current_theme == "dark" else "dark"
        apply_theme(self.current_theme)
        if dpg.does_item_exist("btn_theme_toggle"):
            next_tex = "moon_icon_texture" if self.current_theme == "light" else "sun_icon_texture"
            if dpg.does_item_exist(next_tex):
                dpg.configure_item("btn_theme_toggle", texture_tag=next_tex)

    def on_viewport_resize(self) -> None:
        """Dynamically expand console window to absorb vertical space without outer scrollbars."""
        try:
            client_h = dpg.get_viewport_client_height()
        except Exception:
            return

        content_h = max(440, client_h - 56)
        console_h = max(70, content_h - 325)

        if dpg.does_item_exist("left_panel_child"):
            dpg.configure_item("left_panel_child", height=content_h)
        if dpg.does_item_exist("right_panel_child"):
            dpg.configure_item("right_panel_child", height=content_h)
        if dpg.does_item_exist("console_window"):
            dpg.configure_item("console_window", height=console_h)

    def _append_log(self, message: str) -> None:
        """Append formatted timestamped message to the console widget."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        self._console_lines.append(formatted)
        if len(self._console_lines) > 250:
            self._console_lines.pop(0)

        log_content = "\n".join(self._console_lines)
        if dpg.does_item_exist("console_text"):
            dpg.set_value("console_text", log_content)

    def _clear_console(self) -> None:
        """Clear all messages from the console widget."""
        self._console_lines.clear()
        if dpg.does_item_exist("console_text"):
            dpg.set_value("console_text", "")

    def _on_import_clicked(self, kind: str) -> None:
        """Open native OS file dialog and load dataset if selected."""
        selected = NativeDialogBridge.open_file_dialog(
            title=f"Import {kind.upper()} Dataset",
            file_types=[kind, "yml"] if kind == "yaml" else [kind],
        )
        if selected:
            self._load_file(selected, kind)

    def _on_load_model_clicked(self) -> None:
        """Open native OS directory dialog to load an existing model directory."""
        selected_dir = NativeDialogBridge.open_directory_dialog(
            title="Select Existing Model Directory (containing text_classifier.keras or .tflite)"
        )
        if not selected_dir:
            return

        keras_file = selected_dir / "text_classifier.keras"
        tflite_file = selected_dir / "text_classifier.tflite"

        if not keras_file.is_file() and not tflite_file.is_file():
            self._append_log(
                f"[ERROR] Selected directory does not contain text_classifier.keras or .tflite: {selected_dir}"
            )
            return

        self.custom_model_dir = selected_dir
        dpg.set_value("path_display", f"Model dir: {selected_dir.name} ({selected_dir})")
        self._append_log(f"[MODEL] Model directory loaded into Playground: {selected_dir}")
        dpg.set_value(
            "status_label", f"Status: Model Directory Loaded ({selected_dir.name}) [READY]"
        )
        self._check_existing_artifacts()

    def handle_dropped_paths(self, paths: list[str]) -> None:
        """Handle files or folders dropped into the application window via OS Drag & Drop.

        Args:
            paths: List of absolute paths dropped from Finder or File Explorer.
        """
        for raw_path in paths:
            path = Path(raw_path).resolve()
            if not path.exists():
                continue

            if path.is_dir():
                if (
                    (path / "text_classifier.keras").is_file()
                    or (path / "text_classifier.tflite").is_file()
                    or (path / "model_metadata.json").is_file()
                ):
                    self.custom_model_dir = path
                    dpg.set_value("path_display", f"Model dir: {path.name} ({path})")
                    self._append_log(f"[DROP (Experimental)] Loaded model directory: {path}")
                    dpg.set_value(
                        "status_label", f"Status: Model Directory Loaded ({path.name}) [READY]"
                    )
                    self._check_existing_artifacts()
                    return
                self._append_log(
                    f"[WARN] Drop directory does not contain model artifacts: {path.name}"
                )
            elif path.is_file():
                ext = path.suffix.lower().lstrip(".")
                if ext in {"json", "yaml", "yml", "toml"}:
                    kind = "yaml" if ext == "yml" else ext
                    self._load_file(path, kind)
                    return
                elif ext in {"keras", "tflite"}:
                    self.custom_model_dir = path.parent
                    dpg.set_value("path_display", f"Model dir: {path.parent.name} ({path.parent})")
                    self._append_log(f"[DROP (Experimental)] Loaded model artifact: {path.name}")
                    dpg.set_value(
                        "status_label", f"Status: Model Artifact Loaded ({path.name}) [READY]"
                    )
                    self._check_existing_artifacts()
                    return
                else:
                    self._append_log(
                        f"[WARN] Drop unsupported file extension '{path.suffix}' ({path.name})"
                    )

    def _load_file(self, path: Path, kind: str) -> None:
        """Validate and profile the loaded dataset file, updating UI state."""
        self.selected_path = path
        self.selected_kind = kind
        self.custom_model_dir = None
        self._invalidate_predict_cache()
        dpg.set_value("path_display", str(self.selected_path))
        self._append_log(f"[IMPORT] Loaded {kind.upper()}: {self.selected_path.name}")

        try:
            from tensorsuggestlite.parsers.base import parse_skeleton_str

            skel_str = (
                dpg.get_value("input_skeleton").strip()
                if dpg.does_item_exist("input_skeleton")
                else ""
            )
            skel_dict = parse_skeleton_str(skel_str)

            parser = self.train_use_case.registry.get_parser_for_path(path)
            dataset = parser.parse(path, skeleton=skel_dict)
            profile = DatasetProfiler.profile(dataset)
            cat_summary = ", ".join(
                f"{cat} ({count})" for cat, count in profile.category_counts.items()
            )
            self._append_log(
                f"[READY] Dataset inspection: {profile.total_samples} samples across {len(profile.category_counts)} classes [{cat_summary}]"
            )
            if profile.imbalance_warning:
                self._append_log(f"[WARN] {profile.imbalance_warning}")
            if profile.duplicate_samples:
                self._append_log(
                    f"[INFO] {len(profile.duplicate_samples)} duplicate sample(s) detected."
                )
            if profile.contradictory_samples:
                self._append_log(
                    f"[ALERT] {len(profile.contradictory_samples)} contradictory sample(s) with conflicting labels!"
                )
            dpg.configure_item("btn_train", enabled=True)
            status_text = f"Status: Dataset Ready ({profile.total_samples} samples, {len(profile.category_counts)} classes) [VALIDATED]"
            dpg.set_value("status_label", status_text)
        except Exception as err:
            self._append_log(f"[ERROR] Validation error: {err}")
            dpg.configure_item("btn_train", enabled=False)
            dpg.set_value("status_label", f"Status: Validation Error ({err})")

        self._check_existing_artifacts(update_status=False)

    def _check_existing_artifacts(self, update_status: bool = True) -> None:
        """Inspect generated directory and enable action buttons based on existing artifacts."""
        gen_dir = self._get_generated_dir()
        keras_file = gen_dir / "text_classifier.keras"
        tflite_file = gen_dir / "text_classifier.tflite"

        has_keras = keras_file.is_file()
        has_tflite = tflite_file.is_file()

        if has_tflite:
            dpg.configure_item("btn_convert", enabled=False)
            dpg.configure_item("btn_export", enabled=True, show=True)
            if update_status and not self._active_worker:
                dpg.set_value("status_label", "Status: TFLite Ready [EXPORTABLE]")
        elif has_keras:
            dpg.configure_item("btn_convert", enabled=True)
            dpg.configure_item("btn_export", enabled=False, show=False)
            if update_status and not self._active_worker:
                dpg.set_value("status_label", "Status: Model Trained [READY TO CONVERT]")
        else:
            dpg.configure_item("btn_convert", enabled=False)
            dpg.configure_item("btn_export", enabled=False, show=False)
            if update_status and not self._active_worker:
                if self.selected_path:
                    dpg.set_value("status_label", "Status: Dataset Ready [READY TO TRAIN]")
                else:
                    dpg.set_value("status_label", "Status: Idle (No dataset loaded)")

        can_predict = has_keras or has_tflite
        dpg.configure_item("btn_predict", enabled=can_predict)
        dpg.configure_item("btn_compare", enabled=(has_keras and has_tflite))

    def _on_predict_clicked(self) -> None:
        """Run inference on the text in Playground input using available model artifacts."""
        raw_text = dpg.get_value("input_playground").strip()
        if not raw_text:
            dpg.set_value("playground_result", "Enter text to run classification.")
            return

        gen_dir = self._get_generated_dir()
        try:
            from tensorsuggestlite.use_cases import PredictTextUseCase

            use_case = self._get_predict_use_case(gen_dir)
            assert isinstance(use_case, PredictTextUseCase)
            sentences = [s.strip() for s in raw_text.replace("||", "\n").splitlines() if s.strip()]

            if len(sentences) > 1:
                batch_res = use_case.predict_batch(sentences, prefer_tflite=True)
                self._append_log(f"--- Batch Inference ({len(batch_res)} items) ---")
                for idx, item in enumerate(batch_res, 1):
                    self._append_log(
                        f'  [{idx}] "{item.text}" -> {item.top_category} ({item.top_confidence * 100:.1f}%) [{item.runtime.upper()}]'
                    )
                dpg.set_value(
                    "playground_result",
                    f"✓ Batch Complete: {len(batch_res)} items classified (see Console)",
                )
                dpg.set_value(
                    "status_label",
                    f"Status: Batch Inference Complete ({len(batch_res)} items) [DONE]",
                )
                return

            res = use_case.predict(raw_text, prefer_tflite=True)
            dpg.set_value(
                "playground_result",
                f"✓ Prediction: {res.top_category} ({res.top_confidence * 100:.1f}%) [{res.runtime.upper()} - {res.latency_ms:.1f}ms]",
            )
            dpg.set_value(
                "status_label",
                f"Status: Inference OK ({res.top_category} {res.top_confidence * 100:.1f}%)",
            )
            breakdown = ", ".join(f"{s.category}: {s.confidence * 100:.1f}%" for s in res.scores)
            self._append_log(
                f'[INFERENCE] "{raw_text}" -> {res.top_category} ({res.top_confidence * 100:.1f}%) [{breakdown}]'
            )
        except Exception as err:
            dpg.set_value("playground_result", f"Prediction error: {err}")
            self._append_log(f"[ERROR] Prediction failed: {err}")
            dpg.set_value("status_label", f"Status: Prediction Error ({err})")

    def _on_compare_clicked(self) -> None:
        """Run cross-validation comparing Keras and TFLite inference outputs."""
        text = dpg.get_value("input_playground").strip()
        if not text:
            return

        gen_dir = self._get_generated_dir()
        try:
            from tensorsuggestlite.use_cases import PredictTextUseCase

            use_case = self._get_predict_use_case(gen_dir)
            assert isinstance(use_case, PredictTextUseCase)
            cross = use_case.compare(text)
            k = cross.keras_result
            t = cross.tflite_result
            status = "MATCH" if cross.is_consistent else "MISMATCH"
            dpg.set_value(
                "playground_result",
                f"[{status}] Keras: {k.top_category} ({k.top_confidence * 100:.1f}%) | TFLite: {t.top_category} ({t.top_confidence * 100:.1f}%) | Delta: {cross.delta_confidence * 100:.1f}%",
            )
            dpg.set_value(
                "status_label",
                f"Status: Cross-Validation [{status}] (Δ {cross.delta_confidence * 100:.1f}%)",
            )
            self._append_log(
                f'[COMPARE] "{text}" -> {status} (Delta={cross.delta_confidence * 100:.2f}%) [Keras: {k.top_category} in {k.latency_ms:.1f}ms | TFLite: {t.top_category} in {t.latency_ms:.1f}ms]'
            )
        except Exception as err:
            dpg.set_value("playground_result", f"Comparison error: {err}")
            self._append_log(f"[ERROR] Cross-validation failed: {err}")
            dpg.set_value("status_label", f"Status: Comparison Error ({err})")

    def _on_train_clicked(self) -> None:
        """Launch background worker thread to train model on current dataset."""
        if not self.selected_path:
            return

        dpg.configure_item("btn_train", enabled=False, label="Training in Progress...")
        dpg.configure_item("btn_convert", enabled=False)
        dpg.set_value("status_label", "Status: Training Model... (0%)")
        dpg.set_value("progress_bar", 0.0)

        gen_dir = self._get_generated_dir()
        try:
            cfg = TrainingConfig(
                epochs=int(dpg.get_value("hp_epochs")),
                learning_rate=float(dpg.get_value("hp_lr")),
                dropout_rate=float(dpg.get_value("hp_dropout")),
                batch_size=int(dpg.get_value("hp_batch")),
            )
        except ValueError as err:
            self._append_log(f"[ERROR] Invalid hyperparameter configuration: {err}")
            dpg.set_value("status_label", f"Status: Config Error ({err})")
            dpg.configure_item("btn_train", enabled=True, label="Train Model")
            dpg.configure_item("btn_convert", enabled=True)
            return

        def prog_cb(val: int) -> None:
            pct = max(0, min(100, val))
            dpg.set_value("progress_bar", pct / 100.0)
            dpg.set_value("status_label", f"Status: Training Model... ({pct}%)")

        from tensorsuggestlite.parsers.base import parse_skeleton_str

        skel_str = (
            dpg.get_value("input_skeleton").strip() if dpg.does_item_exist("input_skeleton") else ""
        )
        skel_dict = parse_skeleton_str(skel_str)

        self._append_log(
            f"[TRAIN] Initiating training: {cfg.epochs} epochs, lr={cfg.learning_rate:.4f}, batch={cfg.batch_size}, dropout={cfg.dropout_rate:.2f}"
        )

        self._active_worker = TaskWorker(
            self.train_use_case.execute,
            config_path=self.selected_path,
            output_dir=gen_dir,
            config=cfg,
            skeleton=skel_dict,
            on_progress=prog_cb,
            on_log=self._append_log,
            on_success=self._on_train_success,
            on_failed=self._on_task_failed,
        )
        self._active_worker.start()

    def _on_train_success(self, result: object) -> None:
        """Handle training completion and restore buttons."""
        self._active_worker = None
        self._invalidate_predict_cache()
        if isinstance(result, TrainingResult):
            loss = result.metrics.get("final_loss", result.metrics.get("loss", 0.0))
            acc = result.metrics.get("final_accuracy", result.metrics.get("accuracy", 1.0))
            dur = result.metrics.get("training_duration_seconds", 0.0)
            self._append_log(
                f"[DONE] Training complete in {dur:.2f}s! Loss: {loss:.4f}, Accuracy: {acc * 100:.1f}% ({result.items_serialized} samples)"
            )
            dpg.set_value(
                "status_label",
                f"Status: Model Trained [Acc: {acc * 100:.1f}% | Loss: {loss:.4f}]",
            )
        else:
            dpg.set_value("status_label", "Status: Model Trained [READY TO CONVERT]")

        dpg.configure_item("btn_train", enabled=True, label="Train Model")
        dpg.configure_item("btn_convert", enabled=True)
        dpg.set_value("progress_bar", 1.0)
        self._check_existing_artifacts(update_status=False)

    def _on_convert_clicked(self) -> None:
        """Launch background worker thread to convert Keras model into TFLite."""
        gen_dir = self._get_generated_dir()
        quant_mode = (
            dpg.get_value("combo_quantization")
            if dpg.does_item_exist("combo_quantization")
            else "dynamic"
        )
        dpg.configure_item("btn_convert", enabled=False, label="Converting TFLite...")
        dpg.set_value("status_label", f"Status: Converting to TFLite ({quant_mode})... (0%)")
        dpg.set_value("progress_bar", 0.0)

        def prog_cb(val: int) -> None:
            pct = max(0, min(100, val))
            dpg.set_value("progress_bar", pct / 100.0)
            dpg.set_value(
                "status_label", f"Status: Converting to TFLite ({quant_mode})... ({pct}%)"
            )

        self._append_log(
            f"[TFLITE] Converting Keras model to TFLite (quantization={quant_mode})..."
        )

        self._active_worker = TaskWorker(
            self.export_use_case.execute,
            model_dir=gen_dir,
            quantization=quant_mode,
            on_progress=prog_cb,
            on_log=self._append_log,
            on_success=self._on_convert_success,
            on_failed=self._on_task_failed,
        )
        self._active_worker.start()

    def _on_convert_success(self, dest: object) -> None:
        """Handle TFLite conversion success."""
        self._active_worker = None
        self._invalidate_predict_cache()
        dest_path = Path(str(dest))
        size_kb = dest_path.stat().st_size / 1024.0 if dest_path.is_file() else 0.0
        self._append_log(
            f"[DONE] TFLite model generated successfully ({size_kb:.1f} KB): {dest_path.name}"
        )
        dpg.configure_item("btn_convert", enabled=False, label="Generate TFLite")
        dpg.configure_item("btn_export", enabled=True, show=True)
        dpg.set_value("status_label", f"Status: TFLite Ready ({size_kb:.1f} KB) [EXPORTABLE]")
        dpg.set_value("progress_bar", 1.0)
        self._check_existing_artifacts(update_status=False)

    def _on_export_clicked(self) -> None:
        """Export generated TFLite model using native OS save file dialog."""
        gen_dir = self._get_generated_dir()
        src_tflite = gen_dir / "text_classifier.tflite"
        if not src_tflite.is_file():
            self._append_log("[ERROR] TFLite model file does not exist to export.")
            return

        target = NativeDialogBridge.save_file_dialog(
            title="Export TFLite Model",
            default_name="text_classifier.tflite",
        )
        if not target:
            return

        try:
            shutil.copyfile(src_tflite, target)
            self._append_log(f"[EXPORT] Exported model to: {target}")
            dpg.set_value("status_label", f"Status: Model Exported to {Path(target).name} [SAVED]")
        except Exception as err:
            self._append_log(f"[ERROR] Failed to export model: {err}")
            dpg.set_value("status_label", f"Status: Export Error ({err})")

    def _on_task_failed(self, error_msg: str) -> None:
        """Handle background worker failure by logging error and restoring controls."""
        self._active_worker = None
        self._append_log(f"[ERROR] Operation failed: {error_msg}")
        dpg.set_value("status_label", f"Status: Error ({error_msg})")
        dpg.configure_item(
            "btn_train", enabled=(self.selected_path is not None), label="Train Model"
        )
        dpg.configure_item("btn_convert", label="Generate TFLite")
        self._check_existing_artifacts()

    def _on_clear_generated_clicked(self) -> None:
        """Prompt user with modal confirmation before deleting artifacts."""
        if dpg.does_item_exist("modal_confirm_clear"):
            dpg.configure_item("modal_confirm_clear", show=True)
        else:
            self._perform_clear_folders()

    def _perform_clear_folders(self) -> None:
        """Remove generated artifact files after user confirmation."""
        if dpg.does_item_exist("modal_confirm_clear"):
            dpg.configure_item("modal_confirm_clear", show=False)
        gen_dir = self._get_generated_dir()
        if gen_dir.exists() and gen_dir.is_dir():
            cleared_files: list[str] = []
            for fname in KNOWN_MODEL_ARTIFACTS:
                fpath = gen_dir / fname
                if fpath.is_file():
                    with contextlib.suppress(OSError):
                        fpath.unlink()
                        cleared_files.append(fname)
            self._invalidate_predict_cache()
            if cleared_files:
                self._append_log(
                    f"[CLEAN] Cleared artifacts ({', '.join(cleared_files)}) in: {gen_dir}"
                )
            else:
                self._append_log(f"[CLEAN] No generated artifacts found in: {gen_dir}")
            dpg.set_value("status_label", "Status: Artifacts Cleared [CLEAN]")
        self._check_existing_artifacts(update_status=False)
        dpg.set_value("progress_bar", 0.0)

    def build_ui(self) -> None:
        """Construct visual Dear PyGui layout delegated to UILayoutBuilder."""
        UILayoutBuilder.build(self)


__all__ = ["TensorSuggestLiteUI"]
