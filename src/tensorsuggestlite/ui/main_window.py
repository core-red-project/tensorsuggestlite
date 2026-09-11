"""Main application window using Clean Architecture, native fonts, and drag & drop."""

import shutil
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QCursor, QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from tensorsuggestlite.core.entities import TrainingConfig, TrainingResult
from tensorsuggestlite.core.profiler import DatasetProfiler
from tensorsuggestlite.ui.theme import apply_theme, get_icon, get_mono_font, get_ui_font
from tensorsuggestlite.ui.worker import TaskWorker
from tensorsuggestlite.use_cases import ExportTFLiteUseCase, TrainModelUseCase


class TensorSuggestLiteUI(QWidget):
    """Refactored main application window with polished UX."""

    def __init__(
        self,
        train_use_case: TrainModelUseCase | None = None,
        export_use_case: ExportTFLiteUseCase | None = None,
    ) -> None:
        super().__init__()
        self.train_use_case = train_use_case or TrainModelUseCase()
        self.export_use_case = export_use_case or ExportTFLiteUseCase()

        self.setWindowTitle("TensorSuggestLite")
        self.resize(900, 600)
        self.setMinimumSize(860, 560)
        self.setAcceptDrops(True)

        # State
        self.selected_path: Path | None = None
        self.selected_kind: str | None = None
        self.current_theme: str = "dark"
        self._active_worker: TaskWorker | None = None

        self._build_ui()
        self._apply_current_theme()

    def _get_project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent.parent.parent

    def _get_generated_dir(self) -> Path:
        kind = self.selected_kind or "json"
        return self._get_project_root() / "generated" / kind

    def _apply_current_theme(self) -> None:
        apply_theme(self, self.current_theme)

    def toggle_theme(self) -> None:
        self.current_theme = "light" if self.current_theme == "dark" else "dark"
        self._apply_current_theme()
        self._update_theme_toggle_icon()

    def _update_theme_toggle_icon(self) -> None:
        """Update toggle button icon to reflect opposite theme."""
        icon_name = "sun.svg" if self.current_theme == "dark" else "moon.svg"
        toggle_icon = get_icon(icon_name)
        if not toggle_icon.isNull():
            self.theme_toggle.setIcon(toggle_icon)
            self.theme_toggle.setIconSize(QSize(20, 20))
        else:
            self.theme_toggle.setText("☀️" if self.current_theme == "dark" else "🌙")
        tt = "Switch to light mode" if self.current_theme == "dark" else "Switch to dark mode"
        self.theme_toggle.setToolTip(tt)

    def dragEnterEvent(self, a0: QDragEnterEvent | None) -> None:
        """Handle drag enter for accepted file formats."""
        if a0 is None:
            return
        mime = a0.mimeData()
        if mime and mime.hasUrls():
            for url in mime.urls():
                ext = Path(url.toLocalFile()).suffix.lower()
                if ext in {".json", ".yaml", ".yml", ".toml"}:
                    a0.acceptProposedAction()
                    return
        a0.ignore()

    def dropEvent(self, a0: QDropEvent | None) -> None:
        """Handle drop event to load configuration file."""
        if a0 is None:
            return
        mime = a0.mimeData()
        if mime and mime.hasUrls():
            for url in mime.urls():
                path = Path(url.toLocalFile())
                ext = path.suffix.lower()
                if ext in {".json", ".yaml", ".yml", ".toml"}:
                    kind = "yaml" if ext in {".yaml", ".yml"} else ext.lstrip(".")
                    self._load_file(path, kind)
                    a0.acceptProposedAction()
                    return

    def _build_ui(self) -> None:
        title_font = get_ui_font(16, bold=True)
        copy_font = get_ui_font(9)
        btn_font = get_ui_font(11, bold=True)
        path_font = get_ui_font(10)
        status_font = get_ui_font(10, bold=True)

        app_icon_svg = get_icon("TensorSuggestLite.svg")
        if not app_icon_svg.isNull():
            self.setWindowIcon(app_icon_svg)

        # App bar
        app_bar = QWidget(self)
        app_bar.setObjectName("appBar")
        app_bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        app_bar_layout = QHBoxLayout(app_bar)
        app_bar_layout.setContentsMargins(12, 8, 12, 8)

        self.title_label = QLabel("TensorSuggestLite")
        self.title_label.setFont(title_font)
        self.title_label.setObjectName("appTitle")

        self.app_icon = QLabel(self)
        self.app_icon.setObjectName("appIcon")
        if not app_icon_svg.isNull():
            self.app_icon.setPixmap(app_icon_svg.pixmap(QSize(28, 28)))
            self.app_icon.setFixedSize(28, 28)
            self.app_icon.setScaledContents(True)
            self.app_icon.setVisible(True)
        else:
            self.app_icon.setVisible(False)

        self.theme_toggle = QToolButton(self)
        self.theme_toggle.setObjectName("themeToggle")
        self.theme_toggle.setFixedSize(36, 32)
        self.theme_toggle.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.theme_toggle.clicked.connect(self.toggle_theme)
        self._update_theme_toggle_icon()

        app_bar_layout.addWidget(self.app_icon)
        app_bar_layout.addWidget(self.title_label)
        app_bar_layout.addStretch(1)
        app_bar_layout.addWidget(self.theme_toggle)

        # Left panel
        left_panel = QWidget(self)
        left_panel.setObjectName("leftPanel")
        left_panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)

        self.btn_json = QPushButton("Import JSON")
        self.btn_toml = QPushButton("Import TOML")
        self.btn_yaml = QPushButton("Import YAML")
        for btn, kind in (
            (self.btn_json, "json"),
            (self.btn_toml, "toml"),
            (self.btn_yaml, "yaml"),
        ):
            btn.setFont(btn_font)
            btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            btn.setFixedHeight(38)
            btn.setObjectName("neumoButton")
            btn.clicked.connect(lambda _, k=kind: self._on_import_clicked(k))

        import_row = QHBoxLayout()
        import_row.setSpacing(10)
        import_row.addWidget(self.btn_json)
        import_row.addWidget(self.btn_toml)
        import_row.addWidget(self.btn_yaml)

        self.path_field = QLineEdit()
        self.path_field.setReadOnly(True)
        self.path_field.setFont(path_font)
        self.path_field.setPlaceholderText("Import or drag & drop configuration file...")
        self.path_field.setObjectName("pathField")
        self.path_field.setFixedHeight(38)

        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setFont(get_mono_font(10))
        self.log_widget.setObjectName("consoleBox")
        self.log_widget.setPlaceholderText("Console output...")

        self.copyright_label = QLabel("TensorSuggestLite © Core Red Project 2026")
        self.copyright_label.setFont(copy_font)
        self.copyright_label.setObjectName("copyrightLabel")
        self.copyright_label.setFixedHeight(16)
        self.copyright_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Model Playground section
        playground_box = QWidget()
        playground_box.setObjectName("playgroundBox")
        playground_layout = QVBoxLayout(playground_box)
        playground_layout.setContentsMargins(0, 4, 0, 4)
        playground_layout.setSpacing(6)

        playground_row = QHBoxLayout()
        playground_row.setSpacing(8)

        self.playground_input = QLineEdit()
        self.playground_input.setFont(path_font)
        self.playground_input.setPlaceholderText("Test text (e.g. 'This product is excellent!')...")
        self.playground_input.setFixedHeight(36)
        self.playground_input.setObjectName("pathField")
        self.playground_input.returnPressed.connect(self._on_predict_clicked)

        self.playground_button = QPushButton("Test Inference")
        self.playground_button.setFont(btn_font)
        self.playground_button.setObjectName("neumoButton")
        self.playground_button.setFixedHeight(36)
        self.playground_button.setEnabled(False)
        self.playground_button.clicked.connect(self._on_predict_clicked)

        self.compare_button = QPushButton("Compare Keras vs TFLite")
        self.compare_button.setFont(btn_font)
        self.compare_button.setObjectName("neumoButton")
        self.compare_button.setFixedHeight(36)
        self.compare_button.setEnabled(False)
        self.compare_button.clicked.connect(self._on_compare_clicked)

        playground_row.addWidget(self.playground_input, 1)
        playground_row.addWidget(self.playground_button)
        playground_row.addWidget(self.compare_button)

        self.playground_result = QLabel("Playground: Train or load a model to test inference")
        self.playground_result.setFont(copy_font)
        self.playground_result.setObjectName("playgroundLabel")
        self.playground_result.setAlignment(Qt.AlignmentFlag.AlignLeft)

        playground_layout.addLayout(playground_row)
        playground_layout.addWidget(self.playground_result)

        left_layout.addLayout(import_row)
        left_layout.addWidget(self.path_field)
        left_layout.addWidget(self.log_widget, 1)
        left_layout.addWidget(playground_box)
        left_layout.addWidget(self.copyright_label)

        # Right panel
        right_panel = QWidget(self)
        right_panel.setObjectName("rightPanel")
        right_panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(8)

        # Hyperparameters controls
        hp_group = QGroupBox("Hyperparameters")
        hp_group.setFont(copy_font)
        hp_layout = QFormLayout(hp_group)
        hp_layout.setContentsMargins(6, 6, 6, 6)
        hp_layout.setSpacing(4)

        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 100)
        self.spin_epochs.setValue(12)
        self.spin_epochs.setFont(path_font)

        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setRange(0.0001, 0.1)
        self.spin_lr.setSingleStep(0.0005)
        self.spin_lr.setDecimals(4)
        self.spin_lr.setValue(0.001)
        self.spin_lr.setFont(path_font)

        self.spin_dropout = QDoubleSpinBox()
        self.spin_dropout.setRange(0.0, 0.8)
        self.spin_dropout.setSingleStep(0.05)
        self.spin_dropout.setDecimals(2)
        self.spin_dropout.setValue(0.20)
        self.spin_dropout.setFont(path_font)

        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(4, 128)
        self.spin_batch.setSingleStep(4)
        self.spin_batch.setValue(16)
        self.spin_batch.setFont(path_font)

        hp_layout.addRow("Epochs:", self.spin_epochs)
        hp_layout.addRow("Learning Rate:", self.spin_lr)
        hp_layout.addRow("Dropout:", self.spin_dropout)
        hp_layout.addRow("Batch Size:", self.spin_batch)

        self.train_button = QPushButton("Train Model")
        self.train_button.setFont(btn_font)
        self.train_button.setEnabled(False)
        self.train_button.setObjectName("neumoButton")
        self.train_button.clicked.connect(self._on_train_clicked)

        self.convert_button = QPushButton("Generate TFLite")
        self.convert_button.setFont(btn_font)
        self.convert_button.setEnabled(False)
        self.convert_button.setObjectName("neumoButton")
        self.convert_button.clicked.connect(self._on_convert_clicked)

        self.clear_button = QPushButton("Clear Folders")
        self.clear_button.setFont(btn_font)
        self.clear_button.setEnabled(True)
        self.clear_button.setObjectName("neumoButton")
        self.clear_button.clicked.connect(self._on_clear_generated_clicked)

        self.clear_screen_button = QPushButton("Clear Screen")
        self.clear_screen_button.setFont(btn_font)
        self.clear_screen_button.setEnabled(True)
        self.clear_screen_button.setObjectName("neumoButton")
        self.clear_screen_button.clicked.connect(self._clear_console)

        self.export_button = QPushButton("Export TFLite")
        self.export_button.setFont(btn_font)
        self.export_button.setVisible(False)
        self.export_button.setEnabled(False)
        self.export_button.setObjectName("neumoButton")
        self.export_button.clicked.connect(self._on_export_clicked)

        right_layout.addWidget(hp_group)
        right_layout.addWidget(self.train_button)
        right_layout.addWidget(self.convert_button)
        right_layout.addWidget(self.clear_button)
        right_layout.addWidget(self.clear_screen_button)
        right_layout.addStretch(1)
        right_layout.addWidget(self.export_button)

        self.status_label = QLabel("model not trained")
        self.status_label.setFont(status_font)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setObjectName("statusLabel")
        self.status_label.setFixedHeight(22)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFixedHeight(18)
        self.progress.setObjectName("progressBar")

        right_layout.addWidget(self.status_label)
        right_layout.addWidget(self.progress)

        main_split = QHBoxLayout()
        main_split.setSpacing(12)
        main_split.addWidget(left_panel, 3)
        main_split.addWidget(right_panel, 1)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(app_bar)
        root_layout.addLayout(main_split)
        self.setLayout(root_layout)

    def _append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        self.log_widget.append(formatted)
        cursor = self.log_widget.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.log_widget.setTextCursor(cursor)

    def _clear_console(self) -> None:
        self.log_widget.clear()

    def _on_import_clicked(self, kind: str) -> None:
        filters = {
            "json": "JSON files (*.json)",
            "yaml": "YAML files (*.yaml *.yml)",
            "toml": "TOML files (*.toml)",
        }
        file_filter = filters.get(kind, "All files (*.*)")
        selected, _ = QFileDialog.getOpenFileName(
            self, f"Import {kind.upper()} File", str(self._get_project_root()), file_filter
        )
        if selected:
            self._load_file(Path(selected), kind)

    def _load_file(self, path: Path, kind: str) -> None:
        self.selected_path = path
        self.selected_kind = kind
        self.path_field.setText(str(self.selected_path))
        self._append_log(f"Imported {kind.upper()}: {self.selected_path.name}")

        # Quick validation and deep inspection
        try:
            parser = self.train_use_case.registry.get_parser_for_path(path)
            dataset = parser.parse(path)
            profile = DatasetProfiler.profile(dataset)
            cat_summary = ", ".join(
                f"{cat} ({count})" for cat, count in profile.category_counts.items()
            )
            self._append_log(
                f"Dataset inspection: {profile.total_samples} samples across {len(profile.category_counts)} classes [{cat_summary}]"
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
            self.train_button.setEnabled(True)
        except Exception as err:
            self._append_log(f"Validation error: {err}")
            self.train_button.setEnabled(False)

        self._check_existing_artifacts()

    def _check_existing_artifacts(self) -> None:
        gen_dir = self._get_generated_dir()
        keras_file = gen_dir / "text_classifier.keras"
        tflite_file = gen_dir / "text_classifier.tflite"

        has_keras = keras_file.is_file()
        has_tflite = tflite_file.is_file()

        if has_tflite:
            self.convert_button.setEnabled(False)
            self.export_button.setVisible(True)
            self.export_button.setEnabled(True)
            self.status_label.setText("tflite ready")
        elif has_keras:
            self.convert_button.setEnabled(True)
            self.export_button.setVisible(False)
            self.export_button.setEnabled(False)
            self.status_label.setText("model trained")
        else:
            self.convert_button.setEnabled(False)
            self.export_button.setVisible(False)
            self.export_button.setEnabled(False)
            self.status_label.setText("model not trained")

        self.playground_button.setEnabled(has_keras or has_tflite)
        self.compare_button.setEnabled(has_keras and has_tflite)

        if has_keras and has_tflite:
            self.playground_result.setText(
                "Playground: Keras & TFLite loaded. Ready for cross-validation."
            )
        elif has_keras:
            self.playground_result.setText(
                "Playground: Keras model loaded. Generate TFLite to enable comparison."
            )
        else:
            self.playground_result.setText("Playground: Train or load a model to test inference")

    def _on_predict_clicked(self) -> None:
        text = self.playground_input.text().strip()
        if not text:
            return

        gen_dir = self._get_generated_dir()
        try:
            from tensorsuggestlite.use_cases import PredictTextUseCase

            use_case = PredictTextUseCase(gen_dir)
            res = use_case.predict(text, prefer_tflite=True)
            self.playground_result.setText(
                f"Prediction: {res.top_category} ({res.top_confidence * 100:.1f}%) [{res.runtime.upper()} - {res.latency_ms:.1f}ms]"
            )
            breakdown = ", ".join(f"{s.category}: {s.confidence * 100:.1f}%" for s in res.scores)
            self._append_log(
                f'Playground: "{text}" -> {res.top_category} ({res.top_confidence * 100:.1f}%) [{breakdown}]'
            )
        except Exception as err:
            self.playground_result.setText(f"Prediction error: {err}")
            self._append_log(f"Prediction failed: {err}")

    def _on_compare_clicked(self) -> None:
        text = self.playground_input.text().strip()
        if not text:
            return

        gen_dir = self._get_generated_dir()
        try:
            from tensorsuggestlite.use_cases import PredictTextUseCase

            use_case = PredictTextUseCase(gen_dir)
            cross = use_case.compare(text)
            k = cross.keras_result
            t = cross.tflite_result
            status = "MATCH" if cross.is_consistent else "MISMATCH"
            self.playground_result.setText(
                f"[{status}] Keras: {k.top_category} ({k.top_confidence * 100:.1f}%) | TFLite: {t.top_category} ({t.top_confidence * 100:.1f}%) | Δ: {cross.delta_confidence * 100:.1f}%"
            )
            self._append_log(
                f'Cross-Validation: "{text}" -> {status} (Δ={cross.delta_confidence * 100:.2f}%) [Keras: {k.top_category} {k.top_confidence * 100:.1f}% in {k.latency_ms:.1f}ms | TFLite: {t.top_category} {t.top_confidence * 100:.1f}% in {t.latency_ms:.1f}ms]'
            )
        except Exception as err:
            self.playground_result.setText(f"Comparison error: {err}")
            self._append_log(f"Cross-validation failed: {err}")

    def _on_train_clicked(self) -> None:
        if not self.selected_path:
            return

        self.train_button.setEnabled(False)
        self.convert_button.setEnabled(False)
        self.status_label.setText("training model...")
        self.progress.setValue(0)

        gen_dir = self._get_generated_dir()
        cfg = TrainingConfig(
            epochs=self.spin_epochs.value(),
            learning_rate=self.spin_lr.value(),
            dropout_rate=self.spin_dropout.value(),
            batch_size=self.spin_batch.value(),
        )

        self._active_worker = TaskWorker(
            self.train_use_case.execute,
            config_path=self.selected_path,
            output_dir=gen_dir,
            config=cfg,
        )

        self._active_worker.progress.connect(self.progress.setValue)
        self._active_worker.log_message.connect(self._append_log)
        self._active_worker.finished_success.connect(self._on_train_success)
        self._active_worker.failed.connect(self._on_task_failed)
        self._active_worker.start()

    def _on_train_success(self, result: object) -> None:
        if isinstance(result, TrainingResult):
            self._append_log(
                f"Training complete! {result.items_serialized} samples trained across {result.labels_count} categories."
            )
        self.train_button.setEnabled(True)
        self.convert_button.setEnabled(True)
        self.status_label.setText("model trained")
        self.progress.setValue(100)

    def _on_convert_clicked(self) -> None:
        gen_dir = self._get_generated_dir()
        self.convert_button.setEnabled(False)
        self.status_label.setText("converting tflite...")
        self.progress.setValue(0)

        self._active_worker = TaskWorker(
            self.export_use_case.execute,
            model_dir=gen_dir,
        )
        self._active_worker.progress.connect(self.progress.setValue)
        self._active_worker.log_message.connect(self._append_log)
        self._active_worker.finished_success.connect(self._on_convert_success)
        self._active_worker.failed.connect(self._on_task_failed)
        self._active_worker.start()

    def _on_convert_success(self, dest: object) -> None:
        dest_path = Path(str(dest))
        size_kb = dest_path.stat().st_size / 1024.0 if dest_path.is_file() else 0.0
        self._append_log(f"TFLite generated successfully ({size_kb:.1f} KB): {dest_path.name}")
        self.convert_button.setEnabled(False)
        self.export_button.setVisible(True)
        self.export_button.setEnabled(True)
        self.status_label.setText("tflite ready")
        self.progress.setValue(100)

    def _on_export_clicked(self) -> None:
        gen_dir = self._get_generated_dir()
        src_tflite = gen_dir / "text_classifier.tflite"
        if not src_tflite.is_file():
            self._append_log("Error: TFLite model file does not exist to export.")
            return

        target, _ = QFileDialog.getSaveFileName(
            self, "Export TFLite Model", "text_classifier.tflite", "TFLite model (*.tflite)"
        )
        if not target:
            return

        try:
            shutil.copyfile(src_tflite, target)
            self._append_log(f"Exported model to: {target}")
            self.status_label.setText("model exported")
        except Exception as err:
            self._append_log(f"Failed to export model: {err}")

    def _on_task_failed(self, error_msg: str) -> None:
        self._append_log(f"Operation failed: {error_msg}")
        self.status_label.setText("error")
        self.train_button.setEnabled(self.selected_path is not None)
        self._check_existing_artifacts()

    def _on_clear_generated_clicked(self) -> None:
        gen_dir = self._get_project_root() / "generated"
        if gen_dir.exists():
            try:
                shutil.rmtree(gen_dir)
                self._append_log("Cleared generated output directory.")
            except Exception as err:
                self._append_log(f"Error clearing folders: {err}")
        self._check_existing_artifacts()
        self.progress.setValue(0)
