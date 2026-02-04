"""TensorSuggestLite - Main GUI Application.

This module contains the primary PyQt6 graphical interface for TensorSuggestLite,
a tool for training TensorFlow/Keras text classification models from configuration
files (JSON, YAML, TOML).

Architecture:
    - Main window: TensorSuggestLiteUI (QWidget subclass)
    - Threading: Training and conversion run in daemon threads
    - Callbacks: Progress and logs communicated via Qt signals
    - File watching: QFileSystemWatcher monitors artifact generation

Responsibilities:
    - Import configuration files (JSON/YAML/TOML)
    - Initiate training via interpreter-specific trainers
    - Monitor generated/ directory for artifacts
    - Convert Keras models to TFLite format
    - Export TFLite models
    - Apply and toggle dark/light themes

Notes:
    - Heavy operations delegated to json_interpreter, yaml_interpreter, toml_interpreter
    - UI updates must occur on the main Qt thread (via QTimer.singleShot or signals)
    - TensorFlow is not imported here; trainers handle lazy imports
"""

import sys
import threading
import os
import queue
import shutil

# PyQt imports
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QSize, QFileSystemWatcher
from PyQt6.QtGui import QFont, QIcon, QCursor
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QLineEdit,
    QHBoxLayout,
    QVBoxLayout,
    QFileDialog,
    QProgressBar,
    QTextEdit,
    QToolButton,
)

# trainers locales
from json_interpreter import trainer as json_trainer


class TensorSuggestLiteUI(QWidget):
    """Main application window.

    Responsibilities:
    - Build main layout and widgets.
    - Route user actions (import, train, convert, export) to corresponding trainers.
    - Display logs and progress from trainers (via callbacks).
    - Monitor generated/<kind> to detect artifacts and enable actions.
    """

    training_progress = pyqtSignal(int)  # progress from 0 to 100
    log_signal = pyqtSignal(str)  # logs from threads

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TensorSuggestLite")
        self.setFixedSize(820, 520)

        # State
        self.selected_path = None
        self.selected_kind = None  # 'json' | 'yaml' | 'toml'

        # Progress queue used to transfer updates from background threads
        self._progress_queue = queue.Queue()
        self._progress_poller = QTimer(self)
        self._progress_poller.setInterval(100)
        self._progress_poller.timeout.connect(self._poll_progress_queue)
        self._progress_poller.start()

        # Watcher de archivos + fallback por polling para artefactos generados
        self._watcher = QFileSystemWatcher(self)
        self._watch_paths = set()
        self._watched_kind = None
        self._watch_poll_timer = QTimer(self)
        self._watch_poll_timer.setInterval(500)
        self._watch_poll_timer.timeout.connect(self._poll_generated)
        self._initial_all_exist = False

        # tema por defecto
        self._theme = 'dark'

        # construir interfaz
        self._build_ui()

        # aplicar estilos si existen
        self.apply_stylesheet_if_exists()

        # conectar signals al manejador (no silenciar fallos para detectar problemas)
        self.training_progress.connect(self._handle_progress)
        self.log_signal.connect(self._append_log)

    # ---------- Estilos y tema ----------
    def apply_stylesheet_if_exists(self):
        """Load and apply styles_{dark,light}.qss if it exists in package root.

        Sets 'theme' property on QApplication so QSS can reference it;
        stylesheet application is "best-effort" and failures are ignored
        to prevent UI closure.
        """
        base = os.path.abspath(os.path.dirname(__file__))
        qss_file = os.path.join('styles', 'dark.qss' if self._theme == 'dark' else 'light.qss')
        qss_path = os.path.join(base, qss_file)
        if os.path.exists(qss_path):
            try:
                with open(qss_path, 'r', encoding='utf-8') as f:
                    qss = f.read()
                app = QApplication.instance()
                if app:
                    try:
                        self.setProperty('theme', self._theme)
                        app.setProperty('theme', self._theme)
                    except Exception:
                        pass
                    try:
                        app.setStyleSheet(qss)
                    except Exception:
                        pass
                else:
                    try:
                        self.setStyleSheet(qss)
                    except Exception:
                        pass
            except Exception:
                pass

    def toggle_theme(self):
        """Toggle between 'dark' and 'light' theme and reapply stylesheet.

        The toggle shows the action icon (e.g. sun when current theme is
        dark to indicate switching to light) and updates the tooltip.
        """
        self._theme = 'dark' if self._theme == 'light' else 'light'
        app = QApplication.instance()
        if app:
            try:
                app.setProperty('theme', self._theme)
            except Exception:
                pass

        # Update toggle icon (best effort)
        try:
            icon_path = self.sun_icon_path if self._theme == 'dark' else self.moon_icon_path
            if icon_path and os.path.exists(icon_path):
                try:
                    self.theme_toggle.setIcon(QIcon(icon_path))
                except Exception:
                    pass
        except Exception:
            pass

        try:
            tt = 'Switch to light mode' if self._theme == 'dark' else 'Switch to dark mode'
            try:
                self.theme_toggle.setToolTip(tt)
            except Exception:
                pass
        except Exception:
            pass

        # Reapply QSS
        self.apply_stylesheet_if_exists()

    # ---------- UI Construction ----------
    def _build_ui(self):
        """Create and layout widgets.

        Keep layout code isolated to facilitate future refactors.
        """
        title_font = QFont("Segoe UI", 16, QFont.Weight.Bold)
        copy_font = QFont("Segoe UI", 9)
        btn_font = QFont("Segoe UI", 11)
        path_font = QFont("Segoe UI", 10)

        # Application bar
        app_bar = QWidget(self)
        app_bar.setObjectName('appBar')
        try:
            app_bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        except Exception:
            pass
        app_bar_layout = QHBoxLayout(app_bar)
        app_bar_layout.setContentsMargins(12, 8, 12, 8)

        self.title_label = QLabel('TensorSuggestLite')
        self.title_label.setFont(title_font)
        self.title_label.setObjectName('appTitle')

        # Small app icon to the left of title (leading side)
        self.app_icon = QLabel(self)
        self.app_icon.setObjectName('appIcon')
        try:
            icons_dir = os.path.join(os.path.dirname(__file__), 'icons')
            svg_path = os.path.join(icons_dir, 'TensorSuggestLite.svg')
            if os.path.exists(svg_path):
                pix = QIcon(svg_path).pixmap(QSize(24, 24))
                self.app_icon.setPixmap(pix)
                self.app_icon.setFixedSize(24, 24)
                try:
                    self.app_icon.setAlignment(Qt.AlignmentFlag.AlignVCenter)
                except Exception:
                    pass
            else:
                # If svg doesn't exist, hide widget to not affect layout
                self.app_icon.setVisible(False)
        except Exception:
            try:
                self.app_icon.setVisible(False)
            except Exception:
                pass

        # Theme toggle button (right side)
        self.theme_toggle = QToolButton(self)
        icons_dir = os.path.join(os.path.dirname(__file__), 'icons')
        self.sun_icon_path = os.path.join(icons_dir, 'sun.svg')
        self.moon_icon_path = os.path.join(icons_dir, 'moon.svg')
        try:
            if os.path.exists(self.moon_icon_path) and self._theme == 'light':
                self.theme_toggle.setIcon(QIcon(self.moon_icon_path))
            elif os.path.exists(self.sun_icon_path) and self._theme == 'dark':
                self.theme_toggle.setIcon(QIcon(self.sun_icon_path))
        except Exception:
            pass
        try:
            self.theme_toggle.setText('')
            self.theme_toggle.setToolButtonStyle(Qt.ToolButtonStyle.IconOnly)
            self.theme_toggle.setIconSize(QSize(20, 20))
        except Exception:
            pass
        try:
            tt = 'Switch to light mode' if self._theme == 'dark' else 'Switch to dark mode'
            self.theme_toggle.setToolTip(tt)
        except Exception:
            pass
        self.theme_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.theme_toggle.clicked.connect(self.toggle_theme)
        self.theme_toggle.setObjectName('themeToggle')
        self.theme_toggle.setFixedSize(36, 32)

        # Place icon first (leading) then title
        app_bar_layout.addWidget(self.app_icon)
        app_bar_layout.addWidget(self.title_label)
        app_bar_layout.addStretch(1)
        app_bar_layout.addWidget(self.theme_toggle)
        self.app_bar = app_bar

        # Left panel: import, path and console
        left_panel = QWidget(self)
        left_panel.setObjectName('leftPanel')
        try:
            left_panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        except Exception:
            pass
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)

        self.btn_json = QPushButton("Import JSON")
        self.btn_toml = QPushButton("Import TOML")
        self.btn_yaml = QPushButton("Import YAML")
        for b, kind in ((self.btn_json, 'json'), (self.btn_toml, 'toml'), (self.btn_yaml, 'yaml')):
            b.setFont(btn_font)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setFixedHeight(40)
            b.setObjectName('neumoButton')
            b.clicked.connect(lambda _, k=kind: self._import_file_dialog(k))

        import_row = QHBoxLayout()
        import_row.setSpacing(10)
        import_row.addWidget(self.btn_json)
        import_row.addWidget(self.btn_toml)
        import_row.addWidget(self.btn_yaml)

        self.path_field = QLineEdit()
        self.path_field.setReadOnly(True)
        self.path_field.setFont(path_font)
        self.path_field.setPlaceholderText('Imported file path...')
        self.path_field.setObjectName('pathField')
        self.path_field.setFixedHeight(40)

        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setObjectName('consoleBox')
        self.log_widget.setPlaceholderText('Console output...')

        left_layout.addLayout(import_row)
        left_layout.addWidget(self.path_field)
        left_layout.addWidget(self.log_widget, 1)

        # Small copyright below console
        self.copyright_label = QLabel('TensorSuggestLite © Core Red 2025')
        self.copyright_label.setFont(copy_font)
        self.copyright_label.setObjectName('copyrightLabel')
        self.copyright_label.setFixedHeight(16)
        self.copyright_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        left_layout.addWidget(self.copyright_label)

        # Panel derecho: acciones y progreso
        right_panel = QWidget(self)
        right_panel.setObjectName('rightPanel')
        try:
            right_panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        except Exception:
            pass
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(10)

        self.train_button = QPushButton('Train Model')
        self.train_button.setFont(btn_font)
        self.train_button.setEnabled(False)
        self.train_button.clicked.connect(self._on_train_clicked)
        self.train_button.setObjectName('neumoButton')

        self.convert_button = QPushButton('Generate TFLite')
        self.convert_button.setFont(btn_font)
        self.convert_button.setEnabled(False)
        self.convert_button.clicked.connect(self._on_convert_clicked)
        self.convert_button.setObjectName('neumoButton')

        self.clear_button = QPushButton('Clear Generated Folders')
        self.clear_button.setFont(btn_font)
        self.clear_button.setEnabled(True)
        self.clear_button.clicked.connect(lambda _: self._clear_generated(self.selected_kind or 'json'))
        self.clear_button.setObjectName('neumoButton')

        # Nuevo: botón para limpiar la consola (consoleBox)
        self.clear_screen_button = QPushButton('Clear Screen')
        self.clear_screen_button.setFont(btn_font)
        self.clear_screen_button.setEnabled(True)
        self.clear_screen_button.clicked.connect(self._clear_console)
        self.clear_screen_button.setObjectName('neumoButton')

        self.export_button = QPushButton('Export TFLite')
        self.export_button.setFont(btn_font)
        self.export_button.setVisible(False)
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._on_export_clicked)
        self.export_button.setObjectName('neumoButton')

        right_layout.addWidget(self.train_button)
        right_layout.addWidget(self.convert_button)
        right_layout.addWidget(self.clear_button)
        right_layout.addWidget(self.clear_screen_button)
        right_layout.addStretch(1)
        right_layout.addWidget(self.export_button)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFixedHeight(18)
        self.progress.setObjectName('progressBar')

        # Label de estado encima del progress
        status_font = QFont("Segoe UI", 10)
        self.status_label = QLabel('model not trained')
        self.status_label.setFont(status_font)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.status_label.setObjectName('statusLabel')
        self.status_label.setFixedHeight(20)

        right_layout.addWidget(self.status_label)
        right_layout.addWidget(self.progress)

        main_split = QHBoxLayout()
        main_split.setSpacing(12)
        main_split.addWidget(left_panel, 3)
        main_split.addWidget(right_panel, 1)

        self.left_panel = left_panel
        self.right_panel = right_panel

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(app_bar)
        root_layout.addLayout(main_split)

        self.setLayout(root_layout)

    # ---------- Helpers: importación/selección/logs ----------
    def _import_file_dialog(self, kind: str):
        """Open file dialog and delegate selection to _on_file_selected."""
        if kind == 'json':
            filter = 'JSON files (*.json)'
        elif kind == 'yaml':
            filter = 'YAML files (*.yaml *.yml)'
        else:
            filter = 'TOML files (*.toml)'
        dlg_title = f'Select file ({kind.upper()})'
        path, _ = QFileDialog.getOpenFileName(self, dlg_title, '', filter)
        if path:
            try:
                self._on_file_selected(path, kind)
            except Exception:
                # Minimal fallback: register selection and enable train button
                self.selected_path = path
                self.selected_kind = kind
                self.path_field.setText(path)
                self._set_button_enabled(self.train_button, True)
                self._append_log(f'File selected: {path}\nType: {kind}')

    def _on_file_selected(self, path: str, kind: str):
        """Central handler when user selects a file.

        Sets internal state, updates buttons based on already-generated artifacts,
        and starts watcher to detect future artifact creation.
        """
        self.selected_path = path
        self.selected_kind = kind
        self.path_field.setText(path)
        # Always start from 'model not trained' state when selecting a file
        QTimer.singleShot(0, lambda: self.status_label.setText('model not trained'))
        # Always enable train button once file is selected
        self._set_button_enabled(self.train_button, True)
        # Reset export visibility until confirming tflite exists
        self.export_button.setVisible(False)
        self._set_button_enabled(self.export_button, False)

        try:
            chk = self._check_generated_files_exist(kind)
            if chk.get('all_exist', False):
                # If artifacts already exist, enable conversion/export but
                # don't change status (user wants to start from 'not trained')
                self._set_button_enabled(self.convert_button, True)
                tflite_path = os.path.join(chk.get('gen_dir', ''), 'text_classifier.tflite')
                if os.path.exists(tflite_path):
                    self.export_button.setVisible(True)
                    self._set_button_enabled(self.export_button, True)
            else:
                self._set_button_enabled(self.convert_button, False)
        except Exception:
            self._set_button_enabled(self.convert_button, False)
            # Keep 'model not trained' in case of error

        try:
            self._start_generated_watcher(kind)
        except Exception:
            pass

        self._append_log(f'File selected: {path}\nType: {kind}')

    def _append_log(self, text: str):
        """Append line to console from any thread (queued to UI thread).

        Also inspects key messages produced by trainers to update
        validations and enable buttons without forcing progress.
        """
        def _do():
            try:
                # Append raw text
                self.log_widget.append(text)
                self.log_widget.moveCursor(self.log_widget.textCursor().End)
            except Exception:
                pass

            # Inspeccionar mensajes útiles del trainer para actualizar UI
            try:
                low = (text or '').lower()
                # Si el trainer informa creación de alguno de los artefactos,
                # comprobar la carpeta generated/<kind> y habilitar conversión si
                # todos los archivos están presentes.
                if 'tokenizer' in low or 'label encoder' in low or 'keras' in low or 'tokenizer.json' in low or 'label_encoder.json' in low or 'text_classifier.keras' in low:
                    try:
                        chk = self._check_generated_files_exist(self.selected_kind or 'json')
                        if chk.get('all_exist', False):
                            self._set_button_enabled(self.convert_button, True)
                            # actualizar estado si procedente
                            if self.status_label.text().lower() != 'model trained':
                                self.status_label.setText('model trained')
                    except Exception:
                        pass

                # Si se guardó el TFLite, mostrar y habilitar el botón de exportar
                if 'tflite saved' in low or ('tflite' in low and ('saved' in low or 'conversion complete' in low or 'text_classifier.tflite' in low)):
                    try:
                        self.export_button.setVisible(True)
                        self._set_button_enabled(self.export_button, True)
                        # si estábamos generando, actualizar status
                        if 'generating' in (self.status_label.text() or '').lower():
                            self.status_label.setText('tflite generated successfully')
                    except Exception:
                        pass

                # Si el trainer emite mensajes de error, reflejar el estado
                if 'error' in low or 'exception' in low:
                    try:
                        # no forzamos progreso, solo mostrar estado de error
                        self.status_label.setText('process error')
                    except Exception:
                        pass
            except Exception:
                pass

        QTimer.singleShot(0, _do)

    def _set_button_enabled(self, button: 'QPushButton', enabled: bool):
        """Enable/disable a QPushButton and force repaint.

        Also sets 'enabled' property so QSS can match,
        forces repolish and adjusts cursor for better UX clarity.
        """
        try:
            button.setEnabled(bool(enabled))
            try:
                button.setProperty('enabled', 'true' if enabled else 'false')
            except Exception:
                try:
                    button.setProperty('enabled', bool(enabled))
                except Exception:
                    pass
            try:
                st = button.style()
                try:
                    st.unpolish(button)
                except Exception:
                    pass
                try:
                    st.polish(button)
                except Exception:
                    pass
                try:
                    button.update()
                except Exception:
                    pass
            except Exception:
                pass
            try:
                if enabled:
                    button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
                else:
                    button.setCursor(QCursor(Qt.CursorShape.ForbiddenCursor))
            except Exception:
                try:
                    button.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
                except Exception:
                    pass
        except Exception:
            pass

    # ---------- Watcher / helpers for generated ----------
    def _check_generated_files_exist(self, kind: str):
        """Return dict indicating if tokenizer/model/label artifacts exist."""
        root = os.path.abspath(os.path.dirname(__file__))
        gen_dir = os.path.join(root, 'generated', kind)
        files = ('tokenizer.json', 'text_classifier.keras', 'label_encoder.json')
        all_exist = all(os.path.exists(os.path.join(gen_dir, f)) for f in files)
        return {'all_exist': all_exist, 'gen_dir': gen_dir}

    def _start_generated_watcher(self, kind: str):
        """Start watching generated/<kind> (or generated/) and start polling.

        Watcher records if files already existed at start so it only
        enables conversion button when transition to "all exist" occurs.
        """
        root = os.path.abspath(os.path.dirname(__file__))
        gen_dir = os.path.join(root, 'generated', kind)
        gen_parent = os.path.join(root, 'generated')
        try:
            for p in list(self._watch_paths):
                try:
                    self._watcher.removePath(p)
                except Exception:
                    pass
            self._watch_paths.clear()
        except Exception:
            pass
        try:
            os.makedirs(gen_parent, exist_ok=True)
        except Exception:
            pass
        try:
            if os.path.exists(gen_dir):
                self._watcher.addPath(gen_dir)
                self._watch_paths.add(gen_dir)
            else:
                self._watcher.addPath(gen_parent)
                self._watch_paths.add(gen_parent)
        except Exception:
            pass
        try:
            try:
                self._watcher.directoryChanged.disconnect(self._on_directory_changed)
            except Exception:
                pass
            self._watcher.directoryChanged.connect(self._on_directory_changed)
        except Exception:
            pass
        try:
            chk0 = self._check_generated_files_exist(kind)
            self._initial_all_exist = bool(chk0.get('all_exist', False))
        except Exception:
            self._initial_all_exist = False
        self._watched_kind = kind
        try:
            self._watch_poll_timer.start()
        except Exception:
            pass

    def _on_directory_changed(self, path: str):
        """Callback for QFileSystemWatcher.directoryChanged; enables conversion on transition."""
        try:
            kind = self._watched_kind or 'json'
            chk = self._check_generated_files_exist(kind)
            if chk.get('all_exist', False) and not self._initial_all_exist:
                self._append_log(f'Generated files detected in: {chk.get("gen_dir")}.')
                self._set_button_enabled(self.convert_button, True)
                self._stop_generated_watcher()
        except Exception:
            pass

    def _poll_generated(self):
        """Polling fallback that replicates directoryChanged behavior."""
        try:
            if not self._watched_kind:
                return
            chk = self._check_generated_files_exist(self._watched_kind)
            if chk.get('all_exist', False) and not self._initial_all_exist:
                self._append_log(f'Generated files detected in: {chk.get("gen_dir")}.')
                self._set_button_enabled(self.convert_button, True)
                self._stop_generated_watcher()
        except Exception:
            pass

    def _stop_generated_watcher(self):
        try:
            for p in list(self._watch_paths):
                try:
                    self._watcher.removePath(p)
                except Exception:
                    pass
            self._watch_paths.clear()
        except Exception:
            pass
        try:
            self._watch_poll_timer.stop()
        except Exception:
            pass
        self._watched_kind = None

    def _clear_generated(self, kind: str):
        """Delete previously generated artifacts for clean retrain."""
        try:
            root = os.path.abspath(os.path.dirname(__file__))
            gen_dir = os.path.join(root, 'generated', kind)
            if os.path.exists(gen_dir):
                for fname in ('tokenizer.json', 'text_classifier.keras', 'label_encoder.json', 'text_classifier.tflite'):
                    p = os.path.join(gen_dir, fname)
                    try:
                        if os.path.exists(p):
                            os.remove(p)
                    except Exception:
                        pass
        except Exception:
            pass
        # Al limpiar los archivos generados, ocultar y deshabilitar el botón de exportar
        try:
            if hasattr(self, 'export_button'):
                self.export_button.setVisible(False)
                self._set_button_enabled(self.export_button, False)
            # También deshabilitar botón de convertir si ya no existen los artefactos
            try:
                chk = self._check_generated_files_exist(kind)
                if not chk.get('all_exist', False):
                    self._set_button_enabled(self.convert_button, False)
            except Exception:
                pass
        except Exception:
            pass

    def _clear_console(self):
        """Clear consoleBox (log_widget) content."""
        try:
            self.log_widget.clear()
        except Exception:
            try:
                # fallback: reemplazar con texto vacío usando QTimer
                QTimer.singleShot(0, lambda: self.log_widget.clear())
            except Exception:
                pass

    # ---------- Progreso ----------
    def _handle_progress(self, value: int):
        try:
            v = int(value)
            self.progress.setValue(v)
            try:
                self.progress.update()
            except Exception:
                pass

            # Comportamiento validatorio basado en callbacks del trainer (no forzar nada):
            # - Si el trainer reporta 100 mientras estábamos entrenando, consideramos
            #   el modelo entrenado y habilitamos la conversión.
            # - Si reporta 100 mientras estábamos generando tflite, consideramos
            #   la conversión completada y mostramos el botón de exportar.
            try:
                current = (self.status_label.text() or '').lower()
            except Exception:
                current = ''
            try:
                if v >= 100:
                    if 'training' in current:
                        # confiar en que trainer llegó a 100 por sí mismo
                        self.status_label.setText('model trained')
                        # habilitar conversión si los archivos existen
                        chk = self._check_generated_files_exist(self.selected_kind or 'json')
                        if chk.get('all_exist', False):
                            self._set_button_enabled(self.convert_button, True)
                    elif 'generating' in current or 'convert' in current or 'tflite' in current:
                        self.status_label.setText('tflite generated successfully')
                        self.export_button.setVisible(True)
                        self._set_button_enabled(self.export_button, True)
            except Exception:
                pass
        except Exception:
            QTimer.singleShot(0, lambda: self.progress.setValue(int(value)))

    def _poll_progress_queue(self):
        """Leer la cola de progreso desde el hilo UI y emitir el último valor."""
        try:
            last = None
            while True:
                last = self._progress_queue.get_nowait()
        except Exception:
            if last is not None:
                try:
                    self.training_progress.emit(int(last))
                except Exception:
                    QTimer.singleShot(0, lambda v=last: self.progress.setValue(int(v)))

    # ---------- Flujos: entrenar / convertir / exportar ----------
    def _on_train_clicked(self):
        if not self.selected_path:
            return
        kind = (self.selected_kind or 'json')
        try:
            self._clear_generated(kind)
        except Exception:
            pass
        try:
            self._start_generated_watcher(kind)
        except Exception:
            pass
        self._set_button_enabled(self.train_button, False)
        self.progress.setValue(0)
        # indicar estado de entrenamiento en la UI
        QTimer.singleShot(0, lambda: self.status_label.setText('training model'))
        thread = threading.Thread(target=self._train_worker, daemon=True)
        thread.start()

    def _train_worker(self):
        try:
            trainer = None
            train_fn = None
            if self.selected_kind == 'json':
                trainer = json_trainer
                train_fn = getattr(trainer, 'train_from_json')
            elif self.selected_kind == 'yaml':
                from yaml_interpreter import trainer as yaml_trainer
                trainer = yaml_trainer
                train_fn = getattr(trainer, 'train_from_yaml')
            elif self.selected_kind == 'toml':
                from toml_interpreter import trainer as toml_trainer
                trainer = toml_trainer
                train_fn = getattr(trainer, 'train_from_toml')

            if trainer is None or train_fn is None:
                raise RuntimeError('No valid trainer selected for imported file.')

            def progress_cb(p):
                v = int(max(0, min(100, int(p))))
                # Emitir progreso y log via señales (thread-safe)
                try:
                    self.training_progress.emit(v)
                except Exception:
                    try:
                        QTimer.singleShot(0, lambda vv=v: self._set_progress(vv))
                    except Exception:
                        pass
                try:
                    self.log_signal.emit(f'Progress: {v}%')
                except Exception:
                    try:
                        QTimer.singleShot(0, lambda vv=v: self._append_log(f'Progress: {vv}%'))
                    except Exception:
                        pass

            # Wrapper for log that is thread-safe and ensures append_log on UI thread
            def _trainer_log_wrapper(msg: str):
                try:
                    # Use signal to send to UI thread
                    self.log_signal.emit(msg)
                except Exception:
                    try:
                        QTimer.singleShot(0, lambda m=msg: self._append_log(m))
                    except Exception:
                        try:
                            self._append_log(msg)
                        except Exception:
                            pass

            # Call trainer passing our callbacks; trainer decides
            # and sends real progress and logs
            try:
                result = train_fn(self.selected_path, progress_cb, log_cb=_trainer_log_wrapper)
            except TypeError:
                result = train_fn(self.selected_path, progress_cb)

            # Post-trainer validation: check if trainer created artifacts
            try:
                chk = self._check_generated_files_exist(self.selected_kind or 'json')
                if chk.get('all_exist', False):
                    QTimer.singleShot(0, lambda: self._append_log(f'Training complete: artifacts detected in {chk.get("gen_dir")}'))
                    QTimer.singleShot(0, lambda: self._set_button_enabled(self.convert_button, True))
                    QTimer.singleShot(0, lambda: self.status_label.setText('model trained'))
                else:
                    # Si trainer devolvió rutas explícitas en el result, comprobarlas también
                    try:
                        tok = result.get('tokenizer_path') if isinstance(result, dict) else None
                        mod = result.get('model_path') if isinstance(result, dict) else None
                        lab = result.get('label_encoder_path') if isinstance(result, dict) else None
                        if tok and mod and lab and all(os.path.exists(p) for p in (tok, mod, lab)):
                            QTimer.singleShot(0, lambda: self._append_log('Training complete: artifacts detected (paths returned by trainer).'))
                            QTimer.singleShot(0, lambda: self._set_button_enabled(self.convert_button, True))
                            QTimer.singleShot(0, lambda: self.status_label.setText('model trained'))
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception as e:
            QTimer.singleShot(0, lambda: self._append_log(f'Training error: {e}'))
            QTimer.singleShot(0, lambda: self.status_label.setText('training error'))
            # No forzar progreso en caso de error; confiar en mensajes del trainer.
        finally:
            QTimer.singleShot(0, lambda: self._set_button_enabled(self.train_button, True))

    def _set_progress(self, value: int):
        # Actualizar directamente la barra de progreso en el hilo UI.
        try:
            QTimer.singleShot(0, lambda v=int(value): self.progress.setValue(int(v)))
            try:
                QTimer.singleShot(0, lambda: self.progress.update())
            except Exception:
                pass
        except Exception:
            try:
                self.progress.setValue(int(value))
            except Exception:
                pass

    def _on_convert_clicked(self):
        self._set_button_enabled(self.convert_button, False)
        self.progress.setValue(0)
        # Indicate conversion start
        QTimer.singleShot(0, lambda: self.status_label.setText('generating tflite'))
        thread = threading.Thread(target=self._convert_worker, daemon=True)
        thread.start()

    def _convert_worker(self):
        try:
            root = os.path.abspath(os.path.dirname(__file__))
            kind = (self.selected_kind or 'json')
            gen_model_path = os.path.join(root, 'generated', kind)

            if kind == 'json':
                trainer = json_trainer
            elif kind == 'yaml':
                from yaml_interpreter import trainer as yaml_trainer
                trainer = yaml_trainer
            elif kind == 'toml':
                from toml_interpreter import trainer as toml_trainer
                trainer = toml_trainer
            else:
                raise RuntimeError(f"Unknown type for conversion: {kind}")

            # Pass callbacks so trainer reports progress and logs
            def _conv_progress(p):
                v = int(max(0, min(100, int(p))))
                try:
                    self.training_progress.emit(v)
                except Exception:
                    try:
                        QTimer.singleShot(0, lambda vv=v: self._set_progress(vv))
                    except Exception:
                        pass
                try:
                    self.log_signal.emit(f'Conversion progress: {v}%')
                except Exception:
                    try:
                        QTimer.singleShot(0, lambda pp=v: self._append_log(f'Conversion progress: {pp}%'))
                    except Exception:
                        pass

            def _conv_log(m):
                try:
                    self.log_signal.emit(m)
                except Exception:
                    try:
                        QTimer.singleShot(0, lambda mm=m: self._append_log(mm))
                    except Exception:
                        try:
                            self._append_log(m)
                        except Exception:
                            pass

            out_path = trainer.convert_to_tflite(model_dir=gen_model_path, progress_cb=_conv_progress, log_cb=_conv_log)

            # Update UI based on return (without forcing progress). Trainer
            # should send callbacks for progress and logs; here just show
            # success message and enable export if appropriate
            QTimer.singleShot(0, lambda: self._append_log(f'Conversion complete: {out_path}'))
            QTimer.singleShot(0, lambda: self.status_label.setText('tflite generated successfully'))
            QTimer.singleShot(0, lambda: self.export_button.setVisible(True))
            QTimer.singleShot(0, lambda: self._set_button_enabled(self.export_button, True))
        except Exception as e:
            QTimer.singleShot(0, lambda: self._append_log(f'Conversion error: {e}'))
            QTimer.singleShot(0, lambda: self.status_label.setText('tflite generation error'))
            # Don't force progress on error; trust trainer messages
        finally:
            QTimer.singleShot(0, lambda: self._set_button_enabled(self.convert_button, True))

    def _on_export_clicked(self):
        dlg_title = "Save TFLite"
        kind = (self.selected_kind or 'json')
        src_default = os.path.join(os.getcwd(), 'generated', kind, 'text_classifier.tflite')
        dst, _ = QFileDialog.getSaveFileName(self, dlg_title, src_default, "TFLite files (*.tflite)")
        if dst:
            try:
                shutil.copy(src_default, dst)
                self._append_log(f"Exported: {dst}")
            except Exception as e:
                self._append_log(f"Export error: {e}")


def main():
    app = QApplication(sys.argv)
    win = TensorSuggestLiteUI()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

