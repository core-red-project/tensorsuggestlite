# File Manifest

Complete inventory of TensorSuggestLite repository structure and file purposes.

**Last Updated**: 2026-02-03

## Root Directory

| File | Type | Purpose |
|------|------|---------|
| `tensorsuggestlite.py` | Python | Main entry point, PyQt6 GUI application |
| `requirements.txt` | Config | Python dependencies (unpinned) |
| `README.md` | Doc | User documentation and quickstart guide |
| `LICENSE` | Legal | CR-NC-1.0 license text (English/Spanish) |
| `CONTRIBUTING.md` | Doc | Contribution guidelines |
| `CODE_OF_CONDUCT.md` | Doc | Community conduct expectations |
| `SECURITY.md` | Doc | Security policy and vulnerability reporting |
| `CHANGELOG.md` | Doc | Version history and changes |
| `REPOSITORY_STATUS.md` | Doc | Project status, stability, roadmap |
| `FILE_MANIFEST.md` | Doc | This file - complete file inventory |
| `.gitignore` | Config | Git exclusion rules |
| `build_mac_app.sh` | Build | macOS .app builder using PyInstaller |
| `build_windows_exe.sh` | Build | Windows .exe builder using PyInstaller |
| `TensorSuggestLite.png` | Asset | Application screenshot for documentation |
| `styles_dark.qss` | UI | Qt stylesheet for dark theme |
| `styles_light.qss` | UI | Qt stylesheet for light theme |

## `/icons/`

Application icons and UI assets.

| File | Purpose |
|------|---------|
| `TensorSuggestLite.svg` | Primary application icon |
| `sun.svg` | Theme toggle icon (light mode) |
| `moon.svg` | Theme toggle icon (dark mode) |

## `/json_interpreter/`

JSON format handler and trainer.

| File | Purpose |
|------|---------|
| `__init__.py` | Package initializer, exports `trainer` |
| `trainer.py` | JSON parsing, model training, TFLite conversion |

**Functions**:
- `train_from_json(path, progress_cb, log_cb, epochs)` - Parse JSON and train model
- `convert_to_tflite(model_dir, progress_cb, log_cb)` - Convert Keras to TFLite

## `/yaml_interpreter/`

YAML format handler and trainer.

| File | Purpose |
|------|---------|
| `__init__.py` | Package initializer, exports `trainer` |
| `trainer.py` | YAML parsing, model training, TFLite conversion |

**Functions**:
- `train_from_yaml(path, progress_cb, log_cb, epochs)` - Parse YAML and train model
- `convert_to_tflite(model_dir, progress_cb, log_cb)` - Convert Keras to TFLite

## `/toml_interpreter/`

TOML format handler and trainer.

| File | Purpose |
|------|---------|
| `__init__.py` | Package initializer, exports `trainer` |
| `trainer.py` | TOML parsing, model training, TFLite conversion |

**Functions**:
- `train_from_toml(path, progress_cb, log_cb, epochs)` - Parse TOML and train model
- `convert_to_tflite(model_dir, progress_cb, log_cb)` - Convert Keras to TFLite

## `/test/`

Test data samples for each supported format.

| File | Purpose |
|------|---------|
| `question_data.json` | Sample JSON training data |
| `question_data.yaml` | Sample YAML training data |
| `question_data.toml` | Sample TOML training data |

**Data Structure**:
```
{
  "CATEGORY_NAME": {
    "respuestas": [
      {"respuesta": "training text"},
      ...
    ],
    "sinonimos": ["synonym1", "synonym2", ...]
  },
  ...
}
```

## `/generated/` (Runtime)

Generated at runtime during training. **Not in git** (see `.gitignore`).

Structure per format:
```
generated/
  json/
    tokenizer.json
    text_classifier.keras
    label_encoder.json
    text_classifier.tflite
  yaml/
    (same structure)
  toml/
    (same structure)
```

## `/.github/`

GitHub-specific configurations.

| File | Purpose |
|------|---------|
| `FUNDING.yml` | Funding/sponsorship links |

**Future**: Will contain CI/CD workflows (`.github/workflows/`).

## `/build/` (Build-time)

Created during packaging process. **Not in git**.

Contains:
- PyInstaller spec files
- Intermediate build artifacts
- Icon conversions (.icns, .ico)

## `/dist/` (Build-time)

Created during packaging process. **Not in git**.

Contains:
- Final packaged applications (.app, .exe)
- Distribution-ready bundles

## Hidden/System Files

| File/Dir | Purpose |
|----------|---------|
| `.git/` | Git repository data |
| `.idea/` | JetBrains IDE configuration |
| `.venv/` | Python virtual environment |
| `.venv_build/` | Build-specific virtual environment |
| `.DS_Store` | macOS filesystem metadata |
| `.qodo/` | Qodo (Codium) tool configuration |

## Architecture Overview

```
User
  ↓
tensorsuggestlite.py (GUI)
  ↓
[json|yaml|toml]_interpreter/trainer.py
  ↓
TensorFlow/Keras → Artifacts → generated/<format>/
  ↓
TFLite Converter → text_classifier.tflite
```

## Data Flow

1. User selects input file (JSON/YAML/TOML)
2. GUI invokes appropriate interpreter's `train_from_*` function
3. Trainer parses file into training data
4. TensorFlow trains classification model
5. Artifacts saved to `generated/<format>/`
6. Optional TFLite conversion for deployment
7. User exports TFLite file

## Key Dependencies (External)

- **TensorFlow**: Model training and conversion
- **PyQt6**: GUI framework
- **NumPy**: Numerical operations
- **scikit-learn**: Train/test split, class weights
- **PyYAML**: YAML parsing
- **toml**: TOML parsing

## Notable Patterns

### Threading
- Training and conversion run in daemon threads
- UI callbacks via `QTimer.singleShot` for thread safety
- Progress/log updates via Qt signals

### Error Handling
- Extensive try/except blocks (overly broad in some areas)
- Best-effort execution (errors often silently caught)

### File Watching
- `QFileSystemWatcher` monitors `generated/<format>/` directories
- Polling fallback for filesystem watcher reliability

### Theming
- QSS (Qt StyleSheets) loaded dynamically
- Property-based theme switching

---

**Maintainer**: houjou.sxnnyside@sxnnysideproject.com  
**Report Errors**: Open GitHub Issue or email support.sxnnyside@sxnnysideproject.com

