# Changelog

All notable changes to **TensorSuggestLite** are documented here.

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [2.1.0] — 2026-11-18

### Added

- **Ultra-Flexible Skeleton Dataset Parser**: Dynamic schema parser recognizing custom keywords (`in`, `out`, `alt`, `prompt`, `label`), inverted lookup mappings (`{"text": "CATEGORY"}`), direct/nested lists, and container dictionaries without rigid schemas.
- **Configurable TFLite Quantization**: Configurable optimization targets (`dynamic`, `float16`, `none`) across `ExportTFLiteUseCase`, CLI (`--quantize`), and GUI combo dropdown.
- **High-Throughput Batch Inference**: Vectorized batch prediction in `KerasPredictor` and `TFLitePredictor`, supporting file inputs (`--file`) and stdin streaming (`-`) in CLI, plus multi-line input in GUI Playground.
- **Extended Training & Evaluation Metrics**: Per-class precision, recall, f1-score, and support metrics alongside macro/weighted averages, training duration in seconds, parameter count, and vocabulary statistics recorded in `model_metadata.json` and inspectable via `tensorsuggestlite history`.
- **Dear PyGui 2.3.1 Desktop Framework**: Migrated entire GUI from PySide6/PyQt6 to lightweight Dear PyGui with Zero Qt runtime dependencies.
- **Native OS Dialog Bridge (`NativeDialogBridge`)**: Out-of-process native system file dialogs for macOS (`osascript`) and Windows (`comdlg32.dll`) providing fast file and directory selection.
- **Dynamic Viewport Responsiveness**: Elastic layout with zero outer scrollbars where the console window smoothly absorbs vertical viewport expansions while keeping all controls, status labels, and progress bars visible.

### Changed

- Adopted split-license model: MIT for Core Engine & CLI (`src/tensorsuggestlite/{core,cli,ml,parsers,use_cases}`), MPL-2.0 for Graphical User Interface (`src/tensorsuggestlite/ui`).
- Replaced misleading drag-and-drop copy with direct native OS dialog invocation.
- Vertically centered and pixel-aligned the App Bar theme toggle button with a dedicated translucent hover theme.
- Increased hyperparameter sliders and controls to 100% column width for balanced UX hierarchy.

### Removed

- Permanently retired `PyQt6` and `PySide6` dependencies, reducing distribution bundle sizes and eliminating Qt DLL/framework overhead.
- Removed deprecated `.svg` assets in favor of sharp 256x256 `.png` textures.

---

## [2.0.0] — 2026-09-11

### Added

- Clean Architecture refactoring under `src/tensorsuggestlite` separating domain entities, parsers, machine learning, use cases, and presentation layers.
- Headless CLI with subcommands: `inspect`, `train`, `convert`, `predict`, and `history`.
- Interactive live inference Playground in GUI and CLI with cross-runtime validation (`--compare`) measuring confidence delta and parity between `.keras` and `.tflite`.
- Dataset profiling detecting class imbalance, duplicate sentences, and contradictory labeling.
- Structured Confusion Matrix calculation and telemetry export in `model_metadata.json`.
- Zero-TensorFlow lightweight inference runner (`lite_runtime.py` and CLI `--tflite`) with sub-2ms latency.
- Dual-asset packaging with separate desktop app bundles (`.dmg` on macOS, `.exe` on Windows, Linux executable) and headless CLI binaries (`tensorsuggestlite-cli`).
- Multi-platform release automation matrix in `.github/workflows/release.yml`.
- Optional `[gui]` packaging extra separating the core headless CLI from desktop dependencies.

### Changed

- Replaced legacy tooling and `make` with modern task runner (`Justfile`), `uv`, Ruff, Mypy strict mode, and Pytest.
- Unified configuration parsers across JSON, YAML, and TOML via `BaseConfigParser` and `ParserRegistry`.
- Optimized macOS UI font resolution using native `.AppleSystemUIFont` and `Menlo` eliminating startup font indexing lag.

### Removed

- Deprecated legacy duplicated interpreter folders (`json_interpreter/`, `yaml_interpreter/`, `toml_interpreter/`, root legacy scripts).
- Removed raw requirements files in favor of `pyproject.toml` and deterministic `uv.lock`.

---

[Unreleased]: https://github.com/core-red-project/tensorsuggestlite/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/core-red-project/tensorsuggestlite/releases/tag/v2.0.0
