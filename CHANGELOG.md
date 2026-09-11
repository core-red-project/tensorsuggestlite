# Changelog

All notable changes to **TensorSuggestLite** are documented here.

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [2.0.0] — 2026-09-11

### Added

- Clean Architecture refactoring under `src/tensorsuggestlite` separating domain entities, parsers, machine learning, use cases, and presentation layers.
- Headless CLI with subcommands: `inspect`, `train`, `convert`, `predict`, and `history`.
- Interactive live inference Playground in GUI and CLI with cross-runtime validation (`--compare`) measuring confidence delta and parity between `.keras` and `.tflite`.
- Dataset profiling detecting class imbalance, duplicate sentences, and contradictory labeling.
- Structured Confusion Matrix calculation and telemetry export in `model_metadata.json`.
- Zero-TensorFlow lightweight inference runner (`lite_runtime.py` and `scripts/standalone_infer.py`) with sub-2ms latency.
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
