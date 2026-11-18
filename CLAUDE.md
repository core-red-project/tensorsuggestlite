# TensorSuggestLite — Project Guide

## Overview
TensorSuggestLite is a Python application that trains TensorFlow/Keras text classification models directly from configuration files (JSON, YAML, TOML) and exports optimized TFLite artifacts for embedded or mobile deployment.

## Repository Topology
- **Topology**: Monolithic repository (single deployable/publishable unit).
- **Toolchain**: Python 3.11 managed via `uv`, with `just` as the canonical task runner.
- **Architecture**: Clean Architecture / SOLID / DRY under `src/tensorsuggestlite`.
- **Licensing**: 100% strict MIT (Dear PyGui + native OS bridges, zero GPL/Qt dependencies).

## Where things go (Placement Table)
| Kind | Location | Naming Pattern |
|---|---|---|
| Domain entities & core | `src/tensorsuggestlite/core/` | `snake_case.py` |
| File parsers (JSON, YAML, TOML) | `src/tensorsuggestlite/parsers/` | `<format>_parser.py` |
| Machine learning engines | `src/tensorsuggestlite/ml/` | `snake_case.py` |
| Application use cases | `src/tensorsuggestlite/use_cases/` | `snake_case.py` |
| Graphical user interface (Dear PyGui) | `src/tensorsuggestlite/ui/` | `snake_case.py` |
| CLI & application entry point | `src/tensorsuggestlite/cli/` | `snake_case.py` |
| Automated mirrored tests | `tests/<package>/` | `test_<module>.py` |
| Automation & validation scripts | `scripts/` | `kebab-case.sh` or `snake_case.py` |
| Visual assets & icons | `icons/` | `app_icon.png`, `moon.png`, `sun.png` |
| Configuration dataset examples | `examples/` | `*.json`, `*.yaml`, `*.toml` |
| Documentation | Root or `docs/` | `UPPERCASE.md` or `kebab-case.md` |

## Directory Structure
```
├── src/tensorsuggestlite/
│   ├── cli/                   # Argument parser, POSIX streams, commands & completions
│   ├── core/                  # Domain entities, interfaces, exceptions & profiler
│   ├── ml/                    # Keras trainer, TFLite exporter, predictors & zero-TF lite runtime
│   ├── parsers/               # Extensible JSON, YAML, TOML parsers with skeleton mapping
│   ├── ui/                    # Dear PyGui UI (main window, layout builder, theme, native dialogs)
│   ├── use_cases/             # TrainModel, ExportTFLite, and Predict use cases
│   └── main.py                # Main application entry point
├── examples/                  # Real-world dataset examples (JSON, YAML, TOML)
├── icons/                     # UI icons and logo
├── scripts/                   # Conventional commit hooks and automation
├── tests/                     # Automated test suite mirroring src/
│   ├── cli/                   # CLI command and argument parser tests
│   ├── core/                  # Entities, profiler, and metadata tests
│   ├── ml/                    # Trainer, exporter, predictor, and lite runtime tests
│   ├── parsers/               # JSON, YAML, TOML, and skeleton schema tests
│   ├── ui/                    # Presentation, layout, theme, and DnD tests
│   ├── use_cases/             # Use case orchestration and cross-validation tests
│   └── e2e/                   # End-to-end training and public imports tests
├── pyproject.toml             # Project definition, build backend (Hatchling) & configs
├── Justfile                   # Canonical command surface
└── CLAUDE.md                  # Developer & AI reference
```

## Features & Status
- **Desktop GUI**: Dear PyGui 2.3+ with Core Red theme (high-contrast dark & light modes). Zero outer scrollbars.
- **Drag & Drop**: Native OS file drag-and-drop onto the window. `[Experimental: macOS verified via Cocoa/AppKit; untested on Windows/Linux]`.
- **Skeleton Mapping**: Custom input/output keyword mapping (`_skeleton: {in: "...", out: "..."}`) via dataset headers, CLI (`--skeleton`), or GUI input.
- **Quantization Modes**: `dynamic`, `float16`, and `none` for optimized mobile/edge inference.

## Command Surface (`just`)
All commands are run using `just`:

| Command | Purpose |
|---|---|
| `just install` | Bootstrap environment, sync dependencies and install git hooks |
| `just dev` | Launch the Dear PyGui desktop interface in development mode |
| `just build` | Build distribution packages (wheel and sdist) with `uv build` |
| `just bundle` | Build native standalone executables (Desktop & CLI) |
| `just dmg` | Package macOS DMG installer on Darwin systems |
| `just test` | Run test suite via `pytest` |
| `just typecheck` | Run static type analysis with `mypy` |
| `just lint` | Run static analysis and code checks with `ruff check` |
| `just format` | Automatically format code with `ruff format` and `ruff check --fix` |
| `just format-check` | Verify formatting without modifying files (non-mutating gate) |
| `just check` | Full quality gate: `format-check`, `lint`, `typecheck`, and `test` |
| `just bench` | Benchmark CLI startup and core operations |
| `just clean` | Remove caches, temp files, and build outputs |

## Engineering Conventions
- **Package Management**: Always use `uv` for dependency management. Never install packages directly with global `pip`.
- **Formatting & Linting**: Ruff is the single source of truth for formatting and linting.
- **Type Safety**: Full static typing enforced with `mypy` across `src/` and `tests/`.
- **Testing**: Automated unit and integration tests live in `tests/` strictly mirroring the `src/` structure.
