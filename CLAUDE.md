# TensorSuggestLite — Project Guide

## Overview
TensorSuggestLite is a Python application that trains TensorFlow/Keras text classification models directly from configuration files (JSON, YAML, TOML) and exports optimized TFLite artifacts for embedded or mobile deployment.

## Repository Topology
- **Topology**: Monolithic repository (single deployable/publishable unit).
- **Toolchain**: Python 3.11 managed via `uv`, with `just` as the canonical task runner.
- **Architecture**: Clean Architecture / SOLID / DRY under `src/tensorsuggestlite`.

## Directory Structure
```
├── src/tensorsuggestlite/
│   ├── core/                  # Domain entities, interfaces & exceptions
│   ├── parsers/               # Extensible JSON, YAML, TOML parsers (SRP/OCP)
│   ├── ml/                    # Unified Keras trainer & TFLite exporter
│   ├── use_cases/             # TrainModel and ExportTFLite use cases
│   ├── ui/                    # Decoupled PyQt6 presentation & QThread workers
│   └── main.py                # Main application entry point
├── examples/                  # Configuration dataset examples (JSON, YAML, TOML)
├── styles/                    # PyQt6 QSS themes (dark and light)
├── icons/                     # UI SVG icons
├── tests/                     # Automated test suite (pytest)
├── pyproject.toml             # Project definition, build backend (Hatchling) & configs
├── Justfile                   # Canonical command surface
└── CLAUDE.md                  # Developer & AI reference
```

## Command Surface (`just`)
All commands are run using `just`:

| Command | Purpose |
|---|---|
| `just install` | Bootstrap environment and dependencies via `uv sync` |
| `just dev` | Launch the PyQt6 GUI in development mode |
| `just build` | Build distribution packages (wheel and sdist) with `uv build` |
| `just test` | Run test suite via `pytest` |
| `just typecheck` | Run static type analysis with `mypy` |
| `just lint` | Run static analysis and code checks with `ruff check` |
| `just format` | Automatically format code with `ruff format` |
| `just check` | Full quality gate: format, lint, typecheck, and test |
| `just clean` | Remove caches, temp files, and build outputs |

## Engineering Conventions
- **Package Management**: Always use `uv` for dependency management. Never install packages directly with global `pip`.
- **Formatting & Linting**: Ruff is the single source of truth for formatting and linting.
- **Type Safety**: Full static typing enforced with `mypy` across `src/` and `tests/`.
- **Testing**: Automated unit and integration tests live in `tests/` using `pytest`.
