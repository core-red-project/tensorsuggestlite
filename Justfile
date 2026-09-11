# Task Runner for TensorSuggestLite
set shell := ["bash", "-uc"]

default:
    @just --list

# Bootstrap dependencies and virtual environment
install:
    uv sync --all-extras

# Start GUI application in development mode
dev:
    uv run python -m tensorsuggestlite.main

# Produce distribution packages
build:
    uv build

# Build standalone desktop application bundle
bundle-desktop:
    uv run pyinstaller --noconfirm tensorsuggestlite.spec

# Build standalone headless CLI binary
bundle-cli:
    uv run pyinstaller --noconfirm tensorsuggestlite-cli.spec

# Build all standalone bundles (desktop and CLI)
bundle: bundle-desktop bundle-cli

# Build macOS .dmg installer from app bundle
dmg: bundle-desktop
    @if [ "{{ os() }}" = "macos" ]; then \
        rm -f dist/TensorSuggestLite.dmg; \
        hdiutil create -volname "TensorSuggestLite" -srcfolder dist/TensorSuggestLite.app -ov -format UDZO dist/TensorSuggestLite.dmg; \
        echo "Successfully created dist/TensorSuggestLite.dmg"; \
    else \
        echo "Skipping DMG creation: not running on macOS"; \
    fi


# Run automated tests
test:
    uv run pytest

# Run static type checking
typecheck:
    uv run mypy src/tensorsuggestlite tests

# Run static analysis and linting
lint:
    uv run ruff check src tests

# Apply automatic formatting
format:
    uv run ruff format src tests

# Run full quality gate
check: format lint typecheck test

# Remove build artifacts and temporary caches
clean:
    rm -rf dist/ build/ *.egg-info
    rm -rf generated/
    rm -rf __pycache__ */__pycache__ */*/__pycache__
    rm -rf .pytest_cache .mypy_cache .ruff_cache
    rm -f *.pyc */*.pyc */*/*.pyc
