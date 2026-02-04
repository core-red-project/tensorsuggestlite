.PHONY: help setup run clean lint test

PYTHON := python3
VENV := .venv
VENV_ACTIVATE := $(VENV)/bin/activate

help:
	@echo "TensorSuggestLite - Makefile Commands"
	@echo "======================================"
	@echo ""
	@echo "  make setup    - Create virtual environment and install dependencies"
	@echo "  make run      - Run the GUI application"
	@echo "  make clean    - Remove generated files and artifacts"
	@echo "  make lint     - Run code quality checks (flake8, black)"
	@echo "  make test     - Run tests (when implemented)"
	@echo "  make docs     - Generate documentation (when implemented)"
	@echo ""

setup:
	@echo "Running setup script..."
	@chmod +x SETUP.sh
	@./SETUP.sh

run:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi
	@echo "Launching TensorSuggestLite..."
	@. $(VENV_ACTIVATE) && $(PYTHON) tensorsuggestlite.py

clean:
	@echo "Cleaning generated files..."
	@rm -rf generated/
	@rm -rf __pycache__/ */__pycache__/ */*/__pycache__/
	@rm -rf .pytest_cache/ .mypy_cache/ .coverage
	@rm -f *.pyc */*.pyc */*/*.pyc
	@rm -f *.backup */*.backup */*/*.backup
	@echo "Clean complete"

clean-all: clean
	@echo "Removing virtual environment..."
	@rm -rf $(VENV)
	@echo "Full clean complete"

lint:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi
	@echo "Running linters..."
	@. $(VENV_ACTIVATE) && flake8 --config=.flake8 . || true
	@. $(VENV_ACTIVATE) && black --check --line-length 100 . || true
	@. $(VENV_ACTIVATE) && mypy --config-file=mypy.ini tensorsuggestlite.py || true
	@echo "Lint complete"

test:
	@echo "Tests not yet implemented"
	@echo "To add tests: create tests/ directory and use pytest"

install-dev:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi
	@echo "Installing development dependencies..."
	@. $(VENV_ACTIVATE) && pip install -r requirements-dev.txt
	@echo "Dev dependencies installed"

