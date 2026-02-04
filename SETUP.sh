#!/usr/bin/env bash
# TensorSuggestLite Setup Script
# Automates environment creation and dependency installation

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
VENV_DIR=".venv"
PYTHON_MIN_VERSION="3.8"

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_python() {
    if ! command -v python3 &> /dev/null; then
        log_error "python3 not found. Please install Python ${PYTHON_MIN_VERSION}+"
        exit 1
    fi

    local python_version
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    log_info "Found Python ${python_version}"

    # Basic version check
    local major minor
    major=$(echo "$python_version" | cut -d'.' -f1)
    minor=$(echo "$python_version" | cut -d'.' -f2)

    if [ "$major" -lt 3 ] || { [ "$major" -eq 3 ] && [ "$minor" -lt 8 ]; }; then
        log_error "Python ${PYTHON_MIN_VERSION}+ required, found ${python_version}"
        exit 1
    fi
}

create_venv() {
    if [ -d "$VENV_DIR" ]; then
        log_warn "Virtual environment already exists"
        read -p "Recreate? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Removing existing virtual environment..."
            rm -rf "$VENV_DIR"
        else
            log_info "Using existing virtual environment"
            return 0
        fi
    fi

    log_info "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
}

activate_venv() {
    if [ -f "${VENV_DIR}/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "${VENV_DIR}/bin/activate"
    elif [ -f "${VENV_DIR}/Scripts/activate" ]; then
        # Windows Git Bash
        # shellcheck disable=SC1091
        source "${VENV_DIR}/Scripts/activate"
    else
        log_error "Could not find activation script"
        exit 1
    fi
    log_info "Virtual environment activated"
}

install_dependencies() {
    log_info "Upgrading pip..."
    python -m pip install --upgrade pip --quiet

    local req_file="requirements-pinned.txt"
    if [ ! -f "$req_file" ]; then
        log_warn "${req_file} not found, using requirements.txt"
        req_file="requirements.txt"
    fi

    if [ ! -f "$req_file" ]; then
        log_error "No requirements file found"
        exit 1
    fi

    log_info "Installing dependencies from ${req_file}..."
    pip install -r "$req_file"
}

verify_installation() {
    log_info "Verifying installation..."

    if ! python -c "from json_interpreter import trainer" 2>/dev/null; then
        log_error "Core module import failed"
        exit 1
    fi
    log_info "Core modules verified"

    if python -c "import tensorflow" 2>/dev/null; then
        log_info "TensorFlow available"
    else
        log_warn "TensorFlow not available (OK for code review only)"
    fi
}

print_next_steps() {
    echo
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Setup Complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo
    echo "Activate virtual environment:"
    echo "  ${YELLOW}source ${VENV_DIR}/bin/activate${NC}"
    echo
    echo "Run the application:"
    echo "  ${YELLOW}make run${NC}  or  ${YELLOW}python tensorsuggestlite.py${NC}"
    echo
    echo "See ${YELLOW}make help${NC} for all commands"
    echo
}

# Main execution
main() {
    echo "TensorSuggestLite Setup"
    echo "======================="
    echo

    check_python
    create_venv
    activate_venv
    install_dependencies
    verify_installation
    print_next_steps
}

main

