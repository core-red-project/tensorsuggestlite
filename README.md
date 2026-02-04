# TensorSuggestLite

<p align="center">
  <strong>Configuration-Driven Text Classification</strong><br>
  <em>A CoreRed Experimental Tool</em>
</p>

<p align="center">
  <a href="#about">About</a> •
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#contributing">Contributing</a>
</p>

---

## About

**TensorSuggestLite** transforms structured configuration files (JSON, YAML, TOML) into trained TensorFlow/Keras text classification models with automatic TFLite export for embedded deployment.

Designed for rapid ML prototyping, TensorSuggestLite eliminates boilerplate code by treating configuration as training data. Define your categories and examples in a simple file format, and let the application handle tokenization, training, and model export.

### Philosophy

> *"Configuration as training data."*

TensorSuggestLite embodies pragmatic ML workflows: declarative inputs, transparent processing, portable outputs. No framework lock-in, no hidden complexity—just direct transformation from structured data to deployable models.

This is a **CoreRed** project, part of an experimental toolkit for developers who value simplicity and reproducibility.

## Features

- **Multi-Format Input**: Parse JSON, YAML, and TOML configuration files
- **Automated Training**: TensorFlow/Keras model generation with progress tracking
- **Complete Artifact Export**:
  - `tokenizer.json` - Vocabulary and preprocessing config
  - `text_classifier.keras` - Full Keras model for inference
  - `label_encoder.json` - Category-to-index mappings
  - `text_classifier.tflite` - Quantized model for mobile/edge
- **Dual Interface**:
  - PyQt6 desktop GUI with real-time feedback
  - Python API for scripting and automation
- **Modular Architecture**: Format-specific interpreters with isolated logic
- **Theme Support**: Dark and light UI themes
- **Cross-Platform**: Runs on Linux, macOS, and Windows

## Installation

### Quick Start

```bash
git clone https://github.com/Sxnnyside-Project/TensorSuggestLite.git
cd TensorSuggestLite
make setup
```

### Prerequisites

- Python 3.8 or later
- pip (Python package manager)
- 2 GB RAM minimum (4 GB recommended)

### Manual Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements-pinned.txt
```

### Verification

```bash
# Test core imports
python -c "from json_interpreter import trainer; print('✓ Installation verified')"

# Launch GUI
make run
```

## Usage

### GUI Application

```bash
make run
```

**Workflow**:
1. Click **Import JSON/YAML/TOML** → select configuration file
2. Click **Train Model** → wait for training (1-5 minutes)
3. Click **Generate TFLite** → convert to mobile format
4. Click **Export TFLite** → save for deployment

### Programmatic API

#### Basic Training

```python
from json_interpreter import trainer

result = trainer.train_from_json('config.json')
print(f"Model: {result['model_path']}")
```

#### With Progress Callbacks

```python
from json_interpreter import trainer

def on_progress(percent):
    print(f"Training: {percent}%")

def on_log(message):
    print(f"[LOG] {message}")

result = trainer.train_from_json(
    'config.json',
    progress_cb=on_progress,
    log_cb=on_log,
    epochs=12
)
```

#### TFLite Conversion

```python
from json_interpreter import trainer

tflite_path = trainer.convert_to_tflite(
    model_dir='generated/json',
    progress_cb=on_progress,
    log_cb=on_log
)
print(f"TFLite: {tflite_path}")
```

### Configuration Format

All formats follow this structure:

```json
{
  "CATEGORY_NAME": {
    "respuestas": [
      {"respuesta": "training text example 1"},
      {"respuesta": "training text example 2"}
    ],
    "sinonimos": ["synonym1", "synonym2"]
  }
}
```

**Example** (`test/question_data.json`):
```json
{
  "POSITIVE": {
    "respuestas": [
      {"respuesta": "This is excellent!"},
      {"respuesta": "I love this product"}
    ],
    "sinonimos": ["good", "great", "amazing"]
  },
  "NEGATIVE": {
    "respuestas": [
      {"respuesta": "This is terrible"},
      {"respuesta": "Very disappointing"}
    ],
    "sinonimos": ["bad", "poor", "awful"]
  }
}
```

## Architecture

```
TensorSuggestLite/
├── tensorsuggestlite.py      # GUI entry point
├── json_interpreter/          # JSON format handler
│   └── trainer.py             # Training & conversion
├── yaml_interpreter/          # YAML format handler
│   └── trainer.py
├── toml_interpreter/          # TOML format handler
│   └── trainer.py
├── styles/                    # UI themes
│   ├── dark.qss
│   └── light.qss
├── examples/                  # Functional examples (English schema)
│   ├── example.json           # Recommended format
│   ├── example.yaml
│   └── example.toml
├── test/                      # Test data (legacy, Spanish)
├── generated/                 # Runtime output (git-ignored)
│   ├── json/
│   ├── yaml/
│   └── toml/
└── docs/                      # Documentation
```

### Technology Stack

| Component | Technology |
|-----------|------------|
| ML Framework | TensorFlow/Keras |
| GUI | PyQt6 |
| Data Processing | NumPy, scikit-learn |
| Config Parsing | PyYAML, toml |

### Processing Pipeline

1. **Parse** configuration → extract text samples and labels
2. **Tokenize** → build vocabulary using Keras Tokenizer
3. **Encode** → map labels to numeric IDs
4. **Train** → simple neural network (Embedding → Dense → Softmax)
5. **Export** → save tokenizer, model, label encoder
6. **Convert** → TFLite with quantization optimization

## Documentation

- **[User Guide](docs/USER_GUIDE.md)** - Installation, usage, troubleshooting
- **[Developer Guide](docs/DEVELOPER_GUIDE.md)** - Architecture, extending, API reference
- **[Contributing](CONTRIBUTING.md)** - How to contribute
- **[Security](SECURITY.md)** - Vulnerability reporting
- **[Repository Status](REPOSITORY_STATUS.md)** - Project health and limitations

## Makefile Commands

```bash
make help      # Show all commands
make setup     # Install dependencies
make run       # Launch GUI
make clean     # Remove generated files
make lint      # Run code quality checks
make test      # Run tests (when implemented)
```

## Project Status

**Version**: 2.0 (Experimental)

TensorSuggestLite is functional and stable for prototyping. API is subject to change based on community feedback.

### Known Limitations

- No input validation (well-formed files required)
- Single-threaded training (may freeze GUI briefly)
- No hyperparameter tuning interface
- No real-time evaluation metrics
- Basic model architecture (not SOTA)

See [REPOSITORY_STATUS.md](REPOSITORY_STATUS.md) for complete details.

### Roadmap

- [ ] CLI mode for headless operation
- [ ] Input validation and error recovery
- [ ] Model evaluation metrics display
- [ ] Hyperparameter configuration UI
- [ ] Batch processing support

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup
- Code standards
- Pull request process
- Review criteria

**Priority Areas**:
1. Input validation
2. Test coverage (pytest)
3. CLI mode implementation
4. Documentation improvements

Before contributing, read our [Code of Conduct](CODE_OF_CONDUCT.md).

## Security

For security vulnerabilities, see [SECURITY.md](SECURITY.md).

**Do not** report security issues via GitHub Issues.

**Contact**: security.sxnnyside@sxnnysideproject.com

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

Copyright (c) 2025-2026 CoreRed Project

## Support

- **Documentation**: See `docs/` directory
- **Issues**: [GitHub Issues](https://github.com/Sxnnyside-Project/TensorSuggestLite/issues)
- **Email**: support.sxnnyside@sxnnysideproject.com
- **Maintainer**: houjou.sxnnyside@sxnnysideproject.com

---

<p align="center">
  <sub>Built with discipline by <a href="https://github.com/Sxnnyside-Project">CoreRed Project</a></sub>
</p>

