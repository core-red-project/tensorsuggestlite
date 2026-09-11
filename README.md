# TensorSuggestLite

![Banner](TensorSuggestLite.png)

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![License](https://img.shields.io/badge/License-MIT-green)
[![CI](https://github.com/core-red-project/tensorsuggestlite/workflows/CI/badge.svg)](https://github.com/core-red-project/tensorsuggestlite/actions)

<p align="center">
  <strong>Declarative ML ✦ Edge-Ready TFLite ✦ Zero-TensorFlow Inference</strong><br>
  <em>Transform configuration files into trained text classification models with instant edge deployment.</em>
</p>

<p align="center">
  <a href="#about">About</a> ✦
  <a href="#features">Features</a> ✦
  <a href="#installation">Installation</a> ✦
  <a href="#usage">Usage</a> ✦
  <a href="#architecture">Architecture</a> ✦
  <a href="#contributing">Contributing</a>
</p>

---

## About

**TensorSuggestLite** turns structured configuration files (JSON, YAML, TOML) into trained TensorFlow/Keras text classification models with automated TFLite export for embedded and edge environments.

Machine learning prototypes often suffer from repetitive boilerplate code for tokenization, label encoding, training loops, and quantization. TensorSuggestLite treats configuration directly as training data: declare categories and examples in standard structured formats, and the engine trains, profiles, evaluates, and converts the model into an optimized artifact ready for production.

The pipeline produces both a full Keras model for server workloads and a quantized TFLite binary that runs with sub-2ms latency on edge runtimes.

### Philosophy

> *"Configuration as training data."*

This is a Core Red Project, part of the Sxnnyside Project's ecosystem.

## Features

- **Declarative Multi-Format Input**: Direct parsing of JSON, YAML, and TOML datasets.
- **Dataset Profiling & Diagnostics**: Automated detection of class imbalance, duplicate sentences, and contradictory labeling prior to training.
- **Complete Artifact Bundle**: Exports `tokenizer.json`, `text_classifier.keras`, `label_encoder.json`, `text_classifier.tflite`, and rich `model_metadata.json`.
- **Confusion Matrix & Metrics**: Transparent training telemetry reporting F1-Score, Precision, Recall, Confusion Matrix, and tensor shapes.
- **Live Inference Playground**: Real-time cross-validation in GUI and CLI (`--compare`) verifying prediction parity between Keras and TFLite.
- **Zero-TensorFlow Lightweight Runtime**: Standalone inference engine (`lite_runtime.py`) executing predictions with ~5 MB dependencies and ~1 ms latency.
- **Dual Packaging**: Headless CLI for servers/containers alongside a native desktop GUI with dark/light themes.

## Installation

### Prerequisites

- Python (3.11)
- [uv](https://docs.astral.sh/uv/) (recommended task/package manager) or standard `pip`
- [just](https://github.com/casey/just) (command runner for source builds)

### Package Installation

```bash
# Headless CLI only (lightweight, zero GUI dependencies)
pip install tensorsuggestlite

# Full installation with desktop PyQt6 GUI
pip install "tensorsuggestlite[gui]"
```

### From Source

```bash
git clone https://github.com/core-red-project/tensorsuggestlite.git
cd tensorsuggestlite

# Install dependencies and sync virtual environment
just install

# Run automated test suite
just test
```

## Usage

### Headless CLI

```bash
# Inspect and profile dataset integrity
uv run tensorsuggestlite inspect examples/example.json

# Train model headlessly with custom hyperparameters
uv run tensorsuggestlite train examples/example.json --output-dir generated/json --epochs 12 --learning-rate 0.001

# Convert Keras model to quantized TFLite
uv run tensorsuggestlite convert generated/json

# Run real-time prediction
uv run tensorsuggestlite predict generated/json "This product is fantastic!"

# Cross-validate Keras vs TFLite parity side-by-side
uv run tensorsuggestlite predict generated/json "Outstanding experience!" --compare

# Audit model registry and historical runs
uv run tensorsuggestlite history generated/json
```

### Desktop GUI Application

```bash
# Launch GUI in development mode
just dev
```

1. **Import**: Click **Import JSON/YAML/TOML** or drag & drop configuration files into the window.
2. **Profile**: Review dataset balance and sample counts in the console.
3. **Train**: Adjust hyperparameters (`Epochs`, `Learning Rate`, `Dropout`) and click **Train Model**.
4. **Convert & Export**: Click **Generate TFLite** and save the binary artifact.
5. **Playground**: Test sentences in real-time or click **Compare Keras vs TFLite** to verify quantization consistency.

### Standalone Zero-TensorFlow Inference

For edge deployment or resource-constrained microservices without full TensorFlow:

```bash
uv run python scripts/standalone_infer.py generated/json "Can you help me solve this?"
```

## Architecture

```
tensorsuggestlite/
├── src/tensorsuggestlite/
│   ├── core/         # Domain entities, protocols, profiler, and exceptions
│   ├── parsers/      # Format parsers (JSON, YAML, TOML) with dynamic registry
│   ├── ml/           # Keras training engine, TFLite exporter, and predictors
│   ├── use_cases/    # Application workflows (Train, Export, Predict)
│   ├── ui/           # PyQt6 interface, native fonts, worker threads, and themes
│   ├── lite_runtime.py # Standalone Zero-TF inference runner
│   └── main.py       # Dual-mode CLI/GUI entry point
├── scripts/          # Standalone inference scripts
├── styles/           # Dark and light QSS stylesheets
├── examples/         # Verified multi-format training fixtures
├── tests/            # Automated test suite (29 tests)
└── Justfile          # Canonical engineering task recipes
```

## Contributing

Contributions are accepted. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Before contributing, read the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <strong>TensorSuggestLite</strong> — A Core Red Project<br>
  <em>&copy; 2026 Sxnnyside Project</em>
</p>
