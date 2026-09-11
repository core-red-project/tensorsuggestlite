"""Executable CLI and GUI entry point for TensorSuggestLite."""

import argparse
import os
import sys
from pathlib import Path

# Prevent thread contention and stack pressure in math libraries
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from tensorsuggestlite.core.entities import TrainingConfig
from tensorsuggestlite.use_cases import ExportTFLiteUseCase, PredictTextUseCase, TrainModelUseCase


def run_gui() -> int:
    """Launch the PyQt6 graphical interface."""
    try:
        from PyQt6.QtWidgets import QApplication

        from tensorsuggestlite.ui.main_window import TensorSuggestLiteUI
        from tensorsuggestlite.ui.theme import get_icon, get_ui_font
    except ImportError:
        print("Error: PyQt6 is not installed.", file=sys.stderr)
        print("To run the desktop GUI, install the GUI package with:", file=sys.stderr)
        print("    pip install 'tensorsuggestlite[gui]'", file=sys.stderr)
        return 1

    app = QApplication(sys.argv)
    app.setFont(get_ui_font(10))
    icon = get_icon("TensorSuggestLite.svg")
    if not icon.isNull():
        app.setWindowIcon(icon)

    window = TensorSuggestLiteUI()
    if not icon.isNull():
        window.setWindowIcon(icon)
    window.show()
    return app.exec()


def run_inspect(file_path: str) -> int:
    """Inspect and validate a configuration dataset with deep profiling."""
    path = Path(file_path).resolve()
    print(f"Inspecting configuration: {path.name}")
    try:
        from tensorsuggestlite.core.profiler import DatasetProfiler
        from tensorsuggestlite.parsers import create_default_registry

        registry = create_default_registry()
        parser = registry.get_parser_for_path(path)
        dataset = parser.parse(path)

        print(f"Format: {path.suffix.upper().lstrip('.')}")
        print(f"Total samples: {len(dataset)}")
        print(f"Categories ({len(dataset.unique_categories)}):")
        for cat in dataset.unique_categories:
            count = dataset.categories.count(cat)
            print(f"  - {cat:12s}: {count} samples")

        # Profiling
        profile = DatasetProfiler.profile(dataset)
        print("\nDataset Diagnostics:")
        print(f"  - Unique sentences: {profile.unique_texts}")
        print(f"  - Class imbalance ratio: {profile.imbalance_ratio:.2f}:1")

        if profile.imbalance_warning:
            print(f"  [WARN] {profile.imbalance_warning}")
        if profile.duplicate_samples:
            print(f"  [INFO] Duplicate samples detected: {len(profile.duplicate_samples)} cases.")
        if profile.contradictory_samples:
            print(
                f"  [ALERT] Contradictory samples (same text, different classes): {len(profile.contradictory_samples)} cases:"
            )
            for text, cats in profile.contradictory_samples[:3]:
                print(f'    * "{text}" -> {cats}')

        status = "PASSED WITH WARNINGS" if profile.has_critical_issues else "PASSED (Clean)"
        print(f"\nValidation: {status}")
        return 0
    except Exception as err:
        print(f"Validation: FAILED - {err}")
        return 1


def run_train(
    file_path: str,
    output_dir: str | None = None,
    epochs: int = 12,
    learning_rate: float = 0.001,
    dropout: float = 0.2,
    batch_size: int = 16,
) -> int:
    """Train a model via headless CLI with customizable hyperparameters."""
    path = Path(file_path).resolve()
    print(f"Starting training on: {path.name}")
    print(
        f"Hyperparameters: epochs={epochs}, lr={learning_rate}, dropout={dropout}, batch={batch_size}"
    )

    def log_cb(msg: str) -> None:
        print(f"  [LOG] {msg}")

    def prog_cb(pct: int) -> None:
        if pct % 20 == 0 or pct in {5, 10, 15, 90, 100}:
            print(f"  [PROGRESS] {pct}%")

    try:
        use_case = TrainModelUseCase()
        cfg = TrainingConfig(
            epochs=epochs,
            learning_rate=learning_rate,
            dropout_rate=dropout,
            batch_size=batch_size,
        )
        result = use_case.execute(
            config_path=path,
            output_dir=output_dir,
            config=cfg,
            progress_cb=prog_cb,
            log_cb=log_cb,
        )
        print(f"\nTraining SUCCESS! Artifacts saved to: {result.artifacts.output_dir}")
        print(f"  - Model: {result.artifacts.keras_model_path.name}")
        print(f"  - Tokenizer: {result.artifacts.tokenizer_path.name}")
        print(f"  - Label Encoder: {result.artifacts.label_encoder_path.name}")
        if result.artifacts.metadata_path:
            print(f"  - Metadata: {result.artifacts.metadata_path.name}")
        return 0
    except Exception as err:
        print(f"Training FAILED: {err}")
        return 1


def run_convert(model_dir: str, output: str | None = None) -> int:
    """Convert Keras model to TFLite via headless CLI."""
    path = Path(model_dir).resolve()
    print(f"Converting model from directory: {path}")

    def log_cb(msg: str) -> None:
        print(f"  [LOG] {msg}")

    try:
        use_case = ExportTFLiteUseCase()
        dest = use_case.execute(
            model_dir=path,
            output_path=output,
            log_cb=log_cb,
        )
        size_kb = dest.stat().st_size / 1024.0 if dest.is_file() else 0.0
        print(f"Conversion SUCCESS! TFLite binary ({size_kb:.1f} KB): {dest}")
        return 0
    except Exception as err:
        print(f"Conversion FAILED: {err}")
        return 1


def run_history(target_dir: str) -> int:
    """List and inspect model metadata and training runs."""
    path = Path(target_dir).resolve()
    print(f"Inspecting Model Registry at: {path}")

    metadata_files: list[Path] = []
    if (path / "model_metadata.json").is_file():
        metadata_files.append(path / "model_metadata.json")
    runs_dir = path / "runs"
    if runs_dir.is_dir():
        for run_path in sorted(runs_dir.iterdir(), reverse=True):
            mf = run_path / "model_metadata.json"
            if mf.is_file() and mf not in metadata_files:
                metadata_files.append(mf)

    if not metadata_files:
        print(f"No trained models or metadata found in: {path}")
        return 1

    print(f"Found {len(metadata_files)} model artifact(s):\n")
    print(
        f"{'Run / Timestamp':<22} | {'Samples':<8} | {'Epochs':<7} | {'Accuracy':<9} | {'Loss':<8} | {'SHA-256':<12}"
    )
    print("-" * 76)
    for mf in metadata_files:
        try:
            import json

            with open(mf, encoding="utf-8") as f:
                meta = json.load(f)
            ts = meta.get("created_at", "N/A")[:19].replace("T", " ")
            samples = meta.get("dataset", {}).get("total_samples", "N/A")
            epochs = meta.get("hyperparameters", {}).get("epochs", "N/A")
            metrics = meta.get("metrics", {})
            acc = f"{metrics.get('final_accuracy', 0.0) * 100:.1f}%"
            loss = f"{metrics.get('final_loss', 0.0):.4f}"
            sha = meta.get("dataset", {}).get("sha256", "N/A")[:10]
            print(
                f"{ts:<22} | {str(samples):<8} | {str(epochs):<7} | {acc:<9} | {loss:<8} | {sha:<12}"
            )
        except Exception as err:
            print(f"{mf.parent.name:<22} | Error reading metadata: {err}")
    return 0


def run_predict(
    model_dir: str, text: str, prefer_tflite: bool = False, compare: bool = False
) -> int:
    """Run real-time text classification or cross-runtime comparison via CLI."""
    path = Path(model_dir).resolve()
    try:
        predictor = PredictTextUseCase(path)

        if compare:
            cross = predictor.compare(text)
            k = cross.keras_result
            t = cross.tflite_result
            print("\n--- Cross-Validation: Keras vs TFLite ---")
            print(f'Input: "{text}"')
            print(
                f"Parity:           {'MATCH (Consistent)' if cross.is_consistent else 'MISMATCH (Discrepancy)'}"
            )
            print(f"Confidence Delta: {cross.delta_confidence * 100:.2f}%")
            print(f"Max Class Delta:  {cross.max_class_delta * 100:.2f}%\n")
            print(f"{'Metric':<18} | {'Keras (.keras)':<20} | {'TFLite (.tflite)':<20}")
            print("-" * 64)
            print(f"{'Top Category':<18} | {k.top_category:<20} | {t.top_category:<20}")
            print(
                f"{'Top Confidence':<18} | {f'{k.top_confidence * 100:.2f}%':<20} | {f'{t.top_confidence * 100:.2f}%':<20}"
            )
            print(
                f"{'Latency':<18} | {f'{k.latency_ms:.2f} ms':<20} | {f'{t.latency_ms:.2f} ms':<20}"
            )
            return 0

        result = predictor.predict(text, prefer_tflite=prefer_tflite)
        runtime_label = "TFLite" if result.runtime == "tflite" else "Keras"
        print(f"\nInference Result ({runtime_label} runtime - {result.latency_ms:.2f} ms):")
        print(f'Text: "{result.text}"')
        print(
            f"Predicted Class: {result.top_category} ({result.top_confidence * 100:.1f}% confidence)"
        )
        print("Confidence Breakdown:")
        for score in result.scores:
            bar = "█" * int(score.confidence * 25)
            print(f"  - {score.category:12s}: {score.confidence * 100:5.1f}% {bar}")
        return 0
    except Exception as err:
        print(f"Prediction FAILED: {err}")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="tensorsuggestlite",
        description="Configuration-driven text classification and TFLite model generation.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # GUI command
    subparsers.add_parser("gui", help="Launch PyQt6 graphical interface")

    # Inspect command
    inspect_parser = subparsers.add_parser("inspect", help="Inspect and validate dataset")
    inspect_parser.add_argument("file", help="Path to JSON, YAML, or TOML file")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a classification model")
    train_parser.add_argument("file", help="Path to JSON, YAML, or TOML file")
    train_parser.add_argument("--output-dir", "-o", help="Output directory for artifacts")
    train_parser.add_argument(
        "--epochs", "-e", type=int, default=12, help="Number of training epochs"
    )
    train_parser.add_argument(
        "--learning-rate", "-lr", type=float, default=0.001, help="Learning rate (default: 0.001)"
    )
    train_parser.add_argument(
        "--dropout",
        "-d",
        type=float,
        default=0.2,
        help="Dropout regularization rate (default: 0.2)",
    )
    train_parser.add_argument(
        "--batch-size", "-b", type=int, default=16, help="Training batch size (default: 16)"
    )

    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert Keras model to TFLite")
    convert_parser.add_argument("model_dir", help="Directory containing text_classifier.keras")
    convert_parser.add_argument("--output", "-o", help="Destination path for .tflite file")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict text classification")
    predict_parser.add_argument("model_dir", help="Directory containing trained model artifacts")
    predict_parser.add_argument("text", help="Text sentence to classify")
    predict_parser.add_argument(
        "--tflite", action="store_true", help="Use TFLite binary instead of Keras"
    )
    predict_parser.add_argument(
        "--compare",
        action="store_true",
        help="Run cross-validation comparing Keras vs TFLite side-by-side",
    )

    # History command
    history_parser = subparsers.add_parser(
        "history", help="List and inspect model metadata and training runs"
    )
    history_parser.add_argument("model_dir", help="Directory containing model artifacts or runs")

    args = parser.parse_args()

    if args.command is None or args.command == "gui":
        return run_gui()
    elif args.command == "inspect":
        return run_inspect(args.file)
    elif args.command == "train":
        return run_train(
            file_path=args.file,
            output_dir=args.output_dir,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            dropout=args.dropout,
            batch_size=args.batch_size,
        )
    elif args.command == "convert":
        return run_convert(args.model_dir, args.output)
    elif args.command == "predict":
        return run_predict(
            args.model_dir, args.text, prefer_tflite=args.tflite, compare=args.compare
        )
    elif args.command == "history":
        return run_history(args.model_dir)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
