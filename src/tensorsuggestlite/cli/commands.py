"""Command implementations for the TensorSuggestLite CLI.

Implements POSIX stream separation:
- stdout carries data and structured JSON payloads
- stderr carries diagnostic, progress, and error logging
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from tensorsuggestlite.core.entities import TrainingConfig
from tensorsuggestlite.use_cases import ExportTFLiteUseCase, PredictTextUseCase, TrainModelUseCase


def log_diagnostic(msg: str, quiet: bool = False) -> None:
    """Emit diagnostic or progress text exclusively to stderr.

    Args:
        msg: Text message to write.
        quiet: If True, suppress the output.
    """
    if not quiet:
        sys.stderr.write(f"{msg}\n")
        sys.stderr.flush()


def log_error(msg: str) -> None:
    """Emit an error message to stderr.

    Args:
        msg: Error description text.
    """
    sys.stderr.write(f"error: {msg}\n")
    sys.stderr.flush()


def run_inspect(
    file_path: str,
    skeleton: str | None = None,
    json_mode: bool = False,
    quiet: bool = False,
) -> int:
    """Inspect and validate a configuration dataset with deep profiling.

    Args:
        file_path: Path to dataset configuration.
        skeleton: Optional fixed skeleton mapping string (JSON or in:out).
        json_mode: When True, emit JSON payload to stdout.
        quiet: Suppress diagnostic messages.

    Returns:
        int: 0 on success, 1 on failure.
    """
    path = Path(file_path).resolve()
    log_diagnostic(f"Inspecting configuration: {path.name}", quiet=quiet or json_mode)

    try:
        from tensorsuggestlite.core.profiler import DatasetProfiler
        from tensorsuggestlite.parsers import create_default_registry
        from tensorsuggestlite.parsers.base import parse_skeleton_str

        registry = create_default_registry()
        parser = registry.get_parser_for_path(path)
        skel_dict = parse_skeleton_str(skeleton)
        dataset = parser.parse(path, skeleton=skel_dict)

        profile = DatasetProfiler.profile(dataset)
        status = "PASSED WITH WARNINGS" if profile.has_critical_issues else "PASSED (Clean)"

        if json_mode:
            category_counts = {
                cat: dataset.categories.count(cat) for cat in dataset.unique_categories
            }
            payload = {
                "file": str(path),
                "format": path.suffix.upper().lstrip("."),
                "total_samples": len(dataset),
                "unique_categories": dataset.unique_categories,
                "category_counts": category_counts,
                "unique_sentences": profile.unique_texts,
                "imbalance_ratio": round(profile.imbalance_ratio, 2),
                "has_critical_issues": profile.has_critical_issues,
                "status": status,
                "warnings": [profile.imbalance_warning] if profile.imbalance_warning else [],
                "duplicate_count": len(profile.duplicate_samples),
                "contradiction_count": len(profile.contradictory_samples),
            }
            sys.stdout.write(f"{json.dumps(payload, indent=2)}\n")
            sys.stdout.flush()
            return 0

        # Human-readable output
        sys.stdout.write(f"Format: {path.suffix.upper().lstrip('.')}\n")
        sys.stdout.write(f"Total samples: {len(dataset)}\n")
        sys.stdout.write(f"Categories ({len(dataset.unique_categories)}):\n")
        for cat in dataset.unique_categories:
            count = dataset.categories.count(cat)
            sys.stdout.write(f"  - {cat:12s}: {count} samples\n")

        sys.stdout.write("\nDataset Diagnostics:\n")
        sys.stdout.write(f"  - Unique sentences: {profile.unique_texts}\n")
        sys.stdout.write(f"  - Class imbalance ratio: {profile.imbalance_ratio:.2f}:1\n")

        if profile.imbalance_warning:
            log_diagnostic(f"  [WARN] {profile.imbalance_warning}", quiet=quiet)
        if profile.duplicate_samples:
            log_diagnostic(
                f"  [INFO] Duplicate samples detected: {len(profile.duplicate_samples)} cases.",
                quiet=quiet,
            )
        if profile.contradictory_samples:
            log_diagnostic(
                f"  [ALERT] Contradictory samples (same text, different classes): {len(profile.contradictory_samples)} cases:",
                quiet=quiet,
            )
            for text, cats in profile.contradictory_samples[:3]:
                log_diagnostic(f'    * "{text}" -> {cats}', quiet=quiet)

        sys.stdout.write(f"\nValidation: {status}\n")
        sys.stdout.flush()
        return 0
    except Exception as err:
        log_error(f"Validation FAILED: {err}")
        return 1


def run_train(
    file_path: str,
    output_dir: str | None = None,
    skeleton: str | None = None,
    epochs: int = 12,
    learning_rate: float = 0.001,
    dropout: float = 0.2,
    batch_size: int = 16,
    json_mode: bool = False,
    quiet: bool = False,
) -> int:
    """Train a classification model via headless CLI.

    Args:
        file_path: Path to dataset configuration.
        output_dir: Destination folder for trained artifacts.
        skeleton: Optional fixed schema skeleton mapping.
        epochs: Number of training epochs.
        learning_rate: Optimizer learning rate.
        dropout: Regularization dropout rate.
        batch_size: Training batch size.
        json_mode: Output structured JSON.
        quiet: Suppress progress on stderr.

    Returns:
        int: 0 on success, 1 on failure.
    """
    path = Path(file_path).resolve()
    log_diagnostic(f"Starting training on: {path.name}", quiet=quiet or json_mode)
    log_diagnostic(
        f"Hyperparameters: epochs={epochs}, lr={learning_rate}, dropout={dropout}, batch={batch_size}",
        quiet=quiet or json_mode,
    )

    def log_cb(msg: str) -> None:
        log_diagnostic(f"  [LOG] {msg}", quiet=quiet or json_mode)

    def prog_cb(pct: int) -> None:
        if pct % 20 == 0 or pct in {5, 10, 15, 90, 100}:
            log_diagnostic(f"  [PROGRESS] {pct}%", quiet=quiet or json_mode)

    try:
        from tensorsuggestlite.parsers.base import parse_skeleton_str

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
            skeleton=parse_skeleton_str(skeleton),
            progress_cb=prog_cb,
            log_cb=log_cb,
        )

        if json_mode:
            payload = {
                "status": "success",
                "output_dir": str(result.artifacts.output_dir),
                "keras_model": str(result.artifacts.keras_model_path),
                "tokenizer": str(result.artifacts.tokenizer_path),
                "label_encoder": str(result.artifacts.label_encoder_path),
                "metadata": (
                    str(result.artifacts.metadata_path) if result.artifacts.metadata_path else None
                ),
                "metrics": result.metrics,
            }
            sys.stdout.write(f"{json.dumps(payload, indent=2)}\n")
            sys.stdout.flush()
            return 0

        sys.stdout.write(f"\nTraining SUCCESS! Artifacts saved to: {result.artifacts.output_dir}\n")
        sys.stdout.write(f"  - Model: {result.artifacts.keras_model_path.name}\n")
        sys.stdout.write(f"  - Tokenizer: {result.artifacts.tokenizer_path.name}\n")
        sys.stdout.write(f"  - Label Encoder: {result.artifacts.label_encoder_path.name}\n")
        if result.artifacts.metadata_path:
            sys.stdout.write(f"  - Metadata: {result.artifacts.metadata_path.name}\n")
        sys.stdout.flush()
        return 0
    except Exception as err:
        log_error(f"Training FAILED: {err}")
        return 1


def run_convert(
    model_dir: str,
    output: str | None = None,
    quantization: str = "dynamic",
    json_mode: bool = False,
    quiet: bool = False,
) -> int:
    """Convert a Keras model to TFLite format via headless CLI.

    Args:
        model_dir: Directory containing text_classifier.keras.
        output: Destination path for .tflite binary.
        quantization: Quantization type: dynamic, float16, or none.
        json_mode: Output structured JSON.
        quiet: Suppress progress on stderr.

    Returns:
        int: 0 on success, 1 on failure.
    """
    path = Path(model_dir).resolve()
    log_diagnostic(
        f"Converting model from directory: {path} (quantization: {quantization})",
        quiet=quiet or json_mode,
    )

    def log_cb(msg: str) -> None:
        log_diagnostic(f"  [LOG] {msg}", quiet=quiet or json_mode)

    try:
        use_case = ExportTFLiteUseCase()
        dest = use_case.execute(
            model_dir=path,
            output_path=output,
            quantization=quantization,
            log_cb=log_cb,
        )
        size_kb = dest.stat().st_size / 1024.0 if dest.is_file() else 0.0

        if json_mode:
            payload = {
                "status": "success",
                "tflite_path": str(dest),
                "quantization": quantization,
                "size_kb": round(size_kb, 2),
            }
            sys.stdout.write(f"{json.dumps(payload, indent=2)}\n")
            sys.stdout.flush()
            return 0

        sys.stdout.write(
            f"Conversion SUCCESS! TFLite binary ({size_kb:.1f} KB, {quantization}): {dest}\n"
        )
        sys.stdout.flush()
        return 0
    except Exception as err:
        log_error(f"Conversion FAILED: {err}")
        return 1


def run_history(
    target_dir: str,
    json_mode: bool = False,
    quiet: bool = False,
) -> int:
    """List and inspect model metadata and training runs.

    Args:
        target_dir: Directory containing model artifacts or runs.
        json_mode: Output structured JSON array.
        quiet: Suppress diagnostic headers.

    Returns:
        int: 0 on success, 1 on failure.
    """
    path = Path(target_dir).resolve()
    log_diagnostic(f"Inspecting Model Registry at: {path}", quiet=quiet or json_mode)

    metadata_files: list[Path] = []
    if (path / "model_metadata.json").is_file():
        metadata_files.append(path / "model_metadata.json")
    if path.is_dir():
        for sub in sorted(path.iterdir(), reverse=True):
            if sub.is_dir():
                mf = sub / "model_metadata.json"
                if mf.is_file() and mf not in metadata_files:
                    metadata_files.append(mf)

    if not metadata_files:
        log_error(f"No trained models or metadata found in: {path}")
        return 1

    runs_payload: list[dict[str, Any]] = []
    for mf in metadata_files:
        try:
            with open(mf, encoding="utf-8") as f:
                meta = json.load(f)
            runs_payload.append(meta)
        except Exception as err:
            log_diagnostic(f"  [WARN] Failed to read {mf}: {err}", quiet=quiet)

    if json_mode:
        sys.stdout.write(f"{json.dumps(runs_payload, indent=2)}\n")
        sys.stdout.flush()
        return 0

    sys.stdout.write(f"Found {len(metadata_files)} model artifact(s):\n\n")
    sys.stdout.write(
        f"{'Run / Timestamp':<20} | {'Samples':<7} | {'Epochs':<6} | {'Accuracy':<9} | {'F1-Macro':<8} | {'Duration':<8} | {'SHA-256':<10}\n"
    )
    sys.stdout.write(f"{'-' * 80}\n")
    for meta in runs_payload:
        ts = str(meta.get("created_at", "N/A"))[:19].replace("T", " ")
        samples = str(meta.get("dataset", {}).get("total_samples", "N/A"))
        epochs = str(meta.get("hyperparameters", {}).get("epochs", "N/A"))
        metrics = meta.get("metrics", {})
        acc = f"{metrics.get('final_accuracy', 0.0) * 100:.1f}%"
        f1_macro = (
            f"{metrics.get('macro_avg_f1', 0.0) * 100:.1f}%" if "macro_avg_f1" in metrics else "N/A"
        )
        duration_s = metrics.get("training_duration_seconds")
        dur_str = f"{duration_s:.1f}s" if duration_s is not None else "N/A"
        sha = str(meta.get("dataset", {}).get("sha256", "N/A"))[:10]
        sys.stdout.write(
            f"{ts:<20} | {samples:<7} | {epochs:<6} | {acc:<9} | {f1_macro:<8} | {dur_str:<8} | {sha:<10}\n"
        )

    # Detailed per-class breakdown if single model artifact inspected
    if len(runs_payload) == 1 and runs_payload[0].get("per_class_metrics"):
        pcm = runs_payload[0]["per_class_metrics"]
        sys.stdout.write("\nPer-Class Metrics Evaluation:\n")
        sys.stdout.write(
            f"{'Category':<16} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}\n"
        )
        sys.stdout.write(f"{'-' * 62}\n")
        for cat, scores in pcm.items():
            p = f"{scores.get('precision', 0.0) * 100:.1f}%"
            r = f"{scores.get('recall', 0.0) * 100:.1f}%"
            f1 = f"{scores.get('f1_score', 0.0) * 100:.1f}%"
            sup = int(scores.get("support", 0))
            sys.stdout.write(f"{cat:<16} | {p:<10} | {r:<10} | {f1:<10} | {sup:<8}\n")

    sys.stdout.flush()
    return 0


def run_predict(
    model_dir: str,
    text: str | None = None,
    batch_file: str | None = None,
    prefer_tflite: bool = False,
    compare: bool = False,
    json_mode: bool = False,
) -> int:
    """Classify input text or execute cross-runtime comparison, supporting batch inference.

    Args:
        model_dir: Directory containing trained model artifacts.
        text: Text string to classify, or '-' for stdin streaming.
        batch_file: Optional path to file containing multiple sentences.
        prefer_tflite: Whether to prefer TFLite runtime.
        compare: Whether to execute cross-runtime comparison.
        json_mode: Output structured JSON payload.

    Returns:
        int: 0 on success, 1 on failure.
    """
    path = Path(model_dir).resolve()
    try:
        predictor = PredictTextUseCase(path)

        # 1. Batch Prediction Resolution
        batch_inputs: list[str] = []
        if batch_file:
            bf_path = Path(batch_file).resolve()
            if not bf_path.is_file():
                log_error(f"Batch file not found: {bf_path}")
                return 1
            content = bf_path.read_text(encoding="utf-8")
            if bf_path.suffix.lower() == ".json":
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, str):
                            batch_inputs.append(item.strip())
                        elif isinstance(item, dict):
                            val = item.get("text") or item.get("input") or item.get("in")
                            if val:
                                batch_inputs.append(str(val).strip())
            else:
                batch_inputs = [line.strip() for line in content.splitlines() if line.strip()]
        elif text == "-":
            batch_inputs = [line.strip() for line in sys.stdin.read().splitlines() if line.strip()]

        if batch_inputs:
            batch_results = predictor.predict_batch(batch_inputs, prefer_tflite=prefer_tflite)
            runtime_lbl = (
                "tflite"
                if prefer_tflite and (path / "text_classifier.tflite").is_file()
                else "keras"
            )

            if json_mode:
                batch_payload = [
                    {
                        "text": res.text,
                        "runtime": res.runtime,
                        "top_category": res.top_category,
                        "top_confidence": round(res.top_confidence, 4),
                        "latency_ms": round(res.latency_ms, 2),
                    }
                    for res in batch_results
                ]
                sys.stdout.write(f"{json.dumps(batch_payload, indent=2)}\n")
                sys.stdout.flush()
                return 0

            sys.stdout.write(
                f"\nBatch Inference Results ({len(batch_results)} items - {runtime_lbl.upper()} runtime):\n\n"
            )
            sys.stdout.write(
                f"{'#':<4} | {'Predicted Category':<18} | {'Confidence':<10} | {'Latency':<9} | {'Text'}\n"
            )
            sys.stdout.write(f"{'-' * 78}\n")
            for idx, res in enumerate(batch_results, 1):
                conf = f"{res.top_confidence * 100:.1f}%"
                lat = f"{res.latency_ms:.1f}ms"
                truncated_text = res.text[:32] + "..." if len(res.text) > 35 else res.text
                sys.stdout.write(
                    f"{idx:<4} | {res.top_category:<18} | {conf:<10} | {lat:<9} | {truncated_text}\n"
                )
            sys.stdout.flush()
            return 0

        # 2. Single Text Prediction
        if not text:
            log_error("No input text provided. Provide a text argument, --file, or '-' for stdin.")
            return 1

        if compare:
            cross = predictor.compare(text)
            k = cross.keras_result
            t = cross.tflite_result

            if json_mode:
                cross_payload = {
                    "input": text,
                    "is_consistent": cross.is_consistent,
                    "delta_confidence": round(cross.delta_confidence, 4),
                    "max_class_delta": round(cross.max_class_delta, 4),
                    "keras": {
                        "category": k.top_category,
                        "confidence": round(k.top_confidence, 4),
                        "latency_ms": round(k.latency_ms, 2),
                    },
                    "tflite": {
                        "category": t.top_category,
                        "confidence": round(t.top_confidence, 4),
                        "latency_ms": round(t.latency_ms, 2),
                    },
                }
                sys.stdout.write(f"{json.dumps(cross_payload, indent=2)}\n")
                sys.stdout.flush()
                return 0

            sys.stdout.write("\n--- Cross-Validation: Keras vs TFLite ---\n")
            sys.stdout.write(f'Input: "{text}"\n')
            parity = "MATCH (Consistent)" if cross.is_consistent else "MISMATCH (Discrepancy)"
            sys.stdout.write(f"Parity:           {parity}\n")
            sys.stdout.write(f"Confidence Delta: {cross.delta_confidence * 100:.2f}%\n")
            sys.stdout.write(f"Max Class Delta:  {cross.max_class_delta * 100:.2f}%\n\n")
            sys.stdout.write(
                f"{'Metric':<18} | {'Keras (.keras)':<20} | {'TFLite (.tflite)':<20}\n"
            )
            sys.stdout.write(f"{'-' * 64}\n")
            sys.stdout.write(
                f"{'Top Category':<18} | {k.top_category:<20} | {t.top_category:<20}\n"
            )
            sys.stdout.write(
                f"{'Top Confidence':<18} | {f'{k.top_confidence * 100:.2f}%':<20} | {f'{t.top_confidence * 100:.2f}%':<20}\n"
            )
            sys.stdout.write(
                f"{'Latency':<18} | {f'{k.latency_ms:.2f} ms':<20} | {f'{t.latency_ms:.2f} ms':<20}\n"
            )
            sys.stdout.flush()
            return 0

        result = predictor.predict(text, prefer_tflite=prefer_tflite)
        runtime_label = "TFLite" if result.runtime == "tflite" else "Keras"

        if json_mode:
            single_payload = {
                "text": result.text,
                "runtime": result.runtime,
                "top_category": result.top_category,
                "top_confidence": round(result.top_confidence, 4),
                "latency_ms": round(result.latency_ms, 2),
                "scores": [
                    {"category": s.category, "confidence": round(s.confidence, 4)}
                    for s in result.scores
                ],
            }
            sys.stdout.write(f"{json.dumps(single_payload, indent=2)}\n")
            sys.stdout.flush()
            return 0

        sys.stdout.write(
            f"\nInference Result ({runtime_label} runtime - {result.latency_ms:.2f} ms):\n"
        )
        sys.stdout.write(f'Text: "{result.text}"\n')
        sys.stdout.write(
            f"Predicted Class: {result.top_category} ({result.top_confidence * 100:.1f}% confidence)\n"
        )
        sys.stdout.write("Confidence Breakdown:\n")
        for score in result.scores:
            bar = "█" * int(score.confidence * 25)
            sys.stdout.write(f"  - {score.category:12s}: {score.confidence * 100:5.1f}% {bar}\n")
        sys.stdout.flush()
        return 0
    except Exception as err:
        log_error(f"Prediction FAILED: {err}")
        return 1
