import json
import sys
from pathlib import Path

from tensorsuggestlite.main import run_gui, run_history, run_inspect, run_predict


def test_cli_inspect_valid_json(example_json_path: Path):
    result = run_inspect(str(example_json_path))
    assert result == 0


def test_cli_inspect_valid_yaml(example_yaml_path: Path):
    result = run_inspect(str(example_yaml_path))
    assert result == 0


def test_cli_inspect_valid_toml(example_toml_path: Path):
    result = run_inspect(str(example_toml_path))
    assert result == 0


def test_cli_inspect_nonexistent_file(tmp_path: Path):
    result = run_inspect(str(tmp_path / "nonexistent.json"))
    assert result == 1


def test_cli_predict_nonexistent_dir(tmp_path: Path):
    result = run_predict(str(tmp_path), "sample text")
    assert result == 1


def test_cli_history_empty_dir(tmp_path: Path):
    result = run_history(str(tmp_path))
    assert result == 1


def test_cli_history_with_metadata(tmp_path: Path):
    meta = {
        "created_at": "2026-09-11T12:00:00Z",
        "dataset": {"total_samples": 40, "sha256": "abcdef1234567890"},
        "hyperparameters": {"epochs": 10},
        "metrics": {"final_accuracy": 0.95, "final_loss": 0.12},
    }
    with open(tmp_path / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    result = run_history(str(tmp_path))
    assert result == 0


def test_cli_gui_missing_pyqt6(monkeypatch):
    monkeypatch.setitem(sys.modules, "PyQt6", None)
    monkeypatch.setitem(sys.modules, "PyQt6.QtWidgets", None)

    result = run_gui()
    assert result == 1
