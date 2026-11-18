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


def test_cli_history_with_subdirectories(tmp_path: Path):
    sub = tmp_path / "intent_dataset"
    sub.mkdir()
    meta = {
        "created_at": "2026-11-18T12:00:00Z",
        "dataset": {"total_samples": 50, "sha256": "1234"},
        "hyperparameters": {"epochs": 15},
        "metrics": {"final_accuracy": 0.98, "final_loss": 0.05},
    }
    with open(sub / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    result = run_history(str(tmp_path))
    assert result == 0


def test_cli_gui_missing_dearpygui(monkeypatch):
    monkeypatch.setitem(sys.modules, "dearpygui", None)
    monkeypatch.setitem(sys.modules, "dearpygui.dearpygui", None)

    result = run_gui()
    assert result == 1


def test_cli_version_subcommand(capsys):
    from tensorsuggestlite.main import main

    code = main(["version"])
    assert code == 0
    captured = capsys.readouterr()
    assert "tensorsuggestlite 2.1.0" in captured.out


def test_cli_convert_with_quantization_flag(tmp_path: Path):
    from tensorsuggestlite.main import main

    code = main(["convert", str(tmp_path), "--quantize", "float16", "--json"])
    # Fails gracefully because model file doesn't exist in tmp_path, but argument parsing succeeded
    assert code == 1


def test_cli_predict_batch_file_not_found(tmp_path: Path):
    from tensorsuggestlite.main import main

    code = main(["predict", str(tmp_path), "--file", str(tmp_path / "nonexistent.txt")])
    assert code == 1


def test_cli_inspect_json_mode(example_json_path: Path, capsys):
    from tensorsuggestlite.main import main

    code = main(["inspect", str(example_json_path), "--json"])
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["format"] == "JSON"
    assert data["total_samples"] == 160
    assert "POSITIVE" in data["unique_categories"]


def test_cli_completions(capsys):
    from tensorsuggestlite.main import main

    for shell in ["bash", "zsh", "fish", "powershell"]:
        code = main(["completions", shell])
        assert code == 0
        captured = capsys.readouterr()
        assert len(captured.out) > 20


def test_cli_invalid_command(capsys):
    from tensorsuggestlite.main import main

    code = main(["nonexistent-command-xyz"])
    assert code == 2


def test_cli_did_you_mean_suggestion(capsys):
    from tensorsuggestlite.main import main

    code = main(["trina"])
    assert code == 2
    captured = capsys.readouterr()
    assert "Did you mean 'train'?" in captured.err


def test_cli_help_subcommand(capsys):
    from tensorsuggestlite.main import main

    code = main(["help"])
    assert code == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.out.lower() or "tensorsuggestlite" in captured.out

    code = main(["help", "train"])
    assert code == 0


def test_cli_subcommand_help_flag(capsys):
    """Verify that train --help displays train-specific options rather than root help."""
    from tensorsuggestlite.main import main

    code = main(["train", "--help"])
    assert code == 0
    captured = capsys.readouterr()
    assert "--epochs" in captured.out
    assert "--learning-rate" in captured.out
    assert "--dropout" in captured.out


def test_cli_man_command(capsys):
    from tensorsuggestlite.main import main

    code = main(["man"])
    assert code == 0
    captured = capsys.readouterr()
    assert ".TH TENSORSUGGESTLITE" in captured.out
