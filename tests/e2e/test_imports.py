"""Tests verifying module imports in modern src layout."""

import importlib


def test_import_core_modules():
    """Verify core domain entities, interfaces and exceptions import cleanly."""
    entities = importlib.import_module("tensorsuggestlite.core.entities")
    assert hasattr(entities, "TrainingDataset")
    assert hasattr(entities, "TrainingSample")

    interfaces = importlib.import_module("tensorsuggestlite.core.interfaces")
    assert hasattr(interfaces, "ConfigParserProtocol")
    assert hasattr(interfaces, "ModelTrainerProtocol")


def test_import_parsers():
    """Verify config parsers and registry import cleanly."""
    parsers = importlib.import_module("tensorsuggestlite.parsers")
    assert hasattr(parsers, "JsonConfigParser")
    assert hasattr(parsers, "YamlConfigParser")
    assert hasattr(parsers, "TomlConfigParser")
    assert hasattr(parsers, "create_default_registry")


def test_import_ml():
    """Verify ML trainer and exporter import cleanly."""
    ml = importlib.import_module("tensorsuggestlite.ml")
    assert hasattr(ml, "KerasTextTrainer")
    assert hasattr(ml, "TFLiteExporter")


def test_import_use_cases():
    """Verify application use cases import cleanly."""
    use_cases = importlib.import_module("tensorsuggestlite.use_cases")
    assert hasattr(use_cases, "TrainModelUseCase")
    assert hasattr(use_cases, "ExportTFLiteUseCase")


def test_import_ui():
    """Verify GUI modules import cleanly."""
    ui = importlib.import_module("tensorsuggestlite.ui")
    assert hasattr(ui, "TensorSuggestLiteUI")
    assert hasattr(ui, "launch_gui")
    assert hasattr(ui, "NativeDialogBridge")
