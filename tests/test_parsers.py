"""Tests for configuration parsers in tensorsuggestlite.parsers."""

from pathlib import Path

from tensorsuggestlite.parsers import (
    JsonConfigParser,
    TomlConfigParser,
    YamlConfigParser,
    create_default_registry,
)

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def test_json_parser():
    parser = JsonConfigParser()
    dataset = parser.parse(EXAMPLES_DIR / "example.json")
    assert len(dataset) > 0
    assert "POSITIVE" in dataset.unique_categories
    assert "NEGATIVE" in dataset.unique_categories
    assert "NEUTRAL" in dataset.unique_categories


def test_yaml_parser():
    parser = YamlConfigParser()
    dataset = parser.parse(EXAMPLES_DIR / "example.yaml")
    assert len(dataset) > 0
    assert "POSITIVE" in dataset.unique_categories


def test_toml_parser():
    parser = TomlConfigParser()
    dataset = parser.parse(EXAMPLES_DIR / "example.toml")
    assert len(dataset) > 0
    assert "POSITIVE" in dataset.unique_categories


def test_registry_resolution():
    registry = create_default_registry()
    json_parser = registry.get_parser_for_path("file.json")
    assert isinstance(json_parser, JsonConfigParser)

    yaml_parser = registry.get_parser_for_path("file.yaml")
    assert isinstance(yaml_parser, YamlConfigParser)

    toml_parser = registry.get_parser_for_path("file.toml")
    assert isinstance(toml_parser, TomlConfigParser)
