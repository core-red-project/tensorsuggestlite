"""Tests verifying configuration file parsing and interpreters structure."""

import json
from pathlib import Path

import toml
import yaml

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def test_example_json_syntax():
    """Verify example.json is valid and contains standard structure."""
    json_path = EXAMPLES_DIR / "example.json"
    assert json_path.exists(), f"Missing {json_path}"
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)
    assert len(data) > 0
    for category, content in data.items():
        assert isinstance(category, str)
        assert "responses" in content or "respuestas" in content


def test_example_yaml_syntax():
    """Verify example.yaml is valid and contains standard structure."""
    yaml_path = EXAMPLES_DIR / "example.yaml"
    assert yaml_path.exists(), f"Missing {yaml_path}"
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict)
    assert len(data) > 0
    for category, content in data.items():
        assert isinstance(category, str)
        assert "responses" in content or "respuestas" in content


def test_example_toml_syntax():
    """Verify example.toml is valid and contains standard structure."""
    toml_path = EXAMPLES_DIR / "example.toml"
    assert toml_path.exists(), f"Missing {toml_path}"
    with open(toml_path, encoding="utf-8") as f:
        data = toml.load(f)
    assert isinstance(data, dict)
    assert len(data) > 0
    for category, content in data.items():
        assert isinstance(category, str)
        assert "responses" in content or "respuestas" in content
