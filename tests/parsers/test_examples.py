"""Tests verifying configuration file parsing and interpreters structure."""

import json
import tomllib
from pathlib import Path

import yaml

EXAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "examples"


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
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)
    assert isinstance(data, dict)
    assert len(data) > 0
    for category, content in data.items():
        assert isinstance(category, str)
        assert "responses" in content or "respuestas" in content


def test_rich_example_datasets_parsing():
    """Verify newly introduced rich example datasets parse cleanly through ParserRegistry."""
    from tensorsuggestlite.parsers import create_default_registry

    registry = create_default_registry()

    # 1. intent_dataset.json (embedded _skeleton: prompt -> intent)
    p_json = registry.get_parser_for_path(EXAMPLES_DIR / "intent_dataset.json")
    ds_json = p_json.parse(EXAMPLES_DIR / "intent_dataset.json")
    assert len(ds_json) == 140
    assert set(ds_json.unique_categories) == {
        "BILLING",
        "TECHNICAL_ISSUE",
        "ACCOUNT_SECURITY",
        "FEATURE_REQUEST",
    }

    # 2. feedback_dataset.yaml (embedded _skeleton: utterance -> category)
    p_yaml = registry.get_parser_for_path(EXAMPLES_DIR / "feedback_dataset.yaml")
    ds_yaml = p_yaml.parse(EXAMPLES_DIR / "feedback_dataset.yaml")
    assert len(ds_yaml) == 120
    assert set(ds_yaml.unique_categories) == {
        "DELIVERY_SPEED",
        "PRODUCT_QUALITY",
        "CUSTOMER_SERVICE",
        "PRICING_VALUE",
    }

    # 3. ecommerce_feedback.toml (embedded _skeleton: comment -> label)
    p_toml = registry.get_parser_for_path(EXAMPLES_DIR / "ecommerce_feedback.toml")
    ds_toml = p_toml.parse(EXAMPLES_DIR / "ecommerce_feedback.toml")
    assert len(ds_toml) == 90
    assert set(ds_toml.unique_categories) == {
        "CHECKOUT_EXPERIENCE",
        "RETURN_POLICY",
        "INVENTORY_STOCK",
    }
