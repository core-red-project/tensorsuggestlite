"""Tests for configuration parsers in tensorsuggestlite.parsers."""

from pathlib import Path

from tensorsuggestlite.parsers import (
    JsonConfigParser,
    TomlConfigParser,
    YamlConfigParser,
    create_default_registry,
)

EXAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "examples"


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


def test_direct_list_schema(tmp_path: Path):
    import json

    raw = {
        "GREETING": ["Hello there", "Good morning", "Hi"],
        "FAREWELL": ["Goodbye", "See you later"],
    }
    sample_file = tmp_path / "simple.json"
    with open(sample_file, "w", encoding="utf-8") as f:
        json.dump(raw, f)

    parser = JsonConfigParser()
    dataset = parser.parse(sample_file)
    assert len(dataset) == 5
    assert dataset.unique_categories == ["FAREWELL", "GREETING"]


def test_skeleton_list_in_out(tmp_path: Path):
    import json

    raw = [
        {"in": "hello world", "out": "GREETING"},
        {"in": ["good morning", "hey"], "out": "GREETING"},
        {"in": "see you", "out": "FAREWELL"},
    ]
    f = tmp_path / "skeleton_list.json"
    with open(f, "w", encoding="utf-8") as fp:
        json.dump(raw, fp)

    parser = JsonConfigParser()
    dataset = parser.parse(f)
    assert len(dataset) == 4
    assert set(dataset.unique_categories) == {"GREETING", "FAREWELL"}


def test_skeleton_nested_alt(tmp_path: Path):
    import json

    raw = {
        "GREETING": {"alt": ["hi", "hello there", "greetings"]},
        "FAREWELL": {"alt": ["bye", "see ya"]},
    }
    f = tmp_path / "skeleton_alt.json"
    with open(f, "w", encoding="utf-8") as fp:
        json.dump(raw, fp)

    parser = JsonConfigParser()
    dataset = parser.parse(f)
    assert len(dataset) == 5


def test_skeleton_inverted_dict(tmp_path: Path):
    import json

    raw = {
        "How do I reset my password?": "ACCOUNT_HELP",
        "Where is my package?": "SHIPPING_HELP",
    }
    f = tmp_path / "skeleton_inverted.json"
    with open(f, "w", encoding="utf-8") as fp:
        json.dump(raw, fp)

    parser = JsonConfigParser()
    dataset = parser.parse(f)
    assert len(dataset) == 2
    assert "ACCOUNT_HELP" in dataset.unique_categories


def test_embedded_skeleton_schema_and_parser_str(tmp_path: Path):
    import json

    from tensorsuggestlite.parsers.base import parse_skeleton_str

    # Test shorthand parsing
    assert parse_skeleton_str("in:out") == {"in": "in", "out": "out"}
    assert parse_skeleton_str("prompt:intent") == {"in": "prompt", "out": "intent"}
    assert parse_skeleton_str('{"in": "prompt", "out": "intent"}') == {
        "in": "prompt",
        "out": "intent",
    }
    assert parse_skeleton_str(None) is None
    assert parse_skeleton_str("") is None

    # Test dataset with embedded _skeleton
    raw = {
        "_skeleton": {"in": "query", "out": "intent"},
        "BILLING": ["Where is my invoice?", "Charged twice"],
        "SHIPPING": ["When will it arrive?"],
    }
    f = tmp_path / "embedded_skeleton.json"
    with open(f, "w", encoding="utf-8") as fp:
        json.dump(raw, fp)

    parser = JsonConfigParser()
    dataset = parser.parse(f)
    assert len(dataset) == 3
    assert set(dataset.unique_categories) == {"BILLING", "SHIPPING"}
    assert "_skeleton" not in dataset.unique_categories


def test_numeric_and_boolean_keys(tmp_path: Path):
    """Verify that YAML/dict with numeric or boolean keys parses without AttributeError."""
    yaml_content = """
    1:
      - "hello"
      - "hi"
    2:
      - "bye"
      - "goodbye"
    """
    f = tmp_path / "numeric_keys.yaml"
    f.write_text(yaml_content, encoding="utf-8")

    parser = YamlConfigParser()
    dataset = parser.parse(f)
    assert len(dataset) == 4
    assert "1" in dataset.unique_categories
    assert "2" in dataset.unique_categories
