"""Parsers package for configuration files."""

from tensorsuggestlite.parsers.base import BaseConfigParser, ParserRegistry
from tensorsuggestlite.parsers.json_parser import JsonConfigParser
from tensorsuggestlite.parsers.toml_parser import TomlConfigParser
from tensorsuggestlite.parsers.yaml_parser import YamlConfigParser


def create_default_registry() -> ParserRegistry:
    """Create a registry with default JSON, YAML, and TOML parsers registered."""
    registry = ParserRegistry()
    registry.register(JsonConfigParser())
    registry.register(YamlConfigParser())
    registry.register(TomlConfigParser())
    return registry


__all__ = [
    "BaseConfigParser",
    "JsonConfigParser",
    "ParserRegistry",
    "TomlConfigParser",
    "YamlConfigParser",
    "create_default_registry",
]
