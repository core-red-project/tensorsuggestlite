"""Base parser architecture and registry for dataset configuration files.

Provides extensible base class BaseConfigParser with ultra-flexible Skeleton parsing
(custom keywords, inverted mappings, in/out/alt structures, and nested lists) and
ParserRegistry adhering to Open/Closed Principle.
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from tensorsuggestlite.core.entities import TrainingDataset, TrainingSample
from tensorsuggestlite.core.exceptions import ConfigParseError, EmptyDatasetError
from tensorsuggestlite.core.interfaces import ConfigParserProtocol

# Default keyword sets for flexible dataset skeleton mapping
DEFAULT_INPUT_KEYWORDS: frozenset[str] = frozenset(
    {
        "in",
        "input",
        "text",
        "prompt",
        "sentence",
        "query",
        "x",
        "src",
        "source",
        "phrase",
        "alt",
        "alts",
        "samples",
        "utterance",
        "utterances",
        "examples",
        "comment",
        "comments",
        "feedback",
    }
)

DEFAULT_OUTPUT_KEYWORDS: frozenset[str] = frozenset(
    {
        "out",
        "output",
        "category",
        "label",
        "class",
        "target",
        "y",
        "intent",
        "tag",
        "destination",
        "name",
    }
)


class BaseConfigParser(ABC, ConfigParserProtocol):
    """Abstract base parser standardizing validation, parsing, and flexible sample extraction."""

    @property
    @abstractmethod
    def supported_extensions(self) -> tuple[str, ...]:
        """Tuple of supported file extensions (e.g. ('.json',))."""
        pass

    @abstractmethod
    def load_raw_data(self, file_path: Path) -> Any:
        """Load and parse the file into a Python data structure (dict or list).

        Args:
            file_path: Filesystem path to the target file.

        Returns:
            Any: Parsed raw dictionary or list structure.
        """
        pass

    def parse(
        self,
        file_path: Path | str,
        skeleton: dict[str, str] | None = None,
    ) -> TrainingDataset:
        """Parse configuration file into a validated TrainingDataset.

        Args:
            file_path: Filesystem path to dataset.
            skeleton: Optional mapping specifying custom input/output keys (e.g. {'in': 'out'}).

        Returns:
            TrainingDataset: Dataset containing extracted classification samples.

        Raises:
            ConfigParseError: If file is missing, inaccessible, or has an invalid structure.
            EmptyDatasetError: If no valid classification samples are extracted.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise ConfigParseError(f"Configuration file not found: {path}")
        if not path.is_file():
            raise ConfigParseError(f"Target path is not a regular file: {path}")

        try:
            data = self.load_raw_data(path)
        except Exception as err:
            raise ConfigParseError(f"Failed to parse {path.name}: {err}") from err

        if not isinstance(data, (dict, list)):
            raise ConfigParseError(f"Root structure must be a dictionary or list in {path.name}")

        samples = self.extract_samples(data, skeleton=skeleton)
        if not samples:
            raise EmptyDatasetError(f"No training samples found in {path.name}")

        return TrainingDataset(samples=samples)

    @staticmethod
    def extract_samples(
        data: Any,
        skeleton: dict[str, str] | None = None,
    ) -> list[TrainingSample]:
        """Extract classification samples supporting direct, nested, inverted, and skeleton schemas.

        Supported formats include:
        1. Categorical dictionary: {"GREETING": ["hello", "hi"]}
        2. Nested structures: {"GREETING": {"alt": ["hello", "hi"], "responses": [...]}}
        3. Inverted lookup map: {"hello world": "GREETING", "goodbye": "FAREWELL"}
        4. List of objects with skeleton keys: [{"in": "hello", "out": "GREETING"}]
        5. Top-level container dictionary: {"data": [...]} or {"in": ["a", "b"], "out": "CAT"}

        Args:
            data: Parsed structure (dict or list).
            skeleton: Optional custom keyword mapping (e.g. {"input": "in", "output": "out"}).

        Returns:
            list[TrainingSample]: Extracted and sanitized training samples.
        """
        samples: list[TrainingSample] = []

        in_keys = set(DEFAULT_INPUT_KEYWORDS)
        out_keys = set(DEFAULT_OUTPUT_KEYWORDS)

        # 1. Check for embedded skeleton in dictionary
        if isinstance(data, dict):
            declared = data.get("_skeleton") or data.get("skeleton")
            if isinstance(declared, dict):
                effective_skel = dict(skeleton or {})
                effective_skel.update({str(k): str(v) for k, v in declared.items()})
                skeleton = effective_skel

        if skeleton:
            # Map canonical "in", "out", "alt" aliases if present
            for k, v in skeleton.items():
                k_lower = str(k).lower()
                v_lower = str(v).lower()
                if k_lower in {
                    "in",
                    "input",
                    "x",
                    "src",
                    "source",
                    "prompt",
                    "sentence",
                    "text",
                    "alt",
                }:
                    in_keys.add(v_lower)
                elif k_lower in {
                    "out",
                    "output",
                    "y",
                    "target",
                    "category",
                    "label",
                    "class",
                    "intent",
                }:
                    out_keys.add(v_lower)
                else:
                    in_keys.add(k_lower)
                    out_keys.add(v_lower)

        # Handle top-level list: [{"in": "...", "out": "..."}, ...]
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    samples.extend(
                        BaseConfigParser._extract_from_item_dict(item, in_keys, out_keys)
                    )
            return samples

        if not isinstance(data, dict):
            return samples

        # Filter out meta-configuration keys from category processing
        meta_keys = {"_skeleton", "skeleton"}

        # Check for top-level envelope containers (e.g. {"data": [...]}, {"samples": [...]})
        for container_key in ("data", "dataset", "samples", "items", "rows", "intents"):
            if container_key in data and isinstance(data[container_key], list):
                sub_samples = BaseConfigParser.extract_samples(
                    data[container_key], skeleton=skeleton
                )
                if sub_samples:
                    return sub_samples

        # Check if top-level dict itself is an in/out skeleton: {"in": ["hi", "hey"], "out": "GREETING"}
        top_in = next((k for k in data if str(k).lower() in in_keys), None)
        top_out = next((k for k in data if str(k).lower() in out_keys), None)
        if top_in and top_out and len(data) <= 3:
            return BaseConfigParser._extract_pair(data[top_in], data[top_out])

        # Traverse key-value mappings
        for key, content in data.items():
            key_str = str(key).strip()
            if not key_str or key_str.lower() in meta_keys:
                continue

            # Case A: {"CATEGORY": ["sentence 1", "sentence 2"]}
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, str) and item.strip():
                        samples.append(TrainingSample(text=item.strip(), category=key_str))
                    elif isinstance(item, dict):
                        # Item could be {"in": "...", "out": "..."} or {"response": "..."}
                        sub_samples = BaseConfigParser._extract_from_item_dict(
                            item, in_keys, out_keys
                        )
                        if sub_samples:
                            samples.extend(sub_samples)
                        else:
                            txt = item.get("response") or item.get("respuesta") or item.get("text")
                            if txt and isinstance(txt, str) and txt.strip():
                                samples.append(TrainingSample(text=txt.strip(), category=key_str))
                continue

            # Case B: Inverted dictionary {"sentence 1": "CATEGORY"}
            if isinstance(content, str) and content.strip():
                samples.append(TrainingSample(text=key_str, category=content.strip()))
                continue

            # Case C: Nested dictionary {"CATEGORY": {"responses": [...], "alt": [...]}}
            if isinstance(content, dict):
                found_nested = False

                # Scan for nested input lists: alt, in, responses, synonyms, phrases
                for sub_k, sub_v in content.items():
                    sub_lower = str(sub_k).lower()
                    if (
                        sub_lower in in_keys
                        or sub_lower in {"responses", "respuestas", "synonyms", "sinonimos"}
                    ) and isinstance(sub_v, list):
                        for item in sub_v:
                            if isinstance(item, str) and item.strip():
                                samples.append(TrainingSample(text=item.strip(), category=key_str))
                                found_nested = True
                            elif isinstance(item, dict):
                                txt = (
                                    item.get("response")
                                    or item.get("respuesta")
                                    or item.get("text")
                                    or item.get("in")
                                )
                                if txt and str(txt).strip():
                                    samples.append(
                                        TrainingSample(text=str(txt).strip(), category=key_str)
                                    )
                                    found_nested = True

                # If no known list was found, check if content is an inner {"in": ..., "out": ...}
                if not found_nested:
                    inner_samples = BaseConfigParser._extract_from_item_dict(
                        content, in_keys, out_keys
                    )
                    if inner_samples:
                        samples.extend(inner_samples)

        return samples

    @staticmethod
    def _extract_from_item_dict(
        item: dict[str, Any], in_keys: set[str], out_keys: set[str]
    ) -> list[TrainingSample]:
        """Extract samples from an object containing input and output keys."""
        in_key = next((k for k in item if str(k).lower() in in_keys), None)
        out_key = next((k for k in item if str(k).lower() in out_keys), None)

        if not in_key or not out_key:
            return []

        return BaseConfigParser._extract_pair(item[in_key], item[out_key])

    @staticmethod
    def _extract_pair(input_val: Any, output_val: Any) -> list[TrainingSample]:
        """Convert input value(s) and output category(ies) into TrainingSample list."""
        results: list[TrainingSample] = []

        # Normalize inputs to a list of non-empty strings
        inputs = [input_val] if isinstance(input_val, (str, int, float)) else list(input_val or [])
        clean_inputs = [str(x).strip() for x in inputs if str(x).strip()]

        # Normalize categories to a list of non-empty strings
        cats = [output_val] if isinstance(output_val, (str, int, float)) else list(output_val or [])
        clean_cats = [str(c).strip() for c in cats if str(c).strip()]

        for c in clean_cats:
            for inp in clean_inputs:
                results.append(TrainingSample(text=inp, category=c))

        return results


class ParserRegistry:
    """Registry maintaining associations between file extensions and parser protocols."""

    def __init__(self) -> None:
        """Initialize an empty parser registry."""
        self._parsers: dict[str, ConfigParserProtocol] = {}

    def register(self, parser: ConfigParserProtocol) -> None:
        """Register a parser instance for each of its supported extensions.

        Args:
            parser: Parser instance adhering to ConfigParserProtocol.
        """
        for ext in parser.supported_extensions:
            self._parsers[ext.lower()] = parser

    def get_parser_for_path(self, file_path: Path | str) -> ConfigParserProtocol:
        """Retrieve parser matching the file extension.

        Args:
            file_path: Filesystem path to evaluate.

        Returns:
            ConfigParserProtocol: Registered parser matching the extension.

        Raises:
            ConfigParseError: If no registered parser handles the extension.
        """
        path = Path(file_path)
        ext = path.suffix.lower()
        if ext not in self._parsers:
            supported = ", ".join(sorted(self._parsers.keys()))
            raise ConfigParseError(
                f"Unsupported file format '{ext}'. Supported formats: {supported}"
            )
        return self._parsers[ext]


def parse_skeleton_str(skeleton_str: str | None) -> dict[str, str] | None:
    """Parse skeleton specification from a JSON string or shorthand 'in:out'.

    Examples:
        >>> parse_skeleton_str('{"in": "prompt", "out": "intent"}')
        {'in': 'prompt', 'out': 'intent'}
        >>> parse_skeleton_str('prompt:intent')
        {'in': 'prompt', 'out': 'intent'}

    Args:
        skeleton_str: Input specification string or None.

    Returns:
        dict[str, str] | None: Parsed key mapping dictionary or None if invalid/empty.
    """
    if not skeleton_str or not skeleton_str.strip():
        return None
    s = skeleton_str.strip()
    if s.startswith("{") and s.endswith("}"):
        import json

        with contextlib.suppress(Exception):
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                return {str(k): str(v) for k, v in parsed.items()}
    if ":" in s:
        parts = [p.strip() for p in s.split(":", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            return {"in": parts[0], "out": parts[1]}
    return None


__all__ = [
    "DEFAULT_INPUT_KEYWORDS",
    "DEFAULT_OUTPUT_KEYWORDS",
    "BaseConfigParser",
    "ParserRegistry",
    "parse_skeleton_str",
]
