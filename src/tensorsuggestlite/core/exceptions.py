"""Domain exceptions for TensorSuggestLite.

Establishes a unified exception hierarchy rooted at TensorSuggestLiteError,
enabling callers to isolate domain-specific failures from standard Python runtime errors.
"""

from __future__ import annotations


class TensorSuggestLiteError(Exception):
    """Base exception class for all domain errors within TensorSuggestLite."""


class ConfigParseError(TensorSuggestLiteError):
    """Raised when an input configuration dataset cannot be parsed or decoded."""


class EmptyDatasetError(TensorSuggestLiteError):
    """Raised when an ingested dataset contains zero valid training samples."""


class InvalidDatasetError(TensorSuggestLiteError):
    """Raised when an ingested dataset violates structural or cardinality constraints (e.g. < 2 classes)."""


class ModelTrainingError(TensorSuggestLiteError):
    """Raised when model compilation, gradient fitting, or artifact persistence fails."""


class ModelExportError(TensorSuggestLiteError):
    """Raised when TFLite conversion, quantization, or binary serialization fails."""


class InferenceError(TensorSuggestLiteError):
    """Raised when text classification inference or parity evaluation encounters an error."""


__all__ = [
    "ConfigParseError",
    "EmptyDatasetError",
    "InferenceError",
    "InvalidDatasetError",
    "ModelExportError",
    "ModelTrainingError",
    "TensorSuggestLiteError",
]
