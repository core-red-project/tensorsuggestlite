"""Domain exceptions for TensorSuggestLite."""


class TensorSuggestLiteError(Exception):
    """Base exception for all domain errors."""


class ConfigParseError(TensorSuggestLiteError):
    """Raised when parsing a configuration file fails."""


class EmptyDatasetError(TensorSuggestLiteError):
    """Raised when no training samples are extracted from configuration."""


class ModelTrainingError(TensorSuggestLiteError):
    """Raised when training fails."""


class ModelExportError(TensorSuggestLiteError):
    """Raised when TFLite conversion or export fails."""


class InferenceError(TensorSuggestLiteError):
    """Raised when model inference fails."""
