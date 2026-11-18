"""Domain entities and structured data models for TensorSuggestLite.

Provides immutable and typed value objects representing training datasets,
hyperparameter configurations, evaluation metrics, and inference results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TrainingSample:
    """Represents an individual text utterance paired with a target classification label.

    Attributes:
        text: Normalized raw sentence or phrase.
        category: Ground-truth target class or intent name.
    """

    text: str
    category: str


@dataclass(slots=True)
class TrainingDataset:
    """Encapsulates a collection of training samples parsed from configuration sources.

    Provides utility properties for label distribution extraction, text sequence
    access, and class cardinality.

    Attributes:
        samples: List of TrainingSample instances comprising the dataset.
    """

    samples: list[TrainingSample] = field(default_factory=list)

    @property
    def texts(self) -> list[str]:
        """Extract a flat list of all sample text utterances."""
        return [sample.text for sample in self.samples]

    @property
    def categories(self) -> list[str]:
        """Extract a flat list of all sample categories in order."""
        return [sample.category for sample in self.samples]

    @property
    def unique_categories(self) -> list[str]:
        """Extract sorted list of unique category labels present in the dataset."""
        return sorted({sample.category for sample in self.samples})

    def __len__(self) -> int:
        """Return the total number of training samples."""
        return len(self.samples)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Hyperparameters and runtime parameters governing neural model training.

    Attributes:
        epochs: Number of complete training passes over the dataset.
        learning_rate: Initial learning rate for the Adam optimizer.
        dropout_rate: Dropout regularization probability applied after dense layers.
        test_size: Proportion of dataset reserved for validation split.
        embedding_dim: Dimensionality of token embedding vectors.
        hidden_dim: Number of units in intermediate dense layers.
        batch_size: Mini-batch sample size during gradient updates.
        random_state: Deterministic seed for reproducible train/test splits.
        max_sequence_length: Optional sequence truncation/padding bound.
    """

    epochs: int = 12
    learning_rate: float = 0.001
    dropout_rate: float = 0.2
    test_size: float = 0.1
    embedding_dim: int = 64
    hidden_dim: int = 64
    batch_size: int = 16
    random_state: int = 42
    max_sequence_length: int | None = None

    def __post_init__(self) -> None:
        """Validate hyperparameter constraints."""
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if not (0.0 <= self.dropout_rate < 1.0):
            raise ValueError(f"dropout_rate must be in [0.0, 1.0), got {self.dropout_rate}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0.0, got {self.learning_rate}")
        if not (0.0 <= self.test_size < 1.0):
            raise ValueError(f"test_size must be in [0.0, 1.0), got {self.test_size}")


@dataclass(slots=True)
class ArtifactBundle:
    """Container for filesystem paths to all artifacts emitted during training.

    Attributes:
        output_dir: Base directory housing all run artifacts.
        tokenizer_path: Path to serialized JSON tokenizer config.
        keras_model_path: Path to trained Keras .keras model file.
        label_encoder_path: Path to serialized label index mapping JSON.
        metadata_path: Optional path to model_metadata.json summary.
        tflite_model_path: Optional path to exported .tflite binary.
    """

    output_dir: Path
    tokenizer_path: Path
    keras_model_path: Path
    label_encoder_path: Path
    metadata_path: Path | None = None
    tflite_model_path: Path | None = None


@dataclass(slots=True)
class TrainingResult:
    """Structured report returned upon successful training completion.

    Attributes:
        model_path: Stringified path to the saved Keras model file.
        items_serialized: Total number of training samples ingested.
        labels_count: Number of unique output classes.
        artifacts: ArtifactBundle referencing all generated files.
        metrics: Dictionary mapping metric names to final evaluation values.
    """

    model_path: str
    items_serialized: int
    labels_count: int
    artifacts: ArtifactBundle
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PredictionScore:
    """Individual class prediction confidence.

    Attributes:
        category: Predicted category label.
        confidence: Posterior probability score between 0.0 and 1.0.
    """

    category: str
    confidence: float


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """Comprehensive classification result produced by a model inference engine.

    Attributes:
        text: Raw input text evaluated.
        top_category: Category possessing the highest confidence score.
        top_confidence: Numerical probability of the winning class.
        scores: Complete list of confidence scores across all classes.
        latency_ms: Inference execution time in milliseconds.
        runtime: Execution runtime identifier ('keras' or 'tflite').
    """

    text: str
    top_category: str
    top_confidence: float
    scores: list[PredictionScore]
    latency_ms: float
    runtime: str


@dataclass(frozen=True, slots=True)
class CrossValidationResult:
    """Parity verification report evaluating Keras versus TFLite inference.

    Attributes:
        text: Test sentence evaluated on both runtimes.
        keras_result: Inference result produced by the Keras model.
        tflite_result: Inference result produced by the quantized TFLite binary.
        is_consistent: True if both runtimes assign the exact same winning class.
        delta_confidence: Absolute discrepancy between winning class confidences.
        max_class_delta: Maximum absolute divergence across all class probabilities.
    """

    text: str
    keras_result: PredictionResult
    tflite_result: PredictionResult
    is_consistent: bool
    delta_confidence: float
    max_class_delta: float


@dataclass(frozen=True, slots=True)
class DatasetProfileReport:
    """Diagnostic analysis profiling dataset quality, class balance, and anomalies.

    Attributes:
        total_samples: Gross sample count.
        unique_texts: Count of deduplicated utterances.
        category_counts: Mapping from category name to sample frequency.
        imbalance_ratio: Ratio of largest class sample size to smallest.
        imbalance_warning: Textual warning if class imbalance exceeds thresholds.
        duplicate_samples: List of duplicate utterances with frequency count.
        contradictory_samples: Utterances tagged with conflicting class labels.
    """

    total_samples: int
    unique_texts: int
    category_counts: dict[str, int]
    imbalance_ratio: float
    imbalance_warning: str | None
    duplicate_samples: list[tuple[str, str, int]]
    contradictory_samples: list[tuple[str, list[str]]]

    @property
    def has_critical_issues(self) -> bool:
        """Indicate whether the dataset contains blocking anomalies."""
        return bool(
            (self.total_samples > 0 and len(self.category_counts) < 2)
            or self.contradictory_samples
            or (self.imbalance_ratio > 4.0 and self.total_samples > 10)
        )
