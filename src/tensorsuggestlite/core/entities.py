"""Domain entities and data structures for TensorSuggestLite."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TrainingSample:
    """Represents a single text example associated with a category."""

    text: str
    category: str


@dataclass
class TrainingDataset:
    """Represents the dataset parsed from configuration files."""

    samples: list[TrainingSample] = field(default_factory=list)

    @property
    def texts(self) -> list[str]:
        return [sample.text for sample in self.samples]

    @property
    def categories(self) -> list[str]:
        return [sample.category for sample in self.samples]

    @property
    def unique_categories(self) -> list[str]:
        return sorted(list({sample.category for sample in self.samples}))

    def __len__(self) -> int:
        return len(self.samples)


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters and settings for model training."""

    epochs: int = 12
    learning_rate: float = 0.001
    dropout_rate: float = 0.2
    test_size: float = 0.1
    embedding_dim: int = 64
    hidden_dim: int = 64
    batch_size: int = 16
    random_state: int = 42
    max_sequence_length: int | None = None


@dataclass
class ArtifactBundle:
    """Paths to artifacts generated during training and export."""

    output_dir: Path
    tokenizer_path: Path
    keras_model_path: Path
    label_encoder_path: Path
    metadata_path: Path | None = None
    tflite_model_path: Path | None = None


@dataclass
class TrainingResult:
    """Summary result returned after training."""

    model_path: str
    items_serialized: int
    labels_count: int
    artifacts: ArtifactBundle
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictionScore:
    """Confidence score for an individual category."""

    category: str
    confidence: float


@dataclass(frozen=True)
class PredictionResult:
    """Inference result returned for input text."""

    text: str
    top_category: str
    top_confidence: float
    scores: list[PredictionScore]
    latency_ms: float
    runtime: str  # 'keras' or 'tflite'


@dataclass(frozen=True)
class CrossValidationResult:
    """Side-by-side comparison of Keras and TFLite inference."""

    text: str
    keras_result: PredictionResult
    tflite_result: PredictionResult
    is_consistent: bool
    delta_confidence: float
    max_class_delta: float


@dataclass(frozen=True)
class DatasetProfileReport:
    """Diagnostic report analyzing dataset balance and integrity."""

    total_samples: int
    unique_texts: int
    category_counts: dict[str, int]
    imbalance_ratio: float
    imbalance_warning: str | None
    duplicate_samples: list[tuple[str, str, int]]
    contradictory_samples: list[tuple[str, list[str]]]

    @property
    def has_critical_issues(self) -> bool:
        return bool(
            self.contradictory_samples or (self.imbalance_ratio > 4.0 and self.total_samples > 10)
        )
