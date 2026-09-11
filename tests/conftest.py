"""Pytest fixtures and test utilities for TensorSuggestLite."""

from pathlib import Path

import pytest

from tensorsuggestlite.core.entities import TrainingDataset, TrainingSample


@pytest.fixture
def examples_dir() -> Path:
    """Return path to examples directory."""
    return Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture
def example_json_path(examples_dir: Path) -> Path:
    """Return path to example.json."""
    return examples_dir / "example.json"


@pytest.fixture
def example_yaml_path(examples_dir: Path) -> Path:
    """Return path to example.yaml."""
    return examples_dir / "example.yaml"


@pytest.fixture
def example_toml_path(examples_dir: Path) -> Path:
    """Return path to example.toml."""
    return examples_dir / "example.toml"


@pytest.fixture
def sample_dataset() -> TrainingDataset:
    """Return a clean, balanced in-memory TrainingDataset."""
    samples = [
        TrainingSample(text="This product works wonderfully and is great", category="POSITIVE"),
        TrainingSample(text="Outstanding quality and lovely design", category="POSITIVE"),
        TrainingSample(text="Highly recommended for everyone", category="POSITIVE"),
        TrainingSample(text="I love this item and will buy again", category="POSITIVE"),
        TrainingSample(text="Completely broken and does not work", category="NEGATIVE"),
        TrainingSample(text="Terrible quality and very disappointing", category="NEGATIVE"),
        TrainingSample(text="Do not buy, absolute waste of money", category="NEGATIVE"),
        TrainingSample(text="Poor experience and defective unit", category="NEGATIVE"),
    ]
    return TrainingDataset(samples=samples)


@pytest.fixture
def mock_callbacks():
    """Mock callbacks for progress and log reporting."""
    progress_records: list[int] = []
    log_records: list[str] = []

    def on_progress(p: int) -> None:
        progress_records.append(p)

    def on_log(msg: str) -> None:
        log_records.append(msg)

    return {
        "progress_fn": on_progress,
        "log_fn": on_log,
        "progress_records": progress_records,
        "log_records": log_records,
    }
