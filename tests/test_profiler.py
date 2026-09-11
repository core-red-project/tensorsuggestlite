"""Tests for DatasetProfiler: imbalance, duplicates, and contradictions."""

from tensorsuggestlite.core.entities import TrainingDataset, TrainingSample
from tensorsuggestlite.core.profiler import DatasetProfiler


def test_profiler_clean_dataset() -> None:
    dataset = TrainingDataset(
        samples=[
            TrainingSample("Good morning", "POSITIVE"),
            TrainingSample("Have a nice day", "POSITIVE"),
            TrainingSample("I am very sad", "NEGATIVE"),
            TrainingSample("This is terrible", "NEGATIVE"),
        ]
    )
    report = DatasetProfiler.profile(dataset)
    assert report.total_samples == 4
    assert report.unique_texts == 4
    assert report.imbalance_ratio == 1.0
    assert report.duplicate_samples == []
    assert report.contradictory_samples == []
    assert not report.has_critical_issues


def test_profiler_detects_imbalance() -> None:
    samples = [TrainingSample(f"Positive text {i}", "POSITIVE") for i in range(15)]
    samples.append(TrainingSample("Negative text 1", "NEGATIVE"))
    dataset = TrainingDataset(samples=samples)

    report = DatasetProfiler.profile(dataset)
    assert report.imbalance_ratio == 15.0
    assert report.imbalance_warning is not None
    assert "Critical class imbalance" in report.imbalance_warning
    assert report.has_critical_issues


def test_profiler_detects_duplicates() -> None:
    dataset = TrainingDataset(
        samples=[
            TrainingSample("Duplicate sentence", "POSITIVE"),
            TrainingSample("Duplicate sentence", "POSITIVE"),
            TrainingSample("Unique sentence", "NEGATIVE"),
        ]
    )
    report = DatasetProfiler.profile(dataset)
    assert len(report.duplicate_samples) == 1
    text, cat, count = report.duplicate_samples[0]
    assert text == "Duplicate sentence"
    assert cat == "POSITIVE"
    assert count == 2


def test_profiler_detects_contradictions() -> None:
    dataset = TrainingDataset(
        samples=[
            TrainingSample("This is awesome", "POSITIVE"),
            TrainingSample("This is awesome", "NEGATIVE"),
            TrainingSample("Normal text", "NEUTRAL"),
        ]
    )
    report = DatasetProfiler.profile(dataset)
    assert len(report.contradictory_samples) == 1
    text, cats = report.contradictory_samples[0]
    assert text == "this is awesome"
    assert "NEGATIVE" in cats
    assert "POSITIVE" in cats
    assert report.has_critical_issues
