"""Unit tests for business logic use cases."""

from pathlib import Path

import pytest

from tensorsuggestlite.core.exceptions import ModelTrainingError
from tensorsuggestlite.use_cases import ExportTFLiteUseCase, PredictTextUseCase, TrainModelUseCase


def test_use_cases_instantiation():
    """Verify that use cases initialize with expected default dependencies."""
    train_uc = TrainModelUseCase()
    assert train_uc.trainer is not None
    assert train_uc.registry is not None

    export_uc = ExportTFLiteUseCase()
    assert export_uc.exporter is not None


def test_predict_use_case_missing_model_handling(tmp_path: Path):
    """Verify PredictTextUseCase raises ModelTrainingError when model dir is empty."""
    use_case = PredictTextUseCase(tmp_path)
    with pytest.raises(ModelTrainingError, match="Keras model not found"):
        use_case.predict("test utterance")
