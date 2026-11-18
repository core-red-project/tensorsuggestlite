"""Use case for parsing configuration datasets and training text classification models."""

from __future__ import annotations

from pathlib import Path

from tensorsuggestlite.core.entities import TrainingConfig, TrainingResult
from tensorsuggestlite.core.interfaces import (
    ConfigParserProtocol,
    LogCallback,
    ModelTrainerProtocol,
    ProgressCallback,
)
from tensorsuggestlite.core.paths import get_default_models_dir
from tensorsuggestlite.ml.trainer import KerasTextTrainer
from tensorsuggestlite.parsers import create_default_registry


class TrainModelUseCase:
    """Orchestrates configuration parsing, data validation, and neural model training."""

    def __init__(
        self,
        trainer: ModelTrainerProtocol | None = None,
        parser: ConfigParserProtocol | None = None,
    ) -> None:
        """Initialize the model training use case.

        Args:
            trainer: Custom trainer conforming to ModelTrainerProtocol.
            parser: Optional explicit parser; if omitted, registry auto-detects by extension.
        """
        self.trainer = trainer or KerasTextTrainer()
        self.parser = parser
        self.registry = create_default_registry()

    def execute(
        self,
        config_path: Path | str,
        output_dir: Path | str | None = None,
        config: TrainingConfig | None = None,
        skeleton: dict[str, str] | None = None,
        progress_cb: ProgressCallback | None = None,
        log_cb: LogCallback | None = None,
    ) -> TrainingResult:
        """Execute the end-to-end model training workflow.

        Args:
            config_path: Path to dataset configuration file.
            output_dir: Directory where trained artifacts should be saved.
            config: Optional hyperparameters configuration.
            skeleton: Optional fixed skeleton mapping for custom schemas.
            progress_cb: Optional callback emitting progress updates [0-100].
            log_cb: Optional callback emitting textual log events.

        Returns:
            TrainingResult: Summary of training containing artifact paths and evaluation metrics.

        Raises:
            ConfigParseError: If dataset parsing fails.
            EmptyDatasetError: If dataset contains no valid training samples.
            ModelTrainingError: If compilation or training fails.
        """
        cfg_path = Path(config_path).resolve()

        # Resolve parser
        parser = self.parser or self.registry.get_parser_for_path(cfg_path)

        if log_cb:
            log_cb(f"Parsing configuration from: {cfg_path.name}")

        dataset = parser.parse(cfg_path, skeleton=skeleton)

        # Default output directory based on XDG standard if not specified
        if output_dir is None:
            out_dir = get_default_models_dir() / cfg_path.stem
        else:
            out_dir = Path(output_dir).resolve()

        return self.trainer.train(
            dataset=dataset,
            output_dir=out_dir,
            config=config,
            progress_cb=progress_cb,
            log_cb=log_cb,
        )


__all__ = ["TrainModelUseCase"]
