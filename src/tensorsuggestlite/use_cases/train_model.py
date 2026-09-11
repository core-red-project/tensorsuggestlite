"""Use case for parsing configuration and training models."""

from pathlib import Path

from tensorsuggestlite.core.entities import TrainingConfig, TrainingResult
from tensorsuggestlite.core.interfaces import (
    ConfigParserProtocol,
    LogCallback,
    ModelTrainerProtocol,
    ProgressCallback,
)
from tensorsuggestlite.ml.trainer import KerasTextTrainer
from tensorsuggestlite.parsers import create_default_registry


class TrainModelUseCase:
    """Orchestrates configuration parsing, data validation, and model training."""

    def __init__(
        self,
        trainer: ModelTrainerProtocol | None = None,
        parser: ConfigParserProtocol | None = None,
    ) -> None:
        self.trainer = trainer or KerasTextTrainer()
        self.parser = parser
        self.registry = create_default_registry()

    def execute(
        self,
        config_path: Path | str,
        output_dir: Path | str | None = None,
        config: TrainingConfig | None = None,
        progress_cb: ProgressCallback | None = None,
        log_cb: LogCallback | None = None,
    ) -> TrainingResult:
        cfg_path = Path(config_path).resolve()

        # Resolve parser
        parser = self.parser or self.registry.get_parser_for_path(cfg_path)

        if log_cb:
            log_cb(f"Parsing configuration from: {cfg_path.name}")

        dataset = parser.parse(cfg_path)

        # Default output directory based on format kind if not specified
        if output_dir is None:
            kind = cfg_path.suffix.lstrip(".").lower()
            if kind == "yml":
                kind = "yaml"
            out_dir = cfg_path.parent / "generated" / kind
        else:
            out_dir = Path(output_dir).resolve()

        return self.trainer.train(
            dataset=dataset,
            output_dir=out_dir,
            config=config,
            progress_cb=progress_cb,
            log_cb=log_cb,
        )
