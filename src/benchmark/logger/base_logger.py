import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.algorithms.summarize_algorithms.core.models import (
    DialogueState,
    MemoryBankDialogueState,
    RecsumDialogueState,
    Session,
)
from src.benchmark.models.dtos import BaseRecord, MetricState
from src.benchmark.utils.json_log_utils import JsonLogUtils


class BaseLogger(ABC):
    def __init__(self, logs_dir: str | Path = "logs/memory") -> None:
        os.makedirs(logs_dir, exist_ok=True)
        self.log_dir = Path(logs_dir)
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def log_iteration(
            self,
            system_name: str,
            query: str,
            iteration: int,
            sessions: list[Session],
            state: DialogueState,
            subdirectory: Path,
            metrics: list[MetricState] | None = None,
            save: bool = True,
    ) -> BaseRecord:
        ...

    @abstractmethod
    def fetch_logs(
            self,
            system_names: list[str],
            subdirectory: Path,
    ) -> list[BaseRecord]:
        ...

    def _prepare_and_save_log(
            self,
            record: dict[str, Any],
            subdirectory: Path,
            system_name: str,
            iteration: int,
            metrics: list[MetricState] | None = None
    ) -> None:
        if metrics is not None:
            record["metric"] = BaseLogger.metrics_to_dicts(metrics)

        if subdirectory is not None:
            directory: Path = self.log_dir / subdirectory
            os.makedirs(directory, exist_ok=True)
        else:
            directory = self.log_dir

        self.save_log_dict(directory, iteration, record, system_name)

    def save_log_dict(
            self,
            directory: Path,
            iteration: int,
            record: dict[str, Any],
            system_name: str
    ) -> None:
        # Use overwrite mode: in "evaluate-by-logs" we update the same file in-place.
        # (Previous append mode could create multiple JSON objects in one file.)
        target = directory / (system_name + "-" + str(record["timestamp"]) + ".json")
        JsonLogUtils.write(target, record)

        self.logger.info(f"Saved successfully iteration {iteration} to {self.log_dir}")

    @staticmethod
    def metrics_to_dicts(metrics: list[MetricState]) -> list[dict[str, Any]]:
        return [
            {"metric_name": metric.metric_name.value, "metric_value": metric.metric_value}
            for metric in metrics
        ]

    @staticmethod
    def _serialize_memories(
            state: DialogueState
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}

        if state.code_memory_storage is not None:
            result["code_memory_storage"] = state.code_memory_storage.to_dict()
        if state.tool_memory_storage is not None:
            result["tool_memory_storage"] = state.tool_memory_storage.to_dict()
        if isinstance(state, MemoryBankDialogueState):
            result["text_memory_storage"] = state.text_memory_storage.to_dict()
        if isinstance(state, RecsumDialogueState):
            result["text_memory"] = state.text_memory

        return result
