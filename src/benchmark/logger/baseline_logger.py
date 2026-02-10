from datetime import datetime
from pathlib import Path

from typing_extensions import override

from src.algorithms.summarize_algorithms.core.models import DialogueState, Session
from src.benchmark.logger.base_logger import BaseLogger
from src.benchmark.models.dtos import BaseRecord, MetricState
from src.benchmark.utils.json_log_utils import JsonLogUtils


class BaselineLogger(BaseLogger):
    @override
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
        self.logger.info(f"Logging iteration {iteration} to {self.log_dir}")

        record = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "system": system_name,
            "query": query,
            "response": getattr(state, "response", None),
            "sessions": [s.to_dict() for s in sessions],
            "prepared_messages": [s.model_dump(mode="json") for s in state.prepared_messages],
        }

        if save:
            self._prepare_and_save_log(record, subdirectory, system_name, iteration, metrics)

        return BaseRecord.from_dict(record)

    @override
    def fetch_logs(
        self,
        system_names: list[str],
        subdirectory: Path,
    ) -> list[BaseRecord]:
        """Load saved benchmark log records.

        Logs are expected under:
        `<self.log_dir>/<system_name>/<subdirectory>/*.json`.

        Returns records parsed via `BaseRecord.from_dict()`.
        """
        records: list[BaseRecord] = []
        for system_name in system_names:
            directory = self.log_dir / system_name / subdirectory
            if not directory.exists():
                continue

            for path in sorted(directory.glob("*.json")):
                # Logs are written with `indent=4`, so each record is multi-line JSON.
                # Also keep compatibility with multiple objects appended to the same file.
                for payload in JsonLogUtils.load_log_payloads(path):
                    records.append(BaseRecord.from_dict(payload))

        return records
