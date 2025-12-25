import json
import os

from datetime import datetime
from pathlib import Path

from src.benchmarking.base_logger import BaseLogger
from src.summarize_algorithms.core.models import DialogueState, Session
from src.benchmarking.models.dtos import MetricState, BaseRecord


class BaselineLogger(BaseLogger):
    def log_iteration(
            self,
            system_name: str,
            query: str,
            iteration: int,
            sessions: list[Session],
            state: DialogueState | None = None,
            metrics: list[MetricState] | None = None,
            subdirectory: str | Path | None = None
    ) -> BaseRecord:
        self.logger.info(f"Logging iteration {iteration} to {self.log_dir}")

        record = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "system": system_name,
            "query": query,
            "response": getattr(state, "response", None),
            "sessions": [s.to_dict() for s in sessions],
        }

        if metrics is not None:
            metrics_dict = [
                {"metric_name": metric.metric_name.value, "metric_value": metric.metric_value}
                for metric in metrics
            ]
            record["metric"] = metrics_dict

        if subdirectory is not None:
            directory: Path | str = self.log_dir / subdirectory
            os.makedirs(directory, exist_ok=True)
        else:
            directory: Path | str = self.log_dir

        with open(
                directory / (system_name +'-' + record["timestamp"] + ".json"),
                "a",
                encoding="utf-8"
        ) as f:
            f.write(json.dumps(record, ensure_ascii=False, indent=4))
            f.write("\n")

        self.logger.info(f"Saved successfully iteration {iteration} to {self.log_dir}")

        return BaseRecord.from_dict(record)
