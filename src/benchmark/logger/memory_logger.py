import json
import os

from datetime import datetime
from pathlib import Path

from src.benchmark.logger.base_logger import BaseLogger
from src.benchmark.models.dtos import MemoryRecord, MetricState
from src.algorithms.summarize_algorithms.core.models import DialogueState, Session


class MemoryLogger(BaseLogger):
    def log_iteration(
            self,
            system_name: str,
            query: str,
            iteration: int,
            sessions: list[Session],
            state: DialogueState,
            subdirectory: Path,
            metrics: list[MetricState] | None = None,
    ) -> MemoryRecord:
        self.logger.info(f"Logging iteration {iteration} to {self.log_dir}")

        if state is None:
            raise ValueError("'state' argument is necessary for MemoryLogger")

        record = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "system": system_name,
            "query": query,
            "response": getattr(state, "response", None),
            "memory": MemoryLogger._serialize_memories(state),
            "sessions": [s.to_dict() for s in sessions],
            "prepared_messages": [s.model_dump(mode="json") for s in state.prepared_messages],
        }

        if metrics is not None:
            metrics_dict = [
                {"metric_name": metric.metric_name.value, "metric_value": metric.metric_value}
                for metric in metrics
            ]
            record["metric"] = metrics_dict

        if subdirectory is not None:
            directory: Path = self.log_dir / subdirectory
            os.makedirs(directory, exist_ok=True)
        else:
            directory = self.log_dir

        with open(
                directory / (system_name + "-" + str(record["timestamp"]) + ".json"),
                "a",
                encoding="utf-8"
        ) as f:
            f.write(json.dumps(record, ensure_ascii=False, indent=4))
            f.write("\n")

        self.logger.info(f"Saved successfully iteration {iteration} to {self.log_dir}")

        return MemoryRecord.from_dict(record)
