import json
import logging
import os

from datetime import datetime
from pathlib import Path
from typing import Any

from src.summarize_algorithms.core.models import DialogueState, Session


class BaselineLogger:
    def __init__(self, logs_dir="logs/baseline") -> None:
        os.makedirs(logs_dir, exist_ok=True)
        self.log_dir = Path(logs_dir)
        self.logger = logging.getLogger(__name__)

    def log_iteration(
            self,
            system_name: str,
            query: str,
            iteration: int,
            sessions: list[Session]
    ) -> None:
        self.logger.info(f"Logging iteration {iteration} to {self.log_dir}")

        record = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "system": system_name,
            "query": query,
            "sessions": [s.__dict__() for s in sessions],
        }

        with open(self.log_dir / (system_name + str(iteration) + ".jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, indent=4))
            f.write("\n")

        self.logger.info(f"Saved successfully iteration {iteration} to {self.log_dir}")

    @staticmethod
    def _serialize_memories(state: DialogueState) -> dict[str, Any]:
        result = {}
        for name in ["text_memory_storage", "code_memory_storage", "tool_memory_storage"]:
            storage = getattr(state, name, None)
            if storage is not None:
                result[name] = storage.__dict__()

        text_memory = getattr(state, "text_memory", None)
        if text_memory is not None:
            result["text_memory"] = text_memory

        return result
