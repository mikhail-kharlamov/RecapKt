import hashlib
import logging
import random

from pathlib import Path
from typing import Any

from src.algorithms.dialogue import Dialogue
from src.algorithms.summarize_algorithms.core.models import BaseBlock, Session
from src.benchmark.logger.base_logger import BaseLogger
from src.benchmark.models.dtos import BaseRecord
from src.benchmark.tool_plan_benchmarking.calculator import Calculator
from src.benchmark.tool_plan_benchmarking.evaluators.base_evaluator import BaseEvaluator


class EvaluationLaunchRunner:
    """Runs repeated evaluation launches and returns produced log records."""

    _RUN_ID = "exp_16_02_2026_3_03_am"

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger()

    def run(
        self,
        launch_count: int,
        algorithms: list[Dialogue],
        evaluators: list[BaseEvaluator],
        past_sessions: list[Session],
        session_count: int,
        gold_session: Session,
        prompt: str,
        reference: list[BaseBlock],
        results_logger: BaseLogger,
        subdirectory: Path,
        tools: list[dict[str, Any]] | None = None,
        shuffle: bool = False,
    ) -> list[BaseRecord]:
        records: list[BaseRecord] = []
        for launch_index in range(launch_count):
            prepared_sessions = self._prepare_sessions(
                past_sessions=past_sessions,
                session_count=session_count,
                gold_session=gold_session,
                shuffle=shuffle,
                seed=self._make_seed(launch_count, launch_index),
                # seed=self._make_seed(5, 4),
            )

            self._logger.info("Starting evaluation launch %s", launch_index)
            records.extend(
                Calculator.evaluate(
                    algorithms,
                    evaluators,
                    prepared_sessions,
                    prompt,
                    reference,
                    results_logger,
                    subdirectory,
                    tools,
                    launch_index,
                )
            )

        return records

    @staticmethod
    def _prepare_sessions(
        past_sessions: list[Session],
        session_count: int,
        gold_session: Session,
        shuffle: bool,
        seed: int,
    ) -> list[Session]:
        sessions = past_sessions.copy()
        if len(sessions) > 1 and shuffle:
            random.Random(seed).shuffle(sessions)

        prepared_sessions = sessions[: max(session_count - 1, 0)]
        prepared_sessions.append(gold_session)
        return prepared_sessions

    @classmethod
    def _make_seed(cls, total_launches: int, launch_index: int) -> int:
        digest = hashlib.sha256(
            f"{cls._RUN_ID}:{total_launches}:{launch_index}".encode()
        ).hexdigest()
        return int(digest[:16], 16)
