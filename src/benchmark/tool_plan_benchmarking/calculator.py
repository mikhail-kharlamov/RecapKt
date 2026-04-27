import logging

from pathlib import Path
from typing import Any

from src.algorithms.dialogue import Dialogue
from src.algorithms.summarize_algorithms.core.models import (
    BaseBlock,
    CodeBlock,
    DialogueState,
    Session,
    ToolCallBlock,
)
from src.benchmark.logger.base_logger import BaseLogger
from src.benchmark.models.dtos import BaseRecord, MemoryRecord, MetricState
from src.benchmark.tool_plan_benchmarking.evaluators.base_evaluator import BaseEvaluator
from src.benchmark.tool_plan_benchmarking.tools_and_schemas.parsed_jsons import (
    PLAN_SCHEMA,
)


class Calculator:
    """Run algorithms, evaluate results, and persist/re-evaluate benchmark logs."""

    _LAST_LAUNCH_TIME_WINDOW_SECONDS: int = 120

    @staticmethod
    def evaluate(
        algorithms: list[Dialogue],
        evaluator_functions: list[BaseEvaluator],
        sessions: list[Session],
        prompt: str,
        reference: list[BaseBlock],
        logger: BaseLogger,
        subdirectory: Path,
        tools: list[dict[str, Any]] | None = None,
        iteration: int | None = None,
    ) -> list[BaseRecord]:
        """Run and evaluate an algorithm, then save results via `logger`."""
        system_logger = logging.getLogger()
        metrics: list[BaseRecord] = []

        for algorithm in algorithms:
            system_logger.info(f"Calculating {algorithm.system_name}")
            state: DialogueState = algorithm.process_dialogue(
                sessions, prompt, PLAN_SCHEMA, tools
            )

            algorithm_metrics: list[MetricState] = Calculator.__evaluate_result(
                evaluator_functions,
                sessions,
                prompt,
                state,
                reference,
            )

            record: BaseRecord = logger.log_iteration(
                algorithm.system_name,
                prompt,
                iteration or 1,
                sessions,
                state,
                Path(algorithm.system_name) / subdirectory,
                algorithm_metrics,
            )

            metrics.append(record)

        return metrics

    @staticmethod
    def evaluate_by_logs(
        algorithms: list[Dialogue],
        evaluator_functions: list[BaseEvaluator],
        reference: list[BaseBlock],
        logger: BaseLogger,
        logs_path: Path | str,
        subdirectory: Path,
        iteration: int | None = None,
        recalculate_old_metrics: bool = True,
    ) -> list[BaseRecord]:
        system_logger = logging.getLogger()
        updated_records: list[BaseRecord] = []

        for algorithm in algorithms:
            old_logs: list[BaseRecord] = logger.fetch_logs(
                system_names=[alg.system_name for alg in algorithms],
                subdirectory=subdirectory,
            )

            system_logger.info(f"Calculating {algorithm.system_name} by logs")

            for log in old_logs:
                state = DialogueState(
                    dialogue_sessions=[],
                    prepared_messages=[],
                    code_memory_storage=None,
                    tool_memory_storage=None,
                    query=log.query,
                    _response=log.response,
                )

                sessions: list[Session] = Calculator._deserialize_sessions(log.sessions)

                algorithm_metrics = Calculator.__evaluate_result(
                    evaluator_functions=evaluator_functions,
                    sessions=sessions,
                    prompt=log.query,
                    state=state,
                    reference=reference,
                )

                if log.metric is None:
                    log.metric = []

                existing_metric_names = {m.metric_name.value for m in log.metric}

                for metric_state in algorithm_metrics:
                    metric_key = metric_state.metric_name.value

                    if (
                        not recalculate_old_metrics
                    ) and metric_key in existing_metric_names:
                        continue

                    log.metric = [
                        m for m in log.metric if m.metric_name.value != metric_key
                    ]
                    log.metric.append(metric_state)
                    existing_metric_names.add(metric_key)

                logger.save_log_dict(
                    Path(logs_path) / Path(algorithm.system_name) / subdirectory,
                    iteration or 1,
                    log.to_dict(),
                    algorithm.system_name,
                )

                updated_records.append(log)

        return updated_records

    @staticmethod
    def _record_from_dict(record_dict: dict[str, Any]) -> BaseRecord:
        if record_dict.get("memory") is None:
            return BaseRecord.from_dict(record_dict)
        return MemoryRecord.from_dict(record_dict)

    @staticmethod
    def _deserialize_sessions(raw_sessions: Any) -> list[Session]:
        if not isinstance(raw_sessions, list):
            return []

        sessions: list[Session] = []
        for raw in raw_sessions:
            if not isinstance(raw, dict):
                continue
            sessions.append(Calculator._deserialize_session(raw))

        return sessions

    @staticmethod
    def _deserialize_session(raw_session: dict[str, Any]) -> Session:
        messages: list[BaseBlock] = []

        for msg in raw_session.get("messages", []):
            if not isinstance(msg, dict):
                continue

            msg_type = msg.get("type")
            if msg_type == "text":
                messages.append(
                    BaseBlock(
                        role=str(msg.get("role", "")),
                        content=str(msg.get("content", "")),
                    )
                )
            elif msg_type == "code":
                code = str(msg.get("code", ""))
                messages.append(
                    CodeBlock(role=str(msg.get("role", "")), content=code, code=code)
                )
            elif msg_type == "tool_call":
                messages.append(
                    ToolCallBlock(
                        role="TOOL_RESPONSE",
                        content="",
                        id=str(msg.get("id", "")),
                        name=str(msg.get("name", "")),
                        arguments=str(msg.get("arguments", "")),
                        response=str(msg.get("response", "")),
                    )
                )
            else:
                messages.append(
                    BaseBlock(
                        role=str(msg.get("role", "")),
                        content=str(msg.get("content", "")),
                    )
                )

        return Session(messages)

    @staticmethod
    def __evaluate_result(
        evaluator_functions: list[BaseEvaluator],
        sessions: list[Session],
        prompt: str,
        state: DialogueState,
        reference: list[BaseBlock],
    ) -> list[MetricState]:
        algorithm_metrics: list[MetricState] = []
        for evaluator_function in evaluator_functions:
            metric = evaluator_function.evaluate(sessions, prompt, state, reference)
            algorithm_metrics.append(metric)

        return algorithm_metrics
