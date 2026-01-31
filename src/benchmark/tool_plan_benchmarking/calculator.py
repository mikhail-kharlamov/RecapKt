import logging

from pathlib import Path
from typing import Any

from src.benchmark.logger.base_logger import BaseLogger
from src.benchmark.models.dtos import BaseRecord, MetricState
from src.benchmark.tool_plan_benchmarking.evaluators.base_evaluator import BaseEvaluator
from src.benchmark.tool_plan_benchmarking.json_schemas import PLAN_SCHEMA
from src.algorithms.dialogue import Dialogue
from src.algorithms.summarize_algorithms.core.models import BaseBlock, Session


class Calculator:
    """
    Class for run, evaluate (by compare with reference) and save results of llm's memory algorithms.
    """
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
            iteration: int | None = None
    ) -> list[BaseRecord]:
        """
        The main method for run and evaluate algorithm with the ast sessions.
        :param algorithms: class that implements Dialog protocol.
        :param evaluator_functions: list of class inheritances of BaseEvaluator - for evaluating algorithm's results.
        :param sessions: past user-tool-model interactions.
        :param prompt: the query for model for comparing results
        :param reference: reference model's response for comparing with algorithm's results.
        :param logger: the class for saving results of running and evaluating algorithms.
        :param tools: tools/functions description for function calling.
        :param subdirectory: directory for grouping metric logs.
        :param iteration.
        :return: list[dict[str, Any]]: results of running and evaluating algorithm.
        """
        system_logger = logging.getLogger()
        metrics: list[BaseRecord] = []
        for algorithm in algorithms:
            system_logger.info(f"Calculating {algorithm.system_name}")
            state = algorithm.process_dialogue(sessions, prompt, PLAN_SCHEMA, tools)

            algorithm_metrics: list[MetricState] = []
            for evaluator_function in evaluator_functions:
                metric: MetricState = evaluator_function.evaluate(sessions, prompt, state, reference)
                algorithm_metrics.append(metric)

            record: BaseRecord = logger.log_iteration(
                algorithm.system_name,
                prompt,
                iteration or 1,
                sessions,
                state,
                algorithm.system_name / subdirectory,
                algorithm_metrics,
            )

            metrics.append(record)

        return metrics
