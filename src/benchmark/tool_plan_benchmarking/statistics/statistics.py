import logging

from pathlib import Path
from typing import Any

from src.algorithms.dialogue import Dialogue
from src.algorithms.summarize_algorithms.core.models import BaseBlock, Session
from src.benchmark.logger.base_logger import BaseLogger
from src.benchmark.models.dtos import BaseRecord
from src.benchmark.tool_plan_benchmarking.evaluators.base_evaluator import BaseEvaluator
from src.benchmark.tool_plan_benchmarking.statistics.aggregator import (
    MetricStatisticsAggregator,
)
from src.benchmark.tool_plan_benchmarking.statistics.dtos import StatisticsDto
from src.benchmark.tool_plan_benchmarking.statistics.evaluation_runner import (
    EvaluationLaunchRunner,
)
from src.benchmark.tool_plan_benchmarking.statistics.observations_collector import (
    MetricObservationsCollector,
)
from src.benchmark.tool_plan_benchmarking.statistics.printer import StatisticsPrinter


class Statistics:
    """Facade used by existing call sites (`run.py`, etc.)."""

    @staticmethod
    def calculate(
        count_of_launches: int,
        algorithms: list[Dialogue],
        evaluator_functions: list[BaseEvaluator],
        sessions: list[Session],
        count_of_sessions: int,
        gold_session: Session,
        prompt: str,
        reference: list[BaseBlock],
        logger: BaseLogger,
        subdirectory: Path,
        tools: list[dict[str, Any]] | None = None,
        shuffle: bool = False,
    ) -> StatisticsDto:
        system_logger = logging.getLogger()

        records = EvaluationLaunchRunner(system_logger).run(
            launch_count=count_of_launches,
            algorithms=algorithms,
            evaluators=evaluator_functions,
            past_sessions=sessions,
            session_count=count_of_sessions,
            gold_session=gold_session,
            prompt=prompt,
            reference=reference,
            results_logger=logger,
            subdirectory=subdirectory,
            tools=tools,
            shuffle=shuffle,
        )

        observations = MetricObservationsCollector.collect(records)
        return MetricStatisticsAggregator(normalize=False, logger=system_logger).aggregate(observations)

    @staticmethod
    def calculate_by_logs(
        count_of_launches: int,
        metrics: list[BaseRecord],
        system_logger: logging.Logger | None = None,
        normalize: bool = False,
    ) -> StatisticsDto:
        _ = count_of_launches  # kept for backward-compatible signature
        system_logger = system_logger or logging.getLogger()

        observations = MetricObservationsCollector.collect(metrics)
        return MetricStatisticsAggregator(normalize=normalize, logger=system_logger).aggregate(observations)

    @staticmethod
    def print_statistics(stats: StatisticsDto) -> None:
        StatisticsPrinter.print(stats)
