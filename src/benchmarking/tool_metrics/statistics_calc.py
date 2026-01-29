import hashlib
import logging
import random

from collections import Counter
from logging import Logger
from math import fsum
from pathlib import Path
from typing import Any

from src.benchmarking.base_logger import BaseLogger
from src.benchmarking.models.dtos import (
    AlgorithmRun,
    AlgorithmStatistics,
    BaseRecord,
    StatisticsDto,
)
from src.benchmarking.models.enums import MetricType
from src.benchmarking.tool_metrics.calculator import Calculator
from src.benchmarking.tool_metrics.evaluators.base_evaluator import BaseEvaluator
from src.summarize_algorithms.core.dialogue import Dialogue
from src.summarize_algorithms.core.models import BaseBlock, Session


class Statistics:
    """
    Helpers for running repeated evaluation launches and aggregating metric statistics.

    `calculate()` runs `Calculator.evaluate()` multiple times (optionally shuffling sessions) and computes basic
    descriptive stats (mean/variance) per (algorithm, metric, number_of_sessions).
    """

    RUN_ID = "exp_29_12_2025_1_31_am"

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
            shuffle: bool = False
    ) -> StatisticsDto:
        """
        Run multiple evaluation launches and aggregate results.

        :param count_of_launches: number of repeated runs (useful when randomness is involved).
        :param algorithms: dialogue systems/baselines to evaluate.
        :param evaluator_functions: metric evaluators applied to each run.
        :param sessions: past interactions (may be shuffled if `shuffle=True`).
        :param count_of_sessions: how many sessions to include per run.
        :param gold_session: the final (current) session containing the evaluated query.
        :param prompt: system prompt / query template.
        :param reference: reference blocks used for metric computation.
        :param logger: logger used to persist run artifacts.
        :param subdirectory: path segment for grouping outputs by session count.
        :param tools: optional tool specs.
        :param shuffle: whether to shuffle `sessions` before slicing.
        :return: StatisticsDto: aggregated statistics DTO.
        """
        system_logger = logging.getLogger()

        values_by_alg_metric: dict[tuple[str, MetricType, int], list[float]] = {}
        for i in range(count_of_launches):
            if len(sessions) > 1 and shuffle:
                rnd = random.Random(Statistics.__make_seed(count_of_launches, i))
                ses_0 = sessions[0]
                rnd.shuffle(sessions)
                print(sessions.index(ses_0))
            prepared_sessions: list[Session] = sessions.copy()
            prepared_sessions = prepared_sessions[:(count_of_sessions - 1)]
            prepared_sessions.append(gold_session)
            print("Количество сессий: ", len(prepared_sessions))

            system_logger.info(f"Starting evaluation launch {i}")
            metrics: list[BaseRecord] = Calculator.evaluate(
                algorithms,
                evaluator_functions,
                prepared_sessions,
                prompt,
                reference,
                logger,
                subdirectory,
                tools,
                i
            )

            for record in metrics:
                if record.metric is None:
                    continue
                for metric_state in record.metric:
                    key = (record.system, metric_state.metric_name, len(prepared_sessions))
                    if key not in values_by_alg_metric:
                        values_by_alg_metric[key] = []
                    values_by_alg_metric[key].append(float(metric_state.metric_value))

        return Statistics.__get_statistic_metrics(count_of_launches, system_logger, values_by_alg_metric)

    @staticmethod
    def calculate_by_logs(
            count_of_launches: int,
            metrics: list[BaseRecord],
            system_logger: logging.Logger | None = None
    ) -> StatisticsDto:
        if system_logger is None:
            system_logger = logging.getLogger()

        values_by_alg_metric: dict[tuple[str, MetricType, int], list[float]] = {}
        for _ in range(count_of_launches):
            for record in metrics:
                if record.metric is None:
                    continue
                for metric_state in record.metric:
                    key = (record.system, metric_state.metric_name, len(record.sessions))
                    if key not in values_by_alg_metric:
                        values_by_alg_metric[key] = []
                    values_by_alg_metric[key].append(float(metric_state.metric_value))

        return Statistics.__get_statistic_metrics(count_of_launches, system_logger, values_by_alg_metric)

    @staticmethod
    def __get_statistic_metrics(
            count_of_launches: int,
            system_logger: Logger,
            values_by_alg_metric: dict[tuple[str, MetricType, int], list[float]]
    ) -> StatisticsDto:
        algorithm_stats: list[AlgorithmStatistics] = []
        system_logger.info("Getting statistics...")
        for (alg_name, metric_type, count_of_sessions), values in values_by_alg_metric.items():
            system_logger.info(f"Getting {alg_name} {metric_type.value} statistics")
            n = len(values)
            if n == 0:
                continue

            mean = fsum(values) / n

            variance = fsum((v - mean) ** 2 for v in values) / n

            algorithm_stats.append(
                AlgorithmStatistics(
                    name=alg_name,
                    metric=metric_type,
                    count_of_launches=count_of_launches,
                    mean=mean,
                    variance=variance,
                    runs=[
                        AlgorithmRun(
                            algorithm=alg_name,
                            metric=metric_type,
                            value=value,
                            sessions=count_of_sessions,
                        )
                        for value in values
                    ]
                )
            )
        return StatisticsDto(
            algorithms=algorithm_stats
        )

    @staticmethod
    def calculate_mode(values: list[int | float]) -> int | float:
        if all(isinstance(value, int) for value in values):
            counter = Counter(values)
            mode, _ = counter.most_common(1)[0]
            return mode

        values = [float(value) for value in values]
        values.sort()
        left: float = values[0]
        right: float = values[2]
        last_group: list[float] = []
        max_group: list[float] = []
        for i in range(1, len(values) - 2):
            if abs(values[i] - left) <= abs(values[i] - right):
                last_group.append(values[i])
            elif abs(values[i] - left) > abs(values[i] - right):
                if len(last_group) > len(max_group):
                    max_group = last_group
                last_group = [values[i]]

        return fsum(max_group) / len(max_group)

    @staticmethod
    def print_statistics(stats: StatisticsDto) -> None:
        print("=== Statistics ===")
        for alg_stat in stats.algorithms:
            metric_name = alg_stat.metric.value

            print(f"Algorithm: {alg_stat.name}")
            print(f"  Metric: {metric_name}")
            print(f"  Count of launches: {alg_stat.count_of_launches}")
            print(f"  Math. expectation: {alg_stat.mean:.4f}")
            print(f"  Variance: {alg_stat.variance:.4f}")
            print()

    @staticmethod
    def __make_seed(count_of_iterations: int, iteration: int) -> int:
        h = hashlib.sha256(f"{Statistics.RUN_ID}:{count_of_iterations}:{iteration}".encode()).hexdigest()
        return int(h[:16], 16)
