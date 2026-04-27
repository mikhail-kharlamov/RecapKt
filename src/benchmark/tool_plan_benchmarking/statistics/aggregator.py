import logging

from collections import defaultdict
from collections.abc import Iterable
from math import fsum

from src.benchmark.tool_plan_benchmarking.statistics.dtos import (
    AlgorithmRun,
    AlgorithmStatistics,
    MetricKey,
    MetricObservation,
    StatisticsDto,
)
from src.benchmark.tool_plan_benchmarking.statistics.normalizer import MetricNormalizer


class MetricStatisticsAggregator:
    """Aggregates metric observations into `StatisticsDto` (mean/variance + per-run values)."""

    def __init__(
        self, *, normalize: bool = False, logger: logging.Logger | None = None
    ) -> None:
        self._normalize = normalize
        self._logger = logger or logging.getLogger()

    def aggregate(self, observations: Iterable[MetricObservation]) -> StatisticsDto:
        values_by_key: dict[MetricKey, list[float]] = defaultdict(list)
        for observation in observations:
            values_by_key[observation.key].append(observation.value)

        if self._normalize:
            values_by_key = MetricNormalizer.normalize_by_metric_type(
                dict(values_by_key)
            )

        algorithm_stats: list[AlgorithmStatistics] = []
        self._logger.info("Getting statistics...")

        for key, values in values_by_key.items():
            self._logger.info(
                "Getting %s %s statistics", key.algorithm, key.metric.value
            )

            n = len(values)
            if n == 0:
                continue

            mean = fsum(values) / n
            variance = fsum((v - mean) ** 2 for v in values) / n

            algorithm_stats.append(
                AlgorithmStatistics(
                    name=key.algorithm,
                    metric=key.metric,
                    count_of_launches=n,
                    mean=mean,
                    variance=variance,
                    runs=[
                        AlgorithmRun(
                            algorithm=key.algorithm,
                            metric=key.metric,
                            value=value,
                            sessions=key.session_count,
                        )
                        for value in values
                    ],
                )
            )

        return StatisticsDto(algorithms=algorithm_stats)
