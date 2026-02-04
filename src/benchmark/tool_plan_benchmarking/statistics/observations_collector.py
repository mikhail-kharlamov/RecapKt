from collections.abc import Iterable

from src.benchmark.models.dtos import BaseRecord
from src.benchmark.tool_plan_benchmarking.statistics.dtos import (
    MetricKey,
    MetricObservation,
)


class MetricObservationsCollector:
    """Extracts metric observations from benchmark log records."""

    @staticmethod
    def collect(records: Iterable[BaseRecord]) -> list[MetricObservation]:
        observations: list[MetricObservation] = []
        for record in records:
            if record.metric is None:
                continue

            session_count = len(record.sessions)
            for metric_state in record.metric:
                observations.append(
                    MetricObservation(
                        key=MetricKey(
                            algorithm=record.system,
                            metric=metric_state.metric_name,
                            session_count=session_count,
                        ),
                        value=float(metric_state.metric_value),
                    )
                )

        return observations
