from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from src.benchmark.models.dtos import BaseRecord, MetricState
from src.benchmark.models.enums import MetricType
from src.benchmark.tool_plan_benchmarking.statistics.aggregator import (
    MetricStatisticsAggregator,
)
from src.benchmark.tool_plan_benchmarking.statistics.observations_collector import (
    MetricObservationsCollector,
)
from src.benchmark.tool_plan_benchmarking.statistics.statistics import Statistics


def _make_record(system: str, sessions_count: int, f1: float, strict: float) -> BaseRecord:
    sessions = [{"messages": []} for _ in range(sessions_count)]
    return BaseRecord(
        timestamp=datetime.now().isoformat(),
        iteration=1,
        system=system,
        query="q",
        response={},
        sessions=sessions,
        prepared_messages=[],
        metric=[
            MetricState(metric_name=MetricType.F1_TOOL, metric_value=Decimal(str(f1))),
            MetricState(metric_name=MetricType.F1_TOOL_STRICT, metric_value=Decimal(str(strict))),
        ],
    )


def test_statistics_facade_matches_new_pipeline() -> None:
    records = [
        _make_record("A", 3, 0.1, 0.2),
        _make_record("A", 3, 0.3, 0.4),
        _make_record("B", 3, 0.5, 0.6),
    ]

    # New pipeline
    obs = MetricObservationsCollector.collect(records)
    stats_new = MetricStatisticsAggregator(normalize=False).aggregate(obs)

    # Old facade API
    stats_facade = Statistics.calculate_by_logs(
        count_of_launches=999,
        metrics=records,
        normalize=False,
    )

    assert stats_new.algorithms == stats_facade.algorithms


def test_normalization_is_stable_per_metric_type() -> None:
    records = [
        _make_record("A", 3, 0.0, 1.0),
        _make_record("B", 3, 1.0, 0.0),
    ]

    obs = MetricObservationsCollector.collect(records)
    stats_norm = MetricStatisticsAggregator(normalize=True).aggregate(obs)

    # For each metric type we should get values normalized to [0, 1]
    for alg in stats_norm.algorithms:
        for run in alg.runs:
            assert 0.0 <= run.value <= 1.0
