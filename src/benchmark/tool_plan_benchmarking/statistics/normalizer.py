from collections import defaultdict

from src.benchmark.models.enums import MetricType
from src.benchmark.tool_plan_benchmarking.statistics.dtos import MetricKey


class MetricNormalizer:
    """Normalizes metric series across algorithms.

    Normalization is done *per metric type* (e.g. F1_TOOL, F1_TOOL_STRICT) using global min/max across all
    algorithms and session counts for that metric.
    """

    @staticmethod
    def normalize_by_metric_type(values_by_key: dict[MetricKey, list[float]]) -> dict[MetricKey, list[float]]:
        values_by_metric: dict[MetricType, list[float]] = defaultdict(list)
        for key, values in values_by_key.items():
            values_by_metric[key.metric].extend(values)

        if not values_by_metric:
            return values_by_key

        min_max_by_metric: dict[MetricType, tuple[float, float]] = {}
        for metric, values in values_by_metric.items():
            if not values:
                continue
            min_max_by_metric[metric] = (min(values), max(values))

        normalized: dict[MetricKey, list[float]] = {}
        for key, values in values_by_key.items():
            global_min, global_max = min_max_by_metric.get(key.metric, (0.0, 0.0))
            if global_max == global_min:
                normalized[key] = values
                continue
            normalized[key] = [(v - global_min) / (global_max - global_min) for v in values]

        return normalized
