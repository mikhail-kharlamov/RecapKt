from dataclasses import dataclass

from src.benchmark.models.enums import AlgorithmName, MetricType


@dataclass(frozen=True, slots=True)
class MetricKey:
    """Grouping key for a single metric series.
    A series is identified by:
    - algorithm name
    - metric type
    - number of sessions used in the run
    """
    algorithm: str
    metric: MetricType
    session_count: int


@dataclass(frozen=True, slots=True)
class MetricObservation:
    """A single observed metric value for a given `MetricKey`."""
    key: MetricKey
    value: float


@dataclass
class AlgorithmRun:
    algorithm: str
    metric: MetricType
    value: float
    sessions: int


@dataclass
class AlgorithmStatistics:
    name: str
    metric: MetricType
    count_of_launches: int
    mean: float
    variance: float
    runs: list[AlgorithmRun]
    # mode: int | float


@dataclass
class MetricValues:
    algorithm: AlgorithmName
    metric: MetricType
    values: list[float]


@dataclass
class StatisticsDto:
    algorithms: list[AlgorithmStatistics]
