from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from dataclasses_json import DataClassJsonMixin, dataclass_json

from src.benchmarking.models.enums import MetricType
from src.summarize_algorithms.core.models import BaseBlock, OpenAIModels


@dataclass
class DividedSession:
    reference: list[BaseBlock]
    past_interactions: list[BaseBlock]


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
class StatisticsDto:
    algorithms: list[AlgorithmStatistics]


@dataclass_json
@dataclass
class MetricState:
    metric_name: MetricType
    metric_value: float | int


@dataclass
class BaseRecord(DataClassJsonMixin):
    timestamp: str
    iteration: int
    system: str
    query: str
    response: Any
    sessions: list[dict[str, Any]]
    #prepared_messages: list[dict[str, Any]]
    metric: list[MetricState] | None = field(default=None)


@dataclass
class MemoryRecord(BaseRecord):
    memory: dict[str, Any] = field(default_factory=dict)


@dataclass
class Evaluation:
    memory: dict[str, Any] = field(default_factory=dict)


@dataclass_json
@dataclass(frozen=True)
class ModelPrice:
    input_per_million: Decimal
    output_per_million: Decimal


MODEL_PRICES: dict[OpenAIModels, ModelPrice] = {
    OpenAIModels.GPT_4_O: ModelPrice(
        input_per_million=Decimal("2.50"),
        output_per_million=Decimal("10.00"),
    ),
    OpenAIModels.GPT_5_MINI: ModelPrice(
        input_per_million=Decimal("0.250"),
        output_per_million=Decimal("2.000"),
    ),
    OpenAIModels.GPT_5_NANO: ModelPrice(
        input_per_million=Decimal("0.05"),
        output_per_million=Decimal("0.40"),
    ),
    OpenAIModels.GPT_4_1_MINI: ModelPrice(
        input_per_million=Decimal("0.40"),
        output_per_million=Decimal("1.60"),
    ),
    OpenAIModels.GPT_4_1: ModelPrice(
        input_per_million=Decimal("2.00"),
        output_per_million=Decimal("8.00"),
    ),
    OpenAIModels.GPT_3_5_TURBO: ModelPrice(
        input_per_million=Decimal("0.50"),
        output_per_million=Decimal("1.50"),
    ),
    OpenAIModels.GPT_4_O_MINI: ModelPrice(
        input_per_million=Decimal("0.15"),
        output_per_million=Decimal("0.60"),
    ),
}


@dataclass
class TokenInfo(DataClassJsonMixin):
    model: OpenAIModels
    price: ModelPrice
    input_tokens: int
    output_tokens: int
    input_price: Decimal
    output_price: Decimal
    total_price: Decimal
