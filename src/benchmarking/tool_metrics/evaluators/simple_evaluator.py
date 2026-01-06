from src.benchmarking.models.dtos import MetricState
from src.benchmarking.models.enums import MetricType
from src.benchmarking.tool_metrics.evaluators.base_evaluator import BaseEvaluator
from src.summarize_algorithms.core.models import (
    BaseBlock,
    DialogueState,
    Session,
)


class SimpleEvaluator(BaseEvaluator):
    def evaluate(
            self,
            sessions: list[Session],
            query: BaseBlock,
            state: DialogueState,
            reference: list[BaseBlock] | None = None
    ) -> MetricState:
        return MetricState(
            metric=MetricType.COHERENCE,
            value=1
        )
