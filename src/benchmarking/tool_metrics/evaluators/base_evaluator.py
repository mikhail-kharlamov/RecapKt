from abc import ABC, abstractmethod

from src.benchmarking.models.dtos import MetricState
from src.summarize_algorithms.core.models import (
    BaseBlock,
    DialogueState,
    Session,
)


class BaseEvaluator(ABC):
    """
    Base class for functions that evaluates llm's memory algorithms.
    """
    def __init__(self, tool: str | None = None):
        self._tool = tool
        print(tool)

    @abstractmethod
    def evaluate(
            self,
            sessions: list[Session],
            query: str,
            state: DialogueState,
            reference: list[BaseBlock] | None = None
    ) -> MetricState:
        """
        Returns eval score of llm's answer with for query.
        :param sessions: previous user's sessions (other chats or contexts).
        :param query: the last user's query which response is evaluating.
        :param state: history of interactions with model's answer of the query.
        :param reference: reference model's answer.
        :return: MetricState: dataclass with information about evaluation (metric's name and score).
        """
        ...
