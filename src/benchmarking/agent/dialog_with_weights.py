from typing_extensions import override

from src.benchmarking.agent.baseline import DialogueBaseline
from src.summarize_algorithms.core.models import Session


class DialogWithWeights(DialogueBaseline):
    @override
    @staticmethod
    def _compress(sessions: list[Session]) -> list[Session]:
        ...
