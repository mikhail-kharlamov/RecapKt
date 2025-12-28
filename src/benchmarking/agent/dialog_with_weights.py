from decimal import Decimal
from math import ceil

from langchain_core.messages import BaseMessage
from typing_extensions import override

from src.benchmarking.agent.dialogue_baseline import DialogueBaseline
from src.summarize_algorithms.core.models import Session


class DialogueWithWeights(DialogueBaseline):
    @override
    @staticmethod
    def _compress(sessions: list[Session]) -> list[BaseMessage]:
        messages: list[BaseMessage] = DialogueBaseline._get_context(sessions)
        cropped_messages: list[BaseMessage] = []

        mid: int = (len(messages) - 1) // 2
        step: Decimal = Decimal(1) / Decimal(mid)
        coefficient: Decimal = Decimal(1)

        for i in range(len(messages) - 1):
            if coefficient > 0:
                coefficient -= step
            else:
                coefficient += step

            message = messages[i]
            if message.role in ("USER", "user"):
                continue
            message.content = message.content[ceil(len(message.contenta) * (1 - coefficient)):]
            cropped_messages.append(message)

        return cropped_messages
