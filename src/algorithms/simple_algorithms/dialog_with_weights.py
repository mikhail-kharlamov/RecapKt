from decimal import Decimal
from math import ceil
from typing import override

from langchain_core.messages import BaseMessage, HumanMessage

from src.algorithms.simple_algorithms.dialogue_baseline import DialogueBaseline
from src.algorithms.summarize_algorithms.core.models import Session


class DialogueWithWeights(DialogueBaseline):
    """
    Baseline variant that compresses history by truncating message contents with a positional weight.

    Messages closer to the center of the conversation get truncated more aggressively (triangle-shaped coefficient).
    Human messages are preserved.
    """

    @override
    @staticmethod
    def _compress(sessions: list[Session]) -> list[BaseMessage]:
        """
        Compress sessions by truncating non-human messages based on their position.

        :param sessions: past sessions.
        :return: list[BaseMessage]: flattened history with weighted truncation applied.
        """
        messages: list[BaseMessage] = DialogueBaseline._get_context(sessions)
        cropped_messages: list[BaseMessage] = []

        mid: int = (len(messages) - 1) // 2
        step: Decimal = Decimal(1) / Decimal(mid)
        coefficient: Decimal = Decimal(1)

        for i in range(len(messages)):
            if coefficient > 0 and i != 0:
                coefficient -= step
            else:
                coefficient += step

            message = messages[i]
            if isinstance(message, HumanMessage):
                cropped_messages.append(message)
                continue

            message.content = message.content[:ceil(len(message.content) * coefficient)]
            cropped_messages.append(message)

        return cropped_messages
