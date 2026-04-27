from langchain_core.messages import BaseMessage, ToolMessage
from typing_extensions import override  # noqa: UP035

from src.algorithms.simple_algorithms.dialogue_baseline import DialogueBaseline
from src.algorithms.summarize_algorithms.core.models import Session


class DialogueWithShortTools(DialogueBaseline):
    """
    Baseline variant that shortens tool messages.

    Keeps tool call structure in the history but clears `ToolMessage.content` to reduce context length.
    """

    @override
    @staticmethod
    def _compress(sessions: list[Session]) -> list[BaseMessage]:
        """
        Compress sessions by clearing tool message contents.

        :param sessions: past sessions.
        :return: list[BaseMessage]: flattened history with shortened tool messages.
        """
        messages: list[BaseMessage] = DialogueBaseline._get_context(sessions)

        for message in messages:
            if isinstance(message, ToolMessage):
                message.content = ""

        return messages
