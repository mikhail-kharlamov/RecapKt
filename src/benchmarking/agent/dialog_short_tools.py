
from langchain_core.messages import BaseMessage, ToolMessage
from typing_extensions import override

from src.benchmarking.agent.dialogue_baseline import DialogueBaseline
from src.summarize_algorithms.core.models import Session


class DialogueWithShortTools(DialogueBaseline):
    @override
    @staticmethod
    def _compress(sessions: list[Session]) -> list[BaseMessage]:
        messages: list[BaseMessage] = DialogueBaseline._get_context(sessions)

        for message in messages:
            if isinstance(message, ToolMessage):
                message.content = ""

        return messages
