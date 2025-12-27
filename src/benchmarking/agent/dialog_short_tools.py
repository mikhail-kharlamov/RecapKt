import json
from typing import List

from langchain_core.messages import BaseMessage, ToolMessage
from typing_extensions import override

from src.benchmarking.agent.baseline import DialogueBaseline
from src.summarize_algorithms.core.models import Session


class DialogueWithShortTools(DialogueBaseline):
    @override
    @staticmethod
    def _compress(sessions: list[Session]) -> list[BaseMessage]:
        messages: List[BaseMessage] = DialogueBaseline._get_context(sessions)

        for message in messages:
            if isinstance(message, ToolMessage):
                try:
                    content_json = json.loads(message.content)
                    if isinstance(content_json, dict):
                        shortened_content = {"result": content_json.get("result")}
                        message.content = json.dumps(shortened_content, ensure_ascii=False)

                except (json.JSONDecodeError, TypeError):
                    pass

        return messages
