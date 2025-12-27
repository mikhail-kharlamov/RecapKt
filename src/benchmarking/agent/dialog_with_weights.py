import json
from decimal import Decimal

from typing_extensions import override

from src.benchmarking.agent.baseline import DialogueBaseline
from src.summarize_algorithms.core.models import Session, ToolCallBlock, BaseBlock, CodeBlock


class DialogWithWeights(DialogueBaseline):
    @override
    @staticmethod
    def _compress(sessions: list[Session]) -> str:
        for session in sessions:
            for i in range(len(session.messages)):
                message = session.messages[i]
                if isinstance(message, ToolCallBlock):
                    content: dict[str, str] = json.loads(message.content)
                    new_content: str = json.dumps({"result": content.get("result")})
                    session.messages[i].content = new_content
        return DialogWithWeights.__calculate_with_weights(sessions)

    @staticmethod
    def __calculate_with_weights(sessions: list[Session]) -> str:
        all_messages: list[BaseBlock] = []
        for session in sessions:
            all_messages.extend(session.messages)

        text_messages: list[str] = []
        mid: int = (len(all_messages) - 1) // 2
        step: Decimal = Decimal(1) / Decimal(mid)
        coefficient: Decimal = Decimal(1)
        for i in range((len(all_messages) - 1) // 2):
            coefficient -= step
            message = all_messages[i]
            if message.role in ("USER", "user"):
                continue
            text_messages.append(
                DialogWithWeights.__message_to_str_with_weights(
                    message,
                    coefficient
                )
            )

        return "\n".join(text_messages)


    @staticmethod
    def __message_to_str_with_weights(message: BaseBlock, coefficient: Decimal) -> str:
        text: str = ""
        if isinstance(message, CodeBlock):
             text = f"{message.role}: {message.code}"
        elif isinstance(message, ToolCallBlock):
            text = f"Tool Call [{message.id}]: {message.name} - {message.arguments} -> {message.response}"
        else:
            text = f"{message.role}: {message.content}"
        return text
