import json
import re

from typing import Any, Iterator

from src.summarize_algorithms.core.models import (
    BaseBlock,
    CodeBlock,
    Session,
    ToolCallBlock,
)


class MessageProcessor:
    CODE_PATTERN = re.compile(r"```(?:[a-zA-Z0-9]*)\n(.*?)```", re.DOTALL)

    @classmethod
    def process_message(cls, message: dict[str, Any]) -> list[BaseBlock]:
        role = message["type"]
        message_text = message["content"]

        blocks = []
        last_end = 0

        for match in cls.CODE_PATTERN.finditer(message_text):
            before_code = message_text[last_end : match.start()].strip()
            code_content = match.group(1).strip()

            if before_code:
                blocks.append(BaseBlock(role=role, content=before_code))
                blocks.append(
                    CodeBlock(role=role, content=before_code, code=code_content)
                )
            else:
                blocks.append(
                    CodeBlock(role=role, content=code_content, code=code_content)
                )

            last_end = match.end()

        if not blocks and message_text.strip():
            blocks.append(BaseBlock(role=role, content=message_text.strip()))
        elif last_end < len(message_text):
            tail = message_text[last_end:].strip()
            if tail:
                blocks.append(BaseBlock(role=role, content=tail))

        return blocks

    @classmethod
    def process_tool_calls(cls, messages: list[dict[str, Any]]) -> list[BaseBlock]:
        if len(messages) != 2:
            raise ValueError("Tool call processing requires exactly 2 messages")

        blocks = []
        assistant_message, tool_message = messages

        tool_content = None
        if assistant_message.get("content"):
            blocks.extend(cls.process_message(assistant_message))
            tool_content = assistant_message["content"]

        tool_calls = assistant_message.get("tool_calls", [])
        tool_responses = tool_message.get("tool_responses", [])

        for tool_call, tool_response in zip(tool_calls, tool_responses):
            if tool_content is None:
                tool_content = (
                    f"name: {tool_call['name']}\narguments: {tool_call['arguments']}\n"
                    f"response: {tool_response['responseData']}"
                )

            blocks.append(
                ToolCallBlock(
                    role="tool_call",
                    content=tool_content,
                    id=tool_call["id"],
                    name=tool_call["name"],
                    arguments=tool_call["arguments"],
                    response=tool_response["responseData"],
                )
            )

        return blocks


class ChatDataset:
    def __init__(self, sessions: list[Session] = None) -> None:
        self._sessions = sessions or []

    @property
    def sessions(self) -> list[Session]:
        return self._sessions.copy()

    def __len__(self) -> int:
        return len(self._sessions)

    def __getitem__(self, index: int) -> Session:
        return self._sessions[index]

    def __iter__(self) -> Iterator[Session]:
        return iter(self._sessions)

    def total_messages(self) -> int:
        return sum(len(session) for session in self._sessions)

    @classmethod
    def from_file(
        cls,
        file_name: str = "/home/kush/machine_learning/RecapKt/src/benchmarking/agent_chat/"
        "combined_chat_history_sessions.json",
    ) -> "ChatDataset":
        processor = MessageProcessor()
        sessions = []

        with open(file_name, encoding="utf-8") as f:
            data = json.load(f)

        for session_data in data:
            result_blocks = []
            messages = session_data.get("messages", [])
            message_idx = 0

            while message_idx < len(messages):
                message = messages[message_idx]

                if message["type"] == "USER":
                    result_blocks.extend(processor.process_message(message))
                    message_idx += 1

                elif message["type"] == "ASSISTANT":
                    if message_idx + 1 == len(messages):
                        result_blocks.append(
                            BaseBlock(role=message["type"], content=message["content"])
                        )
                        message_idx += 1
                    elif "tool_calls" in message:
                        tool_blocks = processor.process_tool_calls(
                            [message, messages[message_idx + 1]]
                        )
                        result_blocks.extend(tool_blocks)
                        message_idx += 2
                    else:
                        result_blocks.extend(processor.process_message(message))
                        message_idx += 1

            sessions.append(Session(messages=result_blocks))

        return cls(sessions)
