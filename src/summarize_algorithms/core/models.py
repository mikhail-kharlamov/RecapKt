import json
import logging

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from dataclasses_json import dataclass_json
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


class OpenAIModels(Enum):
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4_1_MINI = "gpt-4.1-mini"
    GPT_4_O = "gpt-4o"
    GPT_4_1 = "gpt-4.1"
    GPT_5_NANO = "gpt-5-nano"
    GPT_5_MINI = "gpt-5-mini"
    GPT_4_O_MINI = "gpt-4o-mini"


class LocalModels(Enum):
    GEMMA_2_9_B = "gemma2:9b"


@dataclass
class BaseBlock:
    role: str
    content: str

    def __str__(self) -> str:
        return f"{self.role}: {self.content}"


@dataclass
class CodeBlock(BaseBlock):
    code: str


@dataclass
class ToolCallBlock(BaseBlock):
    id: str
    name: str
    arguments: str
    response: str


class Session:
    def __init__(self, messages: list[BaseBlock]) -> None:
        self.messages = messages

    def __len__(self) -> int:
        return len(self.messages)

    def __str__(self) -> str:
        if len(self.messages) == 0:
            return "missing"

        result_messages = []
        for msg in self.messages:
            if isinstance(msg, CodeBlock):
                result_messages.append(f"{msg.role}: {msg.code}")
            if isinstance(msg, ToolCallBlock):
                result_messages.append(
                    f"Tool Call [{msg.id}]: {msg.name} - {msg.arguments} -> {msg.response}"
                )
            else:
                result_messages.append(f"{msg.role}: {msg.content}")
        return "\n".join(result_messages)

    def __getitem__(self, index: int) -> BaseBlock:
        return self.messages[index]

    def __iter__(self) -> Iterator[BaseBlock]:
        return iter(self.messages)

    def to_dict(self) -> dict[str, Any]:
        result_messages = []
        for msg in self.messages:
            if isinstance(msg, CodeBlock):
                result_messages.append({
                    "type": "code",
                    "role": msg.role,
                    "code": msg.code,
                })
            elif isinstance(msg, ToolCallBlock):
                result_messages.append({
                    "type": "tool_call",
                    "id": msg.id,
                    "name": msg.name,
                    "arguments": msg.arguments,
                    "response": msg.response,
                })
            else:
                result_messages.append({
                    "type": "text",
                    "role": msg.role,
                    "content": msg.content,
                })
        return {"messages": result_messages}

    def to_langchain_messages(self) -> list[BaseMessage]:
        langchain_messages: list[BaseMessage] = []
        for msg in self.messages:
            if isinstance(msg, CodeBlock):
                langchain_messages.append(AIMessage(content=msg.code))

            elif isinstance(msg, ToolCallBlock):
                try:
                    ai_tool_call = {
                        "name": msg.name,
                        "args": json.loads(msg.arguments)
                                if isinstance(msg.arguments, str) and msg.arguments != ""
                                else {},
                        "id": msg.id
                    }
                except json.decoder.JSONDecodeError as e:
                    logging.error(e)
                    ai_tool_call = {
                        "name": msg.name,
                        "args": {},
                        "id": msg.id
                    }

                langchain_messages.append(AIMessage(
                    content="",
                    tool_calls=[ai_tool_call]
                ))

                langchain_messages.append(ToolMessage(
                    response=msg.response,
                    content=msg.content,
                    tool_call_id=msg.id,
                    name=msg.name
                ))

            else:
                if msg.role.lower() in ["user", "human"]:
                    langchain_messages.append(HumanMessage(content=msg.content))
                elif msg.role.lower() in ["system"]:
                    langchain_messages.append(SystemMessage(content=msg.content))
                else:
                    langchain_messages.append(AIMessage(content=msg.content))

        return langchain_messages

    def get_messages_by_role(self, role: str) -> list[BaseBlock]:
        return [msg for msg in self.messages if msg.role == role]

    def get_text_blocks(self) -> list[BaseBlock]:
        return [
            msg
            for msg in self.messages
            if not (isinstance(msg, CodeBlock) or isinstance(msg, ToolCallBlock))
        ]

    def get_code_blocks(self) -> list[CodeBlock]:
        return [msg for msg in self.messages if isinstance(msg, CodeBlock)]

    def get_tool_calls(self) -> list[ToolCallBlock]:
        return [msg for msg in self.messages if isinstance(msg, ToolCallBlock)]


@dataclass_json
@dataclass
class DialogueState:
    from src.summarize_algorithms.core.memory_storage import MemoryStorage

    dialogue_sessions: list[Session]
    prepared_messages: list[BaseMessage]
    code_memory_storage: MemoryStorage | None
    tool_memory_storage: MemoryStorage | None
    query: str
    current_session_index: int = 0
    _response: str | dict[str, Any] | None = None

    @property
    def response(self) -> str | dict[str, Any]:
        if self._response is None:
            raise ValueError("Response has not been generated yet.")
        return self._response

    @property
    def current_context(self) -> Session:
        return self.dialogue_sessions[-1]


@dataclass_json
@dataclass
class MemoryDialogueState(DialogueState):
    last_session: Session = field(default_factory=lambda: Session([]))


@dataclass_json
@dataclass
class RecsumDialogueState(MemoryDialogueState):
    text_memory: list[list[str]] = field(default_factory=list)

    @property
    def latest_memory(self) -> str:
        return "\n".join(self.text_memory[-1]) if self.text_memory else ""


@dataclass_json
@dataclass
class MemoryBankDialogueState(MemoryDialogueState):
    from src.summarize_algorithms.core.memory_storage import MemoryStorage

    text_memory_storage: MemoryStorage = field(default_factory=MemoryStorage)


class WorkflowNode(Enum):
    UPDATE_MEMORY = "update_memory"
    GENERATE_RESPONSE = "generate_response"


class UpdateState(Enum):
    CONTINUE_UPDATE = "continue_update"
    FINISH_UPDATE = "finish_update"

@dataclass_json
@dataclass
class ResponseContext:
    response: Any
    prepared_history: list[BaseMessage]
