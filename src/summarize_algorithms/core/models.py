from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Optional


class OpenAIModels(Enum):
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4_1_MINI = "gpt-4.1-mini"
    GPT_4_O = "gpt-4o"
    GPT_4_1 = "gpt-4.1"


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


@dataclass
class DialogueState:
    from src.summarize_algorithms.core.memory_storage import MemoryStorage

    dialogue_sessions: list[Session]
    code_memory_storage: Optional[MemoryStorage]
    tool_memory_storage: Optional[MemoryStorage]
    query: str
    current_session_index: int = 0
    _response: Optional[str] = None

    @property
    def response(self) -> str:
        if self._response is None:
            raise ValueError("Response has not been generated yet.")
        return self._response

    @property
    def current_context(self) -> Session:
        return self.dialogue_sessions[-1]


@dataclass
class RecsumDialogueState(DialogueState):
    text_memory: list[list[str]] = field(default_factory=list)

    @property
    def latest_memory(self) -> str:
        return "\n".join(self.text_memory[-1]) if self.text_memory else ""


@dataclass
class MemoryBankDialogueState(DialogueState):
    from src.summarize_algorithms.core.memory_storage import MemoryStorage

    text_memory_storage: MemoryStorage = field(default_factory=MemoryStorage)


class WorkflowNode(Enum):
    UPDATE_MEMORY = "update_memory"
    GENERATE_RESPONSE = "generate_response"


class UpdateState(Enum):
    CONTINUE_UPDATE = "continue_update"
    FINISH_UPDATE = "finish_update"
