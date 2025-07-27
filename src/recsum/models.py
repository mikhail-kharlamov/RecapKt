from dataclasses import dataclass, field


@dataclass
class Message:
    role: str
    message: str

    def __str__(self) -> str:
        return f"{self.role}: {self.message}"


@dataclass
class Session:
    messages: list[Message]

    def __len__(self) -> int:
        return len(self.messages)

    def __str__(self) -> str:
        return "\n".join([str(message) for message in self.messages])


@dataclass
class DialogueState:
    dialogue_sessions: list[Session]
    current_session_index: int
    query: str
    response: str
    memory: list[str] = field(default_factory=list)

    @property
    def current_context(self) -> Session:
        return self.dialogue_sessions[-1]

    @property
    def latest_memory(self) -> str:
        if len(self.memory) == 0:
            return ""
        return self.memory[-1]
