from typing import Any, Protocol

from src.summarize_algorithms.core.models import DialogueState, Session


class Dialogue(Protocol):
    """
    Minimal public interface for a dialogue system used throughout benchmarking.

    Any implementation must expose a `system_name` and provide `process_dialogue()` returning a `DialogueState`.
    """

    system_name: str

    def process_dialogue(
            self,
            sessions: list[Session],
            system_prompt: str,
            structure: dict[str, Any] | None = None,
            tools: list[dict[str, Any]] | None = None
    ) -> DialogueState:
        ...
