from typing import Type

from src.summarize_algorithms.core.base_dialogue_system import BaseDialogueSystem
from src.summarize_algorithms.core.memory_storage import MemoryStorage
from src.summarize_algorithms.core.models import MemoryBankDialogueState, Session
from src.summarize_algorithms.memory_bank.prompts import SESSION_SUMMARY_PROMPT
from src.summarize_algorithms.memory_bank.summarizer import SessionSummarizer


class MemoryBankDialogueSystem(BaseDialogueSystem):
    def _build_summarizer(self) -> SessionSummarizer:
        return SessionSummarizer(self.llm, SESSION_SUMMARY_PROMPT)

    def _get_initial_state(
        self, sessions: list[Session], query: str
    ) -> MemoryBankDialogueState:
        return MemoryBankDialogueState(
            dialogue_sessions=sessions,
            code_memory_storage=(
                MemoryStorage(
                    embeddings=self.embed_model, max_session_id=self.max_session_id
                )
                if self.embed_code
                else None
            ),
            tool_memory_storage=(
                MemoryStorage(
                    embeddings=self.embed_model, max_session_id=self.max_session_id
                )
                if self.embed_tool
                else None
            ),
            query=query,
            text_memory_storage=MemoryStorage(
                embeddings=self.embed_model, max_session_id=self.max_session_id
            ),
        )

    @property
    def _get_dialogue_state_class(self) -> Type:
        return MemoryBankDialogueState
