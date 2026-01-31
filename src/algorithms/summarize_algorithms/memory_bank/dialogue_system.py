from src.algorithms.summarize_algorithms.core.base_dialogue_system import BaseDialogueSystem
from src.algorithms.summarize_algorithms.core.memory_storage.memory_storage import MemoryStorage
from src.algorithms.summarize_algorithms.core.models import MemoryBankDialogueState, Session
from src.algorithms.summarize_algorithms.memory_bank.prompts import SESSION_SUMMARY_PROMPT
from src.algorithms.summarize_algorithms.memory_bank.summarizer import SessionSummarizer


class MemoryBankDialogueSystem(BaseDialogueSystem):
    """
    Implementation of the MemoryBank-style dialogue system.

    Summarizes each session into a compact representation stored in `text_memory_storage` and optionally augments
    the prompt with retrieved code/tool memories (FAISS + embeddings).
    """

    def _build_summarizer(self) -> SessionSummarizer:
        return SessionSummarizer(self.memory_llm, SESSION_SUMMARY_PROMPT)

    def _get_initial_state(
        self, sessions: list[Session], last_session: Session, query: str
    ) -> MemoryBankDialogueState:
        return MemoryBankDialogueState(
            dialogue_sessions=sessions,
            last_session=last_session,
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
            prepared_messages=[]
        )

    @property
    def _get_dialogue_state_class(self) -> type:
        return MemoryBankDialogueState
