
from src.algorithms.summarize_algorithms.core.base_dialogue_system import BaseDialogueSystem
from src.algorithms.summarize_algorithms.core.memory_storage.memory_storage import MemoryStorage
from src.algorithms.summarize_algorithms.core.models import RecsumDialogueState, Session
from src.algorithms.summarize_algorithms.recsum.prompts import MEMORY_UPDATE_PROMPT_TEMPLATE
from src.algorithms.summarize_algorithms.recsum.summarizer import RecursiveSummarizer


class RecsumDialogueSystem(BaseDialogueSystem):
    """
    Implementation of the RecSum-style dialogue system.

    Uses `RecursiveSummarizer` to iteratively update a text memory (and optionally vector-retrieved code/tool memory)
    and then generates a final response via `BaseDialogueSystem`’s graph.
    """

    def _build_summarizer(self) -> RecursiveSummarizer:
        return RecursiveSummarizer(self.memory_llm, MEMORY_UPDATE_PROMPT_TEMPLATE)

    def _get_initial_state(
        self, sessions: list[Session], last_session: Session, query: str
    ) -> RecsumDialogueState:
        return RecsumDialogueState(
            dialogue_sessions=sessions,
            last_session=last_session,
            code_memory_storage=MemoryStorage(
                embeddings=self.embed_model, max_session_id=self.max_session_id
            ),
            tool_memory_storage=MemoryStorage(
                embeddings=self.embed_model, max_session_id=self.max_session_id
            ),
            query=query,
            prepared_messages=[]
        )

    @property
    def _get_dialogue_state_class(self) -> type:
        return RecsumDialogueState
