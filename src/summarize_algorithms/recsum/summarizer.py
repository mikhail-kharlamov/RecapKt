from typing import Any, cast

from langchain_core.runnables import RunnableSerializable

from src.summarize_algorithms.core.base_summarizer import BaseSummarizer
from src.summarize_algorithms.core.models import BaseBlock
from src.summarize_algorithms.memory_bank.summarizer import SessionMemory


class RecursiveSummarizer(BaseSummarizer):
    def _build_chain(self) -> RunnableSerializable[dict[str, Any], SessionMemory]:
        return cast(
            RunnableSerializable[dict, SessionMemory],
            self.prompt | self.llm.with_structured_output(SessionMemory),
        )

    def summarize(self, previous_memory: str, dialogue_context: str) -> list[BaseBlock]:
        try:
            response = self.chain.invoke(
                {
                    "previous_memory": previous_memory,
                    "dialogue_context": dialogue_context,
                }
            )
            return response.summary_messages
        except Exception as e:
            raise ConnectionError(f"API request failed: {e}") from e
