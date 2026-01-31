from typing import Any, cast

from langchain_core.runnables import RunnableSerializable

from src.algorithms.summarize_algorithms.core.base_summarizer import BaseSummarizer
from src.algorithms.summarize_algorithms.core.models import BaseBlock
from src.algorithms.summarize_algorithms.memory_bank.summarizer import SessionMemory


class RecursiveSummarizer(BaseSummarizer):
    """
    RecSum memory updater.

    Given the previous memory + the current dialogue context, produces a list of `BaseBlock` messages representing
    the updated memory.
    """

    def _build_chain(self) -> RunnableSerializable[dict[str, Any], SessionMemory]:
        return cast(
            RunnableSerializable[dict, SessionMemory],
            self.prompt | self.llm.with_structured_output(SessionMemory),
        )

    def summarize(self, previous_memory: str, dialogue_context: str) -> list[BaseBlock]:
        """
        Update recursive memory given previous memory and the latest dialogue context.

        :param previous_memory: previous memory string.
        :param dialogue_context: current dialogue context string.
        :return: list[BaseBlock]: updated memory blocks.
        """
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
