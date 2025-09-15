from typing import Any, cast

from langchain_core.runnables import RunnableSerializable
from pydantic import BaseModel, Field

from src.summarize_algorithms.core.base_summarizer import BaseSummarizer
from src.summarize_algorithms.core.models import BaseBlock


class SessionMemory(BaseModel):
    summary_messages: list[BaseBlock] = Field(description="Summary of session messages")


class SessionSummarizer(BaseSummarizer):
    def _build_chain(self) -> RunnableSerializable[dict[str, Any], SessionMemory]:
        return cast(
            RunnableSerializable[dict, SessionMemory],
            self.prompt | self.llm.with_structured_output(SessionMemory),
        )

    def summarize(self, session_messages: str, session_id: int) -> list[BaseBlock]:
        try:
            response = self.chain.invoke(
                {
                    "session_messages": session_messages,
                    "session_id": session_id,
                }
            )
            return response.summary_messages
        except Exception as e:
            raise ConnectionError(f"API request failed: {e}") from e
