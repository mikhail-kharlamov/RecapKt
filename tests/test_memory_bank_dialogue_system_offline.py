from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from src.algorithms.summarize_algorithms.core.models import BaseBlock, Session
from src.algorithms.summarize_algorithms.core.response_generator import (
    ResponseGenerator,
)
from src.algorithms.summarize_algorithms.memory_bank.dialogue_system import (
    MemoryBankDialogueSystem,
)
from src.benchmark.tool_plan_benchmarking.tools_and_schemas.parsed_jsons import PLAN_SCHEMA


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        t = text.lower()
        return [1.0 if "kafka" in t else 0.0, 1.0 if "redis" in t else 0.0, 0.0]


@dataclass
class Capture:
    invoked_messages: list[BaseMessage] | None = None


class CapturingChain:
    def __init__(self, capture: Capture, response: Any) -> None:
        self._capture = capture
        self._response = response

    def invoke(self, messages: list[BaseMessage]) -> Any:
        self._capture.invoked_messages = messages
        return self._response


class FakeSessionSummarizer:
    def summarize(self, session_messages: str, session_id: int):  # noqa: ANN001
        _ = session_id
        return [BaseBlock(role="SYSTEM", content=f"MEM<{session_messages}>")]


def test_memory_bank_pipeline_retrieves_and_injects_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "offline")

    capture = Capture()

    from src.algorithms.summarize_algorithms.core.base_dialogue_system import (
        BaseDialogueSystem,
    )

    fake_llm = MagicMock(spec=BaseChatModel)
    fake_llm.get_num_tokens_from_messages.return_value = 10

    def _init(self, llm=None, is_local=False):  # noqa: ANN001
        self.llm = fake_llm
        self.memory_llm = fake_llm

    monkeypatch.setattr(BaseDialogueSystem, "_initialize_model", _init, raising=True)

    monkeypatch.setattr(
        MemoryBankDialogueSystem,
        "_build_summarizer",
        lambda self: FakeSessionSummarizer(),
        raising=True,
    )

    def _build_chain(self: ResponseGenerator):
        return CapturingChain(capture, response={"plan_steps": []})

    monkeypatch.setattr(ResponseGenerator, "_build_chain", _build_chain, raising=True)

    system = MemoryBankDialogueSystem(embed_model=FakeEmbeddings(), embed_code=False, embed_tool=False)

    # One session that will be summarized and stored.
    sessions = [Session([BaseBlock(role="USER", content="Kafka is used here")])]

    # Run pipeline.
    state = system.process_dialogue(
        sessions=sessions,
        system_prompt="Where is Kafka used?",
        structure=PLAN_SCHEMA,
        tools=[],
    )

    assert capture.invoked_messages is not None
    assert state.prepared_messages == capture.invoked_messages

    system_msg = capture.invoked_messages[0]
    assert isinstance(system_msg, SystemMessage)
    assert "### MEMORY BANK:" in system_msg.content

    # Should contain our summarized memory content, since query mentions kafka.
    assert "Kafka is used here" in system_msg.content

    assert isinstance(capture.invoked_messages[-1], HumanMessage)
    assert capture.invoked_messages[-1].content == "Where is Kafka used?"
