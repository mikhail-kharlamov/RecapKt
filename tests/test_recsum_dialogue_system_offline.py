from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from src.algorithms.summarize_algorithms.core.models import BaseBlock, Session
from src.algorithms.summarize_algorithms.core.response_generator import (
    ResponseGenerator,
)
from src.algorithms.summarize_algorithms.recsum.dialogue_system import (
    RecsumDialogueSystem,
)
from src.benchmark.tool_plan_benchmarking.tools_and_schemas.parsed_jsons import (
    PLAN_SCHEMA,
)


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


class FakeSummarizer:
    def summarize(self, previous_memory: str, dialogue_context: str):  # noqa: ANN001
        # Return BaseBlock objects, as expected by update_memory_node.
        _ = previous_memory
        return [BaseBlock(role="SYSTEM", content=f"RECSUM<{dialogue_context}>")]


def test_recsum_pipeline_builds_prompt_with_recap_and_invokes_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "offline")

    capture = Capture()

    # Patch BaseDialogueSystem model initialization to avoid network / API keys.
    from src.algorithms.summarize_algorithms.core.base_dialogue_system import (
        BaseDialogueSystem,
    )

    def _fake_init_model(self, llm=None, is_local=False):  # noqa: ANN001
        self.llm = llm or pytest.MonkeyPatch().context  # type: ignore[attr-defined]

    # easier: assign MagicMocks
    import unittest.mock

    fake_llm = unittest.mock.MagicMock(spec=BaseChatModel)
    fake_llm.get_num_tokens_from_messages.return_value = 10

    def _init(self, llm=None, is_local=False):  # noqa: ANN001
        self.llm = fake_llm
        self.memory_llm = fake_llm

    monkeypatch.setattr(BaseDialogueSystem, "_initialize_model", _init, raising=True)

    # Patch summarizer builder
    monkeypatch.setattr(RecsumDialogueSystem, "_build_summarizer", lambda self: FakeSummarizer(), raising=True)

    # Patch ResponseGenerator to use capturing chain
    def _build_chain(self: ResponseGenerator):
        return CapturingChain(capture, response={"plan_steps": []})

    monkeypatch.setattr(ResponseGenerator, "_build_chain", _build_chain, raising=True)

    system = RecsumDialogueSystem(is_local=True)

    sessions = [Session([BaseBlock(role="USER", content="hello")])]
    state = system.process_dialogue(
        sessions=sessions,
        system_prompt="What now?",
        structure=PLAN_SCHEMA,
        tools=[],
    )

    assert capture.invoked_messages is not None
    assert state.prepared_messages == capture.invoked_messages

    # System message must contain recap
    assert isinstance(capture.invoked_messages[0], SystemMessage)
    assert "### RECAP:" in capture.invoked_messages[0].content

    # Latest query must be last human message
    assert isinstance(capture.invoked_messages[-1], HumanMessage)
    assert capture.invoked_messages[-1].content == "What now?"
