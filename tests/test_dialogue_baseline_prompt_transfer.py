from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from src.algorithms.simple_algorithms.dialogue_baseline import DialogueBaseline
from src.algorithms.summarize_algorithms.core.models import BaseBlock, Session
from src.algorithms.summarize_algorithms.core.response_generator import (
    ResponseGenerator,
)
from src.benchmark.tool_plan_benchmarking.tools_and_schemas.parsed_jsons import (
    PLAN_SCHEMA,
    TOOLS,
)
from src.utils.system_prompt_builder import MemorySections, SystemPromptBuilder


@dataclass
class InvocationCapture:
    messages: list[BaseMessage] | None = None


class CapturingChain:
    def __init__(self, capture: InvocationCapture, response: Any = None) -> None:
        self._capture = capture
        # DialogueBaseline uses structured output in this test; return a dict-shaped response by default.
        self._response = {} if response is None else response

    def invoke(self, messages: list[BaseMessage]) -> Any:
        self._capture.messages = messages
        return self._response


class DummyCallback:
    prompt_tokens = 0
    completion_tokens = 0
    total_cost = 0.0

    def __enter__(self) -> DummyCallback:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None


@pytest.fixture
def fake_llm() -> MagicMock:
    llm = MagicMock(spec=BaseChatModel)
    llm.get_num_tokens_from_messages.side_effect = lambda msgs: len(msgs) * 10
    return llm


def test_dialogue_baseline_transfers_full_prompt_as_message_list(
    monkeypatch: pytest.MonkeyPatch, fake_llm: MagicMock
):
    """Baseline must pass a list[BaseMessage] (not dict vars) and use the unified system templates."""

    # Patch LangChain callback context used in DialogueBaseline
    import src.algorithms.simple_algorithms.dialogue_baseline as baseline_mod

    monkeypatch.setattr(baseline_mod, "get_openai_callback", lambda: DummyCallback())

    baseline = DialogueBaseline.__new__(DialogueBaseline)
    baseline.system_name = "Baseline"
    baseline.llm = fake_llm
    baseline._prompt_builder = SystemPromptBuilder()
    baseline.prompt_tokens = 0
    baseline.completion_tokens = 0
    baseline.total_cost = 0.0

    capture = InvocationCapture()
    baseline._build_chain = lambda *_args, **_kwargs: CapturingChain(capture)  # type: ignore[method-assign]

    sessions = [Session([BaseBlock(role="USER", content="Hi")])]
    baseline.process_dialogue(
        sessions=sessions,
        system_prompt="User question",
        structure=PLAN_SCHEMA,
        tools=None,
    )

    assert capture.messages is not None
    assert isinstance(capture.messages, list)
    assert isinstance(capture.messages[0], SystemMessage)
    assert isinstance(capture.messages[-1], HumanMessage)
    assert (
        capture.messages[-1].content == "Hi" or capture.messages[-1].content != ""
    )  # sanity

    expected_system = baseline._prompt_builder.build(
        schema=PLAN_SCHEMA,
        tools=TOOLS,
        memory=MemorySections(),
        memory_mode="baseline",
    )
    assert capture.messages[0].content == expected_system

    # Cross-check with ResponseGenerator system prompt for empty memory.
    rg = ResponseGenerator(fake_llm, structure=PLAN_SCHEMA, max_prompt_tokens=None)
    rg_system = rg._build_system_message(memory=MemorySections()).content
    assert rg_system == expected_system
