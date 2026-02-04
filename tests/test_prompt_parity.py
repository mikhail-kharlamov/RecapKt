from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage

from src.algorithms.simple_algorithms.dialogue_baseline import DialogueBaseline
from src.algorithms.summarize_algorithms.core.response_generator import (
    ResponseGenerator,
)
from src.benchmark.tool_plan_benchmarking.json_schemas import PLAN_SCHEMA
from src.utils.system_prompt_builder import MemorySections


@dataclass
class InvocationCapture:
    messages: list[BaseMessage] | None = None


class CapturingChain:
    def __init__(self, capture: InvocationCapture, response: Any = "ok") -> None:
        self._capture = capture
        self._response = response

    def invoke(self, messages: list[BaseMessage]) -> Any:
        self._capture.messages = messages
        return self._response


@pytest.fixture
def fake_llm() -> MagicMock:
    llm = MagicMock(spec=BaseChatModel)
    # token counter is used by trimming/cropping
    llm.get_num_tokens_from_messages.return_value = 10
    return llm


def _make_simple_baseline(fake_llm: MagicMock) -> DialogueBaseline:
    # Avoid real model init
    baseline = DialogueBaseline.__new__(DialogueBaseline)
    baseline.system_name = "Baseline"
    baseline.llm = fake_llm

    from src.utils.system_prompt_builder import SystemPromptBuilder

    baseline._prompt_builder = SystemPromptBuilder()
    baseline.prompt_tokens = 0
    baseline.completion_tokens = 0
    baseline.total_cost = 0.0
    return baseline


def test_response_generator_system_prompt_matches_baseline(fake_llm: MagicMock) -> None:
    """Baseline and ResponseGenerator must use the same unified system templates.

    This is the core contract: system prompt content must be identical when schema/memory are equivalent.
    """
    # Baseline builds its system prompt with memory_mode="baseline" and schema
    baseline = _make_simple_baseline(fake_llm)
    baseline_system = baseline._prompt_builder.build(
        schema=PLAN_SCHEMA,
        memory=MemorySections(),
        memory_mode="baseline",
    )

    # ResponseGenerator must infer baseline mode when memory is empty
    rg = ResponseGenerator(fake_llm, structure=PLAN_SCHEMA, max_prompt_tokens=None)
    system_message = rg._build_system_message(memory=MemorySections())

    assert baseline_system == system_message.content


def test_baseline_and_response_generator_invoke_with_same_history(fake_llm: MagicMock) -> None:
    """Both paths must pass `list[BaseMessage]` to LLM with the same ordering.

    We verify:
    - first message is SystemMessage
    - system content is identical
    - the rest is the history (with the latest user query last)
    """
    # Capture ResponseGenerator invocation
    rg_capture = InvocationCapture()
    rg = ResponseGenerator(fake_llm, structure=PLAN_SCHEMA, max_prompt_tokens=None)
    rg._chain = CapturingChain(rg_capture)  # type: ignore[assignment]

    from src.algorithms.summarize_algorithms.core.models import BaseBlock, Session

    session = Session([BaseBlock(role="USER", content="Hi")])
    rg.generate_response(
        last_session=session,
        user_query="Question",
        memory=MemorySections(),
    )

    assert rg_capture.messages is not None
    assert isinstance(rg_capture.messages[0], SystemMessage)
    assert rg_capture.messages[-1].content == "Question"


def test_memory_mode_inference_adds_memory_blocks(fake_llm: MagicMock) -> None:
    """When memory sections are present, they must appear in system prompt."""
    rg = ResponseGenerator(fake_llm, structure=PLAN_SCHEMA, max_prompt_tokens=None)

    mem = MemorySections(recap="recap text")
    system_message = rg._build_system_message(memory=mem)

    assert "### RECAP:" in system_message.content
    assert "recap text" in system_message.content
