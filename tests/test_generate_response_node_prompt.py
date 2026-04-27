from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from src.algorithms.summarize_algorithms.core.graph_nodes import generate_response_node
from src.algorithms.summarize_algorithms.core.models import (
    BaseBlock,
    RecsumDialogueState,
    Session,
)
from src.algorithms.summarize_algorithms.core.response_generator import (
    ResponseGenerator,
)
from src.benchmark.tool_plan_benchmarking.tools_and_schemas.parsed_jsons import (
    PLAN_SCHEMA,
)


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


def test_generate_response_node_invokes_llm_with_full_history_and_memory() -> None:
    fake_llm = MagicMock(spec=BaseChatModel)
    fake_llm.get_num_tokens_from_messages.return_value = 10

    capture = InvocationCapture()
    rg = ResponseGenerator(fake_llm, structure=PLAN_SCHEMA, max_prompt_tokens=None)
    rg._chain = CapturingChain(capture, response={"plan_steps": []})  # type: ignore[assignment]

    state = RecsumDialogueState(
        dialogue_sessions=[],
        prepared_messages=[],
        code_memory_storage=None,
        tool_memory_storage=None,
        query="What now?",
        last_session=Session([BaseBlock(role="USER", content="Hi")]),
        text_memory=[["memory line"]],
    )

    out_state = generate_response_node(rg, state)

    assert capture.messages is not None
    assert out_state.prepared_messages == capture.messages
    assert out_state.response == {"plan_steps": []}

    assert isinstance(capture.messages[0], SystemMessage)
    assert "### RECAP:" in capture.messages[0].content
    assert "memory line" in capture.messages[0].content

    assert isinstance(capture.messages[-1], HumanMessage)
    assert capture.messages[-1].content == "What now?"
