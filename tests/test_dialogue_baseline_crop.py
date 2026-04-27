from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.algorithms.simple_algorithms.dialogue_baseline import DialogueBaseline


@pytest.fixture
def fake_llm() -> MagicMock:
    llm = MagicMock(spec=BaseChatModel)
    llm.get_num_tokens_from_messages.side_effect = lambda msgs: len(msgs)
    return llm


def _make_baseline(fake_llm: MagicMock) -> DialogueBaseline:
    baseline = DialogueBaseline.__new__(DialogueBaseline)
    baseline.llm = fake_llm
    return baseline


def test_crop_reinserts_assistant_message_before_tool_when_trim_starts_with_system_tool(
    monkeypatch: pytest.MonkeyPatch,
    fake_llm: MagicMock,
) -> None:
    import src.algorithms.simple_algorithms.dialogue_baseline as baseline_mod

    system = SystemMessage(content="sys")
    old_user = HumanMessage(content="old")
    assistant_before_tool = AIMessage(content="assistant before tool")
    tool = ToolMessage(content="tool output", tool_call_id="call_1")
    after_tool = HumanMessage(content="after")

    original = [system, old_user, assistant_before_tool, tool, after_tool]
    trimmed = [system, tool, after_tool]

    def fake_trim_messages(msgs, *args, **kwargs):  # noqa: ANN001
        # First trim on the full history returns the problematic sequence.
        if msgs is original:
            return trimmed
        # Any subsequent trims return what they got.
        return msgs

    monkeypatch.setattr(baseline_mod, "trim_messages", fake_trim_messages)

    baseline = _make_baseline(fake_llm)
    result = baseline._crop(original, max_tokens=999)

    assert result[:3] == [system, assistant_before_tool, tool]


def test_crop_falls_back_to_tool_call_id_when_tool_identity_differs(
    monkeypatch: pytest.MonkeyPatch,
    fake_llm: MagicMock,
) -> None:
    import src.algorithms.simple_algorithms.dialogue_baseline as baseline_mod

    system = SystemMessage(content="sys")
    assistant_before_tool = AIMessage(content="assistant before tool")
    original_tool = ToolMessage(content="tool output", tool_call_id="call_1")
    after_tool = HumanMessage(content="after")

    original = [system, assistant_before_tool, original_tool, after_tool]

    # Simulate trim returning a different ToolMessage instance with the same tool_call_id.
    trimmed_tool = ToolMessage(content="tool output", tool_call_id="call_1")
    trimmed = [system, trimmed_tool, after_tool]

    def fake_trim_messages(msgs, *args, **kwargs):  # noqa: ANN001
        if msgs is original:
            return trimmed
        return msgs

    monkeypatch.setattr(baseline_mod, "trim_messages", fake_trim_messages)

    baseline = _make_baseline(fake_llm)
    result = baseline._crop(original, max_tokens=999)

    assert result[:3] == [system, assistant_before_tool, trimmed_tool]


def test_crop_does_nothing_when_not_system_followed_by_tool(
    monkeypatch: pytest.MonkeyPatch, fake_llm: MagicMock
) -> None:
    import src.algorithms.simple_algorithms.dialogue_baseline as baseline_mod

    system = SystemMessage(content="sys")
    user = HumanMessage(content="u")
    tool = ToolMessage(content="tool output", tool_call_id="call_1")

    original = [system, user, tool]
    trimmed = [system, user, tool]

    def fake_trim_messages(msgs, *args, **kwargs):  # noqa: ANN001
        return trimmed

    monkeypatch.setattr(baseline_mod, "trim_messages", fake_trim_messages)

    baseline = _make_baseline(fake_llm)
    result = baseline._crop(original, max_tokens=999)

    assert result == trimmed
