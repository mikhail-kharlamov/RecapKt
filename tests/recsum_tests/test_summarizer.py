from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from src.summarize_algorithms.recsum.summarizer import RecursiveSummarizer


@dataclass
class FragmentMemory:
    summary_messages: list[str]


@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def mock_prompt_template():
    return MagicMock()


@pytest.fixture
def summarizer(mock_llm, mock_prompt_template):
    return RecursiveSummarizer(llm=mock_llm, prompt=mock_prompt_template)


def test_initialization(summarizer, mock_llm, mock_prompt_template):
    assert summarizer.llm is mock_llm
    assert hasattr(summarizer, "chain")


def test_summarize_success(summarizer):
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = FragmentMemory(["Test summary"])
    summarizer.chain = mock_chain

    result = summarizer.summarize("Previous memory", "Dialogue context")

    assert result == ["Test summary"]
    mock_chain.invoke.assert_called_once_with(
        {"previous_memory": "Previous memory", "dialogue_context": "Dialogue context"}
    )


def test_summarize_exception(summarizer):
    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = Exception("API error")
    summarizer.chain = mock_chain

    with pytest.raises(ConnectionError) as exc_info:
        summarizer.summarize("Mem", "Context")

    assert "API request failed: API error" in str(exc_info.value)


@pytest.mark.parametrize(
    "memory, context",
    [
        ("", ""),
        ("Long previous memory", "Short context"),
        ("Memory", "Long dialogue context with details"),
    ],
)
def test_argument_passing(summarizer, memory, context):
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = FragmentMemory(["Summary"])
    summarizer.chain = mock_chain

    summarizer.summarize(memory, context)
    mock_chain.invoke.assert_called_once_with(
        {"previous_memory": memory, "dialogue_context": context}
    )
