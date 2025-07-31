from unittest.mock import MagicMock, create_autospec

import pytest

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate

from src.recsum.response_generator import ResponseGenerator


@pytest.fixture
def mock_llm():
    return create_autospec(BaseChatModel)


@pytest.fixture
def mock_prompt_template():
    return create_autospec(PromptTemplate)


@pytest.fixture
def response_generator(mock_llm, mock_prompt_template):
    return ResponseGenerator(llm=mock_llm, prompt_template=mock_prompt_template)


def test_initialization(response_generator, mock_llm, mock_prompt_template):
    assert response_generator.llm is mock_llm
    assert response_generator.prompt_template is mock_prompt_template
    assert hasattr(response_generator, "chain")


def test_generate_response_success(response_generator):
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Test response"
    response_generator.chain = mock_chain

    result = response_generator.generate_response(
        memory="Memory content",
        dialogue_context="Dialogue history",
        query="User question",
    )

    assert result == "Test response"
    mock_chain.invoke.assert_called_once_with(
        {
            "latest_memory": "Memory content",
            "current_dialogue_context": "Dialogue history",
            "query": "User question",
        }
    )


@pytest.mark.parametrize(
    "memory, context, query",
    [
        ("", "", ""),
        ("Short", "Long context with details", "Simple query"),
        ("Very long memory " * 20, "Short context", "Complex?" * 10),
        ("Memory", "", "Query"),
    ],
)
def test_argument_combinations(response_generator, memory, context, query):
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Response"
    response_generator.chain = mock_chain

    response_generator.generate_response(memory, context, query)
    mock_chain.invoke.assert_called_once_with(
        {"latest_memory": memory, "current_dialogue_context": context, "query": query}
    )


def test_generate_response_exception(response_generator):
    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = Exception("Network error")
    response_generator.chain = mock_chain

    with pytest.raises(ConnectionError) as exc_info:
        response_generator.generate_response("mem", "ctx", "q")

    assert "API request failed: Network error" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, Exception)
