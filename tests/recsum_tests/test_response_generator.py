from unittest.mock import MagicMock, create_autospec

import pytest

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate

from src.summarize_algorithms.core.response_generator import ResponseGenerator


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
        dialogue_memory="Memory content",
        code_memory="Code memory",
        tool_memory="Tool memory",
        query="User question",
    )

    assert result == "Test response"
    mock_chain.invoke.assert_called_once_with(
        {
            "dialogue_memory": "Memory content",
            "code_memory": "Code memory",
            "tool_memory": "Tool memory",
            "query": "User question",
        }
    )


@pytest.mark.parametrize(
    "dialogue_memory,code_memory,tool_memory,query",
    [
        ("", "", "", ""),
        ("Short", "Short Code", "Short tool", "Simple query"),
        ("Very long memory " * 20, "Short Code", "Medium Tool" * 5, "Complex?" * 10),
        ("Memory", "", "", "Query"),
    ],
)
def test_argument_combinations(
    dialogue_memory, code_memory, tool_memory, query, response_generator
):
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Response"
    response_generator.chain = mock_chain

    response_generator.generate_response(
        dialogue_memory, code_memory, tool_memory, query
    )
    mock_chain.invoke.assert_called_once_with(
        {
            "dialogue_memory": dialogue_memory,
            "code_memory": code_memory,
            "tool_memory": tool_memory,
            "query": query,
        }
    )


def test_generate_response_exception(response_generator):
    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = Exception("Network error")
    response_generator.chain = mock_chain

    with pytest.raises(ConnectionError) as exc_info:
        response_generator.generate_response("dmem", "cmem", "tmem", "q")

    assert "API request failed: Network error" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, Exception)
