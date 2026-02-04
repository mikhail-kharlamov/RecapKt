from unittest.mock import MagicMock, create_autospec

import pytest

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.algorithms.summarize_algorithms.core.models import (
    BaseBlock,
    ResponseContext,
    Session,
)
from src.algorithms.summarize_algorithms.core.response_generator import (
    ResponseGenerator,
)
from src.utils.system_prompt_builder import MemorySections


@pytest.fixture
def mock_llm():
    llm = create_autospec(BaseChatModel)
    llm.get_num_tokens_from_messages.return_value = 10
    return llm


@pytest.fixture
def response_generator(mock_llm):
    return ResponseGenerator(llm=mock_llm)


@pytest.fixture
def empty_session():
    return Session([])


def test_initialization(response_generator, mock_llm):
    assert response_generator._llm is mock_llm
    assert hasattr(response_generator, "_chain")


def test_generate_response_success(response_generator, empty_session):
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Test response"
    response_generator._chain = mock_chain

    last_session = Session([BaseBlock(role="USER", content="Hi")])

    result = response_generator.generate_response(
        last_session=last_session,
        user_query="User question",
        memory=MemorySections(recap="Some memory"),
    )

    assert isinstance(result, ResponseContext)
    assert result.response == "Test response"
    assert isinstance(result.prepared_history, list)

    mock_chain.invoke.assert_called_once()
    history = mock_chain.invoke.call_args[0][0]

    assert isinstance(history, list)

    assert isinstance(history[0], SystemMessage)
    assert "The System Instruction ends here" in str(history[0].content)

    assert isinstance(history[-1], HumanMessage)
    assert history[-1].content == "User question"


def test_generate_response_exception(response_generator, empty_session):
    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = Exception("Network error")
    response_generator._chain = mock_chain

    with pytest.raises(ConnectionError) as exc_info:
            response_generator.generate_response(
                last_session=empty_session,
                user_query="q",
                memory=MemorySections(),
            )

    assert "API request failed: Network error" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, Exception)


def test_history_structure(response_generator):
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "resp"
    response_generator._chain = mock_chain

    last_ses = Session([BaseBlock(role="ASSISTANT", content="Prev answer")])

    result = response_generator.generate_response(
        last_session=last_ses,
        user_query="New query",
        memory=MemorySections(recap="Memory info"),
    )

    history = result.prepared_history

    assert "Memory info" in str(history[0].content)
    assert history[-1].content == "New query"
