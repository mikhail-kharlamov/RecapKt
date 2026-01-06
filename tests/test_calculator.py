from unittest.mock import MagicMock

import pytest

from src.benchmarking.models.dtos import MetricState
from src.benchmarking.models.enums import MetricType
from src.benchmarking.tool_metrics.calculator import Calculator
from src.summarize_algorithms.core.models import (
    BaseBlock,
    DialogueState,
    Session,
)


@pytest.fixture
def fake_logger():
    logger = MagicMock()
    logger.log_iteration.return_value = {"logged": True, "system_name": "FakeAlgo"}
    return logger


@pytest.fixture
def fake_evaluator():
    evaluator = MagicMock()
    evaluator.evaluate.return_value = MetricState(metric=MetricType.COHERENCE, value=0.95)
    return evaluator


@pytest.fixture
def fake_algorithm():
    algo = MagicMock()
    algo.__class__.__name__ = "FakeAlgorithm"
    fake_state = DialogueState(
        dialogue_sessions=[],
        code_memory_storage=None,
        tool_memory_storage=None,
        query="What is AI?",
    )
    algo.process_dialogue.return_value = fake_state
    return algo


@pytest.fixture
def sessions():
    return [
        Session([BaseBlock(role="USER", content="Hello")]),
        Session([BaseBlock(role="USER", content="What is AI?")]),
        Session([BaseBlock(role="ASSISTANT", content="Artificial intelligence")]),
    ]


@pytest.fixture
def reference_session():
    return Session([
        BaseBlock(role="USER", content="Tell me about AI."),
        BaseBlock(role="ASSISTANT", content="AI stands for artificial intelligence.")
    ])


def test_evaluate_success(fake_logger, fake_evaluator, fake_algorithm, sessions, reference_session):
    calc = Calculator()

    results = calc.evaluate(
        algorithms=[fake_algorithm],
        evaluator_functions=[fake_evaluator],
        sessions=sessions[:1],
        reference=reference_session.messages,
        logger=fake_logger,
        prompt="What is AI?",
    )

    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]["logged"]

    fake_algorithm.process_dialogue.assert_called_once()
    fake_evaluator.evaluate.assert_called_once()
    fake_logger.log_iteration.assert_called_once()

    query_passed = fake_algorithm.process_dialogue.call_args[0][1]
    assert query_passed == "What is AI?"
