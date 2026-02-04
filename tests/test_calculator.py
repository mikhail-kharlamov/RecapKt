from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.algorithms.dialogue import Dialogue
from src.algorithms.summarize_algorithms.core.models import (
    BaseBlock,
    DialogueState,
    Session,
)
from src.benchmark.models.dtos import BaseRecord, MetricState
from src.benchmark.models.enums import MetricType
from src.benchmark.tool_plan_benchmarking.calculator import Calculator


@pytest.fixture
def fake_logger():
    logger = MagicMock()

    logger.log_iteration.return_value = BaseRecord(
        timestamp=datetime.now().isoformat(),
        iteration=1,
        system="FakeAlgo",
        query="What is AI?",
        response={"some": "response"},
        sessions=[],
        prepared_messages=[],
        metric=[MetricState(metric_name=MetricType("COHERENCE"), metric_value=0.95)]
    )
    return logger


@pytest.fixture
def fake_evaluator():
    evaluator = MagicMock()
    evaluator.evaluate.return_value = MetricState(
        metric_name=MetricType("COHERENCE"),
        metric_value=0.95
    )
    return evaluator


@pytest.fixture
def fake_algorithm():
    algo = MagicMock(spec=Dialogue)
    algo.system_name = "FakeAlgorithm"

    fake_state = DialogueState(
        dialogue_sessions=[],
        code_memory_storage=None,
        tool_memory_storage=None,
        query="What is AI?",
        _response={"some": "response"},
        prepared_messages=[]
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
    results = Calculator.evaluate(
        algorithms=[fake_algorithm],
        evaluator_functions=[fake_evaluator],
        sessions=sessions[:1],
        reference=reference_session.messages,
        logger=fake_logger,
        prompt="What is AI?",
        subdirectory=Path("test_subdir"),
        iteration=1
    )

    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], BaseRecord)

    assert results[0].system == "FakeAlgo"
    assert results[0].iteration == 1
    assert results[0].query == "What is AI?"
    assert isinstance(results[0].metric, list)
    assert len(results[0].metric) > 0
    assert results[0].metric[0].metric_value == 0.95

    fake_algorithm.process_dialogue.assert_called_once()
    fake_evaluator.evaluate.assert_called_once()

    logger_call_args = fake_logger.log_iteration.call_args

    assert logger_call_args[0][0] == "FakeAlgorithm"
    assert logger_call_args[0][6][0].metric_value == 0.95
