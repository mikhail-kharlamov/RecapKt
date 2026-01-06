import json

import pytest

from src.benchmarking.memory_logger import MemoryLogger
from src.benchmarking.models.dtos import MetricState
from src.benchmarking.models.enums import MetricType
from src.summarize_algorithms.core.models import (
    BaseBlock,
    DialogueState,
    Session,
)


class FakeStorage:
    def __init__(self, name: str):
        self.name = name

    def to_dict(self):
        return {"storage_name": self.name}


@pytest.fixture
def fake_state():
    s = DialogueState(
        dialogue_sessions=[],
        code_memory_storage=FakeStorage("code"),
        tool_memory_storage=FakeStorage("tool"),
        query="Test query",
    )
    s._response = "Test response"
    s.text_memory = [["memory line 1", "memory line 2"]]
    return s


@pytest.fixture
def sessions():
    return [Session([BaseBlock(role="USER", content="Hello there!")])]


def test_log_iteration_creates_file(tmp_path, fake_state, sessions):
    logger = MemoryLogger(logs_dir=tmp_path)
    metric = MetricState(metric=MetricType.COHERENCE, value=0.87)

    record = logger.log_iteration(
        system_name="FakeSystem",
        query="Hello?",
        iteration=1,
        sessions=sessions,
        state=fake_state,
        metric=metric
    )

    assert isinstance(record, dict)
    assert record["system"] == "FakeSystem"
    assert record["iteration"] == 1
    assert record["query"] == "Hello?"
    assert record["response"] == "Test response"
    assert "metric_name" in record and abs(record["metric_value"] - 0.87) < 0.0001

    expected_file = tmp_path / "FakeSystem1.jsonl"
    assert expected_file.exists()

    content = expected_file.read_text(encoding="utf-8").strip()
    parsed = json.loads(content)
    assert parsed["system"] == "FakeSystem"
    assert parsed["query"] == "Hello?"
    assert "sessions" in parsed
    assert isinstance(parsed["sessions"], list)
    assert parsed["sessions"][0]["messages"][0]["content"] == "Hello there!"


def test_log_iteration_without_metric(tmp_path, fake_state, sessions):
    logger = MemoryLogger(logs_dir=tmp_path)

    record = logger.log_iteration(
        system_name="SystemNoMetric",
        query="No metric case",
        iteration=2,
        sessions=sessions,
        state=fake_state,
        metric=None
    )

    assert "metric_name" not in record
    assert "metric_value" not in record

    expected_file = tmp_path / "SystemNoMetric2.jsonl"
    assert expected_file.exists()
