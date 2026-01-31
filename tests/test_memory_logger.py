import json

from pathlib import Path

import pytest

from src.benchmarking.memory_logger import MemoryLogger
from src.benchmarking.models.dtos import MemoryRecord, MetricState
from src.benchmarking.models.enums import MetricType
from src.algorithms.summarize_algorithms.core.models import (
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
        prepared_messages=[]
    )
    s._response = "Test response"
    s.text_memory = [["memory line 1", "memory line 2"]]
    return s


@pytest.fixture
def sessions():
    return [Session([BaseBlock(role="USER", content="Hello there!")])]


def test_log_iteration_creates_file(tmp_path, fake_state, sessions):
    logger = MemoryLogger(logs_dir=tmp_path)

    metric_list = [MetricState(metric_name=MetricType("COHERENCE"), metric_value=0.87)]

    subdir = Path("test_runs")

    record = logger.log_iteration(
        system_name="FakeSystem",
        query="Hello?",
        iteration=1,
        sessions=sessions,
        state=fake_state,
        metrics=metric_list,
        subdirectory=subdir
    )

    assert isinstance(record, MemoryRecord)
    assert record.system == "FakeSystem"
    assert record.iteration == 1
    assert record.query == "Hello?"
    assert record.response == "Test response"

    assert record.metric is not None
    assert len(record.metric) == 1

    first_metric = record.metric[0]
    if isinstance(first_metric, dict):
        assert abs(first_metric["metric_value"] - 0.87) < 0.0001
    else:
        assert abs(first_metric.metric_value - 0.87) < 0.0001

    expected_dir = tmp_path / subdir
    assert expected_dir.exists()

    files = list(expected_dir.glob("FakeSystem-*.json"))
    assert len(files) == 1
    log_file = files[0]

    content = log_file.read_text(encoding="utf-8").strip()
    parsed = json.loads(content)

    assert parsed["system"] == "FakeSystem"
    assert parsed["query"] == "Hello?"
    assert "sessions" in parsed
    assert isinstance(parsed["sessions"], list)
    assert parsed["sessions"][0]["messages"][0]["content"] == "Hello there!"
    assert parsed["metric"][0]["metric_name"] == "COHERENCE"


def test_log_iteration_without_metric(tmp_path, fake_state, sessions):
    logger = MemoryLogger(logs_dir=tmp_path)
    subdir = Path("no_metric_runs")

    record = logger.log_iteration(
        system_name="SystemNoMetric",
        query="No metric case",
        iteration=2,
        sessions=sessions,
        state=fake_state,
        metrics=None,
        subdirectory=subdir
    )

    assert record.metric is None or record.metric == []

    expected_dir = tmp_path / subdir
    files = list(expected_dir.glob("SystemNoMetric-*.json"))
    assert len(files) == 1

    parsed = json.loads(files[0].read_text(encoding="utf-8"))
    assert "metric" not in parsed or parsed["metric"] is None
