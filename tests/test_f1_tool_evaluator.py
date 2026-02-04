from __future__ import annotations

import json

import pytest

from src.algorithms.summarize_algorithms.core.models import (
    BaseBlock,
    DialogueState,
    Session,
    ToolCallBlock,
)
from src.benchmark.models.enums import MetricType
from src.benchmark.tool_plan_benchmarking.evaluators.f1_tool_evaluator import (
    F1ToolEvaluator,
)


def _state_with_plan(tool_calls: list[tuple[str, dict]]) -> DialogueState:
    state = DialogueState(
        dialogue_sessions=[],
        prepared_messages=[],
        code_memory_storage=None,
        tool_memory_storage=None,
        query="q",
    )
    state._response = {
        "plan_steps": [
            {"kind": "tool_call", "name": name, "args": args, "id": f"s{i}", "description": "", "depends_on": []}
            for i, (name, args) in enumerate(tool_calls, start=1)
        ]
    }
    return state


def _reference(*names_and_args: tuple[str, dict]) -> list[BaseBlock]:
    blocks: list[BaseBlock] = []
    for i, (name, args) in enumerate(names_and_args, start=1):
        blocks.append(
            ToolCallBlock(
                role="ASSISTANT",
                id=f"t{i}",
                name=name,
                arguments=json.dumps(args),
                response="",
                content="",
            )
        )
    return blocks


def test_f1_simple_counts_tool_names() -> None:
    evaluator = F1ToolEvaluator(mode="simple")
    state = _state_with_plan([("list_dir", {}), ("read_file", {})])
    ref = _reference(("read_file", {"a": 1}), ("search_for_text", {}))

    metric = evaluator.evaluate([Session([])], "q", state, ref)

    assert metric.metric_name == MetricType.F1_TOOL
    # predicted={list_dir,read_file}, reference={read_file,search_for_text} => tp=1 fp=1 fn=1 => f1=0.5
    assert metric.metric_value == pytest.approx(0.5)


def test_f1_strict_requires_args_match() -> None:
    evaluator = F1ToolEvaluator(mode="strict")
    state = _state_with_plan([("read_file", {"path": "a"})])
    ref = _reference(("read_file", {"path": "b"}))

    metric = evaluator.evaluate([Session([])], "q", state, ref)

    assert metric.metric_name == MetricType.F1_TOOL_STRICT
    assert metric.metric_value == 0.0
