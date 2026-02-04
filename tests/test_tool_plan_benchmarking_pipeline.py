from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from src.benchmark.logger.baseline_logger import BaselineLogger
from src.benchmark.models.dtos import BaseRecord
from src.benchmark.tool_plan_benchmarking.calculator import Calculator
from src.benchmark.tool_plan_benchmarking.evaluators.f1_tool_evaluator import (
    F1ToolEvaluator,
)


class FakeAlgo:
    system_name = "FakeAlgorithm"

    def process_dialogue(self, sessions, system_prompt, structure=None, tools=None):  # noqa: ANN001
        # Return a state-like object with the minimal surface required by evaluator+logger.
        state = MagicMock()
        state.response = {
            "plan_steps": [
                {"kind": "tool_call", "name": "read_file", "args": {}, "id": "s1", "description": "", "depends_on": []}
            ]
        }
        state.prepared_messages = []
        return state


def test_calculator_end_to_end_logs_metrics(tmp_path: Path) -> None:
    algo = FakeAlgo()
    evaluator = F1ToolEvaluator(mode="simple")
    logger = BaselineLogger(logs_dir=tmp_path)

    sessions = []
    reference = []

    records = Calculator.evaluate(
        algorithms=[algo],
        evaluator_functions=[evaluator],
        sessions=sessions,
        prompt="q",
        reference=reference,
        logger=logger,
        subdirectory=Path("sub"),
        tools=[],
        iteration=1,
    )

    assert len(records) == 1
    assert isinstance(records[0], BaseRecord)
    assert records[0].system == "FakeAlgorithm"

    # logger should have written a json file
    # Note: Calculator passes `algorithm.system_name / subdirectory` to the logger.
    written = list((tmp_path / "FakeAlgorithm" / "sub").glob("FakeAlgorithm-*.json"))
    assert len(written) == 1
