from __future__ import annotations

import importlib

from pathlib import Path

from src.benchmark.tool_plan_benchmarking.tools_and_schemas import parsed_jsons


def test_parsed_jsons_falls_back_to_module_directory(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.delenv("TOOLS_AND_SCHEMAS_PATH", raising=False)

    reloaded = importlib.reload(parsed_jsons)

    assert reloaded.tools_and_schemas_path == Path(reloaded.__file__).resolve().parent
    assert reloaded.TOOLS
    assert reloaded.PLAN_SCHEMA["title"] == "ActionPlan"
