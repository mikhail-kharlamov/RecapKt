from __future__ import annotations

import json

from pathlib import Path

import src.benchmark.tool_plan_benchmarking.run as run_module

from src.benchmark.tool_plan_benchmarking.statistics.dtos import StatisticsDto


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file)


def test_build_graph_skips_tokens_json(monkeypatch, tmp_path: Path) -> None:
    directory = tmp_path / "FullBaseline" / "1"

    _write_json(
        directory / "FullBaseline-2026-02-27T01:20:34.json",
        {
            "timestamp": "2026-02-27T01:20:34",
            "iteration": 1,
            "system": "FullBaseline",
            "query": "q",
            "response": {},
            "sessions": [],
            "prepared_messages": [],
            "metric": [],
        },
    )
    _write_json(
        directory / "FullBaseline-2026-02-27T01:20:34_tokens.json",
        {
            "model": "gpt-4o-mini",
            "input_tokens": 1,
            "output_tokens": 1,
            "total_price": "0.0001",
        },
    )

    observed: dict[str, int] = {"metrics_count": 0}

    def fake_calculate_by_logs(count_of_launches, metrics, system_logger=None, normalize=False):  # noqa: ANN001
        observed["metrics_count"] = len(metrics)
        return StatisticsDto(algorithms=[])

    class DummyGraph:
        @staticmethod
        def build(stats, output_path, mode):  # noqa: ANN001
            return None

    monkeypatch.setattr(run_module, "LOGS_PATH", tmp_path)
    monkeypatch.setattr(run_module.Statistics, "calculate_by_logs", staticmethod(fake_calculate_by_logs))
    monkeypatch.setattr(run_module.Statistics, "print_statistics", staticmethod(lambda _: None))

    run_module.Runner.build_graph(
        graph_types=[DummyGraph],
        directories=[Path("FullBaseline")],
    )

    assert observed["metrics_count"] == 1
