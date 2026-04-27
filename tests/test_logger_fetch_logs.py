from __future__ import annotations

from pathlib import Path

from src.benchmark.logger.baseline_logger import BaselineLogger
from src.benchmark.logger.memory_logger import MemoryLogger
from src.benchmark.utils.json_log_utils import JsonLogUtils


def _write_baseline_like_log(directory: Path, system_name: str, timestamp: str) -> None:
    JsonLogUtils.write(
        directory / f"{system_name}-{timestamp}.json",
        {
            "timestamp": timestamp,
            "iteration": 1,
            "system": system_name,
            "query": "q",
            "response": {},
            "sessions": [],
            "prepared_messages": [],
        },
    )


def _write_tokens_log(directory: Path, system_name: str, timestamp: str) -> None:
    JsonLogUtils.write(
        directory / f"{system_name}-{timestamp}_tokens.json",
        {
            "model": "gpt-4o-mini",
            "input_tokens": 1,
            "output_tokens": 1,
            "total_price": "0.0001",
        },
    )


def test_baseline_logger_fetch_logs_skips_tokens_json(tmp_path: Path) -> None:
    system_name = "FullBaseline"
    subdirectory = Path("1")
    directory = tmp_path / system_name / subdirectory
    directory.mkdir(parents=True)

    _write_baseline_like_log(directory, system_name, "2026-02-27T01:20:34")
    _write_tokens_log(directory, system_name, "2026-02-27T01:20:34")

    logger = BaselineLogger(logs_dir=tmp_path)
    records = logger.fetch_logs(system_names=[system_name], subdirectory=subdirectory)

    assert len(records) == 1
    assert records[0].timestamp == "2026-02-27T01:20:34"


def test_memory_logger_fetch_logs_skips_tokens_json(tmp_path: Path) -> None:
    system_name = "RagMemoryBank"
    subdirectory = Path("1")
    directory = tmp_path / system_name / subdirectory
    directory.mkdir(parents=True)

    _write_baseline_like_log(directory, system_name, "2026-02-27T01:20:34")
    _write_tokens_log(directory, system_name, "2026-02-27T01:20:34")

    logger = MemoryLogger(logs_dir=tmp_path)
    records = logger.fetch_logs(system_names=[system_name], subdirectory=subdirectory)

    assert len(records) == 1
    assert records[0].timestamp == "2026-02-27T01:20:34"
