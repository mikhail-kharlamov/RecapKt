from __future__ import annotations

import json

from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from decimal import Decimal
from enum import Enum
from json import JSONDecodeError
from pathlib import Path
from typing import Any


class JsonLogUtils:
    """Helpers for reading/writing benchmark logs as JSON.

    Key requirements:
    - logs are written pretty-printed (`indent=4`) -> multi-line JSON
    - tolerate legacy files that may contain multiple JSON objects concatenated together
    - handle common non-JSON types (e.g. `Enum`) via a `default=` converter
    """

    _INDENT: int = 4

    @staticmethod
    def iter_json_objects(text: str) -> Iterable[dict[str, Any]]:
        """Iterate JSON objects from a string.

        Supports:
        - a single pretty-printed JSON object (multi-line, e.g. `indent=4`)
        - multiple JSON objects concatenated together

        This intentionally does *not* assume "1 object = 1 line".
        """
        decoder = json.JSONDecoder()
        idx = 0
        length = len(text)

        while True:
            while idx < length and text[idx].isspace():
                idx += 1
            if idx >= length:
                return

            value, end = decoder.raw_decode(text, idx)
            idx = end

            if isinstance(value, dict):
                yield value
                continue

            if isinstance(value, list):
                for item in value:
                    if not isinstance(item, dict):
                        raise ValueError(
                            f"Expected dict items in JSON list, got: {type(item)!r}"
                        )
                    yield item
                continue

            raise ValueError(
                f"Expected JSON object (dict) or list of objects, got: {type(value)!r}"
            )

    @classmethod
    def load_log_payloads(cls, path: Path) -> list[dict[str, Any]]:
        """Load one or more JSON objects from a log file."""
        content = path.read_text(encoding="utf-8")
        if not content.strip():
            return []

        try:
            return list(cls.iter_json_objects(content))
        except JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON log file: {path}") from e

    @staticmethod
    def json_default(obj: Any) -> Any:
        """Fallback conversion for non-JSON-serializable objects."""
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, Decimal):
            # Preserve precision in logs; values are read back as JSON numbers.
            return str(obj)
        if isinstance(obj, Path):
            return str(obj)
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)
        if hasattr(obj, "model_dump"):
            return obj.model_dump(mode="json")
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        raise TypeError(
            f"Object of type {obj.__class__.__name__} is not JSON serializable"
        )

    @classmethod
    def dumps(cls, data: Any) -> str:
        return json.dumps(
            data, ensure_ascii=False, indent=cls._INDENT, default=cls.json_default
        )

    @classmethod
    def write(cls, path: Path, data: Any) -> None:
        path.write_text(cls.dumps(data) + "\n", encoding="utf-8")
