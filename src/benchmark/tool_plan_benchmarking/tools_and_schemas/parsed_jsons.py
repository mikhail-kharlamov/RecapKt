import json
import os

from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def _resolve_tools_and_schemas_path() -> Path:
    configured_path = os.getenv("TOOLS_AND_SCHEMAS_PATH")
    if configured_path:
        candidate = Path(configured_path)
        if candidate.exists():
            return candidate

    return Path(__file__).resolve().parent


tools_and_schemas_path = _resolve_tools_and_schemas_path()

TOOLS: list[dict[str, Any]] = json.loads(
    (tools_and_schemas_path / "tools.json").read_text(encoding="utf-8")
)

PLAN_SCHEMA = {
    "title": "ActionPlan",
    "description": "Schema for the structured action plan output from the agent",
    "type": "object",
    "additionalProperties": False,
    "required": ["version", "user_request", "memory_used", "assumptions", "plan_steps"],
    "properties": {
        "version": {"type": "string", "enum": ["1.0"]},
        "user_request": {"type": "string"},
        "context_summary": {"type": "string"},
        "memory_used": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["source", "key", "excerpt"],
                "properties": {
                    "source": {
                        "type": "string",
                        "enum": ["session", "memory_bank", "tool_result", "note"],
                    },
                    "key": {"type": "string"},
                    "excerpt": {"type": "string"},
                },
            },
        },
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
        "plan_steps": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "kind", "name", "description", "depends_on"],
                "properties": {
                    "id": {"type": "string", "pattern": "^s[0-9]+$"},
                    "kind": {"type": "string", "enum": ["tool_call", "final_answer"]},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "string", "pattern": "^s[0-9]+$"},
                    },
                    "condition": {"type": "string"},
                    "expected": {"type": "string"},
                    "args": {
                        "type": "object",
                        "description": "For kind=tool_call only. Omit for other kinds.",
                        "additionalProperties": True,
                    },
                    "answer_template": {
                        "type": "string",
                        "description": "For kind=final_answer only. Omit otherwise.",
                    },
                    "collect_from": {
                        "type": "array",
                        "description": "For kind=final_answer: step ids whose outputs feed the answer.",
                        "items": {"type": "string", "pattern": "^s[0-9]+$"},
                    },
                },
            },
        },
    },
}
