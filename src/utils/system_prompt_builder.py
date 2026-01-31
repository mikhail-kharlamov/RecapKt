import json

from dataclasses import dataclass
from typing import Any

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape


@dataclass(frozen=True)
class MemorySections:
    """Optional memory sections to inject between introduction and tool/schema blocks."""

    recap: str | None = None
    memory_bank: str | None = None
    code_knowledge: str | None = None
    tool_memory: str | None = None


class SystemPromptBuilder:
    """Builds a single unified system prompt using repository Jinja2 templates."""

    def __init__(self) -> None:
        templates_dir = Path(__file__).resolve().parents[1] / "prompt_templates"
        self._env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=select_autoescape(disabled_extensions=("j2",)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def build(
            self,
            *,
            schema: dict[str, Any] | None,
            memory: MemorySections,
            memory_mode: str,
            examples: str = "",
    ) -> str:
        """
        Build the unified system prompt in the required order.

        Order:
        1) introduction.j2
        2) memory blocks (conditional)
        3) schema_and_tool.j2
        4) bridge_to_conversation.j2

        :param schema: JSON schema for model output (structured output).
        :param memory: optional memory sections.
        :param memory_mode: "baseline" or "memory" (affects MemoryArtifacts description).
        :param examples: optional examples block.
        :return: str: rendered system prompt.
        """
        intro = self._env.get_template("introduction.j2").render().strip()

        memory_text = self._env.get_template("memory_injection.j2").render(
            recap=memory.recap,
            memory_bank=memory.memory_bank,
            code_knowledge=memory.code_knowledge,
            tool_memory=memory.tool_memory,
        ).strip()

        schema_json = json.dumps(schema or {}, ensure_ascii=False, indent=4)

        schema_and_tool = self._env.get_template("schema_and_tool.j2").render(
            schema_json=schema_json,
            memory_mode=memory_mode,
            examples=examples,
        ).strip()

        bridge = self._env.get_template("bridge_to_conversation.j2").render().strip()

        parts = [intro]
        if memory_text != "":
            parts.append(memory_text)
        parts.extend([schema_and_tool, bridge])

        return "\n\n".join(parts).strip()
