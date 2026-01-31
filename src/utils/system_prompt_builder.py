import json

from dataclasses import dataclass
from typing import Any

from jinja2 import Environment, PackageLoader, select_autoescape


@dataclass(frozen=True)
class MemorySections:
    """Optional memory sections to inject between introduction and tool/schema blocks."""

    recap: str | None = None
    memory_bank: str | None = None
    code_knowledge: str | None = None

    def to_blocks(self) -> list[str]:
        blocks: list[str] = []
        if self.recap is not None and self.recap.strip() != "":
            blocks.append(f"### RECAP:\n{self.recap.strip()}")
        if self.memory_bank is not None and self.memory_bank.strip() != "":
            blocks.append(f"### MEMORY BANK:\n{self.memory_bank.strip()}")
        if self.code_knowledge is not None and self.code_knowledge.strip() != "":
            blocks.append(f"### CODE KNOWLEDGE:\n{self.code_knowledge.strip()}")
        return blocks


class SystemPromptBuilder:
    """Builds a single unified system prompt using repository Jinja2 templates."""

    def __init__(self) -> None:
        self._env = Environment(
            loader=PackageLoader("src.prompt_templates"),
            autoescape=select_autoescape(disabled_extensions=("j2",)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def build(
            self,
            *,
            tools_catalog: list[dict[str, Any]] | None,
            schema: dict[str, Any] | None,
            memory: MemorySections,
            memory_artifacts_note: str,
            examples: str = "",
    ) -> str:
        """
        Build the unified system prompt in the required order.

        Order:
        1) introduction.j2
        2) memory blocks (conditional)
        3) schema_and_tool.j2
        4) bridge_to_conversation.j2

        :param tools_catalog: tools available for planning.
        :param schema: JSON schema for model output (structured output).
        :param memory: optional memory sections.
        :param memory_artifacts_note: short description of what MemoryArtifacts means for this agent.
        :param examples: optional examples block.
        :return: str: rendered system prompt.
        """
        intro = self._env.get_template("introduction.j2").render()

        memory_blocks = memory.to_blocks()
        memory_text = ("\n\n" + "\n\n".join(memory_blocks)) if memory_blocks else ""

        tools_catalog_json = json.dumps(tools_catalog or [], ensure_ascii=False, indent=4)
        schema_json = json.dumps(schema or {}, ensure_ascii=False, indent=4)

        schema_and_tool = self._env.get_template("schema_and_tool.j2").render(
            tools_catalog_json=tools_catalog_json,
            schema_json=schema_json,
            memory_artifacts_note=memory_artifacts_note,
            examples=examples,
        )

        bridge = self._env.get_template("bridge_to_conversation.j2").render()

        return "\n".join([intro.rstrip(), memory_text.strip(), schema_and_tool.strip(), bridge.strip()]).strip()
