from dataclasses import dataclass

from typing_extensions import override  # noqa: UP035

from src.algorithms.summarize_algorithms.core.models import (
    BaseBlock,
    CodeBlock,
    ToolCallBlock,
)


@dataclass
class MemoryFragment:
    """Serializable representation of a remembered block used by `MemoryStorage`."""

    embed_content: str
    content: str
    role: str
    session_id: int

    @classmethod
    def from_block(cls, block: BaseBlock, session_id: int) -> "MemoryFragment":
        """
        Create a fragment from a session block.

        :param block: source message block.
        :param session_id: session index.
        :return: MemoryFragment: created fragment.
        """
        return cls(
            embed_content=block.content,
            content=block.content,
            session_id=session_id,
            role=block.role,
        )

    def to_block(self) -> BaseBlock:
        """
        Convert the fragment back to a `BaseBlock` (used when retrieving from the vector store).

        :return: BaseBlock: restored block.
        """
        return BaseBlock(
            role="assistant",
            content=self.content,
        )


@dataclass
class ToolMemoryFragment(MemoryFragment):
    """Specialized fragment for tool calls/responses (keeps tool metadata)."""

    id: str
    name: str
    arguments: str
    response: str

    @override
    @classmethod
    def from_block(
        cls, block: BaseBlock, session_id: int
    ) -> "ToolMemoryFragment":
        if not isinstance(block, ToolCallBlock):
            raise TypeError("ToolMemoryFragment requires a ToolCallBlock")
        return cls(
            embed_content=block.content,
            content=block.content,
            session_id=session_id,
            id=block.id,
            name=block.name,
            arguments=block.arguments,
            response=block.response,
            role=block.role,
        )

    @override
    def to_block(self) -> ToolCallBlock:
        return ToolCallBlock(
            role=self.role,
            content=self.content,
            id=self.id,
            name=self.name,
            arguments=self.arguments,
            response=self.response,
        )


@dataclass
class CodeMemoryFragment(MemoryFragment):
    """Specialized fragment for code blocks (embeds `code`, not `content`)."""

    code: str

    @override
    @classmethod
    def from_block(
        cls, block: BaseBlock, session_id: int
    ) -> "CodeMemoryFragment":
        if not isinstance(block, CodeBlock):
            raise TypeError("CodeMemoryFragment requires a CodeBlock")
        return cls(
            embed_content=block.code,
            content=block.code,
            session_id=session_id,
            code=block.code,
            role=block.role,
        )

    @override
    def to_block(self) -> CodeBlock:
        return CodeBlock(
            role=self.role,
            content="",
            code=self.code,
        )
