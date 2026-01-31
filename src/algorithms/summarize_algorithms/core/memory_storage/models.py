from dataclasses import dataclass
from typing import override

from src.algorithms.summarize_algorithms.core.models import BaseBlock, CodeBlock, ToolCallBlock


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
    def from_block(cls, block: ToolCallBlock, session_id: int):
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
    def to_block(self):
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
    def from_block(cls, block: CodeBlock, session_id: int):
        return cls(
            embed_content=block.code,
            content=block.code,
            session_id=session_id,
            code=block.code,
            role=block.role,
        )

    @override
    def to_block(self):
        return CodeBlock(
            role=self.role,
            content="",
            code=self.code,
        )
