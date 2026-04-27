from __future__ import annotations

import pytest

from langchain_core.embeddings import Embeddings

from src.algorithms.summarize_algorithms.core.memory_storage.memory_storage import (
    MemoryStorage,
)

# IMPORTANT: import models first to avoid circular import issues in `MemoryStorage`.
from src.algorithms.summarize_algorithms.core.models import BaseBlock


class FakeEmbeddings(Embeddings):
    """Deterministic offline embeddings.

    Very small vectors where similarity is driven by keyword presence.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:  # noqa: D401
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:  # noqa: D401
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        t = text.lower()
        return [
            1.0 if "alpha" in t else 0.0,
            1.0 if "beta" in t else 0.0,
            float(len(t) % 7) / 7.0,
        ]


def test_memory_storage_add_and_find_similar_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # MemoryStorage requires OPENAI_API_KEY even when custom embeddings are provided.
    monkeypatch.setenv("OPENAI_API_KEY", "offline")

    storage = MemoryStorage(embeddings=FakeEmbeddings(), max_session_id=10)

    storage.add_memory(
        [
            BaseBlock(role="SYSTEM", content="alpha memory"),
            BaseBlock(role="SYSTEM", content="beta memory"),
        ],
        session_id=0,
    )

    results = storage.find_similar("alpha", top_k=1)
    assert len(results) == 1
    assert "alpha" in results[0].content.lower()
