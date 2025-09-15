import math

from dataclasses import dataclass
from typing import Iterable, Optional

import faiss
import numpy as np

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from src.summarize_algorithms.core.models import BaseBlock, CodeBlock


@dataclass
class MemoryFragment:
    embed_content: str
    content: str
    session_id: int


class MemoryStorage:
    def __init__(
        self,
        embeddings: Optional[Embeddings] = None,
        max_session_id: int = 3,
    ) -> None:
        self.memory_list: list[MemoryFragment] = []
        self.embeddings = embeddings or OpenAIEmbeddings(
            model="text-embedding-3-small", chunk_size=100
        )
        self.max_session_id = max_session_id
        self.index = None
        self._is_initialized = False

    def _initialize_index(self, dimension: int) -> None:
        if self._is_initialized:
            return
        self.index = faiss.IndexFlatIP(dimension)
        self._is_initialized = True

    @staticmethod
    def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms

    def add_memory(self, memories: Iterable[BaseBlock], session_id: int) -> None:
        if not memories:
            return

        memory_embed_contents = [block.content for block in memories]

        embeddings_list = self.embeddings.embed_documents(memory_embed_contents)
        embeddings_array = np.array(embeddings_list, dtype=np.float32)

        self._initialize_index(embeddings_array.shape[1])

        if self.index is None:
            raise ValueError("Index has not been initialized.")

        normalized_embeddings = self._normalize_vectors(embeddings_array)

        importance = math.exp(-0.2 * (1 - (session_id + 1 / self.max_session_id + 1)))

        weighted_embeddings = normalized_embeddings * importance

        self.index.add(weighted_embeddings)

        for memory in memories:
            if isinstance(memory, CodeBlock):
                content = memory.code
            else:
                content = memory.content

            self.memory_list.append(
                MemoryFragment(
                    embed_content=memory.content, content=content, session_id=session_id
                )
            )

    def find_similar(self, query: str, top_k: int = 5) -> list[str]:
        if self.index is None or len(self.memory_list) == 0:
            return []

        query_embedding = self.embeddings.embed_query(query)
        query_vector = np.array([query_embedding], dtype=np.float32)

        normalized_query = self._normalize_vectors(query_vector)

        indices = self.index.search(
            normalized_query, min(top_k, len(self.memory_list))
        )[1]

        results = []
        for idx in indices[0]:
            results.append(self.memory_list[idx].content)

        return results

    def get_memory_count(self) -> int:
        return len(self.memory_list)

    def get_session_memory(self, session_id: int) -> list[str]:
        if session_id < 0 or session_id >= self.max_session_id:
            raise ValueError(
                f"Session ID must be between 0 and {self.max_session_id - 1}."
            )

        return [
            fragment.content
            for fragment in self.memory_list
            if fragment.session_id == session_id
        ]
