import math
import os

from collections.abc import Iterable
from typing import Any

import faiss
import numpy as np

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from src.algorithms.summarize_algorithms.core.memory_storage.models import (
    CodeMemoryFragment,
    MemoryFragment,
    ToolMemoryFragment,
)
from src.algorithms.summarize_algorithms.core.models import (
    BaseBlock,
    CodeBlock,
    ToolCallBlock,
)


class MemoryStorage:
    """
    Vector store for dialogue memory fragments (text/code/tool).

    Stores embeddings in a FAISS inner-product index and keeps the original content in a parallel list.
    Used by dialogue systems to retrieve top-k relevant past fragments for a given query.
    """

    def __init__(
        self,
        embeddings: Embeddings | None = None,
        max_session_id: int = 3,
    ) -> None:
        load_dotenv()

        self.memory_list: list[MemoryFragment] = []
        api_key: str | None = os.getenv("OPENAI_API_KEY")
        if api_key is not None:
            self.embeddings = embeddings or OpenAIEmbeddings(
                model="text-embedding-3-small",
                chunk_size=100,
                api_key=SecretStr(api_key)
            )
        else:
            raise ValueError("OPENAI_API_KEY environment variable is not loaded")

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
        """
        Embed and index new memory blocks for a given session.

        :param memories: blocks to store (text/code/tool). Empty iterables are ignored.
        :param session_id: index of the session these blocks came from (used for time-decay weighting).
        :return: None
        """
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
            if isinstance(memory, ToolCallBlock):
                self.memory_list.append(
                    ToolMemoryFragment.from_block(memory, session_id=session_id)
                )
            elif isinstance(memory, CodeBlock):
                self.memory_list.append(
                    CodeMemoryFragment.from_block(memory, session_id=session_id)
                )
            else:
                self.memory_list.append(
                    MemoryFragment.from_block(memory, session_id=session_id)
                )

    def find_similar(self, query: str, top_k: int = 5) -> list[BaseBlock]:
        """
        Return top-k stored blocks most similar to the query (cosine/IP on normalized vectors).

        :param query: query string to search for.
        :param top_k: maximum number of results to return.
        :return: list[BaseBlock]: retrieved blocks ordered by similarity.
        """
        if self.index is None or len(self.memory_list) == 0:
            return []

        query_embedding = self.embeddings.embed_query(query)
        query_vector = np.array([query_embedding], dtype=np.float32)

        normalized_query = self._normalize_vectors(query_vector)

        indices = self.index.search(
            normalized_query, min(top_k, len(self.memory_list))
        )[1]

        results: list[BaseBlock] = []
        for idx in indices[0]:
            results.append(self.memory_list[idx].to_block())

        return results

    def get_memory_count(self) -> int:
        return len(self.memory_list)

    def get_session_memory(self, session_id: int) -> list[str]:
        """
        Get raw stored contents for a specific session id.

        :param session_id: session index.
        :return: list[str]: stored fragment contents.
        """
        if session_id < 0 or session_id >= self.max_session_id:
            raise ValueError(
                f"Session ID must be between 0 and {self.max_session_id - 1}."
            )

        return [
            fragment.content
            for fragment in self.memory_list
            if fragment.session_id == session_id
        ]

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the storage metadata (memory list + FAISS index info) for logging/debugging.

        :return: dict[str, Any]: JSON-serializable snapshot of the storage.
        """
        return {
            "memory_list": [
                {
                    "embed_content": fragment.embed_content,
                    "content": fragment.content,
                    "session_id": fragment.session_id,
                }
                for fragment in self.memory_list
            ],
            "max_session_id": self.max_session_id,
            "memory_count": len(self.memory_list),
            "is_initialized": self._is_initialized,
            "index_info": {
                "ntotal": int(self.index.ntotal),
                "dimension": int(self.index.d),
            } if self.index is not None else None,
            "embeddings_model": getattr(self.embeddings, "model", str(type(self.embeddings))),
        }
