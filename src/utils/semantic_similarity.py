import json

from dataclasses import dataclass
from typing import Any

import numpy as np
import tiktoken

from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class SemanticSimilarityResult:
    precision: float
    recall: float
    f1: float


class SemanticSimilarity:
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
        use_tokenizer: bool = True,
    ) -> None:
        self.embeddings = OpenAIEmbeddings(model=model, chunk_size=batch_size)
        self.batch_size = batch_size
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.use_tokenizer = use_tokenizer

    @staticmethod
    def _to_text(value: Any) -> str:
        """Convert arbitrary JSON-ish values to text for embedding."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, int | float | bool):
            return str(value)
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return str(value)

    def _tokenize(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            return np.array([])

        token_ids = np.array(self.tokenizer.encode(text))
        if len(token_ids) == 0:
            return np.array([])

        decode_func = np.vectorize(
            lambda token_id: self.tokenizer.decode([token_id]),
        )
        tokens = decode_func(token_ids)

        non_empty_mask = np.vectorize(lambda x: bool(x))(tokens)
        return tokens[non_empty_mask]

    def _get_embeddings_batch(self, tokens: np.ndarray) -> np.ndarray:
        unique_tokens, inverse_indices = np.unique(tokens, return_inverse=True)

        embeddings_list = self.embeddings.embed_documents(unique_tokens.tolist())
        embeddings_array = np.array(embeddings_list)

        return embeddings_array[inverse_indices]

    def compute_similarity(
        self, candidate: Any, reference: Any
    ) -> SemanticSimilarityResult:
        if not candidate or not reference:
            return SemanticSimilarityResult(0.0, 0.0, 0.0)
        if self.use_tokenizer:
            cand_tokens = self._tokenize(candidate)
            ref_tokens = self._tokenize(reference)
        else:
            cand_tokens = np.array([candidate])
            ref_tokens = np.array([reference])

        if len(cand_tokens) == 0 or len(ref_tokens) == 0:
            return SemanticSimilarityResult(0.0, 0.0, 0.0)

        cand_embeddings = self._get_embeddings_batch(cand_tokens)
        ref_embeddings = self._get_embeddings_batch(ref_tokens)

        sim_matrix = cosine_similarity(cand_embeddings, ref_embeddings)

        precisions = np.max(sim_matrix, axis=1)
        recalls = np.max(sim_matrix, axis=0)

        precision = np.mean(precisions)
        recall = np.mean(recalls)

        denominator = precision + recall
        f1 = 2 * precision * recall / denominator if denominator != 0 else 0.0

        return SemanticSimilarityResult(
            precision=float(precision), recall=float(recall), f1=float(f1)
        )

    def calculate(self, sentence_a: str, sentence_b: str) -> float:
        """Embed two sentences and return their cosine similarity."""

        sentence_a = sentence_a.strip()
        sentence_b = sentence_b.strip()
        if not sentence_a and not sentence_b:
            return 1.0
        elif not sentence_a or not sentence_b:
            return 0.0

        vecs = self.embeddings.embed_documents([sentence_a, sentence_b])
        vec_a = np.asarray(vecs[0], dtype=float).reshape(1, -1)
        vec_b = np.asarray(vecs[1], dtype=float).reshape(1, -1)
        return float(cosine_similarity(vec_a, vec_b)[0][0])

    def compare_json(self, json_a: dict[str, Any], json_b: dict[str, Any]) -> float:
        """Compare two JSON objects and return the average similarity score.

        Values are coerced to text via `_to_text()` before embedding.
        """

        common_keys = set(json_a.keys()).intersection(
            set(json_b.keys())
        )  # TODO only common keys??
        if not common_keys:
            return 0.0

        similarities: list[float] = []
        for key in common_keys:
            similarity = self.calculate(
                self._to_text(json_a[key]),
                self._to_text(json_b[key]),
            )
            print(similarity)
            similarities.append(similarity)

        return float(np.mean(similarities))
