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
