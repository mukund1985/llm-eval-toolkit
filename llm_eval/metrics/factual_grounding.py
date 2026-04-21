"""
Factual Grounding Metric
========================
Scores how well a model response is grounded in a provided
reference corpus. Uses entailment-style sentence-level alignment
rather than naive n-gram overlap.
"""
from __future__ import annotations

import re
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from .base import BaseMetric, MetricResult


class FactualGroundingMetric(BaseMetric):
    """
    Measures semantic coverage of claims in the response
    against a reference corpus.

    Strategy: for each sentence in the response, find the
    best-matching reference chunk via cosine similarity.
    Aggregate the top-k alignment scores as the grounding score.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        threshold: float = 0.75,
        chunk_size: int = 2,
    ):
        super().__init__(threshold=threshold, name="factual_grounding")
        self._embedder = SentenceTransformer(model_name)
        self.chunk_size = chunk_size

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s for s in sentences if len(s) > 10]

    def _chunk_reference(self, reference: str) -> list[str]:
        sentences = self._split_sentences(reference)
        chunks = []
        for i in range(0, len(sentences), self.chunk_size):
            chunk = " ".join(sentences[i : i + self.chunk_size])
            chunks.append(chunk)
        return chunks if chunks else [reference]

    def _evaluate(  # type: ignore[override]
        self,
        response: str,
        reference: str,
    ) -> MetricResult:
        response_sentences = self._split_sentences(response)
        reference_chunks = self._chunk_reference(reference)

        if not response_sentences:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                confidence=1.0,
                metadata={"reason": "empty response"},
            )

        resp_embeddings = self._embedder.encode(
            response_sentences, normalize_embeddings=True
        )
        ref_embeddings = self._embedder.encode(
            reference_chunks, normalize_embeddings=True
        )

        sim_matrix = np.dot(resp_embeddings, ref_embeddings.T)
        best_match_per_sentence = sim_matrix.max(axis=1)

        grounding_score = float(np.mean(best_match_per_sentence))
        low_grounding_sentences = [
            response_sentences[i]
            for i, s in enumerate(best_match_per_sentence)
            if s < self.threshold
        ]
        confidence = float(np.clip(1.0 - np.std(best_match_per_sentence), 0.0, 1.0))

        return MetricResult(
            metric_name=self.name,
            score=grounding_score,
            confidence=confidence,
            metadata={
                "num_response_sentences": len(response_sentences),
                "num_reference_chunks": len(reference_chunks),
                "per_sentence_scores": best_match_per_sentence.tolist(),
                "ungrounded_sentences": low_grounding_sentences,
            },
        )
