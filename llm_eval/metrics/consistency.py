"""
Consistency Metric
==================
Measures whether an LLM produces semantically stable outputs
when prompted with paraphrased versions of the same question.

At Meta scale this matters because prompt-surface sensitivity
is a reliability signal — inconsistent models fail silently
in downstream applications.
"""
from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from .base import BaseMetric, MetricResult


class ConsistencyMetric(BaseMetric):
    """
    Semantic consistency across multiple responses to equivalent prompts.

    Parameters
    ----------
    model_name : str
        Sentence-transformer model for embedding responses.
    threshold : float
        Minimum mean pairwise cosine similarity to pass.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        threshold: float = 0.85,
    ):
        super().__init__(threshold=threshold, name="consistency")
        self._embedder = SentenceTransformer(model_name)

    def _evaluate(self, responses: Sequence[str]) -> MetricResult:  # type: ignore[override]
        if len(responses) < 2:
            raise ValueError("Need at least 2 responses to measure consistency.")

        embeddings = self._embedder.encode(responses, normalize_embeddings=True)
        sim_matrix = np.dot(embeddings, embeddings.T)

        n = len(responses)
        upper_tri_indices = np.triu_indices(n, k=1)
        pairwise_sims = sim_matrix[upper_tri_indices]

        mean_sim = float(np.mean(pairwise_sims))
        std_sim = float(np.std(pairwise_sims))
        confidence = float(np.clip(1.0 - std_sim, 0.0, 1.0))

        return MetricResult(
            metric_name=self.name,
            score=mean_sim,
            confidence=confidence,
            metadata={
                "mean_similarity": mean_sim,
                "std_similarity": std_sim,
                "num_responses": n,
                "pairwise_similarities": pairwise_sims.tolist(),
                "response_fingerprints": [
                    hashlib.md5(r.encode()).hexdigest()[:8] for r in responses
                ],
            },
        )
