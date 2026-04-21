"""
Decision Explainability Metric
================================
Scores whether an agent's stated reasoning adequately supports
its final decision. Designed for agentic pipelines where an
LLM must justify tool calls, policy decisions, or plan steps.
"""
from __future__ import annotations

import re

import numpy as np
from sentence_transformers import SentenceTransformer

from .base import BaseMetric, MetricResult


_HEDGE_PATTERNS = re.compile(
    r"\b(maybe|perhaps|might|could be|unclear|uncertain|not sure|I think|I believe)\b",
    re.IGNORECASE,
)

_CAUSAL_PATTERNS = re.compile(
    r"\b(because|therefore|thus|since|given that|as a result|consequently|due to|hence)\b",
    re.IGNORECASE,
)


class ExplainabilityMetric(BaseMetric):
    """
    Measures reasoning quality of an agent's decision explanation.

    Combines:
    - Semantic alignment between reasoning chain and conclusion
    - Presence of causal connectives
    - Absence of excessive hedging
    - Step coverage (# of reasoning steps identified)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        threshold: float = 0.70,
    ):
        super().__init__(threshold=threshold, name="explainability")
        self._embedder = SentenceTransformer(model_name)

    def _evaluate(  # type: ignore[override]
        self,
        reasoning: str,
        conclusion: str,
    ) -> MetricResult:
        emb = self._embedder.encode(
            [reasoning, conclusion], normalize_embeddings=True
        )
        alignment = float(np.dot(emb[0], emb[1]))

        word_count = max(len(reasoning.split()), 1)
        causal_hits = len(_CAUSAL_PATTERNS.findall(reasoning))
        causal_density = min(causal_hits / (word_count / 100), 1.0)

        hedge_hits = len(_HEDGE_PATTERNS.findall(reasoning))
        hedge_penalty = min(hedge_hits / max(word_count / 50, 1), 0.4)

        step_pattern = re.findall(r"(\d+[.)]\s|\n[-•*]\s)", reasoning)
        step_score = min(len(step_pattern) / 5.0, 1.0)

        score = float(
            0.40 * alignment
            + 0.25 * causal_density
            + 0.20 * step_score
            - 0.15 * hedge_penalty
        )
        score = float(np.clip(score, 0.0, 1.0))
        confidence = float(np.clip(alignment, 0.0, 1.0))

        return MetricResult(
            metric_name=self.name,
            score=score,
            confidence=confidence,
            metadata={
                "alignment_score": alignment,
                "causal_density": causal_density,
                "hedge_penalty": hedge_penalty,
                "step_coverage": step_score,
                "word_count": word_count,
                "causal_connective_count": causal_hits,
                "hedge_word_count": hedge_hits,
            },
        )
