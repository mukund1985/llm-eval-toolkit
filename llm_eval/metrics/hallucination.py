"""
Hallucination Detection Metric
================================
Detects hallucinated claims in an LLM response by cross-checking
sentence-level assertions against a grounded knowledge context.

Distinguishes three categories:
  - GROUNDED     : claim supported by context
  - UNSUPPORTED  : claim not contradicted but lacks grounding
  - HALLUCINATED : claim contradicts context
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from .base import BaseMetric, MetricResult


class ClaimStatus(str, Enum):
    GROUNDED = "grounded"
    UNSUPPORTED = "unsupported"
    HALLUCINATED = "hallucinated"


@dataclass
class ClaimEvaluation:
    claim: str
    status: ClaimStatus
    best_context_match: str
    similarity: float


class HallucinationMetric(BaseMetric):
    """
    Sentence-level hallucination detection.

    A claim is considered hallucinated when its semantic similarity
    to the best-matching context chunk is below `hallucination_threshold`
    AND the claim makes a factual assertion (not a question or opinion).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        grounding_threshold: float = 0.70,
        hallucination_threshold: float = 0.40,
        threshold: float = 0.80,
    ):
        super().__init__(threshold=threshold, name="hallucination")
        self._embedder = SentenceTransformer(model_name)
        self.grounding_threshold = grounding_threshold
        self.hallucination_threshold = hallucination_threshold

    @staticmethod
    def _is_factual_claim(sentence: str) -> bool:
        s = sentence.strip()
        if s.endswith("?") or len(s.split()) < 5:
            return False
        opinion_starts = ("i think", "in my opinion", "i believe", "it seems")
        return not any(s.lower().startswith(o) for o in opinion_starts)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        return [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", text)
            if len(s.strip()) > 10
        ]

    def _evaluate(  # type: ignore[override]
        self,
        response: str,
        context: str | Sequence[str],
    ) -> MetricResult:
        if isinstance(context, str):
            context_chunks = self._split_sentences(context)
        else:
            context_chunks = list(context)

        response_sentences = self._split_sentences(response)
        factual_claims = [s for s in response_sentences if self._is_factual_claim(s)]

        if not factual_claims:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                confidence=0.5,
                metadata={"reason": "no factual claims detected"},
            )

        claim_embs = self._embedder.encode(factual_claims, normalize_embeddings=True)
        ctx_embs = self._embedder.encode(context_chunks, normalize_embeddings=True)
        sim_matrix = np.dot(claim_embs, ctx_embs.T)

        evaluations: list[ClaimEvaluation] = []
        for i, claim in enumerate(factual_claims):
            best_idx = int(np.argmax(sim_matrix[i]))
            best_sim = float(sim_matrix[i, best_idx])

            if best_sim >= self.grounding_threshold:
                status = ClaimStatus.GROUNDED
            elif best_sim <= self.hallucination_threshold:
                status = ClaimStatus.HALLUCINATED
            else:
                status = ClaimStatus.UNSUPPORTED

            evaluations.append(
                ClaimEvaluation(
                    claim=claim,
                    status=status,
                    best_context_match=context_chunks[best_idx],
                    similarity=best_sim,
                )
            )

        counts = {s: 0 for s in ClaimStatus}
        for ev in evaluations:
            counts[ev.status] += 1

        grounded_fraction = counts[ClaimStatus.GROUNDED] / len(evaluations)
        hallucinated_fraction = counts[ClaimStatus.HALLUCINATED] / len(evaluations)
        score = float(np.clip(grounded_fraction - hallucinated_fraction * 0.5, 0.0, 1.0))
        confidence = float(np.clip(1.0 - np.std([e.similarity for e in evaluations]), 0.0, 1.0))

        return MetricResult(
            metric_name=self.name,
            score=score,
            confidence=confidence,
            metadata={
                "total_claims": len(evaluations),
                "grounded": counts[ClaimStatus.GROUNDED],
                "unsupported": counts[ClaimStatus.UNSUPPORTED],
                "hallucinated": counts[ClaimStatus.HALLUCINATED],
                "hallucinated_claims": [
                    {"claim": e.claim, "best_match_sim": e.similarity}
                    for e in evaluations
                    if e.status == ClaimStatus.HALLUCINATED
                ],
            },
        )
