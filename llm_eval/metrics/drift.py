"""
Response Drift Metric
======================
Tracks semantic drift in LLM responses over time or across model versions.

Use-cases:
  - Detect silent regressions after fine-tune or RLHF update
  - Track topic drift in long agentic conversations
  - Compare A/B model outputs for consistency guarantees
"""
from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from .base import BaseMetric, MetricResult


@dataclass
class DriftWindow:
    embeddings: deque = field(default_factory=lambda: deque(maxlen=50))
    texts: deque = field(default_factory=lambda: deque(maxlen=50))


class DriftMetric(BaseMetric):
    """
    Measures semantic drift relative to a baseline distribution
    or across a temporal sequence of responses.

    Two modes:
    - snapshot : compare a single new response to a baseline mean embedding
    - sequence : measure trajectory drift across a response sequence
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        threshold: float = 0.85,
        window_size: int = 20,
    ):
        super().__init__(threshold=threshold, name="drift")
        self._embedder = SentenceTransformer(model_name)
        self._window = DriftWindow(
            embeddings=deque(maxlen=window_size),
            texts=deque(maxlen=window_size),
        )

    def update_baseline(self, responses: Sequence[str]) -> None:
        embs = self._embedder.encode(list(responses), normalize_embeddings=True)
        for text, emb in zip(responses, embs):
            self._window.texts.append(text)
            self._window.embeddings.append(emb)

    def _baseline_centroid(self) -> np.ndarray | None:
        if not self._window.embeddings:
            return None
        return np.mean(np.stack(list(self._window.embeddings)), axis=0)

    def _evaluate(  # type: ignore[override]
        self,
        response: str | Sequence[str],
        mode: str = "snapshot",
    ) -> MetricResult:
        if mode == "snapshot":
            return self._snapshot_eval(response if isinstance(response, str) else response[0])
        elif mode == "sequence":
            if isinstance(response, str):
                raise ValueError("sequence mode requires a list of responses")
            return self._sequence_eval(list(response))
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _snapshot_eval(self, response: str) -> MetricResult:
        centroid = self._baseline_centroid()
        new_emb = self._embedder.encode([response], normalize_embeddings=True)[0]

        if centroid is None:
            self._window.embeddings.append(new_emb)
            self._window.texts.append(response)
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                confidence=0.0,
                metadata={"reason": "baseline not yet established"},
            )

        similarity = float(np.dot(new_emb, centroid / np.linalg.norm(centroid)))
        drift_magnitude = 1.0 - similarity

        self._window.embeddings.append(new_emb)
        self._window.texts.append(response)

        return MetricResult(
            metric_name=self.name,
            score=float(np.clip(similarity, 0.0, 1.0)),
            confidence=0.9,
            metadata={
                "similarity_to_baseline": similarity,
                "drift_magnitude": drift_magnitude,
                "baseline_window_size": len(self._window.embeddings),
                "mode": "snapshot",
            },
        )

    def _sequence_eval(self, responses: Sequence[str]) -> MetricResult:
        embeddings = self._embedder.encode(list(responses), normalize_embeddings=True)
        consecutive_sims = [
            float(np.dot(embeddings[i], embeddings[i + 1]))
            for i in range(len(embeddings) - 1)
        ]

        if not consecutive_sims:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                confidence=0.5,
                metadata={"reason": "single response — no drift to measure"},
            )

        mean_sim = statistics.mean(consecutive_sims)
        drift_trend = (
            "increasing" if consecutive_sims[-1] < consecutive_sims[0] else "stable"
        )

        return MetricResult(
            metric_name=self.name,
            score=float(np.clip(mean_sim, 0.0, 1.0)),
            confidence=float(np.clip(1.0 - statistics.stdev(consecutive_sims), 0.0, 1.0)),
            metadata={
                "consecutive_similarities": consecutive_sims,
                "mean_similarity": mean_sim,
                "drift_trend": drift_trend,
                "sequence_length": len(responses),
                "mode": "sequence",
            },
        )
