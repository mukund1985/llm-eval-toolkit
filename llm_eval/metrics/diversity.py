"""
Diversity Metrics
=================
Detect distribution collapse (FM-3) in agentic output sequences.

Distribution collapse occurs when an agent narrows its output distribution
over time — repeatedly recommending the same categories, creators, or topics
instead of exploring the available space. These three metrics operationalise
the failure mode at different resolutions: raw uniqueness, information-theoretic
entropy, and repetition rate weighted by session depth.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .base import BaseMetric, MetricResult


@dataclass
class DiversityReport:
    """Aggregated diversity scores across all three diversity dimensions."""

    intrasession_score: float
    entropy_score: float
    repeat_rate_score: float
    distribution_collapse_detected: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def composite_score(self) -> float:
        """Mean of the three sub-scores."""
        return (self.intrasession_score + self.entropy_score + self.repeat_rate_score) / 3.0


class IntrasessionDiversityMetric(BaseMetric):
    """
    Measures unique categories per session window.

    Score = unique_count / window_size, normalised to [0, 1].
    A score of 1.0 means every output in the window is a distinct category;
    0.0 means every output is the same category (full collapse).

    Parameters
    ----------
    window_size : int
        Maximum number of recent outputs to consider.
    threshold : float
        Minimum diversity ratio to pass.
    """

    def __init__(self, window_size: int = 20, threshold: float = 0.5):
        super().__init__(threshold=threshold, name="intrasession_diversity")
        self.window_size = window_size

    def _evaluate(self, labels: Sequence[str]) -> MetricResult:  # type: ignore[override]
        if not labels:
            raise ValueError("labels sequence must not be empty")

        window = list(labels)[-self.window_size :]
        unique_count = len(set(window))
        score = unique_count / len(window)

        counts = Counter(window)
        dominant_category, dominant_count = counts.most_common(1)[0]

        return MetricResult(
            metric_name=self.name,
            score=float(score),
            confidence=0.9,
            metadata={
                "unique_count": unique_count,
                "window_size": len(window),
                "total_labels": len(labels),
                "dominant_category": dominant_category,
                "dominant_category_count": dominant_count,
                "category_distribution": dict(counts),
            },
        )


class OutputEntropyMetric(BaseMetric):
    """
    Shannon entropy of the output category distribution over a sliding window.

    Declining entropy over successive windows is the primary signal of
    distribution collapse. Score is entropy normalised by log2(unique_categories)
    so it maps to [0, 1] regardless of vocabulary size.

    Parameters
    ----------
    window_size : int
        Sliding window length.
    threshold : float
        Minimum normalised entropy to pass (below this = collapse warning).
    """

    def __init__(self, window_size: int = 20, threshold: float = 0.5):
        super().__init__(threshold=threshold, name="output_entropy")
        self.window_size = window_size

    def _evaluate(self, labels: Sequence[str]) -> MetricResult:  # type: ignore[override]
        if not labels:
            raise ValueError("labels sequence must not be empty")

        window = list(labels)[-self.window_size :]
        counts = Counter(window)
        total = len(window)
        probs = [c / total for c in counts.values()]

        raw_entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        max_entropy = math.log2(len(counts)) if len(counts) > 1 else 0.0
        normalised_entropy = raw_entropy / max_entropy if max_entropy > 0 else 0.0

        return MetricResult(
            metric_name=self.name,
            score=float(np.clip(normalised_entropy, 0.0, 1.0)),
            confidence=0.85,
            metadata={
                "raw_entropy": raw_entropy,
                "max_possible_entropy": max_entropy,
                "normalised_entropy": normalised_entropy,
                "unique_categories": len(counts),
                "window_size": len(window),
            },
        )


class RepeatRateMetric(BaseMetric):
    """
    Frequency with which the same category reappears in top-K outputs,
    paired with session depth to capture recommendation fatigue.

    Score = 1 - adjusted_repeat_rate, so higher is better. A depth weight
    amplifies the penalty when high repetition persists deep into a session.

    Parameters
    ----------
    top_k : int
        Number of recent outputs to consider.
    threshold : float
        Minimum (1 - repeat_rate) score to pass.
    """

    def __init__(self, top_k: int = 10, threshold: float = 0.6):
        super().__init__(threshold=threshold, name="repeat_rate")
        self.top_k = top_k

    def _evaluate(  # type: ignore[override]
        self,
        labels: Sequence[str],
        session_depth: int = 1,
    ) -> MetricResult:
        if not labels:
            raise ValueError("labels sequence must not be empty")

        window = list(labels)[-self.top_k :]
        counts = Counter(window)
        total = len(window)

        most_common_label, most_common_count = counts.most_common(1)[0]
        repeat_rate = most_common_count / total

        # Longer sessions with high repeat rate are penalised proportionally more.
        depth_factor = min(session_depth / 100.0, 1.0)
        adjusted_repeat_rate = min(repeat_rate * (1.0 + 0.5 * depth_factor), 1.0)
        score = 1.0 - adjusted_repeat_rate

        return MetricResult(
            metric_name=self.name,
            score=float(np.clip(score, 0.0, 1.0)),
            confidence=0.8,
            metadata={
                "repeat_rate": repeat_rate,
                "adjusted_repeat_rate": adjusted_repeat_rate,
                "most_repeated_label": most_common_label,
                "most_repeated_count": most_common_count,
                "top_k": len(window),
                "session_depth": session_depth,
                "category_distribution": dict(counts),
            },
        )
