"""
Cascade Uncertainty Metric
===========================
Detect cascading decision errors via uncertainty propagation (FM-1).

In multi-step agentic pipelines, a low-confidence intermediate step can
corrupt all downstream reasoning. The failure mode is subtle: each step
appears individually coherent, but because a shaky premise was never
flagged, all subsequent logic builds on a flawed foundation.

The *coherence illusion* is the gap between high apparent internal
consistency (downstream steps agree with each other) and low external
correctness (the pipeline is wrong relative to ground truth). A pipeline
that looks confident is not necessarily a pipeline that is correct.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .base import BaseMetric, MetricResult


@dataclass
class StepResult:
    """Outcome of a single step in a multi-step agentic pipeline."""

    step_name: str
    input_data: Any
    output_data: Any
    confidence_score: float   # self-reported or derived confidence in [0, 1]
    is_fallback: bool = False  # True when this step used a default / fallback output

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError(
                f"confidence_score must be in [0, 1], got {self.confidence_score}"
            )


class CascadeUncertaintyMetric(BaseMetric):
    """
    Detects uncertainty propagation failures in multi-step pipelines.

    A propagation failure is declared when a step whose confidence is below
    `uncertainty_threshold` passes its output to a subsequent step without
    flagging the uncertainty. All steps downstream of the failure point then
    reason from a potentially flawed premise.

    The coherence illusion score captures how confidently the downstream steps
    proceed despite the upstream failure — a high score means the pipeline
    *looks* certain but should not be trusted.

    Parameters
    ----------
    uncertainty_threshold : float
        Confidence below this level at a non-terminal step triggers a
        propagation failure flag.
    threshold : float
        Minimum pipeline-level score to pass.
    """

    def __init__(self, uncertainty_threshold: float = 0.5, threshold: float = 0.7):
        super().__init__(threshold=threshold, name="cascade_uncertainty")
        self.uncertainty_threshold = uncertainty_threshold

    def _evaluate(  # type: ignore[override]
        self,
        steps: Sequence[StepResult],
        ground_truth_score: float | None = None,
    ) -> MetricResult:
        if not steps:
            raise ValueError("steps sequence must not be empty")

        steps = list(steps)
        confidence_trace = [s.confidence_score for s in steps]
        mean_confidence = float(np.mean(confidence_trace))

        # Locate the first step with confidence below the threshold.
        first_failure_idx: int | None = None
        first_failure_step: str | None = None
        propagation_failure_detected = False

        for i, step in enumerate(steps):
            if step.confidence_score < self.uncertainty_threshold:
                if i < len(steps) - 1:  # non-terminal: output propagates downstream
                    propagation_failure_detected = True
                    first_failure_idx = i
                    first_failure_step = step.step_name
                break

        # Coherence illusion: mean confidence of steps downstream of the failure.
        coherence_illusion_score = 0.0
        if propagation_failure_detected and first_failure_idx is not None:
            downstream = [s.confidence_score for s in steps[first_failure_idx + 1 :]]
            if downstream:
                coherence_illusion_score = float(np.mean(downstream))

        # Internal-vs-external divergence when ground truth is provided.
        internal_external_divergence: float | None = None
        if ground_truth_score is not None:
            internal_external_divergence = abs(mean_confidence - ground_truth_score)

        # Early-failure-masked: failure in first half of pipeline while mean
        # confidence appears healthy.
        early_failure_masked = (
            first_failure_idx is not None
            and first_failure_idx < len(steps) / 2
            and mean_confidence >= self.uncertainty_threshold
        )

        fallback_count = sum(1 for s in steps if s.is_fallback)
        fallback_penalty = 0.1 * (fallback_count / len(steps))
        propagation_penalty = 0.3 if propagation_failure_detected else 0.0
        pipeline_score = float(
            np.clip(mean_confidence - propagation_penalty - fallback_penalty, 0.0, 1.0)
        )

        return MetricResult(
            metric_name=self.name,
            score=pipeline_score,
            confidence=float(np.clip(1.0 - coherence_illusion_score, 0.0, 1.0)),
            metadata={
                "confidence_trace": confidence_trace,
                "mean_confidence": mean_confidence,
                "propagation_failure_detected": propagation_failure_detected,
                "first_failure_step": first_failure_step,
                "coherence_illusion_score": coherence_illusion_score,
                "early_failure_masked": early_failure_masked,
                "fallback_count": fallback_count,
                "internal_external_divergence": internal_external_divergence,
                "ground_truth_score": ground_truth_score,
                "num_steps": len(steps),
                "per_step": [
                    {
                        "step_name": s.step_name,
                        "confidence": s.confidence_score,
                        "is_fallback": s.is_fallback,
                    }
                    for s in steps
                ],
            },
        )
