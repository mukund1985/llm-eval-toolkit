"""
Reliability Metric
==================
Detect silent degradation via availability-truth decoupling (FM-2).

A system exhibiting FM-2 continues returning responses (availability signal
stays green) while the quality of those responses quietly erodes: partial
responses increase, fields go missing, latency creeps up. Because output
accuracy metrics often lag, this failure mode goes undetected until user-
facing harm is observed.

The key diagnostic signature: partial_response_rate rises while an external
accuracy signal remains stable — the system looks healthy on the outside
while internal quality degrades.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

import numpy as np

from .base import BaseMetric, MetricResult


class ToolCallState(Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass
class ToolCall:
    tool_name: str
    state: ToolCallState
    response: Any
    missing_fields: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ReliabilityReport:
    """Per-tool and aggregate reliability breakdown."""

    partial_response_rate: float
    tool_health_scores: dict[str, float]
    latency_drift_detected: bool
    silent_degradation_detected: bool
    per_tool_breakdown: dict[str, dict[str, Any]]
    overall_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class ReliabilityMetric(BaseMetric):
    """
    Tracks a sequence of ToolCall objects and diagnoses silent degradation.

    Parameters
    ----------
    latency_drift_window : int
        Number of recent calls to use for latency trend detection.
    latency_drift_factor : float
        Ratio of late-window to early-window average latency that triggers
        the drift flag (default: 1.25 = 25% increase).
    threshold : float
        Minimum overall reliability score to pass.
    """

    def __init__(
        self,
        latency_drift_window: int = 10,
        latency_drift_factor: float = 1.25,
        threshold: float = 0.7,
    ):
        super().__init__(threshold=threshold, name="reliability")
        self.latency_drift_window = latency_drift_window
        self.latency_drift_factor = latency_drift_factor

    def _evaluate(  # type: ignore[override]
        self,
        calls: Sequence[ToolCall],
        accuracy_signal: float | None = None,
    ) -> MetricResult:
        if not calls:
            raise ValueError("calls sequence must not be empty")

        calls = list(calls)
        total = len(calls)

        partial_count = sum(1 for c in calls if c.state == ToolCallState.PARTIAL)
        failed_count = sum(1 for c in calls if c.state == ToolCallState.FAILED)
        partial_response_rate = partial_count / total

        # Per-tool breakdown
        tool_buckets: dict[str, list[ToolCall]] = {}
        for call in calls:
            tool_buckets.setdefault(call.tool_name, []).append(call)

        tool_health_scores: dict[str, float] = {}
        per_tool_breakdown: dict[str, dict[str, Any]] = {}

        for tool_name, bucket in tool_buckets.items():
            n = len(bucket)
            success_n = sum(1 for c in bucket if c.state == ToolCallState.SUCCESS)
            partial_n = sum(1 for c in bucket if c.state == ToolCallState.PARTIAL)
            avg_latency = float(np.mean([c.latency_ms for c in bucket]))
            all_missing = [f for c in bucket for f in c.missing_fields]

            tool_health_scores[tool_name] = success_n / n
            per_tool_breakdown[tool_name] = {
                "total_calls": n,
                "success_count": success_n,
                "partial_count": partial_n,
                "health_score": success_n / n,
                "avg_latency_ms": avg_latency,
                "common_missing_fields": list(set(all_missing)),
            }

        # Latency drift: compare early vs late half of the recent window.
        latency_drift_detected = False
        if len(calls) >= self.latency_drift_window:
            recent = [c.latency_ms for c in calls[-self.latency_drift_window :]]
            mid = len(recent) // 2
            early_avg = float(np.mean(recent[:mid]))
            late_avg = float(np.mean(recent[mid:]))
            if early_avg > 0 and late_avg > early_avg * self.latency_drift_factor:
                latency_drift_detected = True

        # Silent degradation: partial rate rises while external accuracy looks stable.
        silent_degradation_detected = (
            accuracy_signal is not None
            and partial_response_rate > 0.2
            and accuracy_signal >= 0.7
        )

        success_rate = (total - partial_count - failed_count) / total
        overall_score = float(
            np.clip(
                success_rate
                - 0.5 * partial_response_rate
                - (0.1 if latency_drift_detected else 0.0),
                0.0,
                1.0,
            )
        )

        report = ReliabilityReport(
            partial_response_rate=partial_response_rate,
            tool_health_scores=tool_health_scores,
            latency_drift_detected=latency_drift_detected,
            silent_degradation_detected=silent_degradation_detected,
            per_tool_breakdown=per_tool_breakdown,
            overall_score=overall_score,
            metadata={
                "total_calls": total,
                "partial_count": partial_count,
                "failed_count": failed_count,
                "accuracy_signal": accuracy_signal,
            },
        )

        return MetricResult(
            metric_name=self.name,
            score=overall_score,
            confidence=0.85,
            metadata={
                "partial_response_rate": partial_response_rate,
                "tool_health_scores": tool_health_scores,
                "latency_drift_detected": latency_drift_detected,
                "silent_degradation_detected": silent_degradation_detected,
                "per_tool_breakdown": per_tool_breakdown,
                "total_calls": total,
                "partial_count": partial_count,
                "failed_count": failed_count,
                "accuracy_signal": accuracy_signal,
                "report": report,
            },
        )
