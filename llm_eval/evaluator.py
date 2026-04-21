"""
LLMEvaluator
============
Orchestrates all metrics for a single evaluation request.
Designed to be embedded in CI pipelines, shadow traffic
systems, or interactive eval dashboards.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import structlog

from .metrics.base import MetricResult
from .metrics.consistency import ConsistencyMetric
from .metrics.drift import DriftMetric
from .metrics.explainability import ExplainabilityMetric
from .metrics.factual_grounding import FactualGroundingMetric
from .metrics.hallucination import HallucinationMetric

log = structlog.get_logger()


@dataclass
class EvalRequest:
    response: str
    reference: str | None = None
    context: str | Sequence[str] | None = None
    reasoning: str | None = None
    conclusion: str | None = None
    response_history: list[str] | None = None
    paraphrase_responses: list[str] | None = None
    request_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalReport:
    request_id: str | None
    results: dict[str, MetricResult]
    overall_score: float
    passed: bool
    total_latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "overall_score": self.overall_score,
            "passed": self.passed,
            "total_latency_ms": self.total_latency_ms,
            "metrics": {
                name: {
                    "score": r.score,
                    "confidence": r.confidence,
                    "passed": r.passed,
                    "latency_ms": r.latency_ms,
                    "metadata": r.metadata,
                }
                for name, r in self.results.items()
            },
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class LLMEvaluator:
    """
    Top-level evaluator.

    Runs whichever metrics are applicable given the supplied request
    fields and aggregates results into an EvalReport.

    Usage
    -----
    >>> evaluator = LLMEvaluator()
    >>> report = evaluator.evaluate(EvalRequest(
    ...     response="Paris is the capital of France.",
    ...     context="France is a country in Western Europe. Its capital is Paris.",
    ... ))
    >>> print(report.to_json())
    """

    def __init__(
        self,
        consistency_threshold: float = 0.85,
        grounding_threshold: float = 0.75,
        explainability_threshold: float = 0.70,
        hallucination_threshold: float = 0.80,
        drift_threshold: float = 0.85,
        overall_pass_threshold: float = 0.75,
    ):
        self._metrics = {
            "consistency": ConsistencyMetric(threshold=consistency_threshold),
            "factual_grounding": FactualGroundingMetric(threshold=grounding_threshold),
            "explainability": ExplainabilityMetric(threshold=explainability_threshold),
            "hallucination": HallucinationMetric(threshold=hallucination_threshold),
            "drift": DriftMetric(threshold=drift_threshold),
        }
        self.overall_pass_threshold = overall_pass_threshold

    def evaluate(self, request: EvalRequest) -> EvalReport:
        start = time.perf_counter()
        results: dict[str, MetricResult] = {}

        if request.paraphrase_responses and len(request.paraphrase_responses) >= 2:
            log.info("running consistency metric")
            results["consistency"] = self._metrics["consistency"].evaluate(
                responses=request.paraphrase_responses
            )

        if request.reference:
            log.info("running factual_grounding metric")
            results["factual_grounding"] = self._metrics["factual_grounding"].evaluate(
                response=request.response,
                reference=request.reference,
            )

        if request.reasoning and request.conclusion:
            log.info("running explainability metric")
            results["explainability"] = self._metrics["explainability"].evaluate(
                reasoning=request.reasoning,
                conclusion=request.conclusion,
            )

        if request.context:
            log.info("running hallucination metric")
            results["hallucination"] = self._metrics["hallucination"].evaluate(
                response=request.response,
                context=request.context,
            )

        drift_metric: DriftMetric = self._metrics["drift"]  # type: ignore[assignment]
        if request.response_history and len(request.response_history) >= 2:
            log.info("running drift metric (sequence mode)")
            results["drift"] = drift_metric.evaluate(
                response=request.response_history,
                mode="sequence",
            )
        else:
            log.info("running drift metric (snapshot mode)")
            results["drift"] = drift_metric.evaluate(
                response=request.response,
                mode="snapshot",
            )

        if not results:
            raise ValueError(
                "No metrics could be run — check EvalRequest fields are populated."
            )

        overall_score = sum(r.score for r in results.values()) / len(results)
        total_latency_ms = (time.perf_counter() - start) * 1000

        report = EvalReport(
            request_id=request.request_id,
            results=results,
            overall_score=overall_score,
            passed=overall_score >= self.overall_pass_threshold,
            total_latency_ms=total_latency_ms,
            metadata=request.metadata,
        )

        log.info(
            "eval_complete",
            request_id=request.request_id,
            overall_score=overall_score,
            passed=report.passed,
            metrics_run=list(results.keys()),
        )
        return report
