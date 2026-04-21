"""
EvalPipeline
============
Batch evaluation pipeline with concurrency control and
structured result aggregation. Designed for CI integration
and shadow traffic evaluation at scale.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import structlog

from .evaluator import EvalReport, EvalRequest, LLMEvaluator

log = structlog.get_logger()


@dataclass
class PipelineResult:
    reports: list[EvalReport]
    total_requests: int
    passed: int
    failed: int
    pass_rate: float
    mean_overall_score: float

    def to_dict(self) -> dict:
        return {
            "summary": {
                "total_requests": self.total_requests,
                "passed": self.passed,
                "failed": self.failed,
                "pass_rate": self.pass_rate,
                "mean_overall_score": self.mean_overall_score,
            },
            "reports": [r.to_dict() for r in self.reports],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
        log.info("pipeline_results_saved", path=str(path))


class EvalPipeline:
    """
    Runs LLMEvaluator over a dataset of EvalRequests with
    configurable concurrency and optional pre/post hooks.

    Parameters
    ----------
    evaluator : LLMEvaluator
        Configured evaluator instance.
    max_concurrency : int
        Max simultaneous evaluations.
    on_failure : Callable[[EvalReport], None] | None
        Hook invoked for every failed evaluation.
    """

    def __init__(
        self,
        evaluator: LLMEvaluator | None = None,
        max_concurrency: int = 8,
        on_failure: Callable[[EvalReport], None] | None = None,
    ):
        self.evaluator = evaluator or LLMEvaluator()
        self.max_concurrency = max_concurrency
        self._on_failure = on_failure
        self._semaphore: asyncio.Semaphore | None = None

    async def _eval_one(self, request: EvalRequest) -> EvalReport:
        assert self._semaphore is not None
        async with self._semaphore:
            loop = asyncio.get_running_loop()
            report = await loop.run_in_executor(None, self.evaluator.evaluate, request)
            if not report.passed and self._on_failure:
                self._on_failure(report)
            return report

    async def run_async(self, requests: Sequence[EvalRequest]) -> PipelineResult:
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        tasks = [self._eval_one(req) for req in requests]
        reports = await asyncio.gather(*tasks)

        passed = sum(1 for r in reports if r.passed)
        return PipelineResult(
            reports=list(reports),
            total_requests=len(reports),
            passed=passed,
            failed=len(reports) - passed,
            pass_rate=passed / len(reports) if reports else 0.0,
            mean_overall_score=sum(r.overall_score for r in reports) / len(reports)
            if reports
            else 0.0,
        )

    def run(self, requests: Iterable[EvalRequest]) -> PipelineResult:
        reqs = list(requests)
        return asyncio.run(self.run_async(reqs))
