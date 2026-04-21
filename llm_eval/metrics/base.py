from __future__ import annotations
import abc
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricResult:
    metric_name: str
    score: float                        # normalised [0, 1]
    confidence: float                   # model confidence in the score
    metadata: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    passed: bool = True                 # whether score >= threshold

    def __post_init__(self):
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be in [0, 1], got {self.score}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")


class BaseMetric(abc.ABC):
    """Abstract base for all eval metrics."""

    def __init__(self, threshold: float = 0.7, name: str | None = None):
        self.threshold = threshold
        self.name = name or self.__class__.__name__

    def evaluate(self, *args, **kwargs) -> MetricResult:
        start = time.perf_counter()
        result = self._evaluate(*args, **kwargs)
        result.latency_ms = (time.perf_counter() - start) * 1000
        result.passed = result.score >= self.threshold
        return result

    @abc.abstractmethod
    def _evaluate(self, *args, **kwargs) -> MetricResult:
        ...
