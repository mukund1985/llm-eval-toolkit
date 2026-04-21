"""
Unit tests for all metrics.
Uses lightweight mocking of SentenceTransformer to avoid
requiring GPU or large model downloads in CI.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from llm_eval.metrics.base import MetricResult
from llm_eval.metrics.consistency import ConsistencyMetric
from llm_eval.metrics.drift import DriftMetric
from llm_eval.metrics.explainability import ExplainabilityMetric
from llm_eval.metrics.factual_grounding import FactualGroundingMetric
from llm_eval.metrics.hallucination import HallucinationMetric


def _mock_embedder(embeddings: np.ndarray) -> MagicMock:
    m = MagicMock()
    m.encode.return_value = embeddings
    return m


class TestConsistencyMetric:
    def test_high_consistency(self):
        metric = ConsistencyMetric()
        emb = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        metric._embedder = _mock_embedder(emb)
        result = metric.evaluate(["r1", "r2", "r3"])
        assert result.score == pytest.approx(1.0, abs=1e-3)
        assert result.passed

    def test_low_consistency(self):
        metric = ConsistencyMetric(threshold=0.85)
        emb = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        metric._embedder = _mock_embedder(emb)
        result = metric.evaluate(["r1", "r2", "r3"])
        assert result.score < 0.85
        assert not result.passed

    def test_requires_two_responses(self):
        metric = ConsistencyMetric()
        with pytest.raises(ValueError):
            metric.evaluate(["only one"])


class TestFactualGroundingMetric:
    def test_well_grounded_response(self):
        metric = FactualGroundingMetric()
        emb_resp = np.array([[0.9, 0.1]])
        emb_ref = np.array([[0.9, 0.1], [0.8, 0.2]])
        metric._embedder = MagicMock()
        metric._embedder.encode.side_effect = [emb_resp, emb_ref]
        result = metric.evaluate(
            response="Paris is the capital of France.",
            reference="France's capital city is Paris.",
        )
        assert result.score > 0.7

    def test_empty_response(self):
        metric = FactualGroundingMetric()
        result = metric.evaluate(response="   ", reference="Some reference text.")
        assert result.score == 0.0


class TestExplainabilityMetric:
    def test_causal_reasoning_scores_high(self):
        metric = ExplainabilityMetric()
        emb = np.array([[0.9, 0.1], [0.85, 0.15]])
        metric._embedder = _mock_embedder(emb)
        result = metric.evaluate(
            reasoning=(
                "1. The user asked about X. "
                "2. Because the context confirms Y, therefore Z follows. "
                "3. Since the evidence is clear, the conclusion is valid."
            ),
            conclusion="The answer is Z because Y.",
        )
        assert result.score > 0.0
        assert isinstance(result, MetricResult)

    def test_hedged_reasoning_penalised(self):
        metric = ExplainabilityMetric()
        emb = np.array([[0.5, 0.5], [0.5, 0.5]])
        metric._embedder = _mock_embedder(emb)
        result = metric.evaluate(
            reasoning="I think maybe it could be X, perhaps Y, I'm not sure.",
            conclusion="Maybe X.",
        )
        assert result.metadata["hedge_penalty"] > 0


class TestHallucinationMetric:
    def test_grounded_response(self):
        metric = HallucinationMetric()
        emb_claims = np.array([[1.0, 0.0]])
        emb_ctx = np.array([[0.98, 0.02]])
        metric._embedder = MagicMock()
        metric._embedder.encode.side_effect = [emb_claims, emb_ctx]
        result = metric.evaluate(
            response="Paris is the capital of France.",
            context="France's capital is Paris.",
        )
        assert result.score > 0.5

    def test_hallucinated_response(self):
        metric = HallucinationMetric(hallucination_threshold=0.40)
        emb_claims = np.array([[1.0, 0.0]])
        emb_ctx = np.array([[0.0, 1.0]])
        metric._embedder = MagicMock()
        metric._embedder.encode.side_effect = [emb_claims, emb_ctx]
        result = metric.evaluate(
            response="The moon is made of cheese.",
            context="The moon is a natural satellite of Earth.",
        )
        assert result.metadata["hallucinated"] >= 1


class TestDriftMetric:
    def test_no_baseline_returns_neutral(self):
        metric = DriftMetric()
        emb = np.array([[1.0, 0.0]])
        metric._embedder = _mock_embedder(emb)
        result = metric.evaluate(response="Some response", mode="snapshot")
        assert result.score == 1.0
        assert result.confidence == 0.0

    def test_sequence_drift_detected(self):
        metric = DriftMetric()
        embs = np.array([[1.0, 0.0], [0.95, 0.05], [0.0, 1.0]])
        metric._embedder = _mock_embedder(embs)
        result = metric.evaluate(
            response=["resp1", "resp2", "resp3"],
            mode="sequence",
        )
        assert result.metadata["sequence_length"] == 3
        assert result.metadata["drift_trend"] in ("increasing", "stable")
