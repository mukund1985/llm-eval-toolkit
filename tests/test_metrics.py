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
from llm_eval.metrics.cascade import CascadeUncertaintyMetric, StepResult
from llm_eval.metrics.consistency import ConsistencyMetric
from llm_eval.metrics.diversity import (
    IntrasessionDiversityMetric,
    OutputEntropyMetric,
    RepeatRateMetric,
)
from llm_eval.metrics.drift import DriftMetric
from llm_eval.metrics.explainability import ExplainabilityMetric
from llm_eval.metrics.factual_grounding import FactualGroundingMetric
from llm_eval.metrics.hallucination import HallucinationMetric
from llm_eval.metrics.perturbation import PerturbationConsistencyMetric
from llm_eval.metrics.reliability import ReliabilityMetric, ToolCall, ToolCallState


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


# ---------------------------------------------------------------------------
# Diversity metrics (FM-3)
# ---------------------------------------------------------------------------


class TestIntrasessionDiversityMetric:
    def test_all_unique_labels_perfect_score(self):
        metric = IntrasessionDiversityMetric()
        result = metric.evaluate(labels=["a", "b", "c", "d", "e"])
        assert result.score == pytest.approx(1.0, abs=1e-6)

    def test_all_same_labels_minimum_score(self):
        metric = IntrasessionDiversityMetric()
        result = metric.evaluate(labels=["a", "a", "a", "a"])
        assert result.score == pytest.approx(0.25, abs=1e-6)

    def test_window_truncation_uses_tail(self):
        # Last 3 labels are all "a", so unique=1/3 despite diverse prefix.
        metric = IntrasessionDiversityMetric(window_size=3)
        result = metric.evaluate(labels=["b", "c", "d", "a", "a", "a"])
        assert result.score == pytest.approx(1 / 3, abs=1e-6)
        assert result.metadata["window_size"] == 3

    def test_empty_labels_raises(self):
        metric = IntrasessionDiversityMetric()
        with pytest.raises(ValueError):
            metric.evaluate(labels=[])


class TestOutputEntropyMetric:
    def test_uniform_distribution_maximum_entropy(self):
        metric = OutputEntropyMetric()
        result = metric.evaluate(labels=["a", "b", "c", "d"])
        assert result.score == pytest.approx(1.0, abs=1e-6)

    def test_single_category_zero_entropy(self):
        metric = OutputEntropyMetric()
        result = metric.evaluate(labels=["a", "a", "a", "a"])
        assert result.score == pytest.approx(0.0, abs=1e-6)

    def test_skewed_distribution_low_entropy_fails_threshold(self):
        metric = OutputEntropyMetric(threshold=0.5)
        result = metric.evaluate(labels=["a"] * 9 + ["b"])
        assert result.score < 0.5
        assert not result.passed


class TestRepeatRateMetric:
    def test_all_unique_high_score(self):
        metric = RepeatRateMetric(top_k=5)
        result = metric.evaluate(labels=["a", "b", "c", "d", "e"])
        assert result.score > 0.6

    def test_all_same_very_low_score(self):
        metric = RepeatRateMetric(top_k=5)
        result = metric.evaluate(labels=["a", "a", "a", "a", "a"])
        assert result.score < 0.1

    def test_deep_session_penalises_more_than_shallow(self):
        metric = RepeatRateMetric(top_k=5)
        labels = ["a", "a", "a", "b", "c"]
        r_shallow = metric.evaluate(labels=labels, session_depth=1)
        r_deep = metric.evaluate(labels=labels, session_depth=100)
        assert r_shallow.score >= r_deep.score


# ---------------------------------------------------------------------------
# Reliability metric (FM-2)
# ---------------------------------------------------------------------------


class TestReliabilityMetric:
    def _make_calls(self, state: ToolCallState, n: int, latency: float = 50.0) -> list[ToolCall]:
        return [ToolCall("tool", state, "resp", [], latency) for _ in range(n)]

    def test_all_success_high_score(self):
        metric = ReliabilityMetric()
        calls = self._make_calls(ToolCallState.SUCCESS, 5)
        result = metric.evaluate(calls=calls)
        assert result.score > 0.7
        assert not result.metadata["silent_degradation_detected"]

    def test_silent_degradation_detected_when_partial_rate_high_and_accuracy_stable(self):
        metric = ReliabilityMetric()
        calls = self._make_calls(ToolCallState.PARTIAL, 5)
        result = metric.evaluate(calls=calls, accuracy_signal=0.9)
        assert result.metadata["silent_degradation_detected"]
        assert result.metadata["partial_response_rate"] == pytest.approx(1.0)

    def test_latency_drift_detected(self):
        metric = ReliabilityMetric(latency_drift_window=4)
        calls = [
            ToolCall("tool", ToolCallState.SUCCESS, "ok", [], 100.0),
            ToolCall("tool", ToolCallState.SUCCESS, "ok", [], 100.0),
            ToolCall("tool", ToolCallState.SUCCESS, "ok", [], 200.0),
            ToolCall("tool", ToolCallState.SUCCESS, "ok", [], 200.0),
        ]
        result = metric.evaluate(calls=calls)
        assert result.metadata["latency_drift_detected"]

    def test_per_tool_breakdown_present(self):
        metric = ReliabilityMetric()
        calls = [
            ToolCall("search", ToolCallState.SUCCESS, "ok", [], 30.0),
            ToolCall("search", ToolCallState.PARTIAL, "partial", ["field_x"], 60.0),
            ToolCall("db_lookup", ToolCallState.SUCCESS, "ok", [], 20.0),
        ]
        result = metric.evaluate(calls=calls)
        assert "search" in result.metadata["per_tool_breakdown"]
        assert "db_lookup" in result.metadata["per_tool_breakdown"]

    def test_empty_calls_raises(self):
        metric = ReliabilityMetric()
        with pytest.raises(ValueError):
            metric.evaluate(calls=[])


# ---------------------------------------------------------------------------
# Perturbation consistency metric (FM-5)
# ---------------------------------------------------------------------------


class TestPerturbationConsistencyMetric:
    @staticmethod
    def _linear_predict(features: dict) -> float:
        """Toy model: 'important' drives output, 'noise' barely matters."""
        return features.get("important", 0.0) * 2.0 + features.get("noise", 0.0) * 0.01

    def test_consistent_attribution_not_flagged(self):
        metric = PerturbationConsistencyMetric(low_impact_threshold=0.05)
        features = {"important": 1.0, "noise": 0.5}
        result = metric.evaluate(
            predict_fn=self._linear_predict,
            feature_dict=features,
            top_k_features=["important", "noise"],
        )
        assert not result.metadata["flagged_as_decoupled"]
        assert result.metadata["top_feature_impact"] > 0.05

    def test_decoupled_attribution_flagged(self):
        metric = PerturbationConsistencyMetric(low_impact_threshold=0.05)
        features = {"important": 1.0, "noise": 0.5}
        # Claim noise is the most important feature — it barely affects output.
        result = metric.evaluate(
            predict_fn=self._linear_predict,
            feature_dict=features,
            top_k_features=["noise", "important"],
        )
        assert result.metadata["flagged_as_decoupled"]
        assert result.metadata["top_feature_impact"] <= 0.05

    def test_mean_perturbation_strategy_produces_valid_result(self):
        metric = PerturbationConsistencyMetric(perturbation_strategy="mean")
        features = {"a": 1.0, "b": 3.0}
        result = metric.evaluate(
            predict_fn=lambda f: f.get("a", 0.0),
            feature_dict=features,
            top_k_features=["a"],
        )
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 1.0

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            PerturbationConsistencyMetric(perturbation_strategy="invalid")

    def test_per_feature_impact_keys_match_top_k(self):
        metric = PerturbationConsistencyMetric()
        features = {"x": 1.0, "y": 2.0, "z": 0.5}
        result = metric.evaluate(
            predict_fn=lambda f: f.get("x", 0.0) + f.get("y", 0.0),
            feature_dict=features,
            top_k_features=["x", "y"],
        )
        assert set(result.metadata["per_feature_impact"].keys()) == {"x", "y"}


# ---------------------------------------------------------------------------
# Cascade uncertainty metric (FM-1)
# ---------------------------------------------------------------------------


class TestCascadeUncertaintyMetric:
    def test_all_high_confidence_no_failure(self):
        metric = CascadeUncertaintyMetric(uncertainty_threshold=0.5)
        steps = [
            StepResult("step1", "in", "out", 0.9),
            StepResult("step2", "in", "out", 0.85),
            StepResult("step3", "in", "out", 0.8),
        ]
        result = metric.evaluate(steps=steps)
        assert not result.metadata["propagation_failure_detected"]
        assert result.metadata["first_failure_step"] is None
        assert result.score > 0.5

    def test_propagation_failure_detected_mid_pipeline(self):
        metric = CascadeUncertaintyMetric(uncertainty_threshold=0.5)
        steps = [
            StepResult("step1", "in", "out", 0.9),
            StepResult("step2", "in", "out", 0.3),   # low confidence, non-terminal
            StepResult("step3", "in", "out", 0.85),
        ]
        result = metric.evaluate(steps=steps)
        assert result.metadata["propagation_failure_detected"]
        assert result.metadata["first_failure_step"] == "step2"

    def test_coherence_illusion_score_high_after_early_failure(self):
        metric = CascadeUncertaintyMetric(uncertainty_threshold=0.5)
        steps = [
            StepResult("step1", "in", "out", 0.4),   # early failure
            StepResult("step2", "in", "out", 0.95),
            StepResult("step3", "in", "out", 0.95),
        ]
        result = metric.evaluate(steps=steps)
        assert result.metadata["propagation_failure_detected"]
        assert result.metadata["coherence_illusion_score"] == pytest.approx(0.95, abs=1e-3)

    def test_ground_truth_divergence_computed(self):
        metric = CascadeUncertaintyMetric()
        steps = [StepResult("s1", "in", "out", 0.9)]
        result = metric.evaluate(steps=steps, ground_truth_score=0.4)
        assert result.metadata["internal_external_divergence"] == pytest.approx(0.5, abs=1e-3)

    def test_fallback_step_penalises_score(self):
        metric = CascadeUncertaintyMetric()
        steps_no_fallback = [StepResult("s1", "in", "out", 0.8)]
        steps_with_fallback = [StepResult("s1", "in", "out", 0.8, is_fallback=True)]
        r1 = metric.evaluate(steps=steps_no_fallback)
        r2 = metric.evaluate(steps=steps_with_fallback)
        assert r1.score > r2.score

    def test_terminal_low_confidence_not_flagged_as_propagation(self):
        # Low confidence on the last step cannot propagate.
        metric = CascadeUncertaintyMetric(uncertainty_threshold=0.5)
        steps = [
            StepResult("step1", "in", "out", 0.9),
            StepResult("step2", "in", "out", 0.2),   # last step
        ]
        result = metric.evaluate(steps=steps)
        assert not result.metadata["propagation_failure_detected"]

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError):
            StepResult("bad", "in", "out", confidence_score=1.5)
