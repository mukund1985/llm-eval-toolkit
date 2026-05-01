"""
Perturbation Consistency Metric
=================================
Detect explanation-decision decoupling (FM-5) in model predictions.

FM-5 occurs when an attribution method (SHAP, LIME, attention, etc.) claims
a feature is important to the model's decision, but perturbing that feature
produces little or no change in the output. The mismatch — high attributed
importance, low actual perturbation impact — exposes a gap between the
explanation surface and the real decision function.

A well-calibrated explanation should rank features whose perturbation
maximally shifts the prediction at the top of the importance list. When
the most-cited feature causes near-zero prediction change, the explanation
is flagged as decoupled.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .base import BaseMetric, MetricResult


@dataclass
class PerturbationResult:
    feature_name: str
    original_value: Any
    perturbed_value: Any
    baseline_prediction: float
    perturbed_prediction: float
    impact: float          # |perturbed_prediction - baseline_prediction|
    claimed_rank: int      # position in top_k_features (0 = most important)


class PerturbationConsistencyMetric(BaseMetric):
    """
    Validates attribution explanations by measuring actual feature perturbation impact.

    For each feature in top_k_features, the metric nullifies or replaces the
    feature's value, calls predict_fn, and measures the prediction change.
    The attribution_consistency_score is the Pearson correlation between
    claimed importance order and measured impact, normalised to [0, 1].

    A low score — especially when the top-attributed feature has low impact —
    flags explanation-decision decoupling.

    Parameters
    ----------
    perturbation_strategy : str
        'zero'   — set numeric features to 0, non-numeric to None.
        'mean'   — replace with the mean of all numeric feature values.
        'random' — sample uniformly from the observed numeric value range.
    low_impact_threshold : float
        Maximum perturbation impact that triggers the decoupling flag for
        the top-attributed feature.
    threshold : float
        Minimum attribution consistency score to pass.
    """

    def __init__(
        self,
        perturbation_strategy: str = "zero",
        low_impact_threshold: float = 0.05,
        threshold: float = 0.6,
    ):
        if perturbation_strategy not in ("zero", "mean", "random"):
            raise ValueError(f"Unknown perturbation_strategy: {perturbation_strategy!r}")
        super().__init__(threshold=threshold, name="perturbation_consistency")
        self.perturbation_strategy = perturbation_strategy
        self.low_impact_threshold = low_impact_threshold

    def _perturb_value(self, value: Any, all_values: list[Any]) -> Any:
        numeric_values = [v for v in all_values if isinstance(v, (int, float))]
        if self.perturbation_strategy == "zero":
            return 0 if isinstance(value, (int, float)) else None
        elif self.perturbation_strategy == "mean":
            return float(np.mean(numeric_values)) if numeric_values else None
        else:  # random
            if numeric_values:
                lo, hi = min(numeric_values), max(numeric_values)
                return float(np.random.uniform(lo, hi)) if lo < hi else lo
            return None

    def _evaluate(  # type: ignore[override]
        self,
        predict_fn: Callable[[dict[str, Any]], float],
        feature_dict: dict[str, Any],
        top_k_features: list[str],
    ) -> MetricResult:
        if not top_k_features:
            raise ValueError("top_k_features must not be empty")
        if not feature_dict:
            raise ValueError("feature_dict must not be empty")

        baseline_prediction = float(predict_fn(feature_dict))
        all_values = list(feature_dict.values())

        perturbation_results: list[PerturbationResult] = []
        per_feature_impact: dict[str, float] = {}

        for rank, feature_name in enumerate(top_k_features):
            if feature_name not in feature_dict:
                continue

            original_value = feature_dict[feature_name]
            perturbed_value = self._perturb_value(original_value, all_values)

            perturbed_features = {**feature_dict, feature_name: perturbed_value}
            perturbed_prediction = float(predict_fn(perturbed_features))
            impact = abs(perturbed_prediction - baseline_prediction)

            perturbation_results.append(
                PerturbationResult(
                    feature_name=feature_name,
                    original_value=original_value,
                    perturbed_value=perturbed_value,
                    baseline_prediction=baseline_prediction,
                    perturbed_prediction=perturbed_prediction,
                    impact=impact,
                    claimed_rank=rank,
                )
            )
            per_feature_impact[feature_name] = impact

        # Attribution consistency score: Pearson correlation between
        # importance order (higher = more important) and measured impact.
        if len(perturbation_results) >= 2:
            # claimed_rank 0 → most important, so importance = -claimed_rank
            importance = np.array(
                [-r.claimed_rank for r in perturbation_results], dtype=float
            )
            impacts = np.array([r.impact for r in perturbation_results], dtype=float)

            corr_matrix = np.corrcoef(importance, impacts)
            corr_val = corr_matrix[0, 1]
            corr = 0.0 if np.isnan(corr_val) else float(corr_val)
            consistency_score = float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))
        elif perturbation_results:
            top_impact = perturbation_results[0].impact
            consistency_score = 1.0 if top_impact > self.low_impact_threshold else 0.0
        else:
            consistency_score = 0.0

        # Decoupling flag: top-attributed feature has low actual impact.
        top_feature = top_k_features[0] if top_k_features else None
        top_feature_impact = per_feature_impact.get(top_feature, 0.0) if top_feature else 0.0
        flagged_as_decoupled = top_feature_impact <= self.low_impact_threshold

        return MetricResult(
            metric_name=self.name,
            score=consistency_score,
            confidence=float(np.clip(consistency_score, 0.0, 1.0)),
            metadata={
                "per_feature_impact": per_feature_impact,
                "consistency_score": consistency_score,
                "flagged_as_decoupled": flagged_as_decoupled,
                "top_feature": top_feature,
                "top_feature_impact": top_feature_impact,
                "low_impact_threshold": self.low_impact_threshold,
                "perturbation_strategy": self.perturbation_strategy,
                "baseline_prediction": baseline_prediction,
                "perturbation_results": [
                    {
                        "feature": r.feature_name,
                        "claimed_rank": r.claimed_rank,
                        "impact": r.impact,
                        "original_value": r.original_value,
                        "perturbed_value": r.perturbed_value,
                    }
                    for r in perturbation_results
                ],
            },
        )
