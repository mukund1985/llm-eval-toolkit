# llm-eval-toolkit

![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/status-active-brightgreen?style=flat-square)
![Code style](https://img.shields.io/badge/code%20style-ruff-orange?style=flat-square)

> Production-grade evaluation framework for LLM agent outputs. Built for the class of problems that matter at scale: silent regressions, hallucination under distribution shift, and reasoning quality in autonomous pipelines.

---

## The Problem

Evaluating LLMs in production is not a benchmark problem. It is an observability problem.

When a model update ships to a billion-user system, the failure modes are not dramatic. They are gradual: a slight increase in unsupported claims, a drift in reasoning style, a consistency drop under paraphrased prompts that previously passed. Standard benchmarks do not catch these. Shadow traffic eval pipelines do.

This toolkit provides the metric primitives and pipeline infrastructure for that kind of evaluation — composable, locally executable, CI-ready, and structured for integration with your observability stack.

---

## Production Failure Modes Covered

This toolkit detects the failure modes that matter in production — not the ones that show up on benchmarks.

| Failure Mode | Module | What It Detects |
|---|---|---|
| **Distribution Collapse** | `diversity.py` | Output entropy collapse, intra-session diversity loss, repeat rate spikes — catches the failure where metrics look healthy but users see repetitive outputs |
| **Tool Failure Cascade** | `reliability.py` | Silent degradation via availability-truth decoupling — tool returns schema-valid stale/partial data, system proceeds confidently with wrong inputs |
| **Cascading Decision Error** | `cascade.py` | Coherence illusion in multi-step pipelines — incorrect early decision propagates, each step adds evidence making output look internally consistent but be systematically wrong |
| **Explanation-Decision Decoupling** | `perturbation.py` | Post-hoc attribution proxy errors — model correct, explanation wrong; perturbation consistency checks catch SHAP citing proxy features rather than causal signals |
| **Semantic Consistency** | `consistency.py` | Output instability under paraphrase |
| **Response Drift** | `drift.py` | Temporal semantic drift, baseline deviation |
| **Factual Grounding** | `factual_grounding.py` | Claim-level grounding against reference corpus |
| **Hallucination** | `hallucination.py` | Unsupported claim detection |

## Metrics

| Metric | What it measures | Primary use case |
|---|---|---|
| **Diversity** | Output entropy, intra-session diversity, repeat rate | Distribution collapse detection |
| **Reliability** | Tool call state tracking, partial-response rate, latency-drift correlation | Silent degradation in tool-augmented agents |
| **Perturbation** | Attribution-perturbation consistency score | Explanation validity in regulated/auditable systems |
| **Cascade** | Uncertainty propagation across pipeline steps | Multi-step agent coherence illusion detection |
| **Consistency** | Semantic stability across paraphrased prompts | Prompt-surface sensitivity testing |
| **Factual Grounding** | Sentence-level coverage against a reference corpus | RAG pipeline quality, knowledge grounding |
| **Explainability** | Reasoning chain quality relative to a conclusion | Agentic decision audit, CoT evaluation |
| **Hallucination** | Unsupported factual claims vs. grounded context | Claim-level entailment checking |
| **Drift** | Semantic shift over time or across model versions | Post-deploy regression, A/B model comparison |

All metrics are locally executable — no external API calls, no rate limits. Everything runs on sentence-transformers.

---

## Installation

```bash
git clone https://github.com/mukund1985/llm-eval-toolkit.git
cd llm-eval-toolkit
pip install -r requirements.txt
pip install -e .
```

---

## Quick Start

```python
from llm_eval import LLMEvaluator
from llm_eval.evaluator import EvalRequest

evaluator = LLMEvaluator()

report = evaluator.evaluate(EvalRequest(
    request_id="demo-001",
    response="The Eiffel Tower is in Paris, built 1887–1889.",

    # Factual grounding + hallucination
    context=["The Eiffel Tower is in Paris.", "Built from 1887 to 1889."],
    reference="The Eiffel Tower is a landmark in Paris, built 1887–1889.",

    # Explainability
    reasoning=(
        "1. User asked about Eiffel Tower location and history. "
        "2. Context confirms Paris and construction dates. "
        "3. Therefore I report both facts."
    ),
    conclusion="Eiffel Tower is in Paris, built 1887–1889.",

    # Consistency across paraphrase variants
    paraphrase_responses=[
        "The Eiffel Tower, located in Paris, was erected in the 1880s.",
        "Paris's Eiffel Tower dates to 1887–1889.",
        "Built 1887–1889, the Eiffel Tower stands in Paris.",
    ],
))

print(report.to_json())
# {
#   "overall_score": 0.87,
#   "passed": true,
#   "metrics": {
#     "consistency": { "score": 0.94, "passed": true, ... },
#     "hallucination": { "score": 0.90, "passed": true, ... },
#     ...
#   }
# }
```

---

## Batch Evaluation Pipeline

For shadow traffic evaluation or CI dataset runs:

```python
from llm_eval.pipeline import EvalPipeline
from llm_eval.evaluator import EvalRequest

def on_failure(report):
    alert_channel.send(f"EVAL FAILURE: {report.request_id} score={report.overall_score:.2f}")

pipeline = EvalPipeline(max_concurrency=16, on_failure=on_failure)

result = pipeline.run([
    EvalRequest(request_id=f"req-{i}", response=resp, context=ctx)
    for i, (resp, ctx) in enumerate(shadow_traffic_dataset)
])

print(f"Pass rate: {result.pass_rate:.1%}")
print(f"Mean score: {result.mean_overall_score:.3f}")
result.save("eval_results.json")
```

The pipeline is async-first (`asyncio.gather` under the hood), runs the sync evaluator in a thread pool, and accepts a configurable failure hook for alerting.

---

## Architecture

```
llm_eval/
├── metrics/
│   ├── base.py               # MetricResult dataclass, BaseMetric ABC
│   ├── diversity.py          # Entropy, intra-session diversity, repeat rate (FM-3)
│   ├── reliability.py        # Tool call state tracking, partial-response rate (FM-2)
│   ├── perturbation.py       # Attribution-perturbation consistency checker (FM-5)
│   ├── cascade.py            # Uncertainty propagation across pipeline steps (FM-1)
│   ├── consistency.py        # Pairwise cosine similarity over paraphrases
│   ├── factual_grounding.py  # Sentence-to-chunk alignment vs. reference
│   ├── explainability.py     # Causal density + alignment + step coverage
│   ├── hallucination.py      # Claim-level grounding with GROUNDED/UNSUPPORTED/HALLUCINATED
│   └── drift.py              # Snapshot (vs. baseline centroid) + sequence modes
├── evaluator.py              # LLMEvaluator: runs applicable metrics, returns EvalReport
├── pipeline.py               # EvalPipeline: async batch runner with concurrency control
└── utils/
    └── logging.py            # structlog configuration (JSON or console)
```

### Design decisions

**No external API calls in metrics.** All scoring uses sentence-transformers locally. This means eval pipelines can run in CI without credentials, in air-gapped environments, and without per-token cost.

**Composable metrics.** Each metric is a standalone class with a single `evaluate()` method. Use one, use all, or subclass `BaseMetric` to add your own. The evaluator only runs metrics for which the request has the required fields.

**Production-observable results.** Every `MetricResult` carries `score`, `confidence`, `passed`, `latency_ms`, and structured `metadata`. The metadata keys are consistent across runs — designed to be indexed into a metrics database or logged to your observability stack.

**Structured logging.** Uses `structlog` with `json_output=True` for production pipelines, console renderer for development.

---

## Metric Details

### Consistency
Encodes multiple responses with `sentence-transformers` and computes mean pairwise cosine similarity. A model with `mean_similarity < 0.85` across paraphrase variants is showing prompt-surface sensitivity — a reliability signal worth investigating.

### Factual Grounding
Splits the response into sentences, splits the reference into chunks, and finds the best-matching reference chunk per response sentence. Low per-sentence scores surface specific claims that lack grounding.

### Explainability
Weighted combination of: semantic alignment between reasoning and conclusion (40%), causal connective density (25%), reasoning step coverage (20%), hedge word penalty (15%). Designed for chain-of-thought and agentic decision justification.

### Hallucination
Classifies each factual claim in the response as GROUNDED (≥0.70 similarity to context), UNSUPPORTED (between thresholds), or HALLUCINATED (≤0.40). Score = grounded_fraction − 0.5×hallucinated_fraction.

### Drift
Two modes: **snapshot** compares a new response to the rolling baseline centroid (useful for regression detection); **sequence** measures consecutive similarity across a response trajectory (useful for long agentic conversations and topic drift).

---

## Running Tests

```bash
pytest tests/ -v
```

Tests use mocked embeddings — no model download required in CI.

---

## Contributing

Contributions are welcome. Areas where the toolkit would benefit most:

- **New metrics** — coherence, citation accuracy, task completion rate
- **New sinks** — direct integration with MLflow, W&B, or Prometheus
- **Async metrics** — non-blocking evaluation for high-throughput pipelines
- **Calibration tooling** — threshold tuning against human-labelled datasets

Please open an issue before submitting a large PR.

---

## Research

This toolkit is the reference implementation for:

> *Evaluating Agentic AI in the Wild: Failure Modes, Drift Patterns, and a Production Evaluation Framework*
> Mukund Pandey — preprint forthcoming

The failure modes and detection methods are grounded in production observations from systems operating at billion-event scale. If you are working on agentic evaluation, LLM observability, or AI safety in production systems, contributions and issue reports are welcome.

---

## License

MIT — see [LICENSE](LICENSE)
