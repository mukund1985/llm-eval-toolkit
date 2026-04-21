# llm-eval-toolkit

![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/status-active-brightgreen?style=flat-square)
![Code style](https://img.shields.io/badge/code%20style-ruff-orange?style=flat-square)

A production-grade framework for evaluating LLM agent outputs. Built around the patterns that matter at scale: consistency under prompt variation, factual grounding, hallucination detection, decision explainability, and response drift over time.

This is not a research demo. It is the kind of eval infrastructure you build when silent model regressions cost you at billion-user scale.

---

## Metrics

| Metric | What it measures | Key signal |
|---|---|---|
| **Consistency** | Semantic stability across paraphrased prompts | Prompt-surface sensitivity |
| **Factual Grounding** | Sentence-level coverage against a reference corpus | Evidence alignment |
| **Explainability** | Reasoning chain quality relative to a conclusion | Causal density, step coverage |
| **Hallucination** | Unsupported factual claims against a grounded context | Claim-level entailment |
| **Drift** | Semantic shift over time or across model versions | Snapshot & sequence modes |

---

## Installation

```bash
git clone https://github.com/mukund1985/llm-eval-toolkit.git
cd llm-eval-toolkit
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```python
from llm_eval import LLMEvaluator
from llm_eval.evaluator import EvalRequest

evaluator = LLMEvaluator()

report = evaluator.evaluate(EvalRequest(
    request_id="demo-001",
    response="The Eiffel Tower is in Paris, built in 1887–1889.",
    context=["The Eiffel Tower is in Paris.", "Built from 1887 to 1889."],
    reference="The Eiffel Tower is a landmark in Paris, built 1887–1889.",
    reasoning="1. User asked about Eiffel Tower. 2. Context confirms Paris and dates. 3. Therefore I report both.",
    conclusion="Eiffel Tower is in Paris, built 1887–1889.",
    paraphrase_responses=[
        "The Eiffel Tower, located in Paris, was erected in the 1880s.",
        "Paris's Eiffel Tower dates to 1887–1889.",
        "Built 1887–1889, the Eiffel Tower stands in Paris.",
    ],
))

print(report.to_json())
```

## Batch Evaluation (Pipeline)

```python
from llm_eval.pipeline import EvalPipeline
from llm_eval.evaluator import EvalRequest

pipeline = EvalPipeline(max_concurrency=16)
result = pipeline.run([
    EvalRequest(request_id=f"req-{i}", response=resp, context=ctx)
    for i, (resp, ctx) in enumerate(dataset)
])

print(f"Pass rate: {result.pass_rate:.1%}")
result.save("eval_results.json")
```

## Architecture

```
llm_eval/
├── metrics/
│   ├── base.py               # MetricResult, BaseMetric
│   ├── consistency.py        # Semantic stability
│   ├── factual_grounding.py  # Reference alignment
│   ├── explainability.py     # Reasoning quality
│   ├── hallucination.py      # Claim-level grounding
│   └── drift.py              # Temporal drift
├── evaluator.py              # LLMEvaluator orchestrator
├── pipeline.py               # Async batch pipeline
└── utils/
    └── logging.py            # Structured logging (structlog)
```

## Design Principles

- **No external API calls in metrics** — all scoring runs locally with sentence-transformers. Fast, cheap, offline-capable.
- **Composable** — each metric is standalone. Use one or all.
- **Production-observable** — every result carries `latency_ms`, `confidence`, and structured `metadata`. Pipe directly into your observability stack.
- **CI-ready** — `EvalPipeline` is async-first with configurable concurrency. Designed to plug into shadow traffic systems.

## Running Tests

```bash
pytest tests/ -v
```

## License

MIT — see [LICENSE](LICENSE)
