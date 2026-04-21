"""
Basic evaluation example demonstrating all five metrics.

Run: python examples/basic_eval.py
"""
from llm_eval import LLMEvaluator
from llm_eval.evaluator import EvalRequest
from llm_eval.utils import configure_logging

configure_logging(level="INFO")

evaluator = LLMEvaluator()

request = EvalRequest(
    request_id="demo-001",
    response=(
        "The Eiffel Tower is located in Paris, France. "
        "It was constructed between 1887 and 1889 as the entrance arch "
        "for the 1889 World's Fair."
    ),
    reference=(
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars "
        "in Paris, France. It was named after the engineer Gustave Eiffel, whose "
        "company designed and built the tower from 1887 to 1889 as the entrance "
        "arch for the 1889 World's Fair."
    ),
    context=[
        "The Eiffel Tower stands in Paris, France.",
        "Construction began in 1887 and finished in 1889.",
        "It was built for the World's Fair.",
    ],
    reasoning=(
        "1. The user asked about the Eiffel Tower's location and history. "
        "2. The context confirms it is in Paris. "
        "3. Therefore I provided the location and construction dates."
    ),
    conclusion="The Eiffel Tower is in Paris and was built in 1887–1889.",
    paraphrase_responses=[
        "The Eiffel Tower is a famous landmark in Paris, France, built in the late 1880s.",
        "Paris is home to the Eiffel Tower, which was erected between 1887 and 1889.",
        "Located in Paris, the Eiffel Tower was constructed in 1887–1889 for the World's Fair.",
    ],
    response_history=[
        "The Eiffel Tower is in France.",
        "Specifically, the Eiffel Tower is located in Paris, France.",
        "The Eiffel Tower in Paris was built from 1887 to 1889.",
    ],
)

report = evaluator.evaluate(request)
print(report.to_json())
