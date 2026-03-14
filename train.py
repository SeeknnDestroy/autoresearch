"""
Mutable experiment file for classification autoresearch.

All strategy changes should happen here. `prepare.py` owns the fixed evaluator.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter, defaultdict

from prepare import (
    DEFAULT_MODEL_NAME,
    Example,
    ExperimentContext,
    InferenceRequest,
    humanize_label,
    run_experiment,
)

# ---------------------------------------------------------------------------
# Mutable experiment surface
# ---------------------------------------------------------------------------

MODEL_NAME = DEFAULT_MODEL_NAME
SHORTLIST_SIZE = 8
MAX_EXAMPLES_PER_LABEL = 80
TEMPERATURE = 0.0
TOP_P = 1.0
NUM_PREDICT = 24

SYSTEM_PROMPT = (
    "You are a precise banking intent classifier. "
    "Choose exactly one label from the allowed shortlist. "
    "Return valid JSON only."
)

USER_PROMPT_TEMPLATE = """Choose the best matching banking intent from the shortlist below.

Allowed labels:
{labels_block}

Few-shot examples:
{few_shot_block}

Return JSON with exactly this shape: {{"label": "one_of_the_allowed_labels"}}

Message:
{text}"""

FEW_SHOT_EXAMPLES = (
    ("My cash withdrawal was charged twice.", "cash_withdrawal"),
    ("I need to change the phone number on my account.", "change_phone_number"),
    ("Why did my transfer fail?", "beneficiary_not_allowed"),
)

TOKEN_RE = re.compile(r"[a-z0-9']+")
_LABEL_VECTOR_CACHE: dict[str, Counter[str]] | None = None


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower().replace("_", " "))


def cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(token, 0) for token, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def build_label_vectors(context: ExperimentContext) -> dict[str, Counter[str]]:
    global _LABEL_VECTOR_CACHE
    if _LABEL_VECTOR_CACHE is not None:
        return _LABEL_VECTOR_CACHE

    buckets: dict[str, list[Example]] = defaultdict(list)
    for example in context.train_examples:
        buckets[example.label].append(example)

    label_vectors: dict[str, Counter[str]] = {}
    for label, examples in buckets.items():
        counts = Counter(tokenize(label))
        for example in examples[:MAX_EXAMPLES_PER_LABEL]:
            counts.update(tokenize(example.text))
        label_vectors[label] = counts

    _LABEL_VECTOR_CACHE = label_vectors
    return label_vectors


def shortlist_labels(example: Example, context: ExperimentContext) -> list[str]:
    label_vectors = build_label_vectors(context)
    query_vector = Counter(tokenize(example.text))
    ranked = sorted(
        (
            (label, cosine_similarity(query_vector, label_vectors.get(label, Counter())))
            for label in context.labels
        ),
        key=lambda item: (-item[1], item[0]),
    )
    shortlist = [label for label, _score in ranked[:SHORTLIST_SIZE]]
    return shortlist or context.labels[:SHORTLIST_SIZE]


def render_labels_block(labels: list[str]) -> str:
    return "\n".join(f"- {label}: {humanize_label(label)}" for label in labels)


def render_few_shot_block() -> str:
    return "\n\n".join(f'Text: "{text}"\nLabel: {label}' for text, label in FEW_SHOT_EXAMPLES)


def build_request(example: Example, context: ExperimentContext) -> InferenceRequest:
    candidate_labels = shortlist_labels(example, context)
    prompt = USER_PROMPT_TEMPLATE.format(
        labels_block=render_labels_block(candidate_labels),
        few_shot_block=render_few_shot_block(),
        text=example.text,
    )
    return InferenceRequest(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=prompt,
        allowed_labels=candidate_labels,
        options={
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "num_predict": NUM_PREDICT,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one classification autoresearch experiment.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional smoke-test limit for the number of validation examples to evaluate. "
        "Omit for the full fixed validation run.",
    )
    args = parser.parse_args()
    run_experiment(build_request, model_name=MODEL_NAME, limit=args.limit)


if __name__ == "__main__":
    main()
