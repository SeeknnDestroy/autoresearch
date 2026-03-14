"""Lightweight label shortlist retrieval for Banking77 classification."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from .banking77 import Example, canonicalize_label_name

TOKEN_RE = re.compile(r"[a-z0-9']+")


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


@dataclass(frozen=True)
class RetrieverConfig:
    top_k: int = 8
    max_examples_per_label: int = 80


class BowShortlistRetriever:
    def __init__(self, label_vectors: dict[str, Counter[str]], config: RetrieverConfig) -> None:
        self.label_vectors = label_vectors
        self.config = config

    @classmethod
    def from_examples(
        cls,
        examples: Iterable[Example],
        *,
        top_k: int = 8,
        max_examples_per_label: int = 80,
    ) -> "BowShortlistRetriever":
        buckets: dict[str, list[Example]] = defaultdict(list)
        for example in examples:
            buckets[canonicalize_label_name(example.label)].append(example)

        label_vectors: dict[str, Counter[str]] = {}
        for label, label_examples in buckets.items():
            counts: Counter[str] = Counter(tokenize(label))
            for example in label_examples[:max_examples_per_label]:
                counts.update(tokenize(example.text))
            label_vectors[label] = counts

        return cls(
            label_vectors=label_vectors,
            config=RetrieverConfig(top_k=top_k, max_examples_per_label=max_examples_per_label),
        )

    def shortlist(self, text: str, labels: list[str]) -> list[str]:
        query_vector = Counter(tokenize(text))
        ranked = sorted(
            (
                (label, cosine_similarity(query_vector, self.label_vectors.get(label, Counter())))
                for label in labels
            ),
            key=lambda item: (-item[1], item[0]),
        )
        shortlist = [label for label, _score in ranked[: self.config.top_k]]
        return shortlist or labels[: self.config.top_k]
