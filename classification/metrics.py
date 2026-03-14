"""Metrics for classification experiments."""

from __future__ import annotations

from typing import Iterable


def compute_metrics(records: Iterable[dict[str, object]], labels: list[str]) -> dict[str, float | int]:
    rows = list(records)
    total = len(rows)
    if total == 0:
        raise ValueError("Need at least one record to compute metrics")

    accuracy_hits = 0
    invalid = 0
    per_label_f1: list[float] = []
    latencies = [float(row["latency_ms"]) for row in rows if row.get("latency_ms") is not None]

    for row in rows:
        if row.get("is_valid"):
            if row.get("gold_label") == row.get("pred_label"):
                accuracy_hits += 1
        else:
            invalid += 1

    for label in labels:
        tp = fp = fn = 0
        for row in rows:
            gold = row.get("gold_label")
            pred = row.get("pred_label")
            if gold == label and pred == label:
                tp += 1
            elif gold != label and pred == label:
                fp += 1
            elif gold == label and pred != label:
                fn += 1
        denom = (2 * tp) + fp + fn
        per_label_f1.append(0.0 if denom == 0 else (2 * tp) / denom)

    return {
        "total_examples": total,
        "macro_f1": sum(per_label_f1) / len(labels),
        "accuracy": accuracy_hits / total,
        "invalid_rate": invalid / total,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
    }
