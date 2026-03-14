"""Evaluation helpers for prompt-based classification."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .banking77 import DEFAULT_DATASET_DIR, Example, canonicalize_label_name, load_examples, load_labels
from .metrics import compute_metrics
from .ollama import OllamaClient
from .profile import ClassificationProfile, DEFAULT_PROFILE_PATH, load_profile
from .retrieval import BowShortlistRetriever

DEFAULT_OUTPUT_ROOT = Path("results") / "classification"
CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
    },
    "required": ["label"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class PredictionRecord:
    record_id: str
    text: str
    gold_label: str
    pred_label: str | None
    raw_output: str
    is_valid: bool
    latency_ms: float
    error: str | None = None


def _label_aliases(labels: list[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for label in labels:
        aliases[canonicalize_label_name(label)] = label
        aliases[canonicalize_label_name(label.replace("_", " "))] = label
    return aliases


def parse_prediction(raw_output: str, labels: list[str]) -> tuple[str | None, bool, str | None]:
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        return None, False, f"invalid_json:{exc.msg}"
    if not isinstance(data, dict):
        return None, False, "invalid_payload:not_object"
    label_value = data.get("label")
    if not isinstance(label_value, str) or not label_value.strip():
        return None, False, "invalid_payload:missing_label"
    aliases = _label_aliases(labels)
    normalized = aliases.get(canonicalize_label_name(label_value))
    if normalized is None:
        return None, False, "unknown_label"
    return normalized, True, None


def _default_output_dir(split: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_ROOT / f"{timestamp}-{split}"


def _write_jsonl(path: Path, rows: list[PredictionRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=True))
            handle.write("\n")


def evaluate_examples(
    examples: list[Example],
    *,
    labels: list[str],
    profile: ClassificationProfile,
    client: OllamaClient,
    retriever: BowShortlistRetriever | None = None,
    limit: int | None = None,
) -> list[PredictionRecord]:
    predictions: list[PredictionRecord] = []
    active_examples = examples[:limit] if limit is not None else examples
    for example in active_examples:
        candidate_labels = retriever.shortlist(example.text, labels) if retriever is not None else labels
        prompt = profile.render_prompt(example.text, candidate_labels)
        generation = client.generate_json(
            model=profile.model,
            system_prompt=profile.system_prompt,
            prompt=prompt,
            options=profile.options,
            schema=CLASSIFICATION_SCHEMA,
        )
        pred_label, is_valid, error = parse_prediction(generation.raw_output, candidate_labels)
        predictions.append(
            PredictionRecord(
                record_id=example.record_id,
                text=example.text,
                gold_label=example.label,
                pred_label=pred_label,
                raw_output=generation.raw_output,
                is_valid=is_valid,
                latency_ms=generation.latency_ms,
                error=error,
            )
        )
    return predictions


def evaluate_split(
    split: str,
    *,
    dataset_dir: Path | str = DEFAULT_DATASET_DIR,
    profile_path: Path | str = DEFAULT_PROFILE_PATH,
    output_dir: Path | str | None = None,
    limit: int | None = None,
    base_url: str = "http://127.0.0.1:11434",
    timeout_seconds: int = 120,
    ensure_model: bool = True,
) -> dict[str, object]:
    labels = load_labels(dataset_dir)
    examples = load_examples(dataset_dir, split)
    train_examples = load_examples(dataset_dir, "train")
    profile = load_profile(profile_path)
    client = OllamaClient(base_url=base_url, timeout_seconds=timeout_seconds)
    if ensure_model:
        client.ensure_model_available(profile.model)
    retriever = None
    if profile.candidate_selector and profile.candidate_selector.get("type") == "bow_shortlist":
        retriever = BowShortlistRetriever.from_examples(
            train_examples,
            top_k=int(profile.candidate_selector.get("top_k", 8)),
            max_examples_per_label=int(profile.candidate_selector.get("max_examples_per_label", 80)),
        )

    predictions = evaluate_examples(
        examples,
        labels=labels,
        profile=profile,
        client=client,
        retriever=retriever,
        limit=limit,
    )
    metric_rows = [
        {
            "gold_label": row.gold_label,
            "pred_label": row.pred_label,
            "is_valid": row.is_valid,
            "latency_ms": row.latency_ms,
        }
        for row in predictions
    ]
    metrics = compute_metrics(metric_rows, labels)
    metrics.update(
        {
            "split": split,
            "model": profile.model,
            "dataset_dir": str(Path(dataset_dir)),
            "profile_path": str(Path(profile_path)),
        }
    )

    destination = Path(output_dir) if output_dir is not None else _default_output_dir(split)
    destination.mkdir(parents=True, exist_ok=True)
    predictions_path = destination / "predictions.jsonl"
    metrics_path = destination / "metrics.json"
    _write_jsonl(predictions_path, predictions)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")

    return {
        "metrics": metrics,
        "predictions_path": predictions_path,
        "metrics_path": metrics_path,
        "output_dir": destination,
    }
