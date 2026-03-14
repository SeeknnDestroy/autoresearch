"""Prompt-search utility for the Ollama classification lane."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .banking77 import Example, load_examples, load_labels
from .eval import PredictionRecord, evaluate_examples
from .metrics import compute_metrics
from .ollama import OllamaClient
from .profile import ClassificationProfile, DEFAULT_PROFILE_PATH, load_profile
from .retrieval import BowShortlistRetriever

SEARCH_OUTPUT_ROOT = Path("results") / "classification" / "search"

SYSTEM_PROMPTS = [
    "You are a precise banking intent classifier. Choose exactly one label from the allowed shortlist. Return valid JSON only.",
    "You classify banking support messages into Banking77 intents. Pick the single closest label from the allowed shortlist and return JSON only.",
    "Map the message to the most specific banking intent in the shortlist. Prefer the closest semantic match, not a broad related label. Return JSON only.",
    "You are a digital-banking label matcher. Select exactly one allowed shortlist label and return strict JSON only.",
    "Choose the best shortlist label for the banking support message. Prefer labels that explicitly match the product, action, or failure mentioned. Return JSON only.",
]

USER_PROMPT_TEMPLATES = [
    "Choose the best matching banking intent from the shortlist below.\n\nAllowed labels:\n{labels_block}\n\nReturn JSON with exactly this shape: {{\"label\": \"one_of_the_allowed_labels\"}}\n\nMessage:\n{text}",
    "Choose the closest banking intent from the shortlist below. Use only one allowed label.\n\nAllowed labels:\n{labels_block}\n\nIf several seem related, choose the most specific match.\n\nReturn JSON with exactly this shape: {{\"label\": \"one_of_the_allowed_labels\"}}\n\nMessage:\n{text}",
    "You must classify the message using only one shortlist label.\n\nAllowed labels:\n{labels_block}\n\nFew-shot examples:\n{few_shot_block}\n\nReturn JSON with exactly this shape: {{\"label\": \"one_of_the_allowed_labels\"}}\n\nMessage:\n{text}",
    "Classify the banking support message using exactly one shortlist label.\n\nAllowed labels:\n{labels_block}\n\nFocus on the customer's main request or failure.\n\nReturn JSON with exactly this shape: {{\"label\": \"one_of_the_allowed_labels\"}}\n\nMessage:\n{text}",
    "Pick the best shortlist label for the message.\n\nAllowed labels:\n{labels_block}\n\nUse the shortlist labels verbatim in the JSON response.\n\nFew-shot examples:\n{few_shot_block}\n\nReturn JSON with exactly this shape: {{\"label\": \"one_of_the_allowed_labels\"}}\n\nMessage:\n{text}",
]


def _balanced_examples(examples: Iterable[Example], offset: int = 0) -> list[Example]:
    by_label: dict[str, list[Example]] = {}
    for example in examples:
        by_label.setdefault(example.label, []).append(example)
    selected: list[Example] = []
    for label in sorted(by_label):
        bucket = by_label[label]
        index = min(offset, len(bucket) - 1)
        selected.append(bucket[index])
    return selected


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[PredictionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=True))
            handle.write("\n")


def _build_candidate_profiles(base_profile: ClassificationProfile) -> list[tuple[str, ClassificationProfile]]:
    candidates: list[tuple[str, ClassificationProfile]] = []
    few_shots = tuple(base_profile.few_shot_examples)
    for system_idx, system_prompt in enumerate(SYSTEM_PROMPTS):
        for template_idx, user_prompt_template in enumerate(USER_PROMPT_TEMPLATES):
            for num_predict in (16, 24):
                for top_k in (6, 10):
                    profile = ClassificationProfile(
                        model=base_profile.model,
                        system_prompt=system_prompt,
                        user_prompt_template=user_prompt_template,
                        options={"temperature": 0, "top_p": 1, "num_predict": num_predict},
                        few_shot_examples=few_shots,
                        label_descriptions={},
                        label_order="alphabetical",
                        candidate_selector={
                            "type": "bow_shortlist",
                            "top_k": top_k,
                            "max_examples_per_label": 80,
                        },
                    )
                    key = f"s{system_idx}-t{template_idx}-n{num_predict}-k{top_k}"
                    candidates.append((key, profile))
    return candidates


def _evaluate_profile(
    examples: list[Example],
    *,
    labels: list[str],
    train_examples: list[Example],
    profile: ClassificationProfile,
    client: OllamaClient,
) -> tuple[dict[str, float | int], list[PredictionRecord]]:
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
    )
    metrics = compute_metrics(
        [
            {
                "gold_label": row.gold_label,
                "pred_label": row.pred_label,
                "is_valid": row.is_valid,
                "latency_ms": row.latency_ms,
            }
            for row in predictions
        ],
        labels,
    )
    return metrics, predictions


def _append_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()), delimiter="\t")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_search(
    *,
    dataset_dir: Path,
    profile_path: Path,
    output_dir: Path,
    full_eval: bool,
    install_best: bool,
) -> dict[str, object]:
    labels = load_labels(dataset_dir)
    train_examples = load_examples(dataset_dir, "train")
    val_examples = load_examples(dataset_dir, "val")
    search_examples = _balanced_examples(val_examples, offset=0)
    baseline_examples = list(val_examples)
    client = OllamaClient(timeout_seconds=120)
    base_profile = load_profile(profile_path)
    candidates = _build_candidate_profiles(base_profile)
    output_dir.mkdir(parents=True, exist_ok=True)
    search_tsv = output_dir / "search_results.tsv"

    best_key = None
    best_profile = None
    best_metrics = None
    for experiment_index, (key, profile) in enumerate(candidates, start=1):
        metrics, _preds = _evaluate_profile(
            search_examples,
            labels=labels,
            train_examples=train_examples,
            profile=profile,
            client=client,
        )
        row = {
            "experiment": str(experiment_index),
            "key": key,
            "search_macro_f1": f"{float(metrics['macro_f1']):.6f}",
            "search_accuracy": f"{float(metrics['accuracy']):.6f}",
            "invalid_rate": f"{float(metrics['invalid_rate']):.6f}",
            "avg_latency_ms": f"{float(metrics['avg_latency_ms']):.1f}",
            "system_prompt": profile.system_prompt,
            "top_k": str(profile.candidate_selector["top_k"]),
            "num_predict": str(profile.options["num_predict"]),
        }
        _append_row(search_tsv, row)
        if best_metrics is None or float(metrics["macro_f1"]) > float(best_metrics["macro_f1"]):
            best_key = key
            best_profile = profile
            best_metrics = metrics
            _write_json(output_dir / "best_profile.json", profile_to_dict(profile))

    assert best_profile is not None
    assert best_metrics is not None

    baseline_metrics = None
    best_full_metrics = None
    if full_eval:
        baseline_metrics, baseline_predictions = _evaluate_profile(
            baseline_examples,
            labels=labels,
            train_examples=train_examples,
            profile=base_profile,
            client=client,
        )
        best_full_metrics, best_predictions = _evaluate_profile(
            baseline_examples,
            labels=labels,
            train_examples=train_examples,
            profile=best_profile,
            client=client,
        )
        _write_json(output_dir / "baseline_full_metrics.json", baseline_metrics)
        _write_json(output_dir / "best_full_metrics.json", best_full_metrics)
        _write_jsonl(output_dir / "baseline_full_predictions.jsonl", baseline_predictions)
        _write_jsonl(output_dir / "best_full_predictions.jsonl", best_predictions)

        if install_best and float(best_full_metrics["macro_f1"]) > float(baseline_metrics["macro_f1"]):
            _write_json(profile_path, profile_to_dict(best_profile))

    summary = {
        "search_examples": len(search_examples),
        "num_experiments": len(candidates),
        "best_key": best_key,
        "best_search_metrics": best_metrics,
        "baseline_full_metrics": baseline_metrics,
        "best_full_metrics": best_full_metrics,
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def profile_to_dict(profile: ClassificationProfile) -> dict:
    return {
        "model": profile.model,
        "system_prompt": profile.system_prompt,
        "user_prompt_template": profile.user_prompt_template,
        "options": profile.options,
        "label_order": profile.label_order,
        "label_descriptions": profile.label_descriptions,
        "few_shot_examples": [
            {"text": example.text, "label": example.label}
            for example in profile.few_shot_examples
        ],
        "candidate_selector": profile.candidate_selector,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prompt search for the classification lane")
    parser.add_argument("--dataset-dir", type=Path, default=Path.home() / ".cache" / "autoresearch" / "classification" / "banking77")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip-full-eval", action="store_true")
    parser.add_argument("--install-best", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (SEARCH_OUTPUT_ROOT / timestamp)
    summary = run_search(
        dataset_dir=args.dataset_dir,
        profile_path=args.profile,
        output_dir=output_dir,
        full_eval=not args.skip_full_eval,
        install_best=args.install_best,
    )
    print(json.dumps(summary, indent=2))
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
