"""
Fixed dataset prep, runtime utilities, and evaluation harness for classification autoresearch.

This file is intentionally read-only during experiments. The agent may edit only `train.py`.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import requests

CACHE_DIR = Path.home() / ".cache" / "autoresearch" / "classification"
DATASET_DIR = CACHE_DIR / "banking77"
RAW_DIR = DATASET_DIR / "raw"
TRAIN_PATH = DATASET_DIR / "train.jsonl"
VAL_PATH = DATASET_DIR / "val.jsonl"
LABELS_PATH = DATASET_DIR / "labels.json"
METADATA_PATH = DATASET_DIR / "metadata.json"
DEFAULT_MODEL_NAME = "qwen3.5:0.8b"
TRAIN_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
VAL_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"
RESULTS_ROOT = Path("results") / "classification"
CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string"}},
    "required": ["label"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Example:
    record_id: str
    text: str
    label: str


@dataclass(frozen=True)
class ExperimentContext:
    labels: list[str]
    train_examples: list[Example]


@dataclass(frozen=True)
class InferenceRequest:
    system_prompt: str
    user_prompt: str
    allowed_labels: list[str]
    options: dict[str, object]


@dataclass(frozen=True)
class OllamaGeneration:
    raw_output: str
    latency_ms: float
    model: str


@dataclass(frozen=True)
class PredictionRecord:
    record_id: str
    text: str
    gold_label: str
    pred_label: str | None
    raw_output: str
    is_valid: bool
    latency_ms: float
    candidate_labels: list[str]
    error: str | None = None


class OllamaError(RuntimeError):
    """Raised when the local Ollama server is unavailable or misconfigured."""


def canonicalize_label_name(value: str) -> str:
    return "_".join(value.strip().lower().replace("-", " ").split())


def humanize_label(label: str) -> str:
    return label.replace("_", " ")


def _download_csv_text(url: str, timeout_seconds: int = 60) -> str:
    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    return response.text


def _parse_csv_rows(csv_text: str) -> list[dict[str, str]]:
    reader = csv.DictReader(csv_text.splitlines())
    rows: list[dict[str, str]] = []
    for row in reader:
        text = (row.get("text") or "").strip()
        label = canonicalize_label_name(row.get("category") or row.get("label") or "")
        if text and label:
            rows.append({"text": text, "label": label})
    return rows


def _write_jsonl(path: Path, examples: list[Example]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example), ensure_ascii=True))
            handle.write("\n")


def _attach_record_ids(rows: list[dict[str, str]], split: str) -> list[Example]:
    return [
        Example(record_id=f"{split}-{index:05d}", text=row["text"], label=canonicalize_label_name(row["label"]))
        for index, row in enumerate(rows)
    ]


def _write_raw_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "category"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"text": row["text"], "category": row["label"]})


def prepare_dataset(
    dataset_dir: Path | str = DATASET_DIR,
    *,
    force: bool = False,
    train_rows: list[dict[str, str]] | None = None,
    val_rows: list[dict[str, str]] | None = None,
) -> dict[str, int | str]:
    dataset_path = Path(dataset_dir)
    raw_dir = dataset_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if train_rows is None:
        train_rows = _parse_csv_rows(_download_csv_text(TRAIN_URL))
    if val_rows is None:
        val_rows = _parse_csv_rows(_download_csv_text(VAL_URL))

    train_examples = _attach_record_ids(train_rows, "train")
    val_examples = _attach_record_ids(val_rows, "val")
    labels = sorted({example.label for example in train_examples + val_examples})

    if force or not (dataset_path / "train.jsonl").exists():
        _write_raw_csv(raw_dir / "train.csv", train_rows)
        _write_raw_csv(raw_dir / "val.csv", val_rows)
        _write_jsonl(dataset_path / "train.jsonl", train_examples)
        _write_jsonl(dataset_path / "val.jsonl", val_examples)
        with (dataset_path / "labels.json").open("w", encoding="utf-8") as handle:
            json.dump(labels, handle, indent=2)
            handle.write("\n")
        metadata = {
            "dataset": "banking77",
            "train_examples": len(train_examples),
            "val_examples": len(val_examples),
            "num_labels": len(labels),
            "train_url": TRAIN_URL,
            "val_url": VAL_URL,
            "validation_definition": "Banking77 official test split is the fixed val set.",
        }
        with (dataset_path / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
            handle.write("\n")
    else:
        with (dataset_path / "metadata.json").open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

    return metadata


def load_examples(split: str, dataset_dir: Path | str = DATASET_DIR) -> list[Example]:
    dataset_path = Path(dataset_dir)
    path = dataset_path / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run `uv run prepare.py` first.")
    examples: list[Example] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            examples.append(Example(record_id=row["record_id"], text=row["text"], label=row["label"]))
    return examples


def load_labels(dataset_dir: Path | str = DATASET_DIR) -> list[str]:
    with Path(dataset_dir, "labels.json").open("r", encoding="utf-8") as handle:
        return [canonicalize_label_name(label) for label in json.load(handle)]


class OllamaClient:
    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: int = 120,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()

    def _request(self, method: str, path: str, **kwargs: object) -> requests.Response:
        url = f"{self.base_url}{path}"
        try:
            response = self.session.request(method, url, timeout=self.timeout_seconds, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            raise OllamaError(f"Failed to reach Ollama at {url}: {exc}") from exc

    def list_models(self) -> list[str]:
        response = self._request("GET", "/api/tags")
        payload = response.json()
        return [str(item.get("name")) for item in payload.get("models", []) if item.get("name")]

    def ensure_model_available(self, model_name: str) -> None:
        models = self.list_models()
        if model_name not in models:
            raise OllamaError(
                f"Model {model_name!r} is not available in local Ollama. "
                f"Run `ollama pull {model_name}` first."
            )

    def generate_json(
        self,
        *,
        model: str,
        system_prompt: str,
        prompt: str,
        options: dict[str, object] | None = None,
        schema: dict[str, object] | None = None,
    ) -> OllamaGeneration:
        payload: dict[str, object] = {
            "model": model,
            "system": system_prompt,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": options or {},
        }
        if schema is not None:
            payload["format"] = schema
        response = self._request("POST", "/api/generate", json=payload)
        data = response.json()
        raw_output = data.get("response")
        if not isinstance(raw_output, str):
            raise OllamaError(f"Ollama response missing string `response`: {data!r}")
        total_duration = data.get("total_duration")
        latency_ms = float(total_duration) / 1_000_000.0 if isinstance(total_duration, (int, float)) else 0.0
        return OllamaGeneration(raw_output=raw_output, latency_ms=latency_ms, model=model)


def ensure_ollama_ready(model_name: str = DEFAULT_MODEL_NAME, client: OllamaClient | None = None) -> None:
    active_client = client or OllamaClient()
    active_client.ensure_model_available(model_name)


def parse_prediction(raw_output: str, allowed_labels: list[str]) -> tuple[str | None, bool, str | None]:
    aliases = {canonicalize_label_name(label): label for label in allowed_labels}
    aliases.update({canonicalize_label_name(humanize_label(label)): label for label in allowed_labels})
    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        return None, False, f"invalid_json:{exc.msg}"
    if not isinstance(payload, dict):
        return None, False, "invalid_payload:not_object"
    label_value = payload.get("label")
    if not isinstance(label_value, str) or not label_value.strip():
        return None, False, "invalid_payload:missing_label"
    normalized = aliases.get(canonicalize_label_name(label_value))
    if normalized is None:
        return None, False, "unknown_label"
    return normalized, True, None


def compute_metrics(records: list[dict[str, object]], labels: list[str]) -> dict[str, float | int]:
    total = len(records)
    if total == 0:
        raise ValueError("Need at least one record to compute metrics")

    accuracy_hits = 0
    invalid = 0
    per_label_f1: list[float] = []
    latencies = [float(record["latency_ms"]) for record in records if record.get("latency_ms") is not None]

    for record in records:
        if record.get("is_valid"):
            if record.get("gold_label") == record.get("pred_label"):
                accuracy_hits += 1
        else:
            invalid += 1

    for label in labels:
        tp = fp = fn = 0
        for record in records:
            gold = record.get("gold_label")
            pred = record.get("pred_label")
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
        "val_macro_f1": sum(per_label_f1) / len(labels),
        "val_accuracy": accuracy_hits / total,
        "invalid_rate": invalid / total,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
    }


def _write_prediction_artifacts(output_dir: Path, predictions: list[PredictionRecord], metrics: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(asdict(prediction), ensure_ascii=True))
            handle.write("\n")
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")


def print_summary(metrics: dict[str, object], total_seconds: float) -> None:
    print("---")
    print(f"val_macro_f1:     {float(metrics['val_macro_f1']):.6f}")
    print(f"val_accuracy:     {float(metrics['val_accuracy']):.6f}")
    print(f"invalid_rate:     {float(metrics['invalid_rate']):.6f}")
    print(f"avg_latency_ms:   {float(metrics['avg_latency_ms']):.1f}")
    print(f"total_examples:   {int(metrics['total_examples'])}")
    print(f"total_seconds:    {total_seconds:.1f}")


def run_experiment(
    build_request: Callable[[Example, ExperimentContext], InferenceRequest],
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    dataset_dir: Path | str = DATASET_DIR,
    output_root: Path | str = RESULTS_ROOT,
    client: OllamaClient | None = None,
) -> dict[str, object]:
    active_client = client or OllamaClient()
    if client is None:
        ensure_ollama_ready(model_name, client=active_client)

    labels = load_labels(dataset_dir)
    train_examples = load_examples("train", dataset_dir)
    val_examples = load_examples("val", dataset_dir)
    context = ExperimentContext(labels=labels, train_examples=train_examples)
    started = time.perf_counter()

    predictions: list[PredictionRecord] = []
    for example in val_examples:
        request = build_request(example, context)
        candidate_labels = [canonicalize_label_name(label) for label in request.allowed_labels if label in labels]
        if not candidate_labels:
            candidate_labels = labels
        generation = active_client.generate_json(
            model=model_name,
            system_prompt=request.system_prompt,
            prompt=request.user_prompt,
            options=request.options,
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
                candidate_labels=candidate_labels,
                error=error,
            )
        )

    metrics = compute_metrics(
        [
            {
                "gold_label": prediction.gold_label,
                "pred_label": prediction.pred_label,
                "is_valid": prediction.is_valid,
                "latency_ms": prediction.latency_ms,
            }
            for prediction in predictions
        ],
        labels,
    )
    total_seconds = time.perf_counter() - started
    metrics.update(
        {
            "model": model_name,
            "dataset_dir": str(Path(dataset_dir)),
            "output_root": str(Path(output_root)),
        }
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(output_root) / timestamp
    _write_prediction_artifacts(output_dir, predictions, metrics)
    print_summary(metrics, total_seconds)
    return {"metrics": metrics, "predictions": predictions, "output_dir": output_dir, "total_seconds": total_seconds}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the fixed Banking77 evaluation harness for autoresearch.")
    parser.add_argument("--force", action="store_true", help="Re-download and rewrite cached dataset artifacts.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Ollama model that must be present locally.")
    args = parser.parse_args()

    metadata = prepare_dataset(force=args.force)
    ensure_ollama_ready(args.model)
    print(json.dumps(metadata, indent=2))
    print(f"ollama_model:      {args.model}")
    print(f"dataset_dir:       {DATASET_DIR}")


if __name__ == "__main__":
    try:
        main()
    except OllamaError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
