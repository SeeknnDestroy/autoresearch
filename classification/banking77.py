"""Dataset preparation utilities for Banking77."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests

DEFAULT_DATASET_DIR = Path.home() / ".cache" / "autoresearch" / "classification" / "banking77"
RAW_DIRNAME = "raw"
SPLITS_DIRNAME = "splits"
TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"
LABELS_FILENAME = "labels.json"
META_FILENAME = "metadata.json"
TRAIN_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
TEST_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"


@dataclass(frozen=True)
class Example:
    record_id: str
    text: str
    label: str


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
        if not text or not label:
            continue
        rows.append({"text": text, "label": label})
    return rows


def _write_raw_csv(path: Path, rows: Iterable[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "category"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"text": row["text"], "category": row["label"]})


def _shuffle_in_place(items: list[dict[str, str]], seed: int, label: str) -> None:
    random.Random(f"{seed}:{label}").shuffle(items)


def stratified_split_train_validation(
    rows: Iterable[dict[str, str]],
    val_fraction: float = 0.2,
    seed: int = 1337,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str]]:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")

    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        label = canonicalize_label_name(row["label"])
        buckets[label].append({"text": row["text"], "label": label})

    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    labels = sorted(buckets)
    for label in labels:
        bucket = list(buckets[label])
        _shuffle_in_place(bucket, seed, label)
        if len(bucket) == 1:
            n_val = 0
        else:
            n_val = max(1, min(len(bucket) - 1, round(len(bucket) * val_fraction)))
        val_rows.extend(bucket[:n_val])
        train_rows.extend(bucket[n_val:])

    return train_rows, val_rows, labels


def _attach_record_ids(rows: Iterable[dict[str, str]], split: str) -> list[Example]:
    return [
        Example(record_id=f"{split}-{index:05d}", text=row["text"], label=canonicalize_label_name(row["label"]))
        for index, row in enumerate(rows)
    ]


def _write_jsonl(path: Path, examples: Iterable[Example]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(
                json.dumps(
                    {
                        "id": example.record_id,
                        "text": example.text,
                        "label": example.label,
                    },
                    ensure_ascii=True,
                )
            )
            handle.write("\n")


def load_examples(dataset_dir: Path | str, split: str) -> list[Example]:
    dataset_path = Path(dataset_dir)
    path = dataset_path / SPLITS_DIRNAME / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}. Run prepare first.")
    examples: list[Example] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            examples.append(Example(record_id=row["id"], text=row["text"], label=row["label"]))
    return examples


def load_labels(dataset_dir: Path | str) -> list[str]:
    dataset_path = Path(dataset_dir)
    path = dataset_path / LABELS_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Missing labels file: {path}. Run prepare first.")
    with path.open("r", encoding="utf-8") as handle:
        labels = json.load(handle)
    return [canonicalize_label_name(label) for label in labels]


def prepare_dataset(
    dataset_dir: Path | str = DEFAULT_DATASET_DIR,
    *,
    val_fraction: float = 0.2,
    seed: int = 1337,
    force: bool = False,
    train_rows: list[dict[str, str]] | None = None,
    test_rows: list[dict[str, str]] | None = None,
) -> dict[str, int | float | str]:
    dataset_path = Path(dataset_dir)
    raw_dir = dataset_path / RAW_DIRNAME
    splits_dir = dataset_path / SPLITS_DIRNAME
    raw_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    if train_rows is None:
        train_rows = _parse_csv_rows(_download_csv_text(TRAIN_URL))
    if test_rows is None:
        test_rows = _parse_csv_rows(_download_csv_text(TEST_URL))

    if force or not (raw_dir / TRAIN_FILENAME).exists():
        _write_raw_csv(raw_dir / TRAIN_FILENAME, train_rows)
    if force or not (raw_dir / TEST_FILENAME).exists():
        _write_raw_csv(raw_dir / TEST_FILENAME, test_rows)

    prepared_train, prepared_val, labels = stratified_split_train_validation(
        train_rows,
        val_fraction=val_fraction,
        seed=seed,
    )
    prepared_test = [
        {"text": row["text"], "label": canonicalize_label_name(row["label"])}
        for row in test_rows
    ]

    split_examples = {
        "train": _attach_record_ids(prepared_train, "train"),
        "val": _attach_record_ids(prepared_val, "val"),
        "test": _attach_record_ids(prepared_test, "test"),
    }
    for split, examples in split_examples.items():
        _write_jsonl(splits_dir / f"{split}.jsonl", examples)

    with (dataset_path / LABELS_FILENAME).open("w", encoding="utf-8") as handle:
        json.dump(labels, handle, indent=2)
        handle.write("\n")

    metadata = {
        "dataset": "banking77",
        "seed": seed,
        "val_fraction": val_fraction,
        "train_examples": len(split_examples["train"]),
        "val_examples": len(split_examples["val"]),
        "test_examples": len(split_examples["test"]),
        "num_labels": len(labels),
        "train_url": TRAIN_URL,
        "test_url": TEST_URL,
    }
    with (dataset_path / META_FILENAME).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    return metadata
