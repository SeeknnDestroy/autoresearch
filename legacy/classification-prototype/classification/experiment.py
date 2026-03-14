"""Experiment loop helpers for evaluation-first classification runs."""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

from .banking77 import DEFAULT_DATASET_DIR
from .eval import evaluate_split
from .profile import DEFAULT_PROFILE_PATH

DEFAULT_RESULTS_PATH = Path("results") / "classification" / "banking77_results.tsv"
RESULT_FIELDS = [
    "commit",
    "val_macro_f1",
    "val_accuracy",
    "invalid_rate",
    "avg_latency_ms",
    "status",
    "description",
]


def _read_results(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader)


def _write_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, delimiter="\t")
        writer.writeheader()


def _append_row(path: Path, row: dict[str, str]) -> None:
    _write_header(path)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, delimiter="\t")
        writer.writerow(row)


def _best_keep_score(rows: list[dict[str, str]]) -> float | None:
    keep_scores = [float(row["val_macro_f1"]) for row in rows if row.get("status") == "keep"]
    return max(keep_scores) if keep_scores else None


def _git_short_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def decide_status(current_score: float, prior_rows: list[dict[str, str]]) -> str:
    best_score = _best_keep_score(prior_rows)
    if best_score is None:
        return "keep"
    return "keep" if current_score > best_score else "discard"


def run_experiment_loop(
    *,
    dataset_dir: Path | str = DEFAULT_DATASET_DIR,
    profile_path: Path | str = DEFAULT_PROFILE_PATH,
    results_path: Path | str = DEFAULT_RESULTS_PATH,
    output_dir: Path | str | None = None,
    limit: int | None = None,
    description: str = "manual eval",
    base_url: str = "http://127.0.0.1:11434",
    timeout_seconds: int = 120,
) -> dict[str, object]:
    results_file = Path(results_path)
    prior_rows = _read_results(results_file)
    evaluation = evaluate_split(
        "val",
        dataset_dir=dataset_dir,
        profile_path=profile_path,
        output_dir=output_dir,
        limit=limit,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        ensure_model=True,
    )
    metrics = evaluation["metrics"]
    status = decide_status(float(metrics["macro_f1"]), prior_rows)
    row = {
        "commit": _git_short_commit(),
        "val_macro_f1": f"{float(metrics['macro_f1']):.6f}",
        "val_accuracy": f"{float(metrics['accuracy']):.6f}",
        "invalid_rate": f"{float(metrics['invalid_rate']):.6f}",
        "avg_latency_ms": f"{float(metrics['avg_latency_ms']):.1f}",
        "status": status,
        "description": description,
    }
    _append_row(results_file, row)
    return {
        "status": status,
        "results_path": results_file,
        "row": row,
        **evaluation,
    }
