"""CLI for the local classification lane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .banking77 import DEFAULT_DATASET_DIR, prepare_dataset
from .eval import evaluate_split
from .experiment import DEFAULT_RESULTS_PATH, run_experiment_loop
from .profile import DEFAULT_PROFILE_PATH


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autoresearch classification lane")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Download and prepare Banking77")
    prepare_parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    prepare_parser.add_argument("--val-fraction", type=float, default=0.2)
    prepare_parser.add_argument("--seed", type=int, default=1337)
    prepare_parser.add_argument("--force", action="store_true")

    eval_parser = subparsers.add_parser("eval", help="Evaluate a dataset split through Ollama")
    eval_parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    eval_parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    eval_parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH)
    eval_parser.add_argument("--output-dir", type=Path)
    eval_parser.add_argument("--limit", type=int)
    eval_parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    eval_parser.add_argument("--timeout-seconds", type=int, default=120)
    eval_parser.add_argument("--skip-model-check", action="store_true")

    loop_parser = subparsers.add_parser("loop", help="Run one keep/discard evaluation iteration")
    loop_parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    loop_parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH)
    loop_parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS_PATH)
    loop_parser.add_argument("--output-dir", type=Path)
    loop_parser.add_argument("--limit", type=int)
    loop_parser.add_argument("--description", default="manual eval")
    loop_parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    loop_parser.add_argument("--timeout-seconds", type=int, default=120)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "prepare":
        metadata = prepare_dataset(
            args.dataset_dir,
            val_fraction=args.val_fraction,
            seed=args.seed,
            force=args.force,
        )
        print(json.dumps(metadata, indent=2))
        return

    if args.command == "eval":
        result = evaluate_split(
            args.split,
            dataset_dir=args.dataset_dir,
            profile_path=args.profile,
            output_dir=args.output_dir,
            limit=args.limit,
            base_url=args.base_url,
            timeout_seconds=args.timeout_seconds,
            ensure_model=not args.skip_model_check,
        )
        print(json.dumps(result["metrics"], indent=2))
        print(f"predictions_path={result['predictions_path']}")
        print(f"metrics_path={result['metrics_path']}")
        return

    if args.command == "loop":
        result = run_experiment_loop(
            dataset_dir=args.dataset_dir,
            profile_path=args.profile,
            results_path=args.results_file,
            output_dir=args.output_dir,
            limit=args.limit,
            description=args.description,
            base_url=args.base_url,
            timeout_seconds=args.timeout_seconds,
        )
        print(json.dumps(result["row"], indent=2))
        print(f"results_path={result['results_path']}")
        print(f"metrics_path={result['metrics_path']}")
        return

    parser.error(f"Unsupported command: {args.command}")
