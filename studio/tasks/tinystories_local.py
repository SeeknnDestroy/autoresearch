from __future__ import annotations

import difflib
import hashlib
import json
import math
import random
import resource
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from studio.tasks.base import TaskBackend, TaskRunResult


DEFAULT_RECIPE = {
    "seed": 7,
    "sequence_len": 48,
    "batch_size": 18,
    "embedding_dim": 48,
    "hidden_dim": 104,
    "num_layers": 1,
    "dropout": 0.10,
    "learning_rate": 0.012,
    "steps": 48,
    "eval_batches": 8,
    "sample_length": 200,
    "temperature": 0.90,
}

FIELD_ORDER = [
    "learning_rate",
    "hidden_dim",
    "sequence_len",
    "dropout",
    "batch_size",
    "embedding_dim",
]

FIELD_BOUNDS = {
    "learning_rate": (0.004, 0.04),
    "hidden_dim": (48, 192),
    "sequence_len": (24, 80),
    "dropout": (0.0, 0.35),
    "batch_size": (8, 32),
    "embedding_dim": (24, 96),
}


class TinyStoryRNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, idx: torch.Tensor, hidden: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(idx)
        x, hidden = self.rnn(x, hidden)
        x = self.dropout(x)
        logits = self.head(x)
        return logits, hidden


class TinyStoriesLocalBackend(TaskBackend):
    task_id = "tinystories-local"

    def __init__(
        self,
        root: Path,
        *,
        device_preference: str = "auto",
        recipe_overrides: dict[str, Any] | None = None,
        corpus_path: Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.workspace_dir = self.root / "task_backend"
        self.recipes_dir = self.workspace_dir / "recipes"
        self.current_recipe_path = self.workspace_dir / "current_recipe.json"
        self.corpus_path = corpus_path or Path(__file__).resolve().parents[1] / "data" / "tiny_stories.txt"
        self.device_preference = device_preference
        self.recipe_overrides = recipe_overrides or {}
        self._rng = random.Random(1337)
        self._dataset_cache: dict[str, Any] | None = None

    def prepare(self) -> dict[str, Any]:
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.recipes_dir.mkdir(parents=True, exist_ok=True)
        recipe = {**DEFAULT_RECIPE, **self.recipe_overrides}
        if not self.current_recipe_path.exists():
            self._write_recipe(recipe)
        return self.get_current_recipe()

    def get_current_recipe(self) -> dict[str, Any]:
        return json.loads(self.current_recipe_path.read_text(encoding="utf-8"))

    def recipe_sha(self, recipe: dict[str, Any]) -> str:
        payload = json.dumps(recipe, sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

    def _recipe_snapshot_path(self, recipe_sha: str) -> Path:
        return self.recipes_dir / f"{recipe_sha}.json"

    def _write_recipe(self, recipe: dict[str, Any]) -> str:
        recipe_sha = self.recipe_sha(recipe)
        self.current_recipe_path.write_text(json.dumps(recipe, indent=2), encoding="utf-8")
        self._recipe_snapshot_path(recipe_sha).write_text(json.dumps(recipe, indent=2), encoding="utf-8")
        return recipe_sha

    def rollback(self, target_sha: str) -> dict[str, Any]:
        snapshot = self._recipe_snapshot_path(target_sha)
        recipe = json.loads(snapshot.read_text(encoding="utf-8"))
        self._write_recipe(recipe)
        return recipe

    def run_baseline(self, run_dir: Path | None = None) -> TaskRunResult:
        self.prepare()
        recipe = self.get_current_recipe()
        if run_dir is None:
            raise ValueError("run_dir is required")
        return self._run_recipe(recipe, run_dir)

    def run_candidate(self, change_spec: dict[str, Any], run_dir: Path | None = None) -> TaskRunResult:
        self.prepare()
        if run_dir is None:
            raise ValueError("run_dir is required")
        base_recipe = self.get_current_recipe()
        recipe = self.apply_change_spec(base_recipe, change_spec)
        return self._run_recipe(recipe, run_dir)

    def extract_metrics(self, run_dir: Path) -> dict[str, Any]:
        metrics_path = Path(run_dir) / "metrics.json"
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    def generate_samples(self, run_dir: Path) -> str:
        sample_path = Path(run_dir) / "sample.txt"
        return sample_path.read_text(encoding="utf-8")

    def propose_change(self, runs: list[Any], current_recipe: dict[str, Any]) -> dict[str, Any]:
        tried = {getattr(run, "novelty_fingerprint", "") for run in runs}
        for offset, field in enumerate(FIELD_ORDER):
            direction = "up" if (len(runs) + offset) % 2 == 0 else "down"
            spec = self._change_spec(current_recipe, field, direction)
            if spec["novelty_fingerprint"] not in tried:
                return spec
        field = FIELD_ORDER[len(runs) % len(FIELD_ORDER)]
        direction = "down" if len(runs) % 2 else "up"
        return self._change_spec(current_recipe, field, direction)

    def apply_change_spec(self, recipe: dict[str, Any], change_spec: dict[str, Any]) -> dict[str, Any]:
        updated = dict(recipe)
        field = change_spec["field"]
        updated[field] = change_spec["to"]
        updated["seed"] = int(recipe["seed"]) + 17
        return updated

    def recipe_diff(self, before: dict[str, Any], after: dict[str, Any]) -> str:
        before_text = json.dumps(before, indent=2, sort_keys=True).splitlines()
        after_text = json.dumps(after, indent=2, sort_keys=True).splitlines()
        diff = difflib.unified_diff(before_text, after_text, fromfile="current_recipe.json", tofile="candidate_recipe.json", lineterm="")
        return "\n".join(diff)

    def _change_spec(self, recipe: dict[str, Any], field: str, direction: str) -> dict[str, Any]:
        current_value = recipe[field]
        low, high = FIELD_BOUNDS[field]
        if field == "learning_rate":
            updated = round(current_value * (1.25 if direction == "up" else 0.8), 5)
        elif field == "dropout":
            updated = round(current_value + (0.04 if direction == "up" else -0.04), 2)
        elif field == "batch_size":
            updated = current_value + (4 if direction == "up" else -4)
        else:
            updated = current_value + (12 if direction == "up" else -12)
        updated = max(low, min(high, updated))
        if isinstance(current_value, int):
            updated = int(updated)
        rationale = {
            "learning_rate": "trade cautious fitting against faster adaptation",
            "hidden_dim": "trade memory against expressiveness",
            "sequence_len": "trade local speed against narrative continuity",
            "dropout": "trade confidence against regularization",
            "batch_size": "trade noisier updates against smoother gradients",
            "embedding_dim": "trade lexical nuance against speed",
        }[field]
        direction_text = "push" if direction == "up" else "trim"
        title = f"{direction_text.title()} {field.replace('_', ' ')}"
        fingerprint = f"{field}:{direction}:{updated}"
        return {
            "field": field,
            "from": current_value,
            "to": updated,
            "direction": direction,
            "title": title,
            "rationale": f"{direction_text} {field.replace('_', ' ')} to {rationale}.",
            "novelty_fingerprint": fingerprint,
        }

    def _load_dataset(self) -> dict[str, Any]:
        if self._dataset_cache is not None:
            return self._dataset_cache
        text = self.corpus_path.read_text(encoding="utf-8").strip()
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}
        encoded = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
        split_idx = int(len(encoded) * 0.9)
        train_data = encoded[:split_idx]
        val_data = encoded[split_idx:]
        dataset = {
            "text": text,
            "stoi": stoi,
            "itos": itos,
            "vocab_size": len(chars),
            "train_data": train_data,
            "val_data": val_data,
        }
        self._dataset_cache = dataset
        return dataset

    def _device(self) -> torch.device:
        if self.device_preference == "cpu":
            return torch.device("cpu")
        if self.device_preference == "mps":
            return torch.device("mps")
        if self.device_preference == "cuda":
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _batch(self, data: torch.Tensor, batch_size: int, sequence_len: int, generator: torch.Generator, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = len(data) - sequence_len - 1
        starts = torch.randint(0, max_start, (batch_size,), generator=generator)
        x = torch.stack([data[start:start + sequence_len] for start in starts.tolist()])
        y = torch.stack([data[start + 1:start + sequence_len + 1] for start in starts.tolist()])
        return x.to(device), y.to(device)

    def _evaluate(self, model: TinyStoryRNN, data: torch.Tensor, recipe: dict[str, Any], generator: torch.Generator, device: torch.device) -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for _ in range(recipe["eval_batches"]):
                x, y = self._batch(data, recipe["batch_size"], recipe["sequence_len"], generator, device)
                logits, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                losses.append(loss.item())
        return float(sum(losses) / len(losses))

    def _generate(self, model: TinyStoryRNN, dataset: dict[str, Any], recipe: dict[str, Any], device: torch.device) -> str:
        prompt = "Once "
        stoi = dataset["stoi"]
        itos = dataset["itos"]
        seed_ids = [stoi.get(ch, 0) for ch in prompt]
        idx = torch.tensor([seed_ids], dtype=torch.long, device=device)
        hidden = None
        model.eval()
        with torch.no_grad():
            _, hidden = model(idx, hidden)
            current = idx[:, -1:]
            generated = prompt
            for _ in range(recipe["sample_length"]):
                logits, hidden = model(current, hidden)
                logits = logits[:, -1, :] / max(recipe["temperature"], 0.2)
                probs = torch.softmax(logits, dim=-1)
                next_idx = torch.multinomial(probs, num_samples=1)
                generated += itos[int(next_idx.item())]
                current = next_idx
        return generated.strip()

    def _peak_memory_mb(self) -> float:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return rss / (1024 * 1024)
        return rss / 1024

    def _run_recipe(self, recipe: dict[str, Any], run_dir: Path) -> TaskRunResult:
        dataset = self._load_dataset()
        device = self._device()
        torch.manual_seed(recipe["seed"])
        if device.type == "mps":
            torch.mps.manual_seed(recipe["seed"])
        generator = torch.Generator(device="cpu")
        generator.manual_seed(recipe["seed"])

        model = TinyStoryRNN(
            vocab_size=dataset["vocab_size"],
            embedding_dim=recipe["embedding_dim"],
            hidden_dim=recipe["hidden_dim"],
            num_layers=recipe["num_layers"],
            dropout=recipe["dropout"],
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["learning_rate"])

        started = time.perf_counter()
        log_lines = [
            f"device: {device.type}",
            f"recipe_sha: {self.recipe_sha(recipe)}",
        ]
        for step in range(1, recipe["steps"] + 1):
            model.train()
            x, y = self._batch(dataset["train_data"], recipe["batch_size"], recipe["sequence_len"], generator, device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step == 1 or step % max(8, recipe["steps"] // 4) == 0 or step == recipe["steps"]:
                log_lines.append(f"step {step:03d}/{recipe['steps']}: train_loss={loss.item():.4f}")

        elapsed = time.perf_counter() - started
        val_loss = self._evaluate(model, dataset["val_data"], recipe, generator, device)
        val_bpb = val_loss / math.log(2)
        sample_text = self._generate(model, dataset, recipe, device)
        tokens_seen = recipe["steps"] * recipe["batch_size"] * recipe["sequence_len"]
        metrics = {
            "train_loss": round(float(loss.item()), 4),
            "val_loss": round(val_loss, 4),
            "val_bpb": round(val_bpb, 4),
            "elapsed_seconds": round(elapsed, 2),
            "peak_memory_mb": round(self._peak_memory_mb(), 1),
            "tokens_seen": tokens_seen,
            "tokens_per_second": round(tokens_seen / max(elapsed, 1e-6), 1),
            "device": device.type,
        }

        run_dir.mkdir(parents=True, exist_ok=True)
        log_text = "\n".join(log_lines)
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (run_dir / "sample.txt").write_text(sample_text, encoding="utf-8")
        (run_dir / "run.log").write_text(log_text, encoding="utf-8")

        recipe_sha = self.recipe_sha(recipe)
        self._recipe_snapshot_path(recipe_sha).write_text(json.dumps(recipe, indent=2), encoding="utf-8")

        return TaskRunResult(
            recipe=recipe,
            recipe_sha=recipe_sha,
            metrics=metrics,
            sample_text=sample_text,
            log_text=log_text,
            artifact_paths={
                "metrics": str(run_dir / "metrics.json"),
                "sample": str(run_dir / "sample.txt"),
                "log": str(run_dir / "run.log"),
            },
        )
