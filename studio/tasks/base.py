from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TaskRunResult:
    recipe: dict[str, Any]
    recipe_sha: str
    metrics: dict[str, Any]
    sample_text: str
    log_text: str
    artifact_paths: dict[str, str]


class TaskBackend(ABC):
    task_id: str

    @abstractmethod
    def prepare(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def run_baseline(self, run_dir: Path | None = None) -> TaskRunResult:
        raise NotImplementedError

    @abstractmethod
    def run_candidate(self, change_spec: dict[str, Any], run_dir: Path | None = None) -> TaskRunResult:
        raise NotImplementedError

    @abstractmethod
    def extract_metrics(self, run_dir: Path) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def generate_samples(self, run_dir: Path) -> str:
        raise NotImplementedError

    @abstractmethod
    def rollback(self, target_sha: str) -> dict[str, Any]:
        raise NotImplementedError
