from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


JSONDict = dict[str, Any]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class BudgetPolicy:
    max_runs: int = 5
    max_runtime_seconds: int = 240
    max_model_calls: int = 0
    max_concurrent_runs: int = 1

    @classmethod
    def from_dict(cls, data: JSONDict | None) -> "BudgetPolicy":
        data = data or {}
        return cls(
            max_runs=int(data.get("max_runs", 3)),
            max_runtime_seconds=int(data.get("max_runtime_seconds", 240)),
            max_model_calls=int(data.get("max_model_calls", 0)),
            max_concurrent_runs=int(data.get("max_concurrent_runs", 1)),
        )


@dataclass
class RunRecord:
    run_id: str
    session_id: str
    task_id: str
    title: str
    hypothesis: str
    git_sha: str
    recipe_sha: str
    parent_recipe_sha: str | None
    status: str
    stage: str
    decision: str
    change_spec: JSONDict
    metrics: JSONDict = field(default_factory=dict)
    artifact_paths: JSONDict = field(default_factory=dict)
    budget_usage: JSONDict = field(default_factory=dict)
    proposal_note: str = ""
    implementer_note: str = ""
    analyst_summary: str = ""
    novelty_fingerprint: str = ""
    follow_up: str = ""
    diff_text: str = ""
    sample_text: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    finished_at: str | None = None

    @classmethod
    def from_dict(cls, data: JSONDict) -> "RunRecord":
        return cls(
            run_id=data["run_id"],
            session_id=data["session_id"],
            task_id=data["task_id"],
            title=data["title"],
            hypothesis=data["hypothesis"],
            git_sha=data["git_sha"],
            recipe_sha=data.get("recipe_sha", ""),
            parent_recipe_sha=data.get("parent_recipe_sha"),
            status=data["status"],
            stage=data.get("stage", "queued"),
            decision=data.get("decision", "pending"),
            change_spec=data.get("change_spec", {}),
            metrics=data.get("metrics", {}),
            artifact_paths=data.get("artifact_paths", {}),
            budget_usage=data.get("budget_usage", {}),
            proposal_note=data.get("proposal_note", ""),
            implementer_note=data.get("implementer_note", ""),
            analyst_summary=data.get("analyst_summary", ""),
            novelty_fingerprint=data.get("novelty_fingerprint", ""),
            follow_up=data.get("follow_up", ""),
            diff_text=data.get("diff_text", ""),
            sample_text=data.get("sample_text", ""),
            created_at=data.get("created_at", utc_now_iso()),
            finished_at=data.get("finished_at"),
        )


@dataclass
class LessonRecord:
    lesson_id: str
    run_id: str
    summary: str
    novelty_fingerprint: str
    decision: str
    follow_up: str
    created_at: str = field(default_factory=utc_now_iso)

    @classmethod
    def from_dict(cls, data: JSONDict) -> "LessonRecord":
        return cls(
            lesson_id=data["lesson_id"],
            run_id=data["run_id"],
            summary=data["summary"],
            novelty_fingerprint=data["novelty_fingerprint"],
            decision=data["decision"],
            follow_up=data["follow_up"],
            created_at=data.get("created_at", utc_now_iso()),
        )


@dataclass
class SessionRecord:
    session_id: str
    task_id: str
    status: str
    repo_sha: str
    budget_policy: BudgetPolicy
    current_recipe: JSONDict
    current_recipe_sha: str
    stage_headline: str = "The lab is waiting for its first baseline."
    runs: list[RunRecord] = field(default_factory=list)
    lessons: list[LessonRecord] = field(default_factory=list)
    stop_requested: bool = False
    best_run_id: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    finished_at: str | None = None

    def to_dict(self) -> JSONDict:
        return {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "status": self.status,
            "repo_sha": self.repo_sha,
            "budget_policy": asdict(self.budget_policy),
            "current_recipe": self.current_recipe,
            "current_recipe_sha": self.current_recipe_sha,
            "stage_headline": self.stage_headline,
            "runs": [asdict(run) for run in self.runs],
            "lessons": [asdict(lesson) for lesson in self.lessons],
            "stop_requested": self.stop_requested,
            "best_run_id": self.best_run_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "finished_at": self.finished_at,
        }

    @classmethod
    def from_dict(cls, data: JSONDict) -> "SessionRecord":
        return cls(
            session_id=data["session_id"],
            task_id=data["task_id"],
            status=data["status"],
            repo_sha=data["repo_sha"],
            budget_policy=BudgetPolicy.from_dict(data.get("budget_policy")),
            current_recipe=data.get("current_recipe", {}),
            current_recipe_sha=data.get("current_recipe_sha", ""),
            stage_headline=data.get("stage_headline", "The lab is waiting for its first baseline."),
            runs=[RunRecord.from_dict(run) for run in data.get("runs", [])],
            lessons=[LessonRecord.from_dict(lesson) for lesson in data.get("lessons", [])],
            stop_requested=bool(data.get("stop_requested", False)),
            best_run_id=data.get("best_run_id"),
            created_at=data.get("created_at", utc_now_iso()),
            updated_at=data.get("updated_at", utc_now_iso()),
            finished_at=data.get("finished_at"),
        )
