from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from studio.models import BudgetPolicy, SessionRecord, utc_now_iso


class StudioStorage:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.sessions_dir = self.root / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def session_dir(self, session_id: str) -> Path:
        return self.sessions_dir / session_id

    def session_file(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "session.json"

    def latest_session_id(self) -> str | None:
        session_ids = sorted(path.name for path in self.sessions_dir.iterdir() if path.is_dir())
        return session_ids[-1] if session_ids else None

    def create_session(
        self,
        session_id: str,
        task_id: str,
        repo_sha: str,
        budget_policy: BudgetPolicy,
        current_recipe: dict[str, Any],
        current_recipe_sha: str,
    ) -> SessionRecord:
        session = SessionRecord(
            session_id=session_id,
            task_id=task_id,
            status="queued",
            repo_sha=repo_sha,
            budget_policy=budget_policy,
            current_recipe=current_recipe,
            current_recipe_sha=current_recipe_sha,
        )
        self.save_session(session)
        return session

    def save_session(self, session: SessionRecord) -> None:
        session.updated_at = utc_now_iso()
        directory = self.session_dir(session.session_id)
        directory.mkdir(parents=True, exist_ok=True)
        self.session_file(session.session_id).write_text(
            json.dumps(session.to_dict(), indent=2),
            encoding="utf-8",
        )

    def load_session(self, session_id: str) -> SessionRecord:
        data = json.loads(self.session_file(session_id).read_text(encoding="utf-8"))
        return SessionRecord.from_dict(data)

    def latest_session(self) -> SessionRecord | None:
        session_id = self.latest_session_id()
        return self.load_session(session_id) if session_id else None

    def write_artifact(self, session_id: str, run_id: str, filename: str, content: str) -> str:
        run_dir = self.session_dir(session_id) / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / filename
        path.write_text(content, encoding="utf-8")
        return str(path)

    def write_json_artifact(self, session_id: str, run_id: str, filename: str, payload: dict[str, Any]) -> str:
        run_dir = self.session_dir(session_id) / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / filename
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(path)
