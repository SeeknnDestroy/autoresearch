from __future__ import annotations

import json
import subprocess
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from studio.events import EventBroker
from studio.models import BudgetPolicy, LessonRecord, RunRecord, SessionRecord, utc_now_iso
from studio.storage import StudioStorage
from studio.tasks.base import TaskBackend


class StudioOrchestrator:
    def __init__(
        self,
        *,
        storage: StudioStorage,
        backend: TaskBackend,
        broker: EventBroker,
        repo_root: Path,
    ) -> None:
        self.storage = storage
        self.backend = backend
        self.broker = broker
        self.repo_root = Path(repo_root)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def recover_sessions(self) -> None:
        latest = self.storage.latest_session()
        if latest and latest.status == "running":
            latest.status = "interrupted"
            latest.stop_requested = True
            latest.finished_at = utc_now_iso()
            self.storage.save_session(latest)

    def repo_sha(self) -> str:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.repo_root,
            capture_output=True,
            check=True,
            text=True,
        )
        return proc.stdout.strip()

    def latest_session(self) -> SessionRecord | None:
        return self.storage.latest_session()

    def list_runs(self) -> list[RunRecord]:
        session = self.latest_session()
        return session.runs if session else []

    def get_run(self, run_id: str) -> tuple[RunRecord, LessonRecord | None]:
        session = self.latest_session()
        if session is None:
            raise KeyError(run_id)
        run = next(run for run in session.runs if run.run_id == run_id)
        lesson = next((lesson for lesson in session.lessons if lesson.run_id == run_id), None)
        return run, lesson

    def report(self) -> dict[str, Any]:
        session = self.latest_session()
        if session is None:
            return {
                "session": None,
                "best_run": None,
                "recent_lessons": [],
                "scoreboard": None,
                "spotlight": None,
                "play_view": self._empty_play_view(),
                "next_move": "Start a session to generate the first baseline.",
            }
        best_run = next((run for run in session.runs if run.run_id == session.best_run_id), None)
        recent_lessons = [asdict(lesson) for lesson in session.lessons[-3:]]
        completed_runs = [run for run in session.runs if run.status == "completed"]
        kept_runs = [run for run in completed_runs if run.decision in {"keep", "baseline"}]
        discarded_runs = [run for run in completed_runs if run.decision == "discard"]
        crash_runs = [run for run in session.runs if run.decision == "crash"]
        spotlight_run = session.runs[-1] if session.runs else None
        scoreboard = {
            "completed": len(completed_runs),
            "kept": len(kept_runs),
            "discarded": len(discarded_runs),
            "crashes": len(crash_runs),
        }
        return {
            "session": {
                "session_id": session.session_id,
                "status": session.status,
                "task_id": session.task_id,
                "repo_sha": session.repo_sha,
                "budget_policy": asdict(session.budget_policy),
                "best_run_id": session.best_run_id,
                "current_recipe_sha": session.current_recipe_sha,
                "stage_headline": session.stage_headline,
                "run_count": len(session.runs),
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "finished_at": session.finished_at,
            },
            "best_run": asdict(best_run) if best_run else None,
            "recent_lessons": recent_lessons,
            "scoreboard": scoreboard,
            "spotlight": asdict(spotlight_run) if spotlight_run else None,
            "play_view": self._build_play_view(
                session=session,
                best_run=best_run,
                spotlight_run=spotlight_run,
                scoreboard=scoreboard,
            ),
            "next_move": recent_lessons[-1]["follow_up"] if recent_lessons else "Baseline first, then widen or tighten one training dial.",
        }

    def start_session(self, budget_policy: BudgetPolicy | None = None) -> SessionRecord:
        with self._lock:
            existing = self.latest_session()
            if existing and existing.status == "running":
                return existing

            budget_policy = budget_policy or BudgetPolicy()
            current_recipe = self.backend.prepare()
            current_recipe_sha = self.backend.recipe_sha(current_recipe)  # type: ignore[attr-defined]
            session_id = time.strftime("studio-%Y%m%d-%H%M%S")
            session = self.storage.create_session(
                session_id=session_id,
                task_id=self.backend.task_id,
                repo_sha=self.repo_sha(),
                budget_policy=budget_policy,
                current_recipe=current_recipe,
                current_recipe_sha=current_recipe_sha,
            )
            session.status = "running"
            session.stage_headline = "The proposer is sketching the opening move."
            self.storage.save_session(session)
            self.broker.publish("session_started", {"session_id": session.session_id, "status": session.status})
            self._thread = threading.Thread(target=self._run_session, args=(session.session_id,), daemon=True)
            self._thread.start()
            return session

    def stop_session(self) -> SessionRecord | None:
        with self._lock:
            session = self.latest_session()
            if session is None:
                return None
            session.stop_requested = True
            if session.status == "running":
                session.status = "stopping"
            self.storage.save_session(session)
            self.broker.publish("session_stopping", {"session_id": session.session_id, "status": session.status})
            return session

    def _run_session(self, session_id: str) -> None:
        session = self.storage.load_session(session_id)
        started = time.perf_counter()

        if not session.runs:
            baseline = self._create_run(
                session=session,
                title="Baseline",
                hypothesis="Establish the local TinyStories baseline before exploring mutations.",
                change_spec={"type": "baseline"},
            )
            self._execute_run(session, baseline, baseline_mode=True)
            session = self.storage.load_session(session_id)

        while len(session.runs) < session.budget_policy.max_runs:
            session = self.storage.load_session(session_id)
            elapsed = time.perf_counter() - started
            if session.stop_requested or elapsed >= session.budget_policy.max_runtime_seconds:
                break
            change_spec = self.backend.propose_change(session.runs, session.current_recipe)  # type: ignore[attr-defined]
            run = self._create_run(
                session=session,
                title=change_spec["title"],
                hypothesis=change_spec["rationale"],
                change_spec=change_spec,
            )
            self._execute_run(session, run, baseline_mode=False)
            session = self.storage.load_session(session_id)

        session = self.storage.load_session(session_id)
        session.status = "stopped" if session.stop_requested else "completed"
        session.finished_at = utc_now_iso()
        session.stage_headline = (
            "The lab wrapped the session and pinned the strongest local recipe."
            if session.runs
            else "The lab stopped before any run landed."
        )
        self.storage.save_session(session)
        self.broker.publish("session_finished", {"session_id": session.session_id, "status": session.status})

    def _create_run(self, session: SessionRecord, title: str, hypothesis: str, change_spec: dict[str, Any]) -> RunRecord:
        run_id = uuid.uuid4().hex[:8]
        run = RunRecord(
            run_id=run_id,
            session_id=session.session_id,
            task_id=session.task_id,
            title=title,
            hypothesis=hypothesis,
            git_sha=session.repo_sha[:12],
            recipe_sha=session.current_recipe_sha,
            parent_recipe_sha=session.current_recipe_sha,
            status="running",
            stage="proposer",
            decision="pending",
            change_spec=change_spec,
            proposal_note=change_spec.get("rationale", "Measure the untouched baseline."),
            implementer_note="Sync the recipe snapshot, run the local model, and capture metrics plus samples.",
        )
        session.runs.append(run)
        session.stage_headline = self._stage_copy(run)
        self.storage.save_session(session)
        self.broker.publish(
            "run_started",
            {"session_id": session.session_id, "run_id": run.run_id, "title": run.title, "stage": run.stage},
        )
        return run

    def _execute_run(self, session: SessionRecord, run: RunRecord, *, baseline_mode: bool) -> None:
        run_dir = self.storage.session_dir(session.session_id) / "runs" / run.run_id
        before_recipe = dict(session.current_recipe)
        try:
            self._set_run_stage(session, run, "implementer")
            self._set_run_stage(session, run, "runner")
            if baseline_mode:
                result = self.backend.run_baseline(run_dir=run_dir)
            else:
                result = self.backend.run_candidate(run.change_spec, run_dir=run_dir)
            self._set_run_stage(session, run, "analyst")
            run.recipe_sha = result.recipe_sha
            run.metrics = result.metrics
            run.artifact_paths = result.artifact_paths
            run.sample_text = result.sample_text
            run.diff_text = self.backend.recipe_diff(before_recipe, result.recipe)  # type: ignore[attr-defined]
            run.status = "completed"
            run.finished_at = utc_now_iso()
            run.budget_usage = {
                "elapsed_seconds": result.metrics["elapsed_seconds"],
                "model_calls": 0,
            }
            self._finalize_decision(session, run, result.recipe)
        except Exception as exc:
            run.status = "crash"
            run.stage = "analyst"
            run.decision = "crash"
            run.analyst_summary = f"Candidate crashed before evaluation completed: {exc}"
            run.follow_up = "Retreat to the last kept recipe and change only one dial next time."
            run.finished_at = utc_now_iso()
            self.backend.rollback(session.current_recipe_sha)
            self.storage.save_session(session)
        self.storage.save_session(session)
        self.broker.publish(
            "run_finished",
            {
                "session_id": session.session_id,
                "run_id": run.run_id,
                "status": run.status,
                "decision": run.decision,
                "stage": run.stage,
            },
        )

    def _set_run_stage(self, session: SessionRecord, run: RunRecord, stage: str) -> None:
        run.stage = stage
        session.stage_headline = self._stage_copy(run)
        self.storage.save_session(session)
        self.broker.publish(
            "run_stage",
            {
                "session_id": session.session_id,
                "run_id": run.run_id,
                "stage": stage,
                "headline": session.stage_headline,
            },
        )

    def _stage_copy(self, run: RunRecord) -> str:
        axis = run.change_spec.get("field", "baseline")
        if run.stage == "proposer":
            if run.change_spec.get("type") == "baseline":
                return "The proposer is locking the first baseline before taking risks."
            return f"The proposer is teeing up a move on {axis.replace('_', ' ')}."
        if run.stage == "implementer":
            return f"The implementer is wiring {run.title.lower()} into the recipe snapshot."
        if run.stage == "runner":
            return f"The runner is stress-testing {run.title.lower()} on the local TinyStories lane."
        if run.stage == "analyst":
            return f"The analyst is judging whether {run.title.lower()} deserves to stay."
        return "The lab is moving."

    def _empty_play_view(self) -> dict[str, Any]:
        return {
            "intro_title": "Play with a tiny AI lab",
            "intro_body": (
                "This repo lets an agent try one training idea at a time and keep it only if the model score gets better."
            ),
            "primary_cta": "Run a tiny experiment",
            "current_moment_title": "What will happen when you press play?",
            "current_moment_body": "The lab will measure the starting model, try one small idea, compare the score, and keep only the winner.",
            "session_summary": {
                "runs": "0 experiments",
                "wins": "0 winners kept",
            },
            "score": {
                "label": "Model score",
                "value": "--",
                "help": "Lower is better.",
            },
            "verdict": {
                "title": "Nothing has happened yet",
                "body": "Start a session and the Studio will explain each step in plain language.",
            },
            "last_change": {
                "title": "No change yet",
                "field_label": "Training setting",
                "before": "--",
                "after": "--",
                "reason": "The first run is just the starting point.",
            },
            "sample": {
                "title": "What the model wrote",
                "body": "A sample will appear here once the first run finishes.",
            },
            "repo_map": self._repo_map(),
            "steps": self._play_steps(None, None),
        }

    def _build_play_view(
        self,
        *,
        session: SessionRecord,
        best_run: RunRecord | None,
        spotlight_run: RunRecord | None,
        scoreboard: dict[str, int],
    ) -> dict[str, Any]:
        last_candidate = next((run for run in reversed(session.runs) if run.change_spec.get("type") != "baseline"), None)
        metric_run = best_run or spotlight_run
        score_value = (
            f"{metric_run.metrics.get('val_bpb', 0):.4f}"
            if metric_run and metric_run.metrics.get("val_bpb") is not None
            else "--"
        )
        return {
            "intro_title": "Play with a tiny AI lab",
            "intro_body": (
                "Autoresearch keeps trying one small training change at a time. If the model score gets better, it keeps the change. If not, it throws it away."
            ),
            "primary_cta": "Run a tiny experiment" if session.status != "running" else "The lab is already running",
            "current_moment_title": "What is happening right now?",
            "current_moment_body": self._play_current_moment(session, spotlight_run),
            "session_summary": {
                "runs": f"{scoreboard['completed']} experiment{'s' if scoreboard['completed'] != 1 else ''} finished",
                "wins": f"{scoreboard['kept']} winner{'s' if scoreboard['kept'] != 1 else ''} kept",
            },
            "score": {
                "label": "Model score",
                "value": score_value,
                "help": "Lower is better. The lab only keeps a change if this number goes down.",
            },
            "verdict": self._play_verdict(best_run, last_candidate),
            "last_change": self._play_last_change(last_candidate),
            "sample": {
                "title": "What the model wrote",
                "body": (
                    (last_candidate or best_run).sample_text if (last_candidate or best_run) else "A sample will appear here once the first run finishes."
                ),
            },
            "repo_map": self._repo_map(),
            "steps": self._play_steps(session, spotlight_run),
        }

    def _play_current_moment(self, session: SessionRecord, spotlight_run: RunRecord | None) -> str:
        if spotlight_run is None:
            return "The lab is waiting for the first baseline run."
        if session.status == "running":
            if spotlight_run.change_spec.get("type") == "baseline":
                return "The lab is measuring the starting model so it has something real to beat."
            if spotlight_run.stage == "proposer":
                return "The lab is choosing one small change idea to test next."
            if spotlight_run.stage == "implementer":
                return "The lab is applying that one change to the training settings."
            if spotlight_run.stage == "runner":
                return "The lab is running the changed model and collecting a new score."
            if spotlight_run.stage == "analyst":
                return "The lab is comparing the new score against the best score so far."
        if spotlight_run.change_spec.get("type") == "baseline":
            return "The baseline is done. From now on, every new idea gets judged against that starting score."
        return "The latest test finished, and the lab has already decided whether to keep or throw away the change."

    def _play_verdict(self, best_run: RunRecord | None, last_candidate: RunRecord | None) -> dict[str, str]:
        if last_candidate is None:
            if best_run is None:
                return {
                    "title": "Nothing has happened yet",
                    "body": "Start a session and the Studio will explain each step in plain language.",
                }
            return {
                "title": "The starting score is locked in",
                "body": "The baseline gives the lab a real number to beat before it starts taking risks.",
            }
        candidate_score = float(last_candidate.metrics.get("val_bpb", 0.0))
        if best_run and last_candidate.decision == "keep":
            return {
                "title": "That change helped",
                "body": f"The score dropped to {candidate_score:.4f}, so the lab kept the new settings.",
            }
        if best_run:
            best_score = float(best_run.metrics.get("val_bpb", 0.0))
            return {
                "title": "That change did not help",
                "body": f"The score moved from {best_score:.4f} to {candidate_score:.4f}, so the lab threw the change away.",
            }
        return {
            "title": "The lab is still learning",
            "body": "A verdict will appear here once a non-baseline run finishes.",
        }

    def _play_last_change(self, last_candidate: RunRecord | None) -> dict[str, str]:
        if last_candidate is None:
            return {
                "title": "No change yet",
                "field_label": "Training setting",
                "before": "--",
                "after": "--",
                "reason": "The first run is just the starting point.",
            }
        spec = last_candidate.change_spec
        return {
            "title": last_candidate.title,
            "field_label": spec.get("field", "setting").replace("_", " ").title(),
            "before": str(spec.get("from", "--")),
            "after": str(spec.get("to", "--")),
            "reason": spec.get("why") or spec.get("rationale") or "The lab is probing one small training change.",
        }

    def _repo_map(self) -> list[dict[str, str]]:
        return [
            {
                "name": "prepare.py",
                "role": "Sets up the data and the scoring rule.",
            },
            {
                "name": "train.py",
                "role": "Contains the model and the training loop the agent experiments on.",
            },
            {
                "name": "program.md",
                "role": "Tells the agent how to behave while it researches.",
            },
        ]

    def _play_steps(self, session: SessionRecord | None, spotlight_run: RunRecord | None) -> list[dict[str, str]]:
        steps = [
            {
                "title": "1. Measure the starting point",
                "body": "Run the untouched model once so the lab has a real score to beat.",
                "state": "upcoming",
            },
            {
                "title": "2. Try one small idea",
                "body": "Change one training setting, not everything at once.",
                "state": "upcoming",
            },
            {
                "title": "3. Compare the new score",
                "body": "Look at the new result and compare it with the best score so far.",
                "state": "upcoming",
            },
            {
                "title": "4. Keep it or throw it away",
                "body": "Keep the idea only if the score got better.",
                "state": "upcoming",
            },
        ]
        if session is None or not session.runs:
            steps[0]["state"] = "active"
            return steps

        steps[0]["state"] = "done"
        if len(session.runs) == 1 and session.status == "running":
            return steps

        steps[1]["state"] = "done" if len(session.runs) > 1 else "active"
        steps[2]["state"] = "done" if spotlight_run and spotlight_run.stage == "analyst" else ("active" if len(session.runs) > 1 else "upcoming")
        steps[3]["state"] = "done" if session.status in {"completed", "stopped"} or (spotlight_run and spotlight_run.status == "completed") else "upcoming"
        return steps

    def _best_completed_run(self, session: SessionRecord, exclude_run_id: str | None = None) -> RunRecord | None:
        completed = [
            run for run in session.runs
            if run.status == "completed" and run.decision in {"keep", "baseline"} and run.run_id != exclude_run_id
        ]
        if not completed:
            return None
        return min(completed, key=lambda run: float(run.metrics.get("val_bpb", 999.0)))

    def _finalize_decision(self, session: SessionRecord, run: RunRecord, recipe: dict[str, Any]) -> None:
        if run.change_spec.get("type") == "baseline":
            run.decision = "baseline"
            session.current_recipe = recipe
            session.current_recipe_sha = run.recipe_sha
            session.best_run_id = run.run_id
            self.backend.rollback(run.recipe_sha)
            summary = "Baseline locked. The Studio now has a grounded metric and a reference sample."
            follow_up = "Move one recipe dial at a time so the analyst can attribute wins cleanly."
        else:
            best_run = self._best_completed_run(session, exclude_run_id=run.run_id)
            incumbent = float(best_run.metrics["val_bpb"]) if best_run else 999.0
            candidate = float(run.metrics["val_bpb"])
            if candidate <= incumbent - 0.01:
                run.decision = "keep"
                session.current_recipe = recipe
                session.current_recipe_sha = run.recipe_sha
                session.best_run_id = run.run_id
                self.backend.rollback(run.recipe_sha)
                summary = (
                    f"Keep: val_bpb improved from {incumbent:.4f} to {candidate:.4f} "
                    f"by changing {run.change_spec['field']}."
                )
                follow_up = "Compound the win carefully: probe an adjacent change rather than a full rewrite."
            else:
                run.decision = "discard"
                self.backend.rollback(session.current_recipe_sha)
                summary = (
                    f"Discard: val_bpb moved from {incumbent:.4f} to {candidate:.4f}. "
                    f"The idea was learnable, but not yet a keeper."
                )
                follow_up = "Try a neighboring setting or a different axis so the loop does not thrash."

        run.analyst_summary = summary
        run.novelty_fingerprint = run.change_spec.get("novelty_fingerprint", "baseline")
        run.follow_up = follow_up
        lesson = LessonRecord(
            lesson_id=uuid.uuid4().hex[:8],
            run_id=run.run_id,
            summary=summary,
            novelty_fingerprint=run.novelty_fingerprint,
            decision=run.decision,
            follow_up=follow_up,
        )
        session.lessons.append(lesson)
        self.storage.write_artifact(session.session_id, run.run_id, "recipe.json", json.dumps(recipe, indent=2))
