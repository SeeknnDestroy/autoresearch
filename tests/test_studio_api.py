import time
from pathlib import Path

from fastapi.testclient import TestClient

from studio.api import create_app
from studio.events import EventBroker
from studio.models import BudgetPolicy
from studio.orchestrator import StudioOrchestrator
from studio.storage import StudioStorage
from studio.tasks.tinystories_local import TinyStoriesLocalBackend


QUICK_OVERRIDES = {
    "sequence_len": 24,
    "batch_size": 8,
    "embedding_dim": 24,
    "hidden_dim": 48,
    "steps": 6,
    "eval_batches": 2,
    "sample_length": 50,
}


def wait_until(predicate, timeout: float = 12.0) -> None:
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        if predicate():
            return
        time.sleep(0.2)
    raise AssertionError("Timed out waiting for background session")


def test_recover_sessions_marks_running_session_interrupted(tmp_path: Path) -> None:
    storage = StudioStorage(tmp_path / ".studio")
    backend = TinyStoriesLocalBackend(tmp_path / ".studio", device_preference="cpu", recipe_overrides=QUICK_OVERRIDES)
    broker = EventBroker()
    orchestrator = StudioOrchestrator(
        storage=storage,
        backend=backend,
        broker=broker,
        repo_root=Path.cwd(),
    )

    recipe = backend.prepare()
    recipe_sha = backend.recipe_sha(recipe)
    session = storage.create_session(
        session_id="studio-test",
        task_id=backend.task_id,
        repo_sha="deadbeef",
        budget_policy=BudgetPolicy(),
        current_recipe=recipe,
        current_recipe_sha=recipe_sha,
    )
    session.status = "running"
    storage.save_session(session)

    orchestrator.recover_sessions()

    recovered = storage.load_session("studio-test")
    assert recovered.status == "interrupted"
    assert recovered.stop_requested is True


def test_api_runs_end_to_end(tmp_path: Path) -> None:
    app = create_app(tmp_path / ".studio")
    backend = app.state.orchestrator.backend
    backend.device_preference = "cpu"
    backend.recipe_overrides = dict(QUICK_OVERRIDES)

    client = TestClient(app)

    initial_report = client.get("/api/report/latest").json()
    assert initial_report["session"] is None

    started = client.post("/api/session/start").json()
    assert started["status"] in {"running", "queued"}

    wait_until(
        lambda: client.get("/api/report/latest").json()["session"]["status"] in {"completed", "stopped"}
    )

    runs = client.get("/api/runs").json()["runs"]
    assert len(runs) >= 2

    detail = client.get(f"/api/runs/{runs[0]['run_id']}").json()
    assert detail["run"]["title"]
    assert detail["run"]["metrics"]["val_bpb"] > 0

    latest_report = client.get("/api/report/latest").json()
    assert latest_report["best_run"] is not None
    assert latest_report["recent_lessons"]
    assert latest_report["scoreboard"]["completed"] >= 2
    assert latest_report["session"]["stage_headline"]


def test_storage_load_session_retries_partial_json(tmp_path: Path, monkeypatch) -> None:
    storage = StudioStorage(tmp_path / ".studio")
    backend = TinyStoriesLocalBackend(tmp_path / ".studio", device_preference="cpu", recipe_overrides=QUICK_OVERRIDES)
    recipe = backend.prepare()
    session = storage.create_session(
        session_id="studio-retry",
        task_id=backend.task_id,
        repo_sha="deadbeef",
        budget_policy=BudgetPolicy(),
        current_recipe=recipe,
        current_recipe_sha=backend.recipe_sha(recipe),
    )

    session_file = storage.session_file("studio-retry")
    original_read_text = Path.read_text
    calls = {"count": 0}

    def flaky_read_text(self: Path, *args, **kwargs):
        if self == session_file and calls["count"] == 0:
            calls["count"] += 1
            return ""
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky_read_text)
    loaded = storage.load_session("studio-retry")

    assert loaded.session_id == session.session_id
