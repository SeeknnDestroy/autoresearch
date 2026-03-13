from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from studio.events import EventBroker
from studio.models import BudgetPolicy
from studio.orchestrator import StudioOrchestrator
from studio.storage import StudioStorage
from studio.tasks.tinystories_local import TinyStoriesLocalBackend


def create_app(studio_root: Path | None = None) -> FastAPI:
    package_root = Path(__file__).resolve().parent
    static_dir = package_root / "static"
    studio_root = studio_root or Path.cwd() / ".studio"

    storage = StudioStorage(studio_root)
    backend = TinyStoriesLocalBackend(studio_root)
    broker = EventBroker()
    orchestrator = StudioOrchestrator(
        storage=storage,
        backend=backend,
        broker=broker,
        repo_root=Path.cwd(),
    )
    orchestrator.recover_sessions()

    app = FastAPI(title="Autoresearch Studio")
    app.state.orchestrator = orchestrator
    app.state.broker = broker
    app.mount("/assets", StaticFiles(directory=static_dir), name="assets")

    @app.get("/", response_class=HTMLResponse)
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.post("/api/session/start")
    def start_session() -> JSONResponse:
        session = orchestrator.start_session(BudgetPolicy())
        return JSONResponse(session.to_dict())

    @app.post("/api/session/stop")
    def stop_session() -> JSONResponse:
        session = orchestrator.stop_session()
        return JSONResponse(session.to_dict() if session else {"session": None})

    @app.get("/api/runs")
    def list_runs() -> JSONResponse:
        runs = [asdict(run) for run in orchestrator.list_runs()]
        return JSONResponse({"runs": runs})

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> JSONResponse:
        run, lesson = orchestrator.get_run(run_id)
        return JSONResponse({"run": asdict(run), "lesson": asdict(lesson) if lesson else None})

    @app.get("/api/report/latest")
    def latest_report() -> JSONResponse:
        return JSONResponse(orchestrator.report())

    @app.get("/api/events")
    async def events(request: Request, after: int = 0) -> StreamingResponse:
        async def generator():
            last_id = after
            while True:
                if await request.is_disconnected():
                    break
                events = await asyncio.to_thread(broker.wait_for_events, last_id, 10.0)
                if not events:
                    yield "event: heartbeat\ndata: {}\n\n"
                    continue
                for event in events:
                    last_id = event["id"]
                    payload = json.dumps(event)
                    yield f"id: {event['id']}\nevent: {event['type']}\ndata: {payload}\n\n"

        return StreamingResponse(generator(), media_type="text/event-stream")

    return app
