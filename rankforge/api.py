"""Read-only model artifact, durable feedback, and same-origin dashboard."""

import json
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import torch
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .engine import Engine
from .store import Conflict, Store, assignment

STATIC = Path(__file__).parent / "static"


def create_app(artifacts: Path | None = None, db_path: Path | None = None, api_token=None):
    directory = artifacts or Path(os.getenv("RANKFORGE_ARTIFACTS", "artifacts"))
    database = db_path or Path(os.getenv("RANKFORGE_DB", "runtime/events.db"))
    token = os.getenv("RANKFORGE_API_TOKEN", "") if api_token is None else api_token

    @asynccontextmanager
    async def lifespan(app):
        torch.set_num_threads(max(1, int(os.getenv("RANKFORGE_TORCH_THREADS", "1"))))
        app.state.store = Store(database)
        app.state.engine = None
        app.state.error = None
        app.state.requests = 0
        app.state.total_seconds = 0.0
        try:
            app.state.engine = Engine.load(directory)
            app.state.report = json.loads((directory / "report.json").read_text())
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            app.state.error = type(error).__name__
        yield

    app = FastAPI(title="RankForge", version="1.0.0", lifespan=lifespan)

    def authorize(authorization: Annotated[str | None, Header()] = None):
        if token and not secrets.compare_digest(authorization or "", f"Bearer {token}"):
            raise HTTPException(401, "Valid bearer token required")

    def engine() -> Engine:
        if app.state.engine is None:
            raise HTTPException(503, "Model unavailable. Run: python -m rankforge.cli demo")
        return app.state.engine

    @app.get("/health")
    def health():
        return {"status": "alive"}

    @app.get("/ready")
    def ready(model: Annotated[Engine, Depends(engine)]):
        return {"status": "ready", "model_version": model.version}

    @app.get("/api/report", dependencies=[Depends(authorize)])
    def report(model: Annotated[Engine, Depends(engine)]):
        return app.state.report

    @app.get("/api/users", dependencies=[Depends(authorize)])
    def users(model: Annotated[Engine, Depends(engine)]):
        return {"users": model.users, "cold_start_user": "new-visitor"}

    @app.get("/api/history/{user_id}", dependencies=[Depends(authorize)])
    def history(user_id: str, model: Annotated[Engine, Depends(engine)]):
        if len(user_id) > 80:
            raise HTTPException(422, "user_id too long")
        profile = model.profile(user_id, app.state.store.recent(user_id))
        return {
            "categories": [
                {"category": c, "share": float(profile[n])} for n, c in enumerate(model.categories)
            ],
            "training_interactions": len(model.histories.get(user_id, [])),
            "live_feedback": len(app.state.store.recent(user_id)),
        }

    @app.get("/api/recommendations", dependencies=[Depends(authorize)])
    def recommendations(
        model: Annotated[Engine, Depends(engine)],
        user_id: Annotated[str, Query(min_length=1, max_length=80)],
        k: Annotated[int, Query(ge=1, le=30)] = 12,
        diversity: Annotated[float, Query(ge=0, le=0.6)] = 0.1,
    ):
        started = time.perf_counter()
        variant = assignment(user_id)
        strategy = "popular" if variant == "A" else "neural"
        rows, method = model.recommend(
            user_id, k, diversity, app.state.store.recent(user_id), strategy, now=int(time.time())
        )
        request_id = str(uuid.uuid4())
        app.state.store.exposure(
            request_id, user_id, variant, model.version, [r["item_id"] for r in rows]
        )
        seconds = time.perf_counter() - started
        app.state.requests += 1
        app.state.total_seconds += seconds
        return {
            "request_id": request_id,
            "user_id": user_id,
            "variant": variant,
            "method": method,
            "model_version": model.version,
            "latency_ms": round(seconds * 1000, 3),
            "items": rows,
        }

    class Feedback(BaseModel):
        event_id: str = Field(min_length=1, max_length=80)
        request_id: str = Field(min_length=1, max_length=80)
        user_id: str = Field(min_length=1, max_length=80)
        item_id: str = Field(min_length=1, max_length=160)

    @app.post("/api/feedback", dependencies=[Depends(authorize)])
    def feedback(body: Feedback):
        try:
            inserted = app.state.store.feedback(**body.model_dump())
        except Conflict as error:
            raise HTTPException(409, str(error)) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        return {"accepted": True, "inserted": inserted}

    @app.get("/api/experiments", dependencies=[Depends(authorize)])
    def experiments():
        return app.state.store.summary()

    @app.get("/metrics", response_class=PlainTextResponse, dependencies=[Depends(authorize)])
    def metrics():
        return (
            "# TYPE rankforge_recommendations_total counter\n"
            f"rankforge_recommendations_total {app.state.requests}\n"
            "# TYPE rankforge_recommendation_seconds_sum counter\n"
            f"rankforge_recommendation_seconds_sum {app.state.total_seconds}\n"
        )

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
