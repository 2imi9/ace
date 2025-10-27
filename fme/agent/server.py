"""FastAPI service exposing ACE agent orchestration endpoints."""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from fastapi import BackgroundTasks, BaseModel, FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

from .cli import build_orchestrator, generate_report
from .orchestrator import AgentOrchestrator
from .session_state import SessionState


class QuestionPayload(BaseModel):
    """Request model for submitting a new analysis question."""

    question: str
    config_path: Optional[str] = None


class JobStatus(BaseModel):
    """Representation of the job status payload."""

    job_id: str
    status: str
    result: Optional[Any] = None


@dataclass
class JobRecord:
    """Internal representation of a background job."""

    job_id: str
    question: str
    status: str = "pending"
    result: Optional[Any] = None
    session: Optional[Dict[str, Any]] = None


@dataclass
class JobStore:
    """Thread-safe in-memory job registry."""

    _records: Dict[str, JobRecord] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def create(self, question: str) -> JobRecord:
        job_id = str(uuid.uuid4())
        record = JobRecord(job_id=job_id, question=question)
        with self._lock:
            self._records[job_id] = record
        return record

    def update(self, job_id: str, **changes: Any) -> JobRecord:
        with self._lock:
            if job_id not in self._records:
                raise KeyError(job_id)
            record = self._records[job_id]
            for key, value in changes.items():
                setattr(record, key, value)
            return record

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._records.get(job_id)


def _execute_question(
    orchestrator_factory: Callable[[Optional[str]], AgentOrchestrator],
    job_store: JobStore,
    job_id: str,
    payload: QuestionPayload,
) -> None:
    """Background worker that executes an analysis request."""

    try:
        orchestrator = orchestrator_factory(payload.config_path)
        response = orchestrator.run_command("ask llm", payload.question)
        session_snapshot = orchestrator.session_state
        job_store.update(
            job_id,
            status="completed",
            result=response,
            session=session_snapshot and json.loads(
                generate_report(session_snapshot, format="json")
            ),
        )
    except Exception as exc:  # pragma: no cover - defensive coding
        job_store.update(job_id, status="failed", result=str(exc))


def create_app(
    orchestrator_factory: Callable[[Optional[str]], AgentOrchestrator] | None = None,
) -> FastAPI:
    """Instantiate the FastAPI application."""

    job_store = JobStore()

    def _factory(config_path: Optional[str]) -> AgentOrchestrator:
        if orchestrator_factory is not None:
            return orchestrator_factory(config_path)
        return build_orchestrator(config_path)

    app = FastAPI(title="ACE Agent Service", version="0.1.0")

    @app.post("/questions", response_model=JobStatus)
    def submit_question(
        payload: QuestionPayload, background_tasks: BackgroundTasks
    ) -> JobStatus:
        record = job_store.create(payload.question)
        background_tasks.add_task(_execute_question, _factory, job_store, record.job_id, payload)
        return JobStatus(job_id=record.job_id, status=record.status)

    @app.get("/jobs/{job_id}", response_model=JobStatus)
    def get_status(job_id: str) -> JobStatus:
        record = job_store.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return JobStatus(job_id=record.job_id, status=record.status, result=record.result)

    @app.get("/jobs/{job_id}/report")
    def get_report(job_id: str, format: str = "json"):
        record = job_store.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if record.session is None:
            raise HTTPException(status_code=409, detail="Report not available yet")
        if format == "json":
            return JSONResponse(record.session)
        if format in {"md", "markdown"}:
            session_state = record.session
            text = generate_report(
                SessionState(**session_state),  # type: ignore[arg-type]
                format="markdown",
            )
            return PlainTextResponse(text)
        raise HTTPException(status_code=400, detail="Unsupported report format")

    return app


__all__ = ["create_app"]
