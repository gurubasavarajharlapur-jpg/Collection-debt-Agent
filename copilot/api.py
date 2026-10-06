"""FastAPI surface for Collections Copilot."""
from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import service, store
from .config import settings
from .demo import DEMO_ACCOUNT_IDS

app = FastAPI(title="Collections Copilot (demo)", version="0.1.0",
              description="Demo collections agent. Synthetic data only; sending is mocked.")


class Decision(BaseModel):
    action: Literal["approve", "edit", "reject"]
    message: str | None = None  # edited text for 'edit', optional note for 'reject'


def _call(fn, *args):
    try:
        return fn(*args)
    except service.RunError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "demo_mode": settings.demo_mode,
        "demo_accounts": list(DEMO_ACCOUNT_IDS) if settings.demo_mode else [],
        "llm": None if settings.demo_mode else f"{settings.llm_provider}:{settings.llm_model}",
    }


@app.get("/accounts")
def accounts() -> list[dict]:
    return store.list_accounts()


@app.get("/runs")
def runs() -> list[dict]:
    return store.list_runs()


@app.post("/runs/{account_id}", status_code=202)
def start_run(account_id: str) -> dict:
    """Start the agent for an account. Returns immediately; poll GET /runs/{id}."""
    return _call(service.start_run, account_id)


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    return _call(service.get_run, run_id)


@app.post("/runs/{run_id}/decision")
def decide(run_id: str, decision: Decision) -> dict:
    """Human decision for a run paused at approval: approve, edit or reject."""
    return _call(service.submit_decision, run_id, decision.action, decision.message)


@app.get("/audit/{run_id}")
def audit(run_id: str) -> list[dict]:
    _call(service.get_run, run_id)  # 404 for an unknown run
    return store.get_audit(run_id)
