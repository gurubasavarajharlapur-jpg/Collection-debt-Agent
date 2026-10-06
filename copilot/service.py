"""Run lifecycle: start a run in the background, pause at approval, resume on a decision."""
from __future__ import annotations

import logging
import sqlite3
import threading
import uuid
from functools import lru_cache

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from . import store
from .config import settings
from .demo import DEMO_ACCOUNT_IDS, DemoLLM
from .graph import build_graph
from .llm import LiveLLM, LLMError, _safe_message

log = logging.getLogger(__name__)

FINAL_STATUSES = {"sent", "rejected", "escalated", "failed"}


class RunError(Exception):
    """A request that cannot be carried out; `status_code` is the HTTP status to return."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@lru_cache(maxsize=1)
def get_graph():
    settings.var_dir.mkdir(parents=True, exist_ok=True)
    # The checkpointer persists paused runs, so an approval can arrive after a restart.
    conn = sqlite3.connect(settings.checkpoint_path, check_same_thread=False)
    llm = DemoLLM() if settings.demo_mode else LiveLLM()
    return build_graph(llm, checkpointer=SqliteSaver(conn))


def _config(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}}


def _execute(run_id: str, payload) -> None:
    """Drive the graph until it finishes, pauses for approval, or fails."""
    graph, config = get_graph(), _config(run_id)
    try:
        for values in graph.stream(payload, config, stream_mode="values"):
            state = {k: v for k, v in values.items() if not k.startswith("__")}
            store.update_run(run_id, state=state)
        snapshot = graph.get_state(config)
        if snapshot.next:  # paused at the human_approval interrupt
            pending = [i.value for task in snapshot.tasks for i in task.interrupts]
            store.update_run(run_id, status="awaiting_approval", pending=pending[0] if pending else {})
        else:
            store.update_run(run_id, status=snapshot.values.get("outcome", "failed"), pending=None)
    except Exception as exc:
        log.exception("Run %s failed", run_id)
        message = str(exc) if isinstance(exc, LLMError) else f"{type(exc).__name__}: {_safe_message(exc)}"
        store.update_run(run_id, status="failed", error=message, pending=None)


def start_run(account_id: str) -> dict:
    if store.get_account(account_id) is None:
        raise RunError(f"Unknown account '{account_id}'", status_code=404)
    if settings.demo_mode and account_id not in DEMO_ACCOUNT_IDS:
        raise RunError(f"Demo mode only supports accounts {', '.join(DEMO_ACCOUNT_IDS)}.")
    run_id = uuid.uuid4().hex[:12]
    store.create_run(run_id, account_id)
    payload = {"run_id": run_id, "account_id": account_id}
    threading.Thread(target=_execute, args=(run_id, payload), daemon=True, name=f"run-{run_id}").start()
    return get_run(run_id)


def get_run(run_id: str) -> dict:
    run = store.get_run(run_id)
    if run is None:
        raise RunError(f"Unknown run '{run_id}'", status_code=404)
    run["steps"] = store.get_audit(run_id)
    run["demo_mode"] = settings.demo_mode
    return run


def submit_decision(run_id: str, action: str, message: str | None = None) -> dict:
    run = get_run(run_id)
    if run["status"] != "awaiting_approval":
        raise RunError(f"Run is '{run['status']}', not awaiting approval.", status_code=409)
    if action == "edit" and not (message or "").strip():
        raise RunError("An edited message is required for action 'edit'.", status_code=422)
    store.update_run(run_id, status="running", pending=None)
    # The remaining steps (mock send, audit) make no LLM calls, so resume inline.
    _execute(run_id, Command(resume={"action": action, "message": message or ""}))
    return get_run(run_id)
