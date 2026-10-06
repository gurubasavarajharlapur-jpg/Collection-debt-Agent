"""Data access: synthetic accounts (JSON) and runtime records (SQLite)."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Iterator

from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    account_id  TEXT NOT NULL,
    status      TEXT NOT NULL,
    error       TEXT,
    state_json  TEXT NOT NULL DEFAULT '{}',
    pending_json TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    step        TEXT NOT NULL,
    status      TEXT NOT NULL,
    input_json  TEXT NOT NULL,
    output_json TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    duration_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS sent_messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    account_id  TEXT NOT NULL,
    language    TEXT NOT NULL,
    message     TEXT NOT NULL,
    mock        INTEGER NOT NULL DEFAULT 1,
    sent_at     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS escalations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    account_id  TEXT NOT NULL,
    reason      TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    settings.var_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


# --- Accounts ----------------------------------------------------------------

@lru_cache(maxsize=1)
def _accounts() -> dict[str, dict]:
    rows = json.loads(settings.accounts_path.read_text(encoding="utf-8"))
    return {row["account_id"]: row for row in rows}


def list_accounts() -> list[dict]:
    return list(_accounts().values())


def get_account(account_id: str) -> dict | None:
    return _accounts().get(account_id)


# --- Runs --------------------------------------------------------------------

def create_run(run_id: str, account_id: str) -> None:
    ts = now_iso()
    with connect() as conn:
        conn.execute(
            "INSERT INTO runs (run_id, account_id, status, created_at, updated_at) VALUES (?,?,?,?,?)",
            (run_id, account_id, "running", ts, ts),
        )


_UNSET: Any = object()


def update_run(run_id: str, *, status: str | None = None, error: Any = _UNSET,
               state: dict | None = None, pending: Any = _UNSET) -> None:
    sets, params = ["updated_at = ?"], [now_iso()]
    if status is not None:
        sets.append("status = ?"); params.append(status)
    if error is not _UNSET:
        sets.append("error = ?"); params.append(error)
    if state is not None:
        sets.append("state_json = ?"); params.append(_dumps(state))
    if pending is not _UNSET:
        sets.append("pending_json = ?"); params.append(None if pending is None else _dumps(pending))
    with connect() as conn:
        conn.execute(f"UPDATE runs SET {', '.join(sets)} WHERE run_id = ?", (*params, run_id))


def _run_row(row: sqlite3.Row) -> dict:
    return {
        "run_id": row["run_id"],
        "account_id": row["account_id"],
        "status": row["status"],
        "error": row["error"],
        "state": json.loads(row["state_json"]),
        "pending": json.loads(row["pending_json"]) if row["pending_json"] else None,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def get_run(run_id: str) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    return _run_row(row) if row else None


def list_runs(limit: int = 50) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT run_id, account_id, status, created_at FROM runs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


# --- Audit log ---------------------------------------------------------------

def write_audit(run_id: str, step: str, status: str, input_: Any, output: Any,
                started_at: str, duration_ms: int) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO audit_log (run_id, step, status, input_json, output_json, started_at, duration_ms)"
            " VALUES (?,?,?,?,?,?,?)",
            (run_id, step, status, _dumps(input_), _dumps(output), started_at, duration_ms),
        )


def get_audit(run_id: str) -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM audit_log WHERE run_id = ? ORDER BY id", (run_id,)).fetchall()
    return [
        {
            "seq": i + 1,
            "step": r["step"],
            "status": r["status"],
            "input": json.loads(r["input_json"]),
            "output": json.loads(r["output_json"]),
            "started_at": r["started_at"],
            "duration_ms": r["duration_ms"],
        }
        for i, r in enumerate(rows)
    ]


# --- Mock outbound channels --------------------------------------------------

def record_sent_message(run_id: str, account_id: str, language: str, message: str) -> dict:
    sent_at = now_iso()
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO sent_messages (run_id, account_id, language, message, mock, sent_at) VALUES (?,?,?,?,1,?)",
            (run_id, account_id, language, message, sent_at),
        )
        return {"message_id": cur.lastrowid, "sent_at": sent_at, "mock": True}


def get_sent_messages(run_id: str) -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM sent_messages WHERE run_id = ?", (run_id,)).fetchall()
    return [dict(r) for r in rows]


def record_escalation(run_id: str, account_id: str, reason: str) -> dict:
    created_at = now_iso()
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO escalations (run_id, account_id, reason, created_at) VALUES (?,?,?,?)",
            (run_id, account_id, reason, created_at),
        )
        return {"escalation_id": cur.lastrowid, "created_at": created_at}
