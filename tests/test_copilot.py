from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from conftest import FakeLLM
from copilot import rag, service, store
from copilot.config import settings
from copilot.graph import build_graph
from copilot.llm import LLMError, call_with_retry
from copilot.policy import load_policy

# ACC-009: English, telecom, 67 days overdue -> 10% discount cap, 6 installments max.
ACCOUNT = "ACC-009"
OVER_CAP = "Dear Grace, settle your AED 1,980 device plan balance this week and we will take 30% off."
COMPLIANT = ("Dear Grace, AED 1,980 is overdue on your device plan. We can spread it over 6 monthly "
             "installments. Reply to this message and we will set it up.")


def start(llm, account_id=ACCOUNT):
    graph = build_graph(llm, MemorySaver())
    config = {"configurable": {"thread_id": "t1"}}
    state = graph.invoke({"run_id": "t1", "account_id": account_id}, config)
    return graph, config, state


def wait_for(client: TestClient, run_id: str, timeout: float = 20) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        run = client.get(f"/runs/{run_id}").json()
        if run["status"] != "running":
            return run
        time.sleep(0.05)
    raise AssertionError("run did not finish in time")


def test_synthetic_data_and_policy_retrieval():
    accounts = store.list_accounts()
    assert len(accounts) == 15
    assert {a["preferred_language"] for a in accounts} == {"en", "ar"}
    assert {a["product_type"] for a in accounts} == {"bank", "telecom"}

    policy = load_policy()
    assert [policy.discount_cap(d) for d in (10, 45, 75, 120, 400)] == [0, 5, 10, 15, 20]
    assert "legal action" in policy.banned_phrases

    chunks, backend = rag.retrieve_for_account(store.get_account(ACCOUNT))
    assert backend == "keyword"
    assert all(c["source"].startswith("policy.md § ") and c["text"] for c in chunks)
    assert any("discount" in c["source"].lower() for c in chunks)


def test_draft_over_discount_cap_is_caught_and_redrafted():
    llm = FakeLLM(drafts=[OVER_CAP, COMPLIANT])
    graph, config, state = start(llm)

    first, second = state["compliance_history"]
    # The fake LLM reviewer said "pass"; the deterministic rule check is what caught it.
    assert first["llm_passed"] and not first["passed"]
    assert [v["rule"] for v in first["rule_violations"]] == ["discount_cap"]
    assert "30%" in first["feedback"] and "10%" in first["feedback"]
    # The feedback reached the second draft, which passed.
    assert "cap for 67 days overdue is 10%" in llm.feedback_seen[1]
    assert second["passed"] and state["draft"] == COMPLIANT and state["draft_attempts"] == 2
    # Paused for a human; nothing has been sent.
    assert graph.get_state(config).next == ("human_approval",)
    assert store.get_sent_messages("t1") == []


def test_escalates_after_two_failed_retries():
    llm = FakeLLM(drafts=[OVER_CAP])  # never fixes the draft
    graph, config, state = start(llm)

    assert state["draft_attempts"] == 1 + settings.max_draft_retries == 3
    assert state["outcome"] == "escalated"
    assert state["escalation"]["reason"].startswith("7.7")
    assert graph.get_state(config).next == ()  # finished without asking for approval
    assert store.get_sent_messages("t1") == []


def test_human_approval_gates_the_mock_send_and_every_step_is_audited():
    graph, config, _ = start(FakeLLM(drafts=[COMPLIANT]))

    # A human edit that breaks the cap is refused and the run stays paused.
    graph.invoke(Command(resume={"action": "edit", "message": "Dear Grace, pay today and get 50% off."}), config)
    pending = graph.get_state(config).tasks[0].interrupts[0].value
    assert [v["rule"] for v in pending["edit_violations"]] == ["discount_cap"]
    assert store.get_sent_messages("t1") == []

    state = graph.invoke(Command(resume={"action": "approve"}), config)
    assert state["outcome"] == "sent" and state["token_usage"]["llm_calls"] == 3
    (sent,) = store.get_sent_messages("t1")
    assert sent["message"] == COMPLIANT and sent["mock"] == 1

    audit = store.get_audit("t1")
    assert [e["step"] for e in audit if e["status"] == "ok"] == [
        "load_account", "retrieve_policy", "decide_strategy", "draft_message",
        "compliance_check", "human_approval", "send_message", "audit_log",
    ]
    assert all(e["started_at"] and e["input"] is not None and e["output"] is not None for e in audit)


def test_api_demo_mode_runs_end_to_end_without_an_api_key(monkeypatch):
    monkeypatch.setattr(settings, "demo_mode", True)
    client = TestClient(service_app())

    assert client.get("/health").json()["demo_mode"] is True
    assert len(client.get("/accounts").json()) == 15
    assert client.post("/runs/ACC-010").status_code == 400  # no demo script for this account
    assert client.post("/runs/NOPE").status_code == 404

    # ACC-002: the scripted first draft offers 25% against a 10% cap, is caught, then fixed.
    run_id = client.post("/runs/ACC-002").json()["run_id"]
    run = wait_for(client, run_id)
    assert run["status"] == "awaiting_approval"
    assert [c["passed"] for c in run["state"]["compliance_history"]] == [False, True]

    done = client.post(f"/runs/{run_id}/decision", json={"action": "approve"}).json()
    assert done["status"] == "sent" and done["state"]["send_receipt"]["mock"] is True
    assert client.post(f"/runs/{run_id}/decision", json={"action": "approve"}).status_code == 409
    assert client.get(f"/audit/{run_id}").json()[-1]["step"] == "audit_log"

    # ACC-003: three broken promises -> escalated, no approval step, nothing sent.
    escalated = wait_for(client, client.post("/runs/ACC-003").json()["run_id"])
    assert escalated["status"] == "escalated" and "draft" not in escalated["state"]


def test_missing_api_key_fails_the_run_with_a_clear_error():
    client = TestClient(service_app())  # live mode, GROQ_API_KEY removed by the fixture
    run = wait_for(client, client.post(f"/runs/{ACCOUNT}").json()["run_id"])

    assert run["status"] == "failed"
    assert "GROQ_API_KEY is not set" in run["error"]
    assert run["steps"][-1]["step"] == "decide_strategy" and run["steps"][-1]["status"] == "error"


def test_rate_limit_retries_with_exponential_backoff():
    class RateLimited(Exception):
        status_code = 429

    sleeps: list[float] = []
    calls = iter([RateLimited(), RateLimited(), "ok"])

    def flaky():
        result = next(calls)
        if isinstance(result, Exception):
            raise result
        return result

    assert call_with_retry(flaky, sleep=sleeps.append) == "ok"
    assert len(sleeps) == 2 and 2.0 <= sleeps[0] <= 2.5 and 4.0 <= sleeps[1] <= 4.5

    def always_limited():
        raise RateLimited()

    with pytest.raises(LLMError, match="429"):
        call_with_retry(always_limited, sleep=lambda _: None)


def service_app():
    from copilot.api import app
    return app
