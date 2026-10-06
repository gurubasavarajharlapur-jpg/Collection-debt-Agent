"""The Collections Copilot workflow as a LangGraph state machine.

load_account -> retrieve_policy -> decide_strategy -> draft_message -> compliance_check
    -> human_approval (interrupt) -> send_message -> audit_log

compliance_check loops back to draft_message with feedback (max 2 retries),
then escalates. Every node is wrapped by `audited`, which records its input,
output, start time and duration in SQLite.
"""
from __future__ import annotations

import functools
import operator
import time
from typing import Annotated, Any, Callable, Literal, TypedDict

from langchain_core.tools import tool
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from . import rag, store
from .compliance import hard_escalation_triggers, rule_checks
from .config import settings
from .llm import NO_USAGE, Usage


def add_usage(left: Usage | None, right: Usage | None) -> Usage:
    left, right = left or NO_USAGE, right or NO_USAGE
    return {key: left.get(key, 0) + right.get(key, 0) for key in NO_USAGE}


class CopilotState(TypedDict, total=False):
    run_id: str
    account_id: str
    account: dict                     # load_account
    policy_chunks: list[dict]         # retrieve_policy
    retriever_backend: str
    strategy: dict                    # decide_strategy
    draft: str                        # draft_message
    draft_attempts: int
    compliance: dict                  # compliance_check (latest result)
    compliance_history: Annotated[list[dict], operator.add]
    human_decision: dict              # human_approval
    final_message: str
    send_receipt: dict                # send_message
    escalation: dict                  # escalate
    outcome: Literal["sent", "rejected", "escalated"]
    summary: dict                     # audit_log
    token_usage: Annotated[Usage, add_usage]


# --- Tools ---------------------------------------------------------------------

@tool
def fetch_account(account_id: str) -> dict:
    """Fetch a debtor account and its contact history by account id."""
    account = store.get_account(account_id)
    if account is None:
        raise ValueError(f"Unknown account '{account_id}'")
    return account


@tool
def mock_send_message(run_id: str, account_id: str, language: str, message: str) -> dict:
    """MOCK send: record the approved message in SQLite. Nothing leaves this machine."""
    return store.record_sent_message(run_id, account_id, language, message)


@tool
def mock_escalate(run_id: str, account_id: str, reason: str) -> dict:
    """MOCK escalation: record a hand-off to a human collections specialist in SQLite."""
    return store.record_escalation(run_id, account_id, reason)


# --- Audit wrapper ---------------------------------------------------------------

def audited(step: str, reads: tuple[str, ...]) -> Callable:
    """Record a node's input (the state keys it reads), output, start time and duration."""

    def decorator(fn: Callable[[CopilotState], dict]) -> Callable[[CopilotState], dict]:
        @functools.wraps(fn)
        def wrapper(state: CopilotState) -> dict:
            started_at, t0 = store.now_iso(), time.perf_counter()
            input_ = {key: state.get(key) for key in reads}

            def record(status: str, output: Any) -> None:
                elapsed = int((time.perf_counter() - t0) * 1000)
                store.write_audit(state["run_id"], step, status, input_, output, started_at, elapsed)

            try:
                output = fn(state)
            except GraphBubbleUp:  # interrupt(): the node is paused, not finished
                record("waiting", {"waiting_for": "human decision"})
                raise
            except Exception as exc:
                record("error", {"error": f"{type(exc).__name__}: {exc}"})
                raise
            record("ok", output)
            return output

        return wrapper

    return decorator


# --- Graph -----------------------------------------------------------------------

def build_graph(llm, checkpointer=None):
    """Compile the workflow. `llm` is LiveLLM, DemoLLM or a test fake."""
    max_drafts = 1 + settings.max_draft_retries

    @audited("load_account", reads=("account_id",))
    def load_account(state: CopilotState) -> dict:
        return {"account": fetch_account.invoke({"account_id": state["account_id"]})}

    @audited("retrieve_policy", reads=("account_id",))
    def retrieve_policy(state: CopilotState) -> dict:
        chunks, backend = rag.retrieve_for_account(state["account"])
        return {"policy_chunks": chunks, "retriever_backend": backend}

    @audited("decide_strategy", reads=("account", "policy_chunks"))
    def decide_strategy(state: CopilotState) -> dict:
        decision, usage = llm.decide_strategy(state["account"], state["policy_chunks"])
        strategy = decision.model_dump()
        # Guardrail: triggers that can be read off the record are not left to the model.
        triggers = hard_escalation_triggers(state["account"])
        if triggers and strategy["strategy"] != "escalate":
            strategy.update(
                overridden_from=strategy["strategy"],
                strategy="escalate",
                reasoning="Overridden by rule check: " + "; ".join(triggers) + ".",
                cited_policy_rule=triggers[0],
            )
        return {"strategy": strategy, "token_usage": usage}

    @audited("draft_message", reads=("strategy", "draft", "compliance", "draft_attempts"))
    def draft_message(state: CopilotState) -> dict:
        attempt = state.get("draft_attempts", 0) + 1
        feedback = (state.get("compliance") or {}).get("feedback", "") if attempt > 1 else ""
        text, usage = llm.draft_message(
            state["account"], state["strategy"], state["policy_chunks"],
            feedback=feedback, previous_draft=state.get("draft", ""), attempt=attempt,
        )
        return {"draft": text, "draft_attempts": attempt, "token_usage": usage}

    @audited("compliance_check", reads=("draft", "draft_attempts"))
    def compliance_check(state: CopilotState) -> dict:
        account, draft, attempt = state["account"], state["draft"], state["draft_attempts"]
        violations = rule_checks(draft, account)                        # deterministic rules
        verdict, usage = llm.review_compliance(account, draft, state["policy_chunks"], attempt=attempt)
        feedback = [v["detail"] for v in violations] + list(verdict.issues)
        if verdict.feedback:
            feedback.append(verdict.feedback)
        result = {
            "attempt": attempt,
            "passed": not violations and verdict.passed,
            "rule_violations": violations,
            "llm_passed": verdict.passed,
            "llm_issues": verdict.issues,
            "feedback": "\n".join(f"- {line}" for line in feedback),
        }
        return {"compliance": result, "compliance_history": [result], "token_usage": usage}

    @audited("human_approval", reads=("draft", "strategy", "compliance"))
    def human_approval(state: CopilotState) -> dict:
        request = {"type": "approval_request", "draft": state["draft"], "edit_violations": []}
        while True:
            decision = interrupt(request)  # pauses here until the API resumes the run
            action = decision.get("action")
            if action == "approve":
                return {"human_decision": {"action": "approve"}, "final_message": state["draft"]}
            if action == "reject":
                return {"human_decision": {"action": "reject", "note": decision.get("message", "")},
                        "outcome": "rejected"}
            # edit: a human edit must still pass the hard rules before it can be sent
            edited = (decision.get("message") or "").strip()
            violations = rule_checks(edited, state["account"])
            if not violations:
                return {"human_decision": {"action": "edit"}, "final_message": edited}
            request = {**request, "rejected_edit": edited, "edit_violations": violations}

    @audited("send_message", reads=("final_message",))
    def send_message(state: CopilotState) -> dict:
        receipt = mock_send_message.invoke({
            "run_id": state["run_id"], "account_id": state["account_id"],
            "language": state["account"]["preferred_language"], "message": state["final_message"],
        })
        return {"send_receipt": receipt, "outcome": "sent"}

    @audited("escalate", reads=("strategy", "compliance", "draft_attempts"))
    def escalate(state: CopilotState) -> dict:
        if state["strategy"]["strategy"] == "escalate":
            reason = f"Strategy: {state['strategy']['cited_policy_rule']}"
        else:
            reason = (f"7.7: draft failed the compliance check {state['draft_attempts']} times. "
                      f"Last feedback:\n{state['compliance']['feedback']}")
        record = mock_escalate.invoke(
            {"run_id": state["run_id"], "account_id": state["account_id"], "reason": reason})
        return {"escalation": {**record, "reason": reason}, "outcome": "escalated"}

    @audited("audit_log", reads=("outcome", "token_usage", "draft_attempts"))
    def audit_log(state: CopilotState) -> dict:
        """Close the run with a summary record (the per-step records are written by `audited`)."""
        return {"summary": {
            "outcome": state["outcome"],
            "strategy": state["strategy"]["strategy"],
            "draft_attempts": state.get("draft_attempts", 0),
            "human_action": (state.get("human_decision") or {}).get("action"),
            "token_usage": state.get("token_usage", NO_USAGE),
            "demo_mode": settings.demo_mode,
        }}

    def after_strategy(state: CopilotState) -> str:
        return "escalate" if state["strategy"]["strategy"] == "escalate" else "draft_message"

    def after_compliance(state: CopilotState) -> str:
        if state["compliance"]["passed"]:
            return "human_approval"
        return "draft_message" if state["draft_attempts"] < max_drafts else "escalate"

    def after_approval(state: CopilotState) -> str:
        return "audit_log" if state.get("outcome") == "rejected" else "send_message"

    graph = StateGraph(CopilotState)
    for node in (load_account, retrieve_policy, decide_strategy, draft_message, compliance_check,
                 human_approval, send_message, escalate, audit_log):
        graph.add_node(node.__name__, node)

    graph.add_edge(START, "load_account")
    graph.add_edge("load_account", "retrieve_policy")
    graph.add_edge("retrieve_policy", "decide_strategy")
    graph.add_conditional_edges("decide_strategy", after_strategy, ["draft_message", "escalate"])
    graph.add_edge("draft_message", "compliance_check")
    graph.add_conditional_edges("compliance_check", after_compliance,
                                ["human_approval", "draft_message", "escalate"])
    graph.add_conditional_edges("human_approval", after_approval, ["send_message", "audit_log"])
    graph.add_edge("send_message", "audit_log")
    graph.add_edge("escalate", "audit_log")
    graph.add_edge("audit_log", END)
    return graph.compile(checkpointer=checkpointer)
