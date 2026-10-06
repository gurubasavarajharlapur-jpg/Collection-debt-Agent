"""Streamlit UI for Collections Copilot. Talks to the FastAPI service only."""
from __future__ import annotations

import html
import json
import os
import time

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
STEP_ICONS = {"ok": "✅", "waiting": "⏸️", "error": "❌"}
STATUS_LABELS = {
    "running": "🔄 Running",
    "awaiting_approval": "⏸️ Waiting for your approval",
    "sent": "✅ Sent (mock)",
    "rejected": "🚫 Rejected by agent",
    "escalated": "🧑‍💼 Escalated to a human specialist",
    "failed": "❌ Failed",
}

st.set_page_config(page_title="Collections Copilot", page_icon="📨", layout="wide")


def api(method: str, path: str, **kwargs):
    """Call the API. Returns (data, error_message)."""
    try:
        response = httpx.request(method, f"{API_URL}{path}", timeout=30, **kwargs)
    except httpx.HTTPError as exc:
        return None, f"Cannot reach the API at {API_URL} ({type(exc).__name__}). Is it running?"
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        return None, f"{detail} (HTTP {response.status_code})"
    return response.json(), None


def show_message(text: str) -> None:
    """Render a customer message; dir=auto gives Arabic its right-to-left layout."""
    st.markdown(
        f'<div dir="auto" style="border:1px solid #8884;border-radius:8px;padding:12px 16px;'
        f'white-space:pre-wrap;line-height:1.7">{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def show_trace(steps: list[dict], running: bool) -> None:
    for step in steps:
        icon = STEP_ICONS.get(step["status"], "•")
        st.markdown(f"{icon} **{step['seq']}. {step['step']}** · {step['duration_ms']} ms"
                    + (" · waiting for human" if step["status"] == "waiting" else ""))
    if running:
        st.markdown("⏳ _working…_")


def show_compliance(history: list[dict]) -> None:
    for result in history:
        verdict = "PASS" if result["passed"] else "FAIL"
        with st.expander(f"Attempt {result['attempt']}: {verdict}", expanded=not result["passed"] or len(history) == 1):
            cols = st.columns(2)
            cols[0].markdown("**Rule checks:** " + ("pass" if not result["rule_violations"] else "fail"))
            cols[1].markdown("**LLM reviewer:** " + ("pass" if result["llm_passed"] else "fail"))
            for violation in result["rule_violations"]:
                st.error(f"`{violation['rule']}` {violation['detail']}")
            for issue in result["llm_issues"]:
                st.warning(issue)


def show_approval(run: dict) -> None:
    run_id, pending = run["run_id"], run["pending"] or {}
    st.subheader("Your decision")
    if pending.get("edit_violations"):
        st.error("Your edit was not sent because it breaks policy rules. Fix it and resubmit:")
        for violation in pending["edit_violations"]:
            st.error(f"`{violation['rule']}` {violation['detail']}")
    default = pending.get("rejected_edit") or pending.get("draft", "")
    edited = st.text_area("Message (edit here if needed)", value=default, height=180, key=f"edit-{run_id}")
    approve, edit, reject = st.columns(3)
    decision = None
    if approve.button("✅ Approve draft", type="primary", width="stretch"):
        decision = {"action": "approve"}
    if edit.button("✏️ Send edited", width="stretch", disabled=edited.strip() == pending.get("draft", "").strip()):
        decision = {"action": "edit", "message": edited}
    if reject.button("🚫 Reject", width="stretch"):
        decision = {"action": "reject"}
    if decision:
        _, error = api("POST", f"/runs/{run_id}/decision", json=decision)
        if error:
            st.error(error)
        else:
            st.rerun()


def show_run(run: dict) -> None:
    state, status = run["state"], run["status"]
    usage = state.get("token_usage") or {}

    top = st.columns([3, 1, 1])
    top[0].markdown(f"### {STATUS_LABELS.get(status, status)}")
    top[0].caption(f"Run `{run['run_id']}` · account {run['account_id']}")
    top[1].metric("Tokens this run", f"{usage.get('total_tokens', 0):,}")
    top[2].metric("LLM calls", usage.get("llm_calls", 0))
    if run["demo_mode"]:
        top[1].caption("Demo mode makes no LLM calls.")

    if status == "failed":
        st.error(f"**The run failed.** {run['error'] or 'Unknown error.'}")
        st.caption("Nothing was sent. Fix the problem and press “Run agent” to try again.")

    trace_col, detail_col = st.columns([1, 2])
    with trace_col:
        st.subheader("Live trace")
        show_trace(run["steps"], running=status == "running")

    with detail_col:
        if strategy := state.get("strategy"):
            st.subheader(f"Strategy: {strategy['strategy'].replace('_', ' ')}")
            st.write(strategy["reasoning"])
            st.info(f"**Cited policy rule:** {strategy['cited_policy_rule']}")

        if chunks := state.get("policy_chunks"):
            with st.expander(f"Retrieved policy chunks ({len(chunks)}) · retriever: {state.get('retriever_backend')}"):
                for chunk in chunks:
                    st.markdown(f"**{chunk['source']}** · score {chunk['score']}")
                    st.code(chunk["text"], language="markdown")

        if state.get("draft"):
            st.subheader(f"Draft message (attempt {state.get('draft_attempts', 1)})")
            show_message(state["draft"])

        if history := state.get("compliance_history"):
            st.subheader("Compliance result")
            show_compliance(history)

        if status == "awaiting_approval":
            show_approval(run)
        elif status == "sent":
            st.success("Approved and recorded by the MOCK sender. No real message was sent.")
            show_message(state["final_message"])
        elif status == "rejected":
            st.warning("You rejected the draft. Nothing was sent.")
        elif status == "escalated":
            st.warning("No automated message was sent. " + state.get("escalation", {}).get("reason", ""))


def agent_tab(health: dict) -> None:
    accounts, error = api("GET", "/accounts")
    if error:
        st.error(error)
        return

    st.subheader("Accounts")
    st.dataframe(
        [{
            "ID": a["account_id"], "Name": a["name"], "Product": f"{a['product_type']} · {a['product']}",
            "Amount due (AED)": a["amount_due"], "Days overdue": a["days_overdue"],
            "Broken promises": a["broken_promises"], "Language": a["preferred_language"],
            "Contacts": len(a["contact_history"]),
        } for a in accounts],
        width="stretch", hide_index=True, height=250,
    )

    runnable = [a for a in accounts if not health["demo_mode"] or a["account_id"] in health["demo_accounts"]]
    pick, button = st.columns([3, 1], vertical_alignment="bottom")
    account = pick.selectbox(
        "Account to work", runnable,
        format_func=lambda a: f"{a['account_id']} · {a['name']} · {a['days_overdue']} days overdue",
        help="Demo mode has pre-written outputs for three accounts only." if health["demo_mode"] else None,
    )
    if button.button("▶ Run agent", type="primary", width="stretch"):
        run, error = api("POST", f"/runs/{account['account_id']}")
        if error:
            st.error(error)
        else:
            st.session_state["run_id"] = run["run_id"]

    run_id = st.session_state.get("run_id")
    if not run_id:
        st.info("Pick an account and press “Run agent”.")
        return
    st.divider()
    run, error = api("GET", f"/runs/{run_id}")
    if error:
        st.error(error)
        return
    show_run(run)
    if run["status"] == "running":  # poll for the live trace
        time.sleep(1)
        st.rerun()


def audit_tab() -> None:
    runs, error = api("GET", "/runs")
    if error:
        st.error(error)
        return
    if not runs:
        st.info("No runs yet.")
        return
    current = st.session_state.get("run_id")
    ids = [r["run_id"] for r in runs]
    labels = {r["run_id"]: f"{r['run_id']} · {r['account_id']} · {r['status']} · {r['created_at'][:19]}" for r in runs}
    run_id = st.selectbox("Run", ids, index=ids.index(current) if current in ids else 0, format_func=labels.get)
    entries, error = api("GET", f"/audit/{run_id}")
    if error:
        st.error(error)
        return
    st.dataframe(
        [{"#": e["seq"], "Step": e["step"], "Status": e["status"], "Started (UTC)": e["started_at"],
          "Duration (ms)": e["duration_ms"]} for e in entries],
        width="stretch", hide_index=True,
    )
    st.download_button("Download audit log (JSON)", json.dumps(entries, ensure_ascii=False, indent=2),
                       file_name=f"audit-{run_id}.json", mime="application/json")
    for entry in entries:
        with st.expander(f"{entry['seq']}. {entry['step']} · {entry['status']}"):
            left, right = st.columns(2)
            left.caption("Input")
            left.json(entry["input"], expanded=1)
            right.caption("Output")
            right.json(entry["output"], expanded=1)


def main() -> None:
    st.title("📨 Collections Copilot")
    st.caption("Demo agent for a collections team · synthetic data only · sending is mocked")

    health, error = api("GET", "/health")
    if error:
        st.error(error)
        st.stop()
    if health["demo_mode"]:
        st.warning("**DEMO MODE** · no LLM calls are made. Pre-written outputs for "
                   + ", ".join(health["demo_accounts"]) + ". Set DEMO_MODE=false for live runs.", icon="🧪")
    else:
        st.caption(f"Live LLM: `{health['llm']}`")

    agent, audit = st.tabs(["Agent", "Audit log"])
    with agent:
        agent_tab(health)
    with audit:
        audit_tab()


main()
