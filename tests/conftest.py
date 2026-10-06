"""Shared fixtures. No test needs a live API key, the network or the embedding model."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot import rag, service  # noqa: E402
from copilot.config import settings  # noqa: E402
from copilot.llm import NO_USAGE, ComplianceVerdict, StrategyDecision  # noqa: E402


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Fresh SQLite files per test, no delays, keyword retriever, no API key."""
    monkeypatch.setattr(settings, "var_dir", tmp_path)
    monkeypatch.setattr(settings, "demo_mode", False)
    monkeypatch.setattr(settings, "demo_step_delay_s", 0)
    monkeypatch.setattr(settings, "llm_min_interval_s", 0)
    monkeypatch.setenv("RAG_BACKEND", "keyword")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    rag.get_retriever.cache_clear()
    service.get_graph.cache_clear()
    yield
    rag.get_retriever.cache_clear()
    service.get_graph.cache_clear()


class FakeLLM:
    """Scripted stand-in for the LLM. Its reviewer always passes, so only the rule checks can fail a draft."""

    def __init__(self, drafts: list[str], strategy: str = "payment_plan") -> None:
        self.drafts = drafts
        self.strategy = strategy
        self.feedback_seen: list[str] = []

    def decide_strategy(self, account, chunks):
        decision = StrategyDecision(strategy=self.strategy, reasoning="scripted", cited_policy_rule="6.2: scripted")
        return decision, dict(NO_USAGE, total_tokens=10, llm_calls=1)

    def draft_message(self, account, strategy, chunks, feedback="", previous_draft="", attempt=1):
        self.feedback_seen.append(feedback)
        return self.drafts[min(attempt, len(self.drafts)) - 1], dict(NO_USAGE, total_tokens=10, llm_calls=1)

    def review_compliance(self, account, message, chunks, attempt=1):
        return ComplianceVerdict(passed=True), dict(NO_USAGE, total_tokens=10, llm_calls=1)
