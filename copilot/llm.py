"""LLM access: provider factory, free-tier protection, and the three LLM tasks.

The graph only talks to an object with three methods (decide_strategy,
draft_message, review_compliance). LiveLLM implements them with a real model,
copilot.demo.DemoLLM with pre-written outputs, and the tests with a fake.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import threading
import time
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from .config import settings

log = logging.getLogger(__name__)

Usage = dict  # {"input_tokens", "output_tokens", "total_tokens", "llm_calls"}
NO_USAGE: Usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "llm_calls": 0}


class LLMError(RuntimeError):
    """An LLM call failed in a way the user should see (safe to display)."""


# --- Structured outputs --------------------------------------------------------

class StrategyDecision(BaseModel):
    strategy: Literal["reminder", "payment_plan", "escalate"] = Field(
        description="The single collections strategy to use for this account.")
    reasoning: str = Field(description="Two or three sentences explaining the choice from the account facts.")
    cited_policy_rule: str = Field(
        description="The policy rule number that justifies the choice, with a short quote, e.g. '6.2: Payment plan ...'.")
    discount_pct: float = Field(default=0, description="Settlement discount to offer in percent, 0 for none.")
    installments: int = Field(default=0, description="Number of monthly installments to offer, 0 for none.")


class ComplianceVerdict(BaseModel):
    passed: bool = Field(description="True only if the message follows every policy rule provided.")
    issues: list[str] = Field(default_factory=list, description="Each policy problem found, with its rule number.")
    feedback: str = Field(default="", description="Concrete instructions for rewriting the message, empty if it passed.")


# --- Provider factory ----------------------------------------------------------

def build_chat_model():
    """Return a LangChain chat model for the configured provider.

    Only langchain-groq is installed by default. For the others, install the
    package named below and set its credentials in .env.
    """
    common = dict(model=settings.llm_model, temperature=settings.llm_temperature,
                  max_tokens=settings.llm_max_tokens, timeout=settings.llm_timeout_s,
                  max_retries=0)  # retries are handled in call_with_retry
    provider = settings.llm_provider
    if provider == "groq":
        if not os.getenv("GROQ_API_KEY"):
            raise LLMError("GROQ_API_KEY is not set. Add it to .env, or set DEMO_MODE=true to run without an LLM.")
        from langchain_groq import ChatGroq
        return ChatGroq(**common, **settings.llm_extra_kwargs)
    if provider == "anthropic":  # pip install langchain-anthropic; ANTHROPIC_API_KEY
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(**common)
    if provider == "azure_openai":  # pip install langchain-openai; AZURE_OPENAI_* variables
        from langchain_openai import AzureChatOpenAI
        common["azure_deployment"] = common.pop("model")
        return AzureChatOpenAI(**common)
    raise LLMError(f"Unknown LLM_PROVIDER '{provider}'. Use groq, anthropic or azure_openai.")


# --- Free-tier protection: spacing, timeouts, backoff ----------------------------

_throttle_lock = threading.Lock()
_last_call_at = 0.0


def _throttle() -> None:
    """Keep at least llm_min_interval_s between any two LLM calls in this process."""
    global _last_call_at
    with _throttle_lock:
        wait = _last_call_at + settings.llm_min_interval_s - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call_at = time.monotonic()


def _status_code(exc: Exception) -> int | None:
    code = getattr(exc, "status_code", None)
    if code is None:
        code = getattr(getattr(exc, "response", None), "status_code", None)
    return code


def _is_retryable(exc: Exception) -> bool:
    code = _status_code(exc)
    if code == 429 or (code is not None and code >= 500):
        return True
    name = type(exc).__name__.lower()
    return "timeout" in name or "connection" in name


def _retry_after(exc: Exception) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    try:
        return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None


def _safe_message(exc: Exception) -> str:
    text = re.sub(r"\b(gsk|sk)[-_][A-Za-z0-9_-]{8,}", "[redacted]", str(exc))
    return text[:300]


def call_with_retry(fn: Callable[[], Any], sleep: Callable[[float], None] = time.sleep) -> Any:
    """Run fn with spacing between calls and exponential backoff on 429 / transient errors."""
    for attempt in range(settings.llm_max_retries + 1):
        _throttle()
        try:
            return fn()
        except LLMError:
            raise
        except Exception as exc:
            code = _status_code(exc)
            if not _is_retryable(exc):
                if code in (401, 403):
                    raise LLMError("The LLM provider rejected the API key. Check GROQ_API_KEY in .env.") from exc
                raise LLMError(f"LLM call failed ({type(exc).__name__}): {_safe_message(exc)}") from exc
            if attempt == settings.llm_max_retries:
                reason = "rate limit (HTTP 429)" if code == 429 else type(exc).__name__
                raise LLMError(
                    f"LLM call failed after {attempt + 1} attempts: {reason}. "
                    "The free tier may be exhausted; wait a minute and retry, or use DEMO_MODE=true."
                ) from exc
            delay = _retry_after(exc) or settings.llm_backoff_base_s * 2 ** attempt
            delay += random.uniform(0, 0.5)
            log.warning("LLM call failed (%s); retry %d in %.1fs", code or type(exc).__name__, attempt + 1, delay)
            sleep(delay)


# --- Prompts ---------------------------------------------------------------------

def _policy_block(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks)


def _account_block(account: dict) -> str:
    return json.dumps(account, ensure_ascii=False, indent=2)


STRATEGY_SYSTEM = (
    "You are a collections strategy assistant for a fictional demo company. Choose exactly one strategy for the "
    "account: reminder, payment_plan or escalate. Base the decision ONLY on the policy excerpts provided and the "
    "account facts, including the notes in the contact history. Escalation triggers override everything else. "
    "Cite the rule number you relied on. If you propose a discount or installments, stay within the policy limits "
    "for the account's overdue bucket; never propose both."
)

DRAFT_SYSTEM = (
    "You write short customer messages for a fictional demo collections team. Follow the policy excerpts exactly. "
    "Write ONLY in the requested language, at most 90 words, respectful and factual, addressed to the customer by "
    "name, stating the product and amount due, and inviting a reply. Never threaten or mention consequences. "
    "Offer only what the strategy specifies. Output the message text only: no preamble, no notes, no placeholders."
)

COMPLIANCE_SYSTEM = (
    "You are an independent compliance reviewer for a fictional demo collections team. Check the draft message "
    "against the policy excerpts: tone, banned phrases, discount cap for the overdue bucket, payment plan limits, "
    "and language. Be strict about threats or pressure, but do not invent rules that are not in the excerpts. "
    "If anything is wrong, list each issue with its rule number and give concrete rewrite instructions."
)

LANGUAGES = {"en": "English", "ar": "Arabic"}


def _text(message) -> str:
    content = message.content
    if isinstance(content, list):  # some providers return content blocks
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return content.strip()


def _usage(message) -> Usage:
    meta = getattr(message, "usage_metadata", None) or {}
    return {
        "input_tokens": meta.get("input_tokens", 0),
        "output_tokens": meta.get("output_tokens", 0),
        "total_tokens": meta.get("total_tokens", 0),
        "llm_calls": 1,
    }


class LiveLLM:
    """The three LLM tasks, backed by the configured chat model."""

    def __init__(self, model=None) -> None:
        self._model = model

    @property
    def model(self):
        if self._model is None:
            self._model = build_chat_model()
        return self._model

    def _structured(self, schema: type[BaseModel], system: str, user: str) -> tuple[Any, Usage]:
        # Structured output is done through tool calling, so any tool-calling model works.
        runnable = self.model.with_structured_output(schema, include_raw=True)
        result = call_with_retry(lambda: runnable.invoke([("system", system), ("human", user)]))
        if result.get("parsed") is None:
            raise LLMError(f"The model did not return a valid {schema.__name__}. Try running the agent again.")
        return result["parsed"], _usage(result["raw"])

    def decide_strategy(self, account: dict, chunks: list[dict]) -> tuple[StrategyDecision, Usage]:
        user = f"POLICY EXCERPTS\n{_policy_block(chunks)}\n\nACCOUNT\n{_account_block(account)}"
        return self._structured(StrategyDecision, STRATEGY_SYSTEM, user)

    def draft_message(self, account: dict, strategy: dict, chunks: list[dict],
                      feedback: str = "", previous_draft: str = "", attempt: int = 1) -> tuple[str, Usage]:
        user = (
            f"POLICY EXCERPTS\n{_policy_block(chunks)}\n\nACCOUNT\n{_account_block(account)}\n\n"
            f"STRATEGY\n{json.dumps(strategy, ensure_ascii=False)}\n\n"
            f"LANGUAGE: {LANGUAGES.get(account['preferred_language'], 'English')}"
        )
        if feedback:
            user += (f"\n\nYour previous draft was rejected by compliance.\nPREVIOUS DRAFT\n{previous_draft}\n\n"
                     f"COMPLIANCE FEEDBACK\n{feedback}\n\nWrite a corrected message that fixes every point.")
        message = call_with_retry(lambda: self.model.invoke([("system", DRAFT_SYSTEM), ("human", user)]))
        text = _text(message)
        if not text:
            raise LLMError("The model returned an empty draft. Try running the agent again.")
        return text, _usage(message)

    def review_compliance(self, account: dict, message: str, chunks: list[dict],
                          attempt: int = 1) -> tuple[ComplianceVerdict, Usage]:
        user = (
            f"POLICY EXCERPTS\n{_policy_block(chunks)}\n\n"
            f"ACCOUNT FACTS\ndays_overdue: {account['days_overdue']}\n"
            f"preferred_language: {LANGUAGES.get(account['preferred_language'], 'English')}\n\n"
            f"DRAFT MESSAGE\n{message}"
        )
        return self._structured(ComplianceVerdict, COMPLIANCE_SYSTEM, user)
