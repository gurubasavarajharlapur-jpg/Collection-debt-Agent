"""Deterministic compliance rules. These never call an LLM.

The LLM reviewer in the graph is a second opinion on tone; these rules are the
hard guardrail. Limits come from data/policy.md via copilot.policy.
"""
from __future__ import annotations

import re

from .policy import load_policy

MAX_WORDS = 120                   # policy 1.6
ESCALATE_BROKEN_PROMISES = 3      # policy 7.1
ESCALATE_DAYS_OVERDUE = 180       # policy 7.2
ESCALATE_AMOUNT_DUE = 50_000      # policy 7.3

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٪٫", "01234567890123456789%.")
_NUM = r"(\d+(?:\.\d+)?)"
_PERCENT = re.compile(
    rf"{_NUM}\s*%|%\s*{_NUM}|{_NUM}\s*(?:percent|per cent|بالمئة|بالمائة|في المئة|في المائة)",
    re.IGNORECASE,
)
_INSTALLMENTS = re.compile(
    r"(\d+)\s+(?:(?:equal|monthly|easy)\s+)*(?:installments?|instalments?|payments)"
    r"|(\d+)\s*(?:أقساط|قسط[اًا]?|دفعات|دفعة)",
    re.IGNORECASE,
)
_ARABIC_LETTER = re.compile(r"[؀-ۿ]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")


def _numbers(pattern: re.Pattern, text: str) -> list[float]:
    return [float(g) for match in pattern.finditer(text) for g in match.groups() if g]


def offered_discount(message: str) -> float:
    """Largest percentage mentioned. Any percentage is treated as a discount (conservative)."""
    return max(_numbers(_PERCENT, message.translate(_ARABIC_DIGITS)), default=0.0)


def offered_installments(message: str) -> int:
    return int(max(_numbers(_INSTALLMENTS, message.translate(_ARABIC_DIGITS)), default=0))


def banned_phrases_in(message: str) -> list[str]:
    found = []
    for phrase in load_policy().banned_phrases:
        if _LATIN_LETTER.search(phrase):  # whole words only, so "court" does not match "courtesy"
            hit = re.search(rf"\b{re.escape(phrase)}\b", message, re.IGNORECASE)
        else:
            hit = phrase in message
        if hit:
            found.append(phrase)
    return found


def rule_checks(message: str, account: dict) -> list[dict]:
    """Return a list of violations ({"rule", "detail"}); empty means the message passes."""
    policy = load_policy()
    days = account["days_overdue"]
    violations: list[dict] = []

    if not message.strip():
        return [{"rule": "empty_message", "detail": "The message is empty."}]

    discount, cap = offered_discount(message), policy.discount_cap(days)
    if discount > cap:
        violations.append({
            "rule": "discount_cap",
            "detail": f"Offers a {discount:g}% discount; the cap for {days} days overdue is {cap:g}% (policy 4).",
        })

    installments, max_installments = offered_installments(message), policy.installment_cap(days)
    if installments > max_installments:
        violations.append({
            "rule": "installment_cap",
            "detail": f"Offers {installments} installments; the limit for {days} days overdue is "
                      f"{max_installments} (policy 5).",
        })
    if discount > 0 and installments > 0:
        violations.append({
            "rule": "discount_with_plan",
            "detail": "Combines a discount with a payment plan (policy 4.2).",
        })

    banned = banned_phrases_in(message)
    if banned:
        violations.append({
            "rule": "banned_phrase",
            "detail": "Contains banned phrase(s): " + ", ".join(f"'{p}'" for p in banned) + " (policy 2).",
        })

    arabic = len(_ARABIC_LETTER.findall(message))
    latin = len(_LATIN_LETTER.findall(message))
    if account["preferred_language"] == "ar" and arabic <= latin:
        violations.append({"rule": "language", "detail": "The customer prefers Arabic; write in Arabic (policy 1.5)."})
    if account["preferred_language"] == "en" and arabic > 0:
        violations.append({"rule": "language", "detail": "The customer prefers English; do not use Arabic (policy 1.5)."})

    words = len(message.split())
    if words > MAX_WORDS:
        violations.append({"rule": "length", "detail": f"{words} words; the limit is {MAX_WORDS} (policy 1.6)."})

    return violations


def hard_escalation_triggers(account: dict) -> list[str]:
    """Escalation triggers that can be read straight off the account record (policy 7.1-7.3)."""
    triggers = []
    if account["broken_promises"] >= ESCALATE_BROKEN_PROMISES:
        triggers.append(f"7.1: {account['broken_promises']} broken promises to pay")
    if account["days_overdue"] > ESCALATE_DAYS_OVERDUE:
        triggers.append(f"7.2: {account['days_overdue']} days overdue")
    if account["amount_due"] > ESCALATE_AMOUNT_DUE:
        triggers.append(f"7.3: amount due {account['amount_due']:,.0f} is above {ESCALATE_AMOUNT_DUE:,}")
    return triggers
