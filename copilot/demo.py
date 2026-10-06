"""DEMO_MODE: pre-written LLM outputs for three sample accounts. No network.

Only the LLM calls are replaced. Retrieval, the rule checks, the retry loop,
human approval, the mock send and the audit log all run for real, so ACC-002's
first draft (25% discount against a 10% cap) is caught by the actual rules.
"""
from __future__ import annotations

import time

from .config import settings
from .llm import NO_USAGE, ComplianceVerdict, LLMError, StrategyDecision, Usage

PASS = {"passed": True, "issues": [], "feedback": ""}

SCRIPTS: dict[str, dict] = {
    # Straightforward reminder, English, passes first time.
    "ACC-001": {
        "strategy": {
            "strategy": "reminder",
            "reasoning": "The account is 12 days overdue with no broken promises and no previous contact, "
                         "so a first, friendly reminder is the proportionate step. No discount applies in the "
                         "1-30 day bucket.",
            "cited_policy_rule": "6.1: Reminder: use for accounts 1-30 days overdue with no broken promises, "
                                 "or as the first contact on any account.",
        },
        "drafts": [
            "Dear Daniel, this is a friendly reminder from Sandbar Financial Services that AED 1,850.00 on "
            "your credit card is now 12 days overdue. If you have already paid, thank you and please ignore "
            "this message. Otherwise you can pay through the app, or simply reply here and we will be glad "
            "to help.",
        ],
        "reviews": [PASS],
    },
    # Payment plan, Arabic. The first draft breaks the discount cap and is sent back.
    "ACC-002": {
        "strategy": {
            "strategy": "payment_plan",
            "reasoning": "The account is 75 days overdue, a reminder has already been sent and the customer "
                         "broke one promise to pay after a delayed salary. That signals difficulty paying in "
                         "full, so a payment plan within the 6-installment limit fits. No escalation trigger applies.",
            "cited_policy_rule": "6.2: Payment plan: use for accounts more than 30 days overdue where a reminder "
                                 "has already been sent, or where the customer has one or two broken promises.",
            "installments": 6,
        },
        "drafts": [
            "عزيزتي ليلى، نود تذكيرك بأن مبلغ 2,400 درهم مستحق على باقة الهاتف المحمول الخاصة بك. يسعدنا أن "
            "نقدم لك خصماً بنسبة 25% إذا سددتِ كامل المبلغ خلال 14 يوماً. للاستفسار أو طلب المساعدة يمكنك "
            "الرد على هذه الرسالة.",
            "عزيزتي ليلى، نتفهم أن الظروف قد تتغير. المبلغ المستحق على باقة الهاتف المحمول الخاصة بك هو "
            "2,400 درهم. يمكننا مساعدتك بخطة سداد من 6 أقساط شهرية متساوية بقيمة 400 درهم لكل قسط، على أن "
            "يُسدد القسط الأول خلال 7 أيام من الموافقة. إذا كانت هذه الخطة تناسبك أو رغبتِ في مناقشة خيار "
            "آخر، يكفي الرد على هذه الرسالة وسنكون سعداء بمساعدتك.",
        ],
        "reviews": [
            {
                "passed": False,
                "issues": [
                    "4: a 25% discount is above the 10% cap for the 61-90 day bucket.",
                    "6.2: the chosen strategy is a payment plan, but the draft offers a settlement discount instead.",
                ],
                "feedback": "Remove the discount. Offer a payment plan of at most 6 equal monthly installments "
                            "with the first due within 7 days, in Arabic.",
            },
            PASS,
        ],
    },
    # Escalation: no message is drafted, the account goes to a human specialist.
    "ACC-003": {
        "strategy": {
            "strategy": "escalate",
            "reasoning": "The customer has broken three promises to pay and the account is 204 days overdue. "
                         "Both are escalation triggers, and escalation overrides the other strategies, so no "
                         "automated message should be sent.",
            "cited_policy_rule": "7.1: The customer has three or more broken promises to pay. "
                                 "(Also 7.2: more than 180 days overdue.)",
        },
        "drafts": [],
        "reviews": [],
    },
}

DEMO_ACCOUNT_IDS = tuple(SCRIPTS)


class DemoLLM:
    """Same interface as copilot.llm.LiveLLM, returning the scripts above."""

    def _script(self, account: dict) -> dict:
        time.sleep(settings.demo_step_delay_s)  # so the live trace is visible
        script = SCRIPTS.get(account["account_id"])
        if script is None:
            raise LLMError(
                f"Demo mode only has pre-written outputs for {', '.join(DEMO_ACCOUNT_IDS)}. "
                "Set DEMO_MODE=false and add GROQ_API_KEY to run other accounts."
            )
        return script

    def decide_strategy(self, account: dict, chunks: list[dict]) -> tuple[StrategyDecision, Usage]:
        return StrategyDecision(**self._script(account)["strategy"]), dict(NO_USAGE)

    def draft_message(self, account: dict, strategy: dict, chunks: list[dict],
                      feedback: str = "", previous_draft: str = "", attempt: int = 1) -> tuple[str, Usage]:
        drafts = self._script(account)["drafts"]
        return drafts[min(attempt, len(drafts)) - 1], dict(NO_USAGE)

    def review_compliance(self, account: dict, message: str, chunks: list[dict],
                          attempt: int = 1) -> tuple[ComplianceVerdict, Usage]:
        reviews = self._script(account)["reviews"]
        return ComplianceVerdict(**reviews[min(attempt, len(reviews)) - 1]), dict(NO_USAGE)
