"""Parse data/policy.md once into (a) RAG chunks and (b) machine-checkable rules.

The Markdown file is the single source of truth: the discount caps, installment
caps and banned phrases enforced in code are read from it, not duplicated.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from .config import settings

# (low, high) inclusive day range; high is None for an open-ended bucket ("181+").
Bucket = tuple[int, int | None]


@dataclass(frozen=True)
class Policy:
    chunks: tuple[dict, ...]                       # {"id", "source", "text"}
    discount_caps: tuple[tuple[Bucket, float], ...]
    installment_caps: tuple[tuple[Bucket, int], ...]
    banned_phrases: tuple[str, ...]

    def discount_cap(self, days_overdue: int) -> float:
        return _lookup(self.discount_caps, days_overdue, default=0.0)

    def installment_cap(self, days_overdue: int) -> int:
        return _lookup(self.installment_caps, days_overdue, default=0)


def _lookup(table, days: int, default):
    for (low, high), value in table:
        if days >= low and (high is None or days <= high):
            return value
    return default


def _sections(markdown: str) -> list[tuple[str, str]]:
    """Split on level-2 headings -> [(heading, body)]."""
    parts = re.split(r"^## +(.+)$", markdown, flags=re.MULTILINE)
    return [(parts[i].strip(), parts[i + 1].strip()) for i in range(1, len(parts) - 1, 2)]


def _table(body: str) -> list[tuple[Bucket, str]]:
    rows = []
    for match in re.finditer(r"^\|\s*(\d+)\s*(?:-\s*(\d+)|\+)\s*\|\s*([^|]+?)\s*\|", body, flags=re.MULTILINE):
        low, high, value = match.groups()
        rows.append(((int(low), int(high) if high else None), value))
    return rows


def _find(sections: list[tuple[str, str]], keyword: str) -> str:
    for heading, body in sections:
        if keyword.lower() in heading.lower():
            return body
    raise ValueError(f"policy.md has no section containing '{keyword}'")


@lru_cache(maxsize=1)
def load_policy() -> Policy:
    markdown = settings.policy_path.read_text(encoding="utf-8")
    sections = _sections(markdown)
    filename = settings.policy_path.name

    chunks = tuple(
        {"id": f"policy-{i}", "source": f"{filename} § {heading}", "text": f"{heading}\n{body}"}
        for i, (heading, body) in enumerate(sections, start=1)
    )
    discount_caps = tuple(
        (bucket, float(value.rstrip("%"))) for bucket, value in _table(_find(sections, "discount"))
    )
    installment_caps = tuple(
        (bucket, int(value)) for bucket, value in _table(_find(sections, "payment plan"))
    )
    banned = tuple(
        line[2:].strip() for line in _find(sections, "banned").splitlines() if line.startswith("- ")
    )
    return Policy(chunks, discount_caps, installment_caps, banned)
