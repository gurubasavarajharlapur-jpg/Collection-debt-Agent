"""RAG over the sample policy: MiniLM embeddings + FAISS, with an offline fallback.

The embedding model runs locally. If it is not available (never downloaded and
no internet, e.g. a fresh machine in demo mode) retrieval falls back to a small
keyword scorer so the workflow still runs. Each result carries its source.
"""
from __future__ import annotations

import logging
import math
import os
import re
from collections import Counter
from functools import lru_cache

from .config import settings
from .policy import load_policy

log = logging.getLogger(__name__)


class FaissRetriever:
    backend = "faiss"

    def __init__(self, chunks: tuple[dict, ...], model) -> None:
        import faiss

        self.chunks = chunks
        self.model = model
        vectors = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(vectors.shape[1])  # cosine, vectors are normalised
        self.index.add(vectors)

    def search(self, query: str, k: int) -> list[dict]:
        vector = self.model.encode([query], normalize_embeddings=True)
        scores, ids = self.index.search(vector, min(k, len(self.chunks)))
        return [
            {**self.chunks[i], "score": round(float(s), 4)}
            for s, i in zip(scores[0], ids[0]) if i >= 0
        ]


class KeywordRetriever:
    """TF-IDF-style overlap scorer. Used only when embeddings are unavailable."""

    backend = "keyword"

    def __init__(self, chunks: tuple[dict, ...]) -> None:
        self.chunks = chunks
        self.docs = [Counter(_tokens(c["text"])) for c in chunks]
        df = Counter(token for doc in self.docs for token in doc)
        self.idf = {t: math.log(1 + len(chunks) / n) for t, n in df.items()}

    def search(self, query: str, k: int) -> list[dict]:
        terms = set(_tokens(query))
        scored = []
        for chunk, doc in zip(self.chunks, self.docs):
            score = sum(self.idf[t] * (1 + math.log(doc[t])) for t in terms if t in doc)
            scored.append((score / math.sqrt(sum(doc.values())), chunk))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [{**chunk, "score": round(score, 4)} for score, chunk in scored[:k] if score > 0]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _load_embedding_model():
    from sentence_transformers import SentenceTransformer

    try:  # already cached locally: no network needed
        return SentenceTransformer(settings.embedding_model, local_files_only=True)
    except Exception:
        if settings.demo_mode:
            raise  # demo mode must not depend on the internet
        return SentenceTransformer(settings.embedding_model)  # first run: download once


@lru_cache(maxsize=1)
def get_retriever():
    chunks = load_policy().chunks
    if os.getenv("RAG_BACKEND", "").lower() != "keyword":
        try:
            return FaissRetriever(chunks, _load_embedding_model())
        except Exception as exc:  # missing model, offline, blocked native library
            log.warning("Embedding retriever unavailable (%s); using keyword fallback.", type(exc).__name__)
    return KeywordRetriever(chunks)


def account_queries(account: dict) -> list[str]:
    days, promises = account["days_overdue"], account["broken_promises"]
    return [
        f"Which strategy applies, reminder, payment plan or escalation trigger, for an account "
        f"{days} days overdue with {promises} broken promises and amount due {account['amount_due']}?",
        f"Maximum discount and payment plan installment limits for an account {days} days overdue",
        "Tone rules and banned phrases for a customer message",
    ]


def retrieve_for_account(account: dict) -> tuple[list[dict], str]:
    """Run one query per decision the agent has to make; merge, keeping the best score."""
    retriever = get_retriever()
    best: dict[str, dict] = {}
    for query in account_queries(account):
        for hit in retriever.search(query, settings.rag_top_k):
            if hit["id"] not in best or hit["score"] > best[hit["id"]]["score"]:
                best[hit["id"]] = {**hit, "query": query}
    chunks = sorted(best.values(), key=lambda c: c["score"], reverse=True)
    return chunks, retriever.backend
