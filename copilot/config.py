"""Central configuration. Everything tunable lives here.

Secrets are read from the environment (loaded from .env by python-dotenv) and
are never stored on the Settings object, logged or printed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # --- LLM -----------------------------------------------------------------
    # To swap provider, change these two values (or set LLM_PROVIDER / LLM_MODEL)
    # and supply that provider's credentials in .env. See copilot/llm.py.
    llm_provider: str = os.getenv("LLM_PROVIDER", "groq")  # groq | azure_openai | anthropic
    llm_model: str = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
    llm_temperature: float = 0.2
    llm_max_tokens: int = 2048
    # Provider-specific extras passed to the chat model constructor.
    llm_extra_kwargs: dict = field(default_factory=lambda: {"reasoning_effort": "low"})

    # --- Free-tier protection ------------------------------------------------
    llm_timeout_s: float = 30.0       # per-request timeout
    llm_min_interval_s: float = 2.0   # minimum gap between any two LLM calls
    llm_max_retries: int = 4          # retries on 429 / transient errors
    llm_backoff_base_s: float = 2.0   # 2s, 4s, 8s, 16s (+ jitter)

    # --- RAG -----------------------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_top_k: int = 3                # chunks kept per query

    # --- Workflow ------------------------------------------------------------
    max_draft_retries: int = 2        # re-drafts after a failed compliance check
    demo_mode: bool = field(default_factory=lambda: _flag("DEMO_MODE"))
    demo_step_delay_s: float = 0.6    # makes the live trace visible in demo mode

    # --- Paths ---------------------------------------------------------------
    accounts_path: Path = ROOT / "data" / "accounts.json"
    policy_path: Path = ROOT / "data" / "policy.md"
    var_dir: Path = Path(os.getenv("COPILOT_VAR_DIR", str(ROOT / "var")))

    api_url: str = os.getenv("API_URL", "http://127.0.0.1:8000")

    @property
    def db_path(self) -> Path:
        return self.var_dir / "copilot.db"

    @property
    def checkpoint_path(self) -> Path:
        return self.var_dir / "checkpoints.db"


settings = Settings()
