"""Generation / RAG config from environment. Embeddings stay in indexing.config."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_LLM_MODEL = "gemini-2.5-flash"
# Qdrant Fusion.RRF score is rank-based (~1/(k+rank) sum). With k≈2 and two
# prefetches, a single top hit is often ~0.03–0.05; calibrate after ingest.
DEFAULT_MIN_FUSED_SCORE = 0.02


def gemini_api_key() -> str | None:
    raw = os.getenv("GEMINI_API_KEY", "").strip()
    return raw or None


def llm_model() -> str:
    return os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL).strip() or DEFAULT_LLM_MODEL


def min_fused_score() -> float:
    raw = os.getenv("MIN_FUSED_SCORE", str(DEFAULT_MIN_FUSED_SCORE)).strip()
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"MIN_FUSED_SCORE must be a float, got {raw!r}") from exc
