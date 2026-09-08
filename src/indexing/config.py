"""Indexing config from environment (embeddings + Qdrant). No Gemini key here."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_EMBEDDING_DIM = 1024
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_QDRANT_COLLECTION = "atlantic_policies"

DENSE_VECTOR_NAME = "dense"
BM25_VECTOR_NAME = "bm25"

PREFETCH_LIMIT = 20
DEFAULT_SEARCH_LIMIT = 8


def embedding_model() -> str:
    return os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip() or DEFAULT_EMBEDDING_MODEL


def embedding_dim() -> int:
    raw = os.getenv("EMBEDDING_DIM", str(DEFAULT_EMBEDDING_DIM)).strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"EMBEDDING_DIM must be an int, got {raw!r}") from exc


def qdrant_url() -> str:
    return os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL).strip() or DEFAULT_QDRANT_URL


def qdrant_collection() -> str:
    return (
        os.getenv("QDRANT_COLLECTION", DEFAULT_QDRANT_COLLECTION).strip()
        or DEFAULT_QDRANT_COLLECTION
    )


def auto_ingest() -> bool:
    """True when AUTO_INGEST is 1/true/yes. Default off (host pytest must not download BGE-M3)."""
    raw = os.getenv("AUTO_INGEST", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}
