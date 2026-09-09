"""Local dense embeddings via sentence-transformers BGE-M3 (GPU when available).

Uses the reference ``BAAI/bge-m3`` checkpoint — not FastEmbed's custom ONNX hack.
FastEmbed remains only for BM25 sparse vectors in ``store.py``.
"""

from __future__ import annotations

import logging
from typing import Sequence

from src.indexing.config import embedding_dim, embedding_model

# BGE retrieval query prefix (documents are encoded without this string).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

logger = logging.getLogger(__name__)

_model = None


def _resolve_device() -> str:
    """Prefer CUDA; fall back to CPU with a clear path for ops."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def resolve_embedding_device() -> str:
    """Public helper for startup logs (demo: show cuda vs cpu)."""
    return _resolve_device()


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        device = _resolve_device()
        logger.info("Embedding device: %s", device)
        _model = SentenceTransformer(embedding_model(), device=device)
    return _model


def reset_embedding_model() -> None:
    """Clear cached encoder (tests)."""
    global _model
    _model = None


def _validate_vector(vec: list[float], *, expected_dim: int) -> list[float]:
    if len(vec) != expected_dim:
        raise ValueError(
            f"Embedding length {len(vec)} != expected EMBEDDING_DIM={expected_dim}"
        )
    if not any(v != 0.0 for v in vec):
        raise ValueError("Embedding is an all-zero vector; refusing to index")
    return list(vec)


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Embed document chunks. Fail loud on empty input or wrong length."""
    if not texts:
        raise ValueError("embed_texts requires a non-empty list of texts")
    for i, t in enumerate(texts):
        if t is None or not str(t).strip():
            raise ValueError(f"embed_texts: empty text at index {i}")

    expected = embedding_dim()
    model = _get_model()
    raw = model.encode(
        [str(t) for t in texts],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    if len(raw) != len(texts):
        raise ValueError(
            f"Embedding model returned {len(raw)} vectors for {len(texts)} texts"
        )
    return [
        _validate_vector(list(map(float, v)), expected_dim=expected) for v in raw
    ]


def embed_query(query: str) -> list[float]:
    """Embed a search query (BGE query prefix; documents stay prefix-free)."""
    if query is None or not str(query).strip():
        raise ValueError("embed_query requires a non-empty query string")
    return embed_texts([QUERY_PREFIX + str(query).strip()])[0]
