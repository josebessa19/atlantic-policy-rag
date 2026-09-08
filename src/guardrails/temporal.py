"""Temporal assembly: drop superseded legacy chunks before the LLM (Option A)."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from qdrant_client import QdrantClient

from src.indexing.config import qdrant_collection, qdrant_url
from src.indexing.models import Hit
from src.indexing.store import get_client


def load_active_supersedes(
    client: QdrantClient | None = None,
    *,
    collection: str | None = None,
) -> set[str]:
    """Return document_ids superseded by at least one *active* doc in the index.

    Scrolls payload only (document_id, status, supersedes). Small corpus —
    fine to call once per request or cache in app lifespan.
    """
    client = client or get_client(qdrant_url())
    name = collection or qdrant_collection()
    superseded: set[str] = set()

    offset: Any = None
    while True:
        points, offset = client.scroll(
            collection_name=name,
            limit=128,
            offset=offset,
            with_payload=["document_id", "status", "supersedes"],
            with_vectors=False,
        )
        for point in points:
            payload = dict(point.payload or {})
            status = str(payload.get("status", "active")).lower()
            if status != "active":
                continue
            for doc_id in payload.get("supersedes") or []:
                if doc_id:
                    superseded.add(str(doc_id))
        if offset is None:
            break
    return superseded


def drop_superseded(
    hits: Sequence[Hit],
    active_supersedes: Iterable[str],
) -> list[Hit]:
    """Remove hits whose document_id is superseded by an active indexed document.

    Strict A+D: if POLICY-2025-002 (active) lists POLICY-2024-001 in
    ``supersedes``, every 2024 hit is dropped even when 2025 is absent from
    *this* hit list. Index still holds both editions; assembly never sends
    legacy text to the LLM.
    """
    blocked = {str(x) for x in active_supersedes if x}
    if not blocked:
        return list(hits)
    return [h for h in hits if h.document_id not in blocked]
