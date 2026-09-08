"""Map human chunk ids to Qdrant point UUIDs (stable upsert keys)."""

from __future__ import annotations

import uuid

# Fixed namespace so the same chunk.id always maps to the same point id.
_POINT_NAMESPACE = uuid.UUID("a7c3e9f1-2b4d-4e6f-8a0c-1d3e5f708192")


def point_id(chunk_id: str) -> uuid.UUID:
    """Deterministic UUID5 from Step 1 chunk id (Qdrant cannot use arbitrary strings)."""
    if not chunk_id or not chunk_id.strip():
        raise ValueError("chunk_id must be a non-empty string")
    return uuid.uuid5(_POINT_NAMESPACE, chunk_id.strip())
