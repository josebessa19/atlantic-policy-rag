"""Default chunking strategy for structured policy documents."""

from __future__ import annotations

from src.ingestion.chunking.models import ChunkStrategy

# Structure-aware walk of ParsedDocument elements (tables stay atomic).
LOCKED_CHUNK_STRATEGY: ChunkStrategy = "structure"


def get_chunk_strategy() -> ChunkStrategy:
    """Return the default chunking strategy name."""
    return LOCKED_CHUNK_STRATEGY
