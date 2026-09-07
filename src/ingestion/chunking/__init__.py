"""Document chunking: structure-aware walk of ParsedDocument elements."""

from __future__ import annotations

from src.ingestion.chunking.default import LOCKED_CHUNK_STRATEGY, get_chunk_strategy
from src.ingestion.chunking.models import Chunk, ChunkConfig, ChunkStrategy, ChunkType
from src.ingestion.chunking.structure import chunk_structure
from src.ingestion.models import ParsedDocument

__all__ = [
    "Chunk",
    "ChunkConfig",
    "ChunkStrategy",
    "ChunkType",
    "LOCKED_CHUNK_STRATEGY",
    "chunk_document",
    "chunk_structure",
    "get_chunk_strategy",
]


def chunk_document(
    doc: ParsedDocument,
    config: ChunkConfig | None = None,
) -> list[Chunk]:
    """Chunk a ParsedDocument with the structure-aware strategy."""
    config = config or ChunkConfig()
    return chunk_structure(doc, config)
