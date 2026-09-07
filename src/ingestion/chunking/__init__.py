"""Document chunking: structure-aware (default) and Markdown-section helper."""

from __future__ import annotations

from pathlib import Path

from src.ingestion.chunking.default import LOCKED_CHUNK_STRATEGY, get_chunk_strategy
from src.ingestion.chunking.markdown import chunk_markdown
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
    "chunk_markdown",
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


def chunk_markdown_file(
    md_path: Path | str,
    config: ChunkConfig | None = None,
) -> list[Chunk]:
    """Convenience: chunk a Markdown file with the markdown-section helper."""
    path = Path(md_path)
    return chunk_markdown(path.read_text(encoding="utf-8"), source=str(path), config=config)
