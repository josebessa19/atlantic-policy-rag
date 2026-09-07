"""Ingest pipeline: parse policy ``.txt`` files → retrieval-ready chunks."""

from __future__ import annotations

from pathlib import Path

from src.ingestion.chunking import Chunk, ChunkConfig, chunk_document
from src.ingestion.policy_parser import parse_policy_file


def parse_dir(
    raw_dir: Path | str,
    config: ChunkConfig | None = None,
) -> list[Chunk]:
    """Parse all ``*.txt`` under ``raw_dir`` (sorted) into a deterministic chunk list."""
    root = Path(raw_dir)
    paths = sorted(root.glob("*.txt"))
    chunks: list[Chunk] = []
    for path in paths:
        doc = parse_policy_file(path)
        chunks.extend(chunk_document(doc, config=config))
    return chunks
