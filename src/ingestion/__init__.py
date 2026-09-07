"""Ingestion package: policy parse + structure-aware chunking."""

from src.ingestion.chunking import (
    LOCKED_CHUNK_STRATEGY,
    Chunk,
    ChunkConfig,
    chunk_document,
    get_chunk_strategy,
)
from src.ingestion.models import ParsedDocument, PolicyMetadata
from src.ingestion.pipeline import parse_dir
from src.ingestion.policy_parser import parse_policy_file, parse_policy_text

__all__ = [
    "ParsedDocument",
    "PolicyMetadata",
    "Chunk",
    "ChunkConfig",
    "chunk_document",
    "get_chunk_strategy",
    "LOCKED_CHUNK_STRATEGY",
    "parse_dir",
    "parse_policy_file",
    "parse_policy_text",
]
