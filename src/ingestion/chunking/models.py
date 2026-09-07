"""Chunk data model and config for retrieval-ready document pieces."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ChunkType(str, Enum):
    SECTION = "section"
    TABLE = "table"
    PARTIAL_SECTION = "partial_section"


class ChunkConfig(BaseModel):
    """Token budget for section splits (tables stay atomic)."""

    max_tokens: int = Field(default=500, ge=1)
    overlap_tokens: int = Field(default=50, ge=0)
    min_tokens: int | None = Field(default=None, ge=1)


class Chunk(BaseModel):
    id: str
    text: str
    section_path: str
    source: str
    chunk_type: Literal["section", "table", "partial_section"]
    token_estimate: int
    page: int | None = None
    element_index: int | None = None


ChunkStrategy = Literal["structure", "markdown"]
