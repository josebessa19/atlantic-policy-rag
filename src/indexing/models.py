"""Retrieval hit model returned by hybrid_search."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Hit(BaseModel):
    """One fused hybrid-search result (payload + RRF score)."""

    score: float
    document_id: str
    section: str
    chunk_id: str
    status: str = "active"
    chunk_type: str = "section"
    text: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
