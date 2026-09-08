"""Query API request/response models.

A future multi-tenant deploy could add ``tenant_id`` as a Qdrant payload
filter at retrieval time — not implemented in this single-tenant PoC.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language policy question")


class Citation(BaseModel):
    document_id: str
    section: str
    relevance_score: float


class QueryResponse(BaseModel):
    answer: str
    refused: bool
    citations: list[Citation] = Field(default_factory=list)
