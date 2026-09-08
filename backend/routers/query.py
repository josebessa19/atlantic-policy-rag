"""POST /query and GET /health — thin HTTP over the RAG service."""

from __future__ import annotations

from fastapi import APIRouter, Request

from backend.schemas.query import QueryRequest, QueryResponse
from backend.services.rag import answer_query
from src.indexing.config import qdrant_url
from src.indexing.store import get_client

router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict:
    """API liveness + optional Qdrant ping."""
    qdrant_status = "ok"
    try:
        client = get_client(qdrant_url())
        client.get_collections()
    except Exception:
        qdrant_status = "unavailable"
    status = "ok" if qdrant_status == "ok" else "degraded"
    return {"status": status, "qdrant": qdrant_status}


@router.post("/query", response_model=QueryResponse)
def query(body: QueryRequest, request: Request) -> QueryResponse:
    """Stateless policy Q&A: retrieve → temporal drop → guardrails → generate."""
    supersedes = getattr(request.app.state, "active_supersedes", None)
    return answer_query(
        body.query,
        active_supersedes=supersedes if supersedes is not None else None,
    )
