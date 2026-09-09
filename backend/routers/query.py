"""POST /query and GET /health — thin HTTP over the RAG service."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from backend.schemas.query import QueryRequest, QueryResponse
from backend.services.rag import UNAVAILABLE_TEMPLATE, answer_query
from src.indexing.config import qdrant_url
from src.indexing.store import get_client

logger = logging.getLogger(__name__)

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
    try:
        return answer_query(
            body.query,
            active_supersedes=supersedes if supersedes is not None else None,
        )
    except Exception:
        logger.exception("POST /query failed")
        return QueryResponse(answer=UNAVAILABLE_TEMPLATE, refused=True, citations=[])
