"""FastAPI application entrypoint.

Run on the host so BGE-M3 can use the GPU; Qdrant stays in Docker::

    uvicorn backend.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.routers.query import router as query_router
from src.guardrails.temporal import load_active_supersedes
from src.indexing.config import qdrant_url
from src.indexing.store import get_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Cache supersession map so assembly does not scroll on every request.
    # None if Qdrant is down at startup — answer_query reloads per request.
    try:
        get_client(qdrant_url()).get_collections()
        app.state.active_supersedes = load_active_supersedes()
        app.state.qdrant_ok = True
    except Exception:
        app.state.active_supersedes = None
        app.state.qdrant_ok = False
    yield


app = FastAPI(
    title="Atlantic Policy RAG API",
    description=(
        "Stateless policy Q&A over hybrid-retrieved corporate documents. "
        "OpenAPI at /docs."
    ),
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(query_router)
