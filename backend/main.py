"""FastAPI application entrypoint.

Host (GPU conda) or Docker Compose::

    uvicorn backend.main:app --reload
    docker compose up --build
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from backend.routers.query import router as query_router
from src.guardrails.temporal import load_active_supersedes
from src.indexing.config import auto_ingest, qdrant_url
from src.indexing.embeddings import resolve_embedding_device
from src.indexing.ingest import ingest_if_empty
from src.indexing.store import get_client

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Corpus path relative to repo root (WORKDIR /app in Docker).
_RAW_DIR = Path("data/raw")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Log device early so the demo can show cuda vs cpu without waiting for a query.
    logger.info("Embedding device: %s", resolve_embedding_device())

    # Cache supersession map so assembly does not scroll on every request.
    # None if Qdrant is down at startup — answer_query reloads per request.
    # AUTO_INGEST is compose-only (default off) so pytest TestClient does not download BGE-M3.
    try:
        client = get_client(qdrant_url())
        client.get_collections()
        if auto_ingest():
            ingest_if_empty(data_dir=_RAW_DIR, client=client)
        app.state.active_supersedes = load_active_supersedes()
        app.state.qdrant_ok = True
    except Exception:
        logger.exception("Qdrant unavailable or ingest failed at startup")
        app.state.active_supersedes = None
        app.state.qdrant_ok = False
        if auto_ingest():
            # Fail loud when compose expects a populated index.
            raise
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
