"""Qdrant collection: named dense + BM25 vectors, upsert, hybrid RRF search."""

from __future__ import annotations

from typing import Any, Sequence

from qdrant_client import QdrantClient, models

from src.ingestion.chunking.models import Chunk
from src.indexing.config import (
    BM25_VECTOR_NAME,
    DEFAULT_SEARCH_LIMIT,
    DENSE_VECTOR_NAME,
    PREFETCH_LIMIT,
    embedding_dim,
    qdrant_collection,
    qdrant_url,
)
from src.indexing.embeddings import embed_query, embed_texts
from src.indexing.ids import point_id
from src.indexing.models import Hit

_bm25 = None


def _get_bm25():
    global _bm25
    if _bm25 is None:
        from fastembed import SparseTextEmbedding

        _bm25 = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _bm25


def reset_bm25_model() -> None:
    """Clear cached BM25 encoder (tests)."""
    global _bm25
    _bm25 = None


def get_client(url: str | None = None) -> QdrantClient:
    # Client may be newer than the pinned compose image; search still works.
    return QdrantClient(url=url or qdrant_url(), check_compatibility=False)


def chunk_to_payload(chunk: Chunk) -> dict[str, Any]:
    """Payload for citations (now) and temporal drop (Step 3)."""
    section = chunk.section_path.strip() if chunk.section_path.strip() else (chunk.title or "")
    return {
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
        "source": chunk.source,
        "section": section,
        "chunk_type": chunk.chunk_type,
        "text": chunk.text,
        "status": chunk.status,
        "effective_date": chunk.effective_date.isoformat() if chunk.effective_date else None,
        "supersedes": list(chunk.supersedes),
        "title": chunk.title or "",
    }


def _sparse_vectors(texts: Sequence[str]) -> list[models.SparseVector]:
    model = _get_bm25()
    out: list[models.SparseVector] = []
    for emb in model.embed(list(texts)):
        out.append(
            models.SparseVector(
                indices=list(map(int, emb.indices)),
                values=list(map(float, emb.values)),
            )
        )
    return out


def ensure_collection(
    client: QdrantClient | None = None,
    *,
    collection: str | None = None,
    recreate: bool = False,
) -> str:
    """Create named vectors dense (cosine) + bm25 (IDF). Fail on dim mismatch."""
    client = client or get_client()
    name = collection or qdrant_collection()
    dim = embedding_dim()

    exists = client.collection_exists(name)
    if exists and recreate:
        client.delete_collection(name)
        exists = False

    if exists:
        info = client.get_collection(name)
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            dense_cfg = vectors.get(DENSE_VECTOR_NAME)
        else:
            dense_cfg = None
        if dense_cfg is None or getattr(dense_cfg, "size", None) != dim:
            raise ValueError(
                f"Collection {name!r} dense vector missing or size mismatch "
                f"(expected named {DENSE_VECTOR_NAME!r} size={dim}). "
                f"Re-run with --recreate."
            )
        sparse = info.config.params.sparse_vectors or {}
        if BM25_VECTOR_NAME not in sparse:
            raise ValueError(
                f"Collection {name!r} missing sparse vector {BM25_VECTOR_NAME!r}. "
                f"Re-run with --recreate."
            )
        return name

    client.create_collection(
        collection_name=name,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=dim,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            BM25_VECTOR_NAME: models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        },
    )
    return name


def index_chunks(
    chunks: Sequence[Chunk],
    *,
    client: QdrantClient | None = None,
    collection: str | None = None,
    recreate: bool = False,
    dense_vectors: Sequence[Sequence[float]] | None = None,
) -> int:
    """Embed (unless dense_vectors provided) + BM25 + upsert by stable chunk id."""
    if not chunks:
        return 0

    client = client or get_client()
    name = ensure_collection(client, collection=collection, recreate=recreate)

    texts = [c.text for c in chunks]
    if dense_vectors is None:
        dense = embed_texts(texts)
    else:
        if len(dense_vectors) != len(chunks):
            raise ValueError("dense_vectors length must match chunks")
        dense = [list(map(float, v)) for v in dense_vectors]
    sparse = _sparse_vectors(texts)

    points: list[models.PointStruct] = []
    for chunk, dvec, svec in zip(chunks, dense, sparse, strict=True):
        points.append(
            models.PointStruct(
                id=str(point_id(chunk.id)),
                vector={
                    DENSE_VECTOR_NAME: dvec,
                    BM25_VECTOR_NAME: svec,
                },
                payload=chunk_to_payload(chunk),
            )
        )

    client.upsert(collection_name=name, points=points)
    return len(points)


def hybrid_search(
    query: str,
    *,
    limit: int = DEFAULT_SEARCH_LIMIT,
    client: QdrantClient | None = None,
    collection: str | None = None,
    dense_vector: Sequence[float] | None = None,
    prefetch_limit: int = PREFETCH_LIMIT,
) -> list[Hit]:
    """Prefetch dense + BM25, fuse with Qdrant RRF. No temporal drop (Step 3)."""
    if query is None or not str(query).strip():
        raise ValueError("hybrid_search requires a non-empty query")

    client = client or get_client()
    name = collection or qdrant_collection()
    q = str(query).strip()

    dense = list(map(float, dense_vector)) if dense_vector is not None else embed_query(q)
    bm25_q = _sparse_vectors([q])[0]

    result = client.query_points(
        collection_name=name,
        prefetch=[
            models.Prefetch(
                query=dense,
                using=DENSE_VECTOR_NAME,
                limit=prefetch_limit,
            ),
            models.Prefetch(
                query=bm25_q,
                using=BM25_VECTOR_NAME,
                limit=prefetch_limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )

    hits: list[Hit] = []
    for point in result.points:
        payload = dict(point.payload or {})
        hits.append(
            Hit(
                score=float(point.score),
                document_id=str(payload.get("document_id", "")),
                section=str(payload.get("section", "")),
                chunk_id=str(payload.get("chunk_id", "")),
                status=str(payload.get("status", "active")),
                chunk_type=str(payload.get("chunk_type", "section")),
                text=str(payload.get("text", "")),
                payload=payload,
            )
        )
    return hits
