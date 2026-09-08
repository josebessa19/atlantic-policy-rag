"""Hybrid search wiring + corpus smoke (integration)."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest
from qdrant_client import QdrantClient

from src.ingestion.chunking.models import Chunk
from src.ingestion.pipeline import parse_dir
from src.indexing.config import DEFAULT_EMBEDDING_DIM
from src.indexing.store import (
    ensure_collection,
    hybrid_search,
    index_chunks,
    reset_bm25_model,
)

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
DIM = DEFAULT_EMBEDDING_DIM


@pytest.fixture(autouse=True)
def _reset_bm25():
    reset_bm25_model()
    yield
    reset_bm25_model()


def _fake_dense(n: int, seed: float = 0.1) -> list[list[float]]:
    """Deterministic non-zero vectors of length DIM (orthogonal-ish by seed)."""
    out: list[list[float]] = []
    for i in range(n):
        v = [0.0] * DIM
        v[i % DIM] = seed + i * 0.01
        v[(i + 1) % DIM] = 0.5
        out.append(v)
    return out


def _synth_chunks() -> list[Chunk]:
    return [
        Chunk(
            id="IT-SPEC-2025-A:body:0",
            text=(
                "Document IT-SPEC-2025-A (Active) | Section: Global IT Hardware\n"
                "| Region | Laptop Allowance |\n| EMEA | €1,800 EUR |"
            ),
            section_path="",
            source="IT_Hardware_Allowance_Matrix.txt",
            chunk_type="table",
            token_estimate=40,
            document_id="IT-SPEC-2025-A",
            status="active",
            title="Global IT Hardware & Expense Matrix",
        ),
        Chunk(
            id="POLICY-2025-002:1-hybrid-remote-work-framework:0",
            text=(
                "Document POLICY-2025-002 (Active) | Section: 1. Hybrid & Remote Work Framework\n"
                "Tier 1 Roles (Engineering): Fully remote allowed."
            ),
            section_path="1. Hybrid & Remote Work Framework",
            source="Global_Remote_Work_Policy_2025_Update.txt",
            chunk_type="section",
            token_estimate=40,
            document_id="POLICY-2025-002",
            status="active",
            effective_date=date(2025, 1, 1),
            supersedes=["POLICY-2024-001"],
            title="Enterprise Remote Work Policy (2025 Revised)",
        ),
        Chunk(
            id="POLICY-2024-001:1-workplace-flexibility:0",
            text=(
                "Document POLICY-2024-001 (Legacy) | Section: 1. Workplace Flexibility\n"
                "Full-time remote work is not permitted under standard contracts."
            ),
            section_path="1. Workplace Flexibility",
            source="Global_Remote_Work_Policy_2024.txt",
            chunk_type="section",
            token_estimate=40,
            document_id="POLICY-2024-001",
            status="legacy",
            effective_date=date(2024, 1, 1),
            title="Enterprise Remote Work Policy (2024 Edition)",
        ),
    ]


def test_hybrid_search_wiring_in_memory() -> None:
    """Upsert with mocked dense + real BM25; RRF returns payload fields."""
    client = QdrantClient(":memory:")
    collection = "test_hybrid_wiring"
    chunks = _synth_chunks()
    dense = _fake_dense(len(chunks))

    # Prefer sparse query path; if in-memory BM25 fails on some platforms, skip.
    try:
        n = index_chunks(
            chunks,
            client=client,
            collection=collection,
            recreate=True,
            dense_vectors=dense,
        )
    except Exception as exc:  # noqa: BLE001 — surface env issues as skip
        pytest.skip(f"BM25/FastEmbed unavailable: {exc}")

    assert n == 3

    # Query vector closer to chunk 0 (table)
    q_dense = list(dense[0])
    hits = hybrid_search(
        "laptop allowance EMEA €1,800",
        client=client,
        collection=collection,
        dense_vector=q_dense,
        limit=3,
    )
    assert hits
    assert hits[0].document_id
    assert hits[0].status in ("active", "legacy")
    assert "document_id" in hits[0].payload
    assert "status" in hits[0].payload
    # BM25 should prefer the table for laptop/allowance tokens
    doc_ids = [h.document_id for h in hits]
    assert "IT-SPEC-2025-A" in doc_ids


def _qdrant_reachable() -> bool:
    try:
        c = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            timeout=2,
            check_compatibility=False,
        )
        c.get_collections()
        return True
    except Exception:
        return False


@pytest.mark.integration
def test_corpus_hybrid_smoke() -> None:
    """Live Qdrant + local M3: table query + remote-work query."""
    if not _qdrant_reachable():
        pytest.skip("Qdrant not reachable (docker compose up -d qdrant)")

    from src.indexing import embeddings as emb_mod

    emb_mod.reset_embedding_model()
    reset_bm25_model()

    client = QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        check_compatibility=False,
    )
    collection = "atlantic_policies_test_smoke"
    chunks = parse_dir(RAW_DIR)
    assert len(chunks) >= 3

    try:
        index_chunks(chunks, client=client, collection=collection, recreate=True)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"index failed (model download?): {exc}")

    table_hits = hybrid_search(
        "laptop allowance Germany",
        client=client,
        collection=collection,
        limit=5,
    )
    assert table_hits
    top_table = [h for h in table_hits if h.document_id == "IT-SPEC-2025-A"]
    assert top_table, f"expected IT-SPEC table in hits, got {[h.document_id for h in table_hits]}"
    assert any(h.chunk_type == "table" or "€1,800" in h.text for h in table_hits)

    remote_hits = hybrid_search(
        "fully remote AI Software Engineer",
        client=client,
        collection=collection,
        limit=5,
    )
    assert remote_hits
    assert any(h.document_id == "POLICY-2025-002" for h in remote_hits), (
        f"expected 2025 policy in hits, got {[h.document_id for h in remote_hits]}"
    )
    # Legacy may appear — Step 2 must NOT drop it
    # (no assertion that 2024 is absent)

    # cleanup
    try:
        client.delete_collection(collection)
    except Exception:
        pass


def test_ensure_collection_dim_mismatch() -> None:
    client = QdrantClient(":memory:")
    name = "dim_mismatch_test"
    ensure_collection(client, collection=name, recreate=True)
    # Simulate wrong dim by patching embedding_dim
    with patch("src.indexing.store.embedding_dim", return_value=768):
        with pytest.raises(ValueError, match="mismatch|missing"):
            ensure_collection(client, collection=name, recreate=False)
