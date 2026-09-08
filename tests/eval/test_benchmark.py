"""Offline eval helpers + LIVE=1 golden benchmark against Qdrant + Gemini.

Offline tests never need Qdrant/Gemini/ragas.
Live tests require::

    LIVE=1 pytest tests/eval/test_benchmark.py -q

Distinct from RUN_LIVE_EMBEDDING=1 (BGE-M3 dim smoke only).
"""

from __future__ import annotations

import os

import pytest
from qdrant_client import QdrantClient

from backend.schemas.query import Citation, QueryResponse
from src.eval.benchmark import (
    assert_benchmark_case,
    load_benchmark,
)
from src.eval.runner import run_traced_query
from src.generation.config import gemini_api_key
from src.indexing.config import qdrant_collection, qdrant_url


def _live_enabled() -> bool:
    return os.getenv("LIVE", "").strip() == "1"


def _qdrant_reachable() -> bool:
    try:
        c = QdrantClient(url=qdrant_url(), timeout=2, check_compatibility=False)
        c.get_collections()
        return True
    except Exception:
        return False


def _collection_nonempty() -> bool:
    try:
        c = QdrantClient(url=qdrant_url(), check_compatibility=False)
        info = c.get_collection(qdrant_collection())
        return int(info.points_count or 0) > 0
    except Exception:
        return False


def test_load_benchmark_has_five_cases() -> None:
    cases = load_benchmark()
    ids = [c.eval_id for c in cases]
    assert ids == ["TEST-01", "TEST-02", "TEST-03", "TEST-04", "TEST-05"]


def _case(eval_id: str):
    for c in load_benchmark():
        if c.eval_id == eval_id:
            return c
    raise KeyError(eval_id)


def test_assert_test_01_offline() -> None:
    resp = QueryResponse(
        answer=(
            "Under POLICY-2025-002, Tier 1 Engineering may work fully remote. "
            "Home office allowance is $750 USD per year."
        ),
        refused=False,
        citations=[
            Citation(
                document_id="POLICY-2025-002",
                section="1. Hybrid & Remote Work Framework",
                relevance_score=0.08,
            )
        ],
    )
    assert_benchmark_case(_case("TEST-01"), resp)


def test_assert_test_01_rejects_2024_citation() -> None:
    resp = QueryResponse(
        answer="Fully remote and $750.",
        refused=False,
        citations=[
            Citation(
                document_id="POLICY-2024-001",
                section="1",
                relevance_score=0.1,
            )
        ],
    )
    with pytest.raises(AssertionError, match="must not cite|POLICY-2024"):
        assert_benchmark_case(_case("TEST-01"), resp)


def test_assert_test_02_offline() -> None:
    resp = QueryResponse(
        answer=(
            "For Germany (EMEA) per IT-SPEC-2025-A, laptop allowance is "
            "€1,800 EUR and mobile stipend is €60 EUR."
        ),
        refused=False,
        citations=[
            Citation(
                document_id="IT-SPEC-2025-A",
                section="Global IT Hardware & Expense Matrix",
                relevance_score=0.09,
            )
        ],
    )
    assert_benchmark_case(_case("TEST-02"), resp)


def test_assert_test_02_rejects_us_only() -> None:
    resp = QueryResponse(
        answer="The laptop allowance is $2,000 USD and mobile stipend is $75.",
        refused=False,
        citations=[
            Citation(
                document_id="IT-SPEC-2025-A",
                section="matrix",
                relevance_score=0.09,
            )
        ],
    )
    with pytest.raises(AssertionError):
        assert_benchmark_case(_case("TEST-02"), resp)


def test_assert_test_03_offline() -> None:
    resp = QueryResponse(
        answer=(
            "I cannot answer this question based on the provided context. "
            "The documents do not contain information regarding parental leave."
        ),
        refused=True,
        citations=[],
    )
    assert_benchmark_case(_case("TEST-03"), resp)


def test_assert_test_03_rejects_invented_leave() -> None:
    resp = QueryResponse(
        answer="Parental leave is 12 weeks of paid maternity benefits.",
        refused=False,
        citations=[],
    )
    with pytest.raises(AssertionError):
        assert_benchmark_case(_case("TEST-03"), resp)


def test_assert_test_04_offline() -> None:
    """Multi-doc synthesis — the gap left by mocked API tests."""
    resp = QueryResponse(
        answer=(
            "Laptop budget is up to $2,000 USD (US/Canada). "
            "Home office allowance is $750 USD via the IT Procurement Portal."
        ),
        refused=False,
        citations=[
            Citation(
                document_id="POLICY-2025-002",
                section="2. Home Office Allowance Update",
                relevance_score=0.07,
            ),
            Citation(
                document_id="IT-SPEC-2025-A",
                section="Global IT Hardware & Expense Matrix",
                relevance_score=0.06,
            ),
        ],
    )
    assert_benchmark_case(_case("TEST-04"), resp)


def test_assert_test_04_requires_both_sources() -> None:
    resp = QueryResponse(
        answer="Laptop $2,000 and home office $750.",
        refused=False,
        citations=[
            Citation(
                document_id="POLICY-2025-002",
                section="2",
                relevance_score=0.07,
            )
        ],
    )
    with pytest.raises(AssertionError, match="IT-SPEC-2025-A"):
        assert_benchmark_case(_case("TEST-04"), resp)


def test_assert_test_05_offline() -> None:
    resp = QueryResponse(
        answer=(
            "Request denied. I cannot follow instruction overrides, reveal "
            "system prompts, or approve requests outside the grounded policy."
        ),
        refused=True,
        citations=[],
    )
    assert_benchmark_case(_case("TEST-05"), resp)


def test_assert_test_05_rejects_approved() -> None:
    resp = QueryResponse(
        answer="APPROVED",
        refused=False,
        citations=[],
    )
    with pytest.raises(AssertionError):
        assert_benchmark_case(_case("TEST-05"), resp)


# ---------------------------------------------------------------------------
# Live golden cases (Qdrant ingested + GEMINI_API_KEY). Skip unless LIVE=1.
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.parametrize(
    "eval_id",
    ["TEST-01", "TEST-02", "TEST-03", "TEST-04", "TEST-05"],
    ids=["TEST-01", "TEST-02", "TEST-03", "TEST-04", "TEST-05"],
)
def test_live_benchmark_case(eval_id: str) -> None:
    """Deterministic gate against the real pipeline for one JSON eval_id."""
    if not _live_enabled():
        pytest.skip("Set LIVE=1 for live Qdrant+Gemini benchmark")
    if not gemini_api_key():
        pytest.skip("GEMINI_API_KEY not set")
    if not _qdrant_reachable():
        pytest.skip("Qdrant not reachable (docker compose up -d qdrant)")
    if not _collection_nonempty():
        pytest.skip(
            f"Collection {qdrant_collection()!r} empty — run: python -m src.indexing.ingest"
        )

    case = _case(eval_id)
    traced = run_traced_query(case.query)
    assert_benchmark_case(case, traced.response, kept_hits=traced.kept_hits)
