"""Refusal threshold helpers."""

from __future__ import annotations

from src.guardrails.refuse import should_refuse_retrieval
from src.indexing.models import Hit


def test_refuse_empty_hits() -> None:
    assert should_refuse_retrieval([]) is True


def test_refuse_below_threshold() -> None:
    hits = [
        Hit(
            score=0.01,
            document_id="POLICY-2025-002",
            section="x",
            chunk_id="x",
            text="t",
        )
    ]
    assert should_refuse_retrieval(hits, threshold=0.02) is True


def test_accept_at_or_above_threshold() -> None:
    hits = [
        Hit(
            score=0.05,
            document_id="POLICY-2025-002",
            section="x",
            chunk_id="x",
            text="t",
        )
    ]
    assert should_refuse_retrieval(hits, threshold=0.02) is False
