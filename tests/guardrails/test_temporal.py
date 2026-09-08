"""Unit tests for strict A+D temporal drop (Option A)."""

from __future__ import annotations

from src.guardrails.temporal import drop_superseded
from src.indexing.models import Hit


def _hit(document_id: str, *, score: float = 0.5, text: str = "", status: str = "active") -> Hit:
    return Hit(
        score=score,
        document_id=document_id,
        section="1. Workplace Flexibility",
        chunk_id=f"{document_id}:sec:0",
        status=status,
        text=text,
        payload={"document_id": document_id, "status": status, "supersedes": []},
    )


def test_drop_superseded_removes_2024_when_2025_present() -> None:
    hits = [
        _hit("POLICY-2024-001", status="legacy", text="two (2) days per week"),
        _hit("POLICY-2025-002", status="active", text="Fully remote allowed"),
        _hit("IT-SPEC-2025-A", status="active", text="€1,800"),
    ]
    kept = drop_superseded(hits, {"POLICY-2024-001"})
    ids = [h.document_id for h in kept]
    assert "POLICY-2024-001" not in ids
    assert "POLICY-2025-002" in ids
    assert "IT-SPEC-2025-A" in ids
    assert not any("two (2) days" in h.text for h in kept)


def test_drop_superseded_even_if_2025_absent_from_hits() -> None:
    """Strict A+D: index map says active supersedes 2024 → drop even alone."""
    hits = [
        _hit("POLICY-2024-001", status="legacy", text="two (2) days", score=0.9),
        _hit("IT-SPEC-2025-A", status="active", text="€1,800", score=0.4),
    ]
    kept = drop_superseded(hits, {"POLICY-2024-001"})
    assert [h.document_id for h in kept] == ["IT-SPEC-2025-A"]


def test_drop_superseded_noop_when_map_empty() -> None:
    hits = [_hit("POLICY-2024-001", status="legacy")]
    assert drop_superseded(hits, set()) == hits


def test_drop_superseded_preserves_order() -> None:
    hits = [
        _hit("IT-SPEC-2025-A", score=0.8),
        _hit("POLICY-2024-001", score=0.7, status="legacy"),
        _hit("POLICY-2025-002", score=0.6),
    ]
    kept = drop_superseded(hits, {"POLICY-2024-001"})
    assert [h.document_id for h in kept] == ["IT-SPEC-2025-A", "POLICY-2025-002"]
