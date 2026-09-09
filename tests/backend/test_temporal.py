"""Assembler must not leak superseded 2024 text after Option A drop."""

from __future__ import annotations

from src.generation.prompts import assemble_context
from src.guardrails.temporal import drop_superseded
from backend.services.rag import take_context
from src.indexing.models import Hit


def test_assembler_excludes_2024_after_drop() -> None:
    hits = [
        Hit(
            score=0.9,
            document_id="POLICY-2024-001",
            section="1. Workplace Flexibility",
            chunk_id="POLICY-2024-001:1:0",
            status="legacy",
            text=(
                "Document POLICY-2024-001 (Legacy) | Section: 1. Workplace Flexibility\n"
                "Employees may work remotely for up to two (2) days per week."
            ),
        ),
        Hit(
            score=0.85,
            document_id="POLICY-2025-002",
            section="1. Hybrid & Remote Work Framework",
            chunk_id="POLICY-2025-002:1:0",
            status="active",
            text=(
                "Document POLICY-2025-002 (Active) | Section: 1. Hybrid & Remote Work Framework\n"
                "Tier 1 Roles (Engineering): Fully remote allowed."
            ),
            payload={"supersedes": ["POLICY-2024-001"]},
        ),
    ]
    kept = drop_superseded(hits, {"POLICY-2024-001"})
    context = assemble_context(kept)
    assert "POLICY-2024-001" not in context
    assert "two (2) days" not in context
    assert "POLICY-2025-002" in context
    assert "Fully remote" in context
    assert '<document id="POLICY-2025-002"' in context


def test_take_context_keeps_top_three_after_drop() -> None:
    hits = [
        Hit(score=0.09, document_id="POLICY-2024-001", section="1", chunk_id="old", status="legacy", text="two days"),
        Hit(score=0.08, document_id="POLICY-2025-002", section="1", chunk_id="a", text="fully remote"),
        Hit(score=0.07, document_id="POLICY-2025-002", section="2", chunk_id="b", text="$750"),
        Hit(score=0.06, document_id="IT-SPEC-2025-A", section="t", chunk_id="c", text="€1,800"),
        Hit(score=0.05, document_id="IT-SPEC-2025-A", section="n", chunk_id="d", text="VP approval"),
    ]
    kept = take_context(drop_superseded(hits, {"POLICY-2024-001"}), limit=3)
    assert [h.chunk_id for h in kept] == ["a", "b", "c"]
    assert all(h.document_id != "POLICY-2024-001" for h in kept)
